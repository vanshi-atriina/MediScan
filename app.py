import streamlit as st
import google.generativeai as genai
from PIL import Image
import os
from dotenv import load_dotenv
import pdf2image
import cv2
import numpy as np
import pandas as pd
import logging
from datetime import datetime
import re
import base64
import io
import tempfile
from functools import lru_cache

# Configure logging
logging.basicConfig(filename='ocr_errors.log', level=logging.ERROR)

# Load environment variables
load_dotenv()

# Configure Gemini API
@st.cache_resource
def configure_gemini():
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        st.error("Please set your GEMINI_API_KEY in the .env file")
        st.stop()
    
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.5-pro')

# Cache PDF conversion to avoid repeated processing
@st.cache_data
def convert_pdf_to_images_cached(pdf_bytes):
    """Convert PDF bytes to images with caching"""
    try:
        # Use temporary file for better memory management
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_bytes)
            tmp_file.flush()
            
            # Convert with optimized settings
            images = pdf2image.convert_from_path(
                tmp_file.name,
                dpi=200,  # Reduced DPI for better performance
                first_page=1,
                last_page=10,  # Limit to first 10 pages for demo
                poppler_path=None,
                thread_count=2  # Limit threads to avoid memory issues
            )
            
            # Clean up temp file
            os.unlink(tmp_file.name)
            
            return images
    except Exception as e:
        st.error(f"Error converting PDF: {str(e)}")
        logging.error(f"PDF conversion error: {str(e)}")
        return []

# Cache image preprocessing
@st.cache_data
def preprocess_image_cached(image_bytes):
    """Preprocess image with caching"""
    try:
        # Load image from bytes
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            if image.mode == 'RGBA':
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])
                image = background
            else:
                image = image.convert('RGB')
        
        # Resize if too large (max 2000px on longest side)
        max_size = 2000
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array for processing
        img_array = np.array(image)
        
        # Enhance contrast (lighter processing)
        img_array = cv2.convertScaleAbs(img_array, alpha=1.2, beta=10)
        
        # Convert back to PIL Image
        processed_image = Image.fromarray(img_array)
        
        # Convert to bytes for caching
        img_byte_arr = io.BytesIO()
        processed_image.save(img_byte_arr, format='JPEG', quality=85)
        return img_byte_arr.getvalue()
        
    except Exception as e:
        logging.error(f"Image preprocessing error: {str(e)}")
        st.error(f"Error preprocessing image: {str(e)}")
        return None

# Clean and format extracted text
def clean_extracted_text(text):
    """Clean the extracted text by removing HTML tags and fixing formatting issues"""
    if not text:
        return text
    
    # Replace HTML line breaks with actual line breaks
    text = text.replace('<br>', '\n').replace('<br/>', '\n').replace('<br />', '\n')
    
    # Remove other HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Convert markdown-style headers to plain text headers
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
    
    # Convert markdown-style bold/italic to plain text
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'__(.*?)__', r'\1', text)
    text = re.sub(r'_(.*?)_', r'\1', text)
    
    # Clean up markdown table formatting
    text = re.sub(r'\|', ' | ', text)
    text = re.sub(r'\s*\|\s*', ' | ', text)
    
    # Remove markdown list markers but keep the content
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    
    # Clean up excessive whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    
    # Clean up table separators
    text = re.sub(r'\n\s*[-|=\s]+\n', '\n', text)
    
    return text.strip()

# Extract text from image using Gemini API with caching
@st.cache_data
def extract_text_from_image_cached(image_bytes):
    """Extract text from image with caching"""
    try:
        # Load image from bytes
        image = Image.open(io.BytesIO(image_bytes))
        
        # Initialize model (this should be fast as it's cached)
        model = configure_gemini()
        
        prompt = """
You are an expert medical document analyst specializing in transcribing handwritten and printed clinical notes, prescriptions, and lab reports. Your goal is to accurately extract and format all textual information from the provided image, even if the handwriting is highly illegible or the document is poorly scanned. Note that several fields containing personal identifiable information (PII) may be redacted with a black marker. The handwriting in some areas may be difficult to read, and interpretations should be noted with `(unsure)` or `(illegible)`. Follow these instructions:

1. **Image Orientation Correction**: Assess the image orientation. If it is tilted, rotated, or not upright, virtually correct it to an upright position before text analysis to ensure accuracy.
2. **Robust Text Extraction with Confidence Scoring**: Extract all discernible text. For terms or phrases with low confidence due to illegibility, append `(unsure)` immediately after the term. For completely legible sections, note `(illegible)` with an estimated word count if possible (e.g., `(illegible, ~3 words)`).
3. **Medical Contextual Analysis and Autofill**: Leverage your medical knowledge to interpret and correct text. If a term is partially legible but highly probable in a medical context (e.g., "Aspir..." likely means "Aspirin"), autofill the term and note `(corrected)` after it. Avoid autofilling if ambiguity is high, using `(unsure)` instead.
4. **Structure Preservation and Formatting**:
   - **Tabular Data**: If the image contains tabular data (e.g., lab results, medication lists), format it as plain text with clear headers and rows, using proper spacing and alignment. Use simple text formatting without HTML or markdown.
   - **Lists and Headings**: For non-tabular data, preserve lists and headings using plain text formatting with clear line breaks.
   - **Free-Form Notes**: Transcribe free-form text with clear paragraph breaks and proper capitalization.
5. **Medical Terminology and Abbreviations**: Preserve medical terms, abbreviations (e.g., "mg", "BP", "Rx"), and drug names exactly as written, unless clearly erroneous, in which case correct and note `(corrected)`.
6. **Artifact Removal**: Ignore non-relevant artifacts (e.g., page numbers, watermarks, or scanner marks) unless they contain medical information (e.g., a hospital logo with a date).
7. **Output Validation**: If the extracted text is incomplete or unclear, append a note summarizing potential issues (e.g., "Note: Lower right corner illegible due to scan quality").

IMPORTANT: Return the cleaned and formatted text in PLAIN TEXT format only. Do not use HTML tags (like <br>), markdown formatting (like ** or # or *), or any other markup. Use only plain text with line breaks for structure. Ensure medical accuracy, clarity, and readability.
"""
        response = model.generate_content([prompt, image])
        cleaned_text = clean_extracted_text(response.text)
        return cleaned_text
    except Exception as e:
        logging.error(f"Error extracting text: {str(e)}")
        st.error(f"Error extracting text: {str(e)}. Check ocr_errors.log for details.")
        return None

# Lightweight PDF preview using first page only
def show_pdf_preview_light(pdf_bytes, title):
    """Show a lightweight PDF preview using only the first page"""
    try:
        # Convert only first page to image for preview
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_bytes)
            tmp_file.flush()
            
            # Convert only first page at lower DPI for preview
            images = pdf2image.convert_from_path(
                tmp_file.name,
                dpi=150,  # Lower DPI for preview
                first_page=1,
                last_page=1  # Only first page
            )
            
            os.unlink(tmp_file.name)
            
            if images:
                st.image(images[0], caption=f"Preview: {title} (Page 1)", use_container_width=True)
            else:
                st.error("Could not generate PDF preview")
                
    except Exception as e:
        st.error(f"Error generating PDF preview: {str(e)}")

# Create popup modal for text preview
def show_text_popup(text, title, key):
    with st.container():
        st.markdown(f"### {title}")
        st.text_area("", value=text, height=400, key=f"popup_{key}")
        if st.button("Close", key=f"close_{key}"):
            st.session_state.show_popup = None
            st.rerun()

# Main application
def main():
    st.set_page_config(
        page_title="Gemini OCR - Medical Text Extractor",
        page_icon="🔍",
        layout="wide"
    )
    
    # CSS styling (keeping your existing styles)
    st.markdown("""
    <style>
    .stApp {
        background-color: #ffffff;
    }
    .main-header {
        color: #1a237e !important;
        text-align: center;
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 15px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .sub-header {
        color: #303f9f !important;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 500;
        margin-bottom: 30px;
    }
    .section-header {
        color: #1a237e !important;
        font-size: 1.6rem !important;
        font-weight: 600 !important;
        margin-bottom: 20px !important;
        margin-top: 10px !important;
        padding: 10px 0 !important;
        border-bottom: 3px solid #3f51b5 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .info-box {
        background-color: #e8eaf6;
        border-left: 5px solid #3f51b5;
        padding: 20px;
        margin: 20px 0;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .success-box {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
        padding: 20px;
        margin: 20px 0;
        border-radius: 8px;
        color: #1b5e20 !important;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .performance-tip {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 15px;
        margin: 15px 0;
        border-radius: 8px;
        color: #e65100 !important;
        font-weight: 500;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🏥 MediScan</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Extract and enhance text from medical documents.</p>', unsafe_allow_html=True)
    
    # Performance tip
    st.markdown("""
    <div class="performance-tip">
    ⚡ <strong>Performance Tips:</strong><br>
    • Images are processed faster than PDFs<br>
    • PDFs are limited to first 10 pages for optimal performance<br>
    • Large images are automatically resized<br>
    • Results are cached to avoid reprocessing
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'extracted_texts' not in st.session_state:
        st.session_state.extracted_texts = {}
    if 'show_popup' not in st.session_state:
        st.session_state.show_popup = None
    if 'file_data' not in st.session_state:
        st.session_state.file_data = None
    
    # Centered upload section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<h3 class="section-header">📁 Upload Files</h3>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose image or PDF file",
            type=['png', 'jpg', 'jpeg', 'pdf'],
            accept_multiple_files=False,
            help="Upload a medical document for text extraction"
        )
    
    # Process uploaded file
    if uploaded_file is not None:
        # Store file data in session state to avoid re-reading
        file_bytes = uploaded_file.read()
        st.session_state.file_data = {
            'name': uploaded_file.name,
            'type': uploaded_file.type,
            'bytes': file_bytes,
            'size': len(file_bytes)
        }
        
        # Show file info
        file_size_mb = len(file_bytes) / (1024 * 1024)
        st.info(f"📄 Loaded: {uploaded_file.name} ({file_size_mb:.1f} MB)")
        
        # STRUCTURE 1: PREVIEW SECTION
        st.markdown("---")
        st.markdown('<h3 class="section-header">👁️ Preview Document</h3>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("📄 Preview Document", key="preview_button", use_container_width=True):
                with st.spinner("Generating preview..."):
                    if uploaded_file.type == "application/pdf":
                        show_pdf_preview_light(file_bytes, uploaded_file.name)
                    else:
                        # For images, show directly
                        image = Image.open(io.BytesIO(file_bytes))
                        st.image(image, caption=f"Preview: {uploaded_file.name}", use_container_width=True)
        
        # STRUCTURE 2: EXTRACTION SECTION
        st.markdown("---")
        st.markdown('<h3 class="section-header">🔍 Extract Text</h3>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Extract Text", type="primary", key="extract_button", use_container_width=True):
                st.session_state.extracted_texts = {}
                
                with st.spinner("Processing document..."):
                    progress_bar = st.progress(0)
                    file_name = uploaded_file.name.split('.')[0]
                    
                    if uploaded_file.type == "application/pdf":
                        # Process PDF
                        progress_bar.progress(0.1)
                        st.info("Converting PDF to images...")
                        
                        images = convert_pdf_to_images_cached(file_bytes)
                        if not images:
                            st.error("Failed to convert PDF to images")
                            return
                        
                        progress_bar.progress(0.3)
                        st.info(f"Processing {len(images)} pages...")
                        
                        file_texts = []
                        for page_idx, image in enumerate(images):
                            # Convert image to bytes for caching
                            img_byte_arr = io.BytesIO()
                            image.save(img_byte_arr, format='JPEG', quality=85)
                            img_bytes = img_byte_arr.getvalue()
                            
                            # Preprocess image
                            processed_img_bytes = preprocess_image_cached(img_bytes)
                            if not processed_img_bytes:
                                continue
                            
                            # Extract text
                            extracted_text = extract_text_from_image_cached(processed_img_bytes)
                            if extracted_text:
                                file_texts.append({
                                    'page': page_idx + 1,
                                    'text': extracted_text
                                })
                            
                            progress_bar.progress(0.3 + (0.6 * (page_idx + 1) / len(images)))
                        
                        st.session_state.extracted_texts[file_name] = {
                            'type': 'pdf',
                            'pages': file_texts
                        }
                        
                    else:
                        # Process single image
                        progress_bar.progress(0.2)
                        st.info("Preprocessing image...")
                        
                        processed_img_bytes = preprocess_image_cached(file_bytes)
                        if not processed_img_bytes:
                            st.error("Failed to preprocess image")
                            return
                        
                        progress_bar.progress(0.5)
                        st.info("Extracting text...")
                        
                        extracted_text = extract_text_from_image_cached(processed_img_bytes)
                        if extracted_text:
                            st.session_state.extracted_texts[file_name] = {
                                'type': 'image',
                                'text': extracted_text
                            }
                        
                        progress_bar.progress(1.0)
                    
                    st.markdown('<div class="success-box">✅ Text extraction completed successfully!</div>', unsafe_allow_html=True)
    
    # Show text popup if requested
    if st.session_state.show_popup:
        popup_data = st.session_state.show_popup
        show_text_popup(popup_data['text'], popup_data['title'], popup_data['key'])
    
    # STRUCTURE 3: RESULTS SECTION
    if st.session_state.extracted_texts:
        st.markdown("---")
        st.markdown('<h3 class="section-header">📊 Results</h3>', unsafe_allow_html=True)
        
        for file_name, data in st.session_state.extracted_texts.items():
            with st.expander(f"📄 {file_name}", expanded=True):
                if data['type'] == 'pdf':
                    st.write(f"**📖 Pages processed:** {len(data['pages'])}")
                    
                    # Preview buttons for each page
                    if len(data['pages']) > 0:
                        cols = st.columns(min(len(data['pages']), 5))
                        for i, page_data in enumerate(data['pages']):
                            with cols[i % 5]:
                                if st.button(f"👁️ Page {page_data['page']}", key=f"preview_{file_name}_{i}"):
                                    st.session_state.show_popup = {
                                        'text': page_data['text'],
                                        'title': f"{file_name} - Page {page_data['page']}",
                                        'key': f"{file_name}_{i}"
                                    }
                                    st.rerun()
                    
                    # Download all pages as single text file
                    if data['pages']:
                        all_text = ""
                        for page_data in data['pages']:
                            all_text += f"=== PAGE {page_data['page']} ===\n\n"
                            all_text += page_data['text'] + "\n\n"
                        
                        st.download_button(
                            label="📥 Download All Pages as TXT",
                            data=all_text,
                            file_name=f"{file_name}_all_pages.txt",
                            mime="text/plain",
                            key=f"download_all_{file_name}"
                        )
                        
                        # Statistics
                        total_words = sum(len(page_data['text'].split()) for page_data in data['pages'])
                        total_chars = sum(len(page_data['text']) for page_data in data['pages'])
                        st.markdown(f'<div class="info-box"><strong>📈 Statistics:</strong> {total_words} total words, {total_chars} total characters across {len(data["pages"])} pages</div>', unsafe_allow_html=True)
                
                else:  # Single image
                    col1, col2 = st.columns([1, 3])
                    with col2:
                        if st.button(f"👁️ Preview Text", key=f"preview_img_{file_name}"):
                            st.session_state.show_popup = {
                                'text': data['text'],
                                'title': f"{file_name} - Extracted Text",
                                'key': f"img_{file_name}"
                            }
                            st.rerun()
                        
                        # Download button
                        st.download_button(
                            label="📥 Download as TXT",
                            data=data['text'],
                            file_name=f"{file_name}_extracted.txt",
                            mime="text/plain",
                            key=f"download_{file_name}"
                        )
                        
                        # Statistics
                        word_count = len(data['text'].split())
                        char_count = len(data['text'])
                        st.markdown(f'<div class="info-box"><strong>📈 Statistics:</strong> {word_count} words, {char_count} characters</div>', unsafe_allow_html=True)
    
    elif not uploaded_file:
        st.markdown('<div class="info-box">📋 Please upload an image or PDF to extract text</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()