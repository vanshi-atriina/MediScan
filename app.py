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

# Configure logging
logging.basicConfig(filename='ocr_errors.log', level=logging.ERROR)

# Load environment variables
load_dotenv()

# Configure Gemini API
def configure_gemini():
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        st.error("Please set your GEMINI_API_KEY in the .env file")
        st.stop()
    
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.5-pro')

# Update the show_unified_preview function (change use_column_width to use_container_width)
def show_unified_preview(content, title, key):
    try:
        with st.container():
            st.markdown(f"### {title}")
            
            if content['type'] == 'pdf':
                # Convert PDF bytes to images for preview
                try:
                    images = pdf2image.convert_from_bytes(content['pdf_bytes'])
                    for i, img in enumerate(images):
                        st.image(img, caption=f"Page {i+1}", use_container_width=True)
                except Exception as e:
                    st.error(f"Error converting PDF for preview: {str(e)}")
                    logging.error(f"PDF preview conversion error: {str(e)}")
                    return
                
                # Provide a download button for the original PDF
                st.download_button(
                    label="📥 Download Original PDF",
                    data=content['pdf_bytes'],
                    file_name=f"{title}.pdf",
                    mime="application/pdf",
                    key=f"download_pdf_{key}"
                )
            else:
                # Display image preview (unchanged)
                st.image(content['image'], caption=title, use_container_width=True)
            
            if st.button("Close", key=f"close_{key}"):
                st.session_state.show_unified_preview = None
                st.rerun()
    except Exception as e:
        st.error(f"Error displaying preview: {str(e)}")
        logging.error(f"Preview error: {str(e)}")

# Image preprocessing (always enabled)
def preprocess_image(image):
    try:
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        
        # Handle different image modes
        if image.mode == 'RGBA':
            # Create a white background and paste the RGBA image onto it
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])  # Paste using alpha channel as mask
            img_array = np.array(background)
        elif image.mode == 'P':
            # Convert palette-based images to RGB
            img_array = np.array(image.convert('RGB'))
        elif image.mode == 'LA':
            # Convert grayscale + alpha to RGB
            img_array = np.array(image.convert('RGB'))
        elif image.mode != 'RGB':
            # Convert other modes (like L for grayscale) to RGB
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        
        # Enhance contrast
        img_array = cv2.convertScaleAbs(img_array, alpha=1.5, beta=0)
        return Image.fromarray(img_array)
    except Exception as e:
        logging.error(f"Image preprocessing error: {str(e)}")
        st.error(f"Error preprocessing image: {str(e)}")
        return None

# Convert PDF to images
def convert_pdf_to_images(uploaded_file):
    try:
        images = pdf2image.convert_from_bytes(uploaded_file.read())
        return images
    except Exception as e:
        st.error(f"Error converting PDF: {str(e)}")
        logging.error(f"PDF conversion error: {str(e)}")
        return []

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

# Extract text from image using Gemini API
def extract_text_from_image(model, image):
    try:
        prompt = """
You are an expert medical document analyst specializing in transcribing handwritten and printed clinical notes, prescriptions, and lab reports. Your goal is to accurately extract and format all textual information from the provided image, even if the handwriting is highly illegible or the document is poorly scanned. Note that several fields containing personal identifiable information (PII) may be redacted with a black marker. The handwriting in some areas may be difficult to read, and interpretations should be noted with `(unsure)` or `(illegible)`. Follow these instructions:

1. **Image Orientation Correction**: Assess the image orientation. If it is tilted, rotated, or not upright, virtually correct it to an upright position before text analysis to ensure accuracy.
2. **Robust Text Extraction with Confidence Scoring**: Extract all discernible text. For terms or phrases with low confidence due to illegibility, append `(unsure)` immediately after the term. For completely legible sections, note `(illegible)` with an estimated word count if possible (e.g., `(illegible, ~3 words)`).
3. **Medical Contextual Analysis and Autofill**: Leverage your medical knowledge to interpret and correct text. If a term is partially legible but highly probable in a medical context (e.g., "Aspir..." likely means "Aspirin"), autofill the term and note `(corrected)` after it. Avoid autofilling if ambiguity is high, using `(unsure)` instead.
4. **Structure Preservation and Formatting**:
   - **Tabular Data**: If the image contains tabular data (e.g., lab results, medication lists), format it as plain text with clear headers and rows, using proper spacing and alignment. Use simple text formatting without HTML or markdown.
   - **Lists and Headings**: For non-tabular data, preserve lists and headings using plain text formatting with clear line breaks.
   - **Free-Form Notes**: Transcribe free-form textWITH clear paragraph breaks and proper capitalization.
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

# Create popup modal for text preview
def show_text_popup(text, title, key):
    with st.container():
        st.markdown(f"### {title}")
        st.text_area("", value=text, height=400, key=f"popup_{key}")
        if st.button("Close", key=f"close_{key}"):
            st.session_state.show_popup = None
            st.rerun()

# Create popup modal for PDF preview
def show_pdf_popup(pdf_bytes, title, key):
    with st.container():
        st.markdown(f"### {title}")
        
        # Display PDF preview with centered styling
        base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
        pdf_display = f"""
        <div style="display: flex; justify-content: center;">
            <embed src="data:application/pdf;base64,{base64_pdf}" 
                   width="700" 
                   height="1000" 
                   type="application/pdf"
                   style="margin: 0 auto;">
        </div>
        """
        st.markdown(pdf_display, unsafe_allow_html=True)
        
        if st.button("Close", key=f"close_pdf_{key}"):
            st.session_state.show_pdf_popup = None
            st.rerun()

# Main application
def main():
    st.set_page_config(
        page_title="Gemini OCR - Medical Text Extractor",
        page_icon="🔍",
        layout="wide"
    )
    
    # CSS styling
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
    .info-box strong {
        color: #1a237e !important;
        font-weight: 600;
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
    .stButton > button {
        background: #d5eaed !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 8px rgba(63, 81, 181, 0.3) !important;
    }
    .stButton > button:hover {
        background: #d5eaed !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 12px rgba(63, 81, 181, 0.4) !important;
    }
    .stDownloadButton > button {
        background: #d5eaed !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 3px 6px rgba(33, 150, 243, 0.3) !important;
    }
    .stDownloadButton > button:hover {
        background: #d5eaed !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 5px 10px rgba(33, 150, 243, 0.4) !important;
    }
    .streamlit-expanderHeader {
        background-color: #f5f5f5 !important;
        border: 2px solid #e0e0e0 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        color: #1a237e !important;
        font-size: 1.1rem !important;
    }
    .streamlit-expanderContent {
        border: 1px solid #e0e0e0 !important;
        border-radius: 0 0 8px 8px !important;
        background-color: #fafafa !important;
        padding: 20px !important;
    }
    .stProgress > div > div {
        background-color: #3f51b5 !important;
    }
    .stSpinner > div {
        border-top-color: #3f51b5 !important;
    }
    .stTextArea textarea {
        background-color: #ffffff !important;
        border: 2px solid #e0e0e0 !important;
        border-radius: 8px !important;
        color: #212121 !important;
        font-family: 'Courier New', monospace !important;
        font-size: 14px !important;
        line-height: 1.5 !important;
    }
    .stImage {
        border-radius: 8px !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: #1a237e !important;
    }
    .stMarkdown span, .stMarkdown div {
        color: #212121 !important;
    }
    .stMarkdown p {
        color: #212121 !important;
    }
    * {
        color: #212121 !important;
    }
    .main-header, .sub-header, .section-header {
        color: #1a237e !important;
    }
    .success-box {
        color: #1b5e20 !important;
    }
    .info-box strong {
        color: #1a237e !important;
    }
    hr {
        border: none !important;
        height: 2px !important;
        background: linear-gradient(to right, #3f51b5, #9c27b0) !important;
        margin: 30px 0 !important;
    }
    [data-testid="stFileUploader"] > div,
    [data-testid="stFileUploader"] > div > div {
        background: #f0f2f6 !important;
        background-color: #f0f2f6 !important;
        border: none !important;
        padding: 10px !important;
        box-shadow: none !important;
        border-radius: 4px !important;
    }
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] .uploadInstructions,
    [data-testid="stFileUploader"] .small,
    [data-testid="stFileUploader"] .uploadedFileName,
    [data-testid="stFileUploader"] .st-emotion-cache-1aehpvj {
        color: #606060 !important;
        opacity: 1 !important;
        font-family: inherit !important;
    }
    [data-testid="stFileUploader"] button {
        background: #ffffff !important;
        background-color: #ffffff !important;
        color: #606060 !important;
        border: 1px solid #d0d4d9 !important;
        padding: 5px 10px !important;
        box-shadow: none !important;
        border-radius: 4px !important;
    }
    [data-testid="stFileUploader"]:hover,
    [data-testid="stFileUploader"] button:hover,
    [data-testid="stFileUploader"] > div > div:hover {
        background: #e6e8eb !important;
        background-color: #e6e8eb !important;
        border-color: #b0b8c1 !important;
        color: #404040 !important;
        box-shadow: none !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🏥 MediScan</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Extract and enhance text from medical documents.</p>', unsafe_allow_html=True)
    
    # Initialize Gemini model
    model = configure_gemini()
    
    # Initialize session state
    if 'extracted_texts' not in st.session_state:
        st.session_state.extracted_texts = {}
    if 'show_popup' not in st.session_state:
        st.session_state.show_popup = None
    if 'show_pdf_popup' not in st.session_state:
        st.session_state.show_pdf_popup = None
    if 'show_unified_preview' not in st.session_state:
        st.session_state.show_unified_preview = None
    
    # Centered upload section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<h3 class="section-header">📁 Upload Files</h3>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Choose image or PDF file. Upload one at a time.",
            type=['png', 'jpg', 'jpeg', 'pdf'],
            accept_multiple_files=False,
            help="Upload a medical document for text extraction"
        )
    
    # STRUCTURE 1: PREVIEW BUTTON (independent of extraction)
    if uploaded_files:
        st.markdown("---")
        st.markdown('<h3 class="section-header">👁️ Preview Document</h3>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("📄 Preview Document", key="preview_button", use_container_width=True):
                try:
                    if uploaded_files.type == "application/pdf":
                        pdf_bytes = uploaded_files.read()
                        if not pdf_bytes:
                            st.error("Error: Uploaded PDF is empty.")
                            logging.error(f"Empty PDF file uploaded: {uploaded_files.name}")
                            return
                        
                        st.session_state.show_unified_preview = {
                            'type': 'pdf',
                            'pdf_bytes': pdf_bytes,
                            'title': f"Document Preview: {uploaded_files.name}",
                            'key': "doc_preview"
                        }
                    else:
                        try:
                            image = Image.open(uploaded_files)
                            if image.mode != 'RGB':
                                if image.mode == 'RGBA':
                                    background = Image.new('RGB', image.size, (255, 255, 255))
                                    background.paste(image, mask=image.split()[-1])
                                    image = background
                                else:
                                    image = image.convert('RGB')
                            
                            st.session_state.show_unified_preview = {
                                'type': 'image',
                                'image': image,
                                'title': f"Image Preview: {uploaded_files.name}",
                                'key': "img_preview"
                            }
                        except Exception as e:
                            st.error(f"Error loading image: {str(e)}")
                            logging.error(f"Image loading error: {str(e)}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error preparing preview: {str(e)}")
                    logging.error(f"Preview preparation error: {str(e)}")
    
    # STRUCTURE 2: PREVIEW POPUP (shown when requested)
    if st.session_state.get('show_unified_preview'):
        show_unified_preview(
            st.session_state.show_unified_preview,
            st.session_state.show_unified_preview['title'],
            st.session_state.show_unified_preview['key']
        )
    
    # STRUCTURE 3: EXTRACTION BUTTON (works independently of preview)
    if uploaded_files:
        st.markdown("---")
        st.markdown('<h3 class="section-header">🔍 Extract Text</h3>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Extract Text", type="primary", key="extract_button", use_container_width=True):
                st.session_state.extracted_texts = {}
                
                with st.spinner("Processing file..."):
                    progress_bar = st.progress(0)
                    file = uploaded_files  # Single file
                    file_name = file.name.split('.')[0]
                    
                    # Handle PDF or image
                    if file.type == "application/pdf":
                        images = convert_pdf_to_images(file)
                        file_texts = []
                        
                        for page_idx, image in enumerate(images):
                            # Preprocess image
                            processed_image = preprocess_image(image)
                            if not processed_image:
                                continue
                            
                            # Extract text
                            extracted_text = extract_text_from_image(model, processed_image)
                            if extracted_text:
                                file_texts.append({
                                    'page': page_idx + 1,
                                    'text': extracted_text,
                                    'image': image
                                })
                            
                            progress_bar.progress((page_idx + 1) / len(images))
                        
                        st.session_state.extracted_texts[file_name] = {
                            'type': 'pdf',
                            'pages': file_texts
                        }
                    else:
                        try:
                            image = Image.open(file)
                            # Ensure image is in RGB mode before preprocessing
                            if image.mode != 'RGB':
                                if image.mode == 'RGBA':
                                    background = Image.new('RGB', image.size, (255, 255, 255))
                                    background.paste(image, mask=image.split()[-1])
                                    image = background
                                else:
                                    image = image.convert('RGB')
                            
                            processed_image = preprocess_image(image)
                            if not processed_image:
                                st.error("Failed to preprocess image.")
                                return
                            
                            extracted_text = extract_text_from_image(model, processed_image)
                            if extracted_text:
                                st.session_state.extracted_texts[file_name] = {
                                    'type': 'image',
                                    'text': extracted_text,
                                    'image': image,
                                    'file_bytes': file.getvalue()  # Store original bytes for download
                                }
                            progress_bar.progress(1.0)
                        except Exception as e:
                            st.error(f"Error processing image: {str(e)}")
                            logging.error(f"Image processing error: {str(e)}")
                            return
                    
                    st.markdown('<div class="success-box">✅ Text extraction completed successfully!</div>', unsafe_allow_html=True)
    
    # STRUCTURE 4: EXTRACTED TEXT RESULTS
    if st.session_state.extracted_texts:
        st.markdown("---")
        st.markdown('<h3 class="section-header">📊 Results</h3>', unsafe_allow_html=True)
        
        # Show text popup if requested
        if st.session_state.show_popup:
            popup_data = st.session_state.show_popup
            show_text_popup(popup_data['text'], popup_data['title'], popup_data['key'])
        
        for file_name, data in st.session_state.extracted_texts.items():
            with st.expander(f"📄 {file_name}", expanded=True):
                if data['type'] == 'pdf':
                    st.write(f"**📖 Pages found:** {len(data['pages'])}")
                    
                    # Preview buttons for each page
                    cols = st.columns(min(len(data['pages']), 5))
                    for i, page_data in enumerate(data['pages']):
                        with cols[i % 5]:
                            if st.button(f"👁️ Preview Page {page_data['page']}", key=f"preview_{file_name}_{i}"):
                                st.session_state.show_popup = {
                                    'text': clean_extracted_text(page_data['text']),
                                    'title': f"{file_name} - Page {page_data['page']}",
                                    'key': f"{file_name}_{i}"
                                }
                                st.rerun()
                    
                    # Download all pages as single text file
                    all_text = ""
                    for page_data in data['pages']:
                        all_text += f"=== PAGE {page_data['page']} ===\n\n"
                        all_text += clean_extracted_text(page_data['text']) + "\n\n"
                    
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
                    
                    # Structured CSV download
                    structured_data = []
                    for page_data in data['pages']:
                        cleaned_text = clean_extracted_text(page_data['text'])
                        structured_data.append({
                            'File': file_name,
                            'Page': page_data['page'],
                            'Text': cleaned_text,
                            'Word_Count': len(cleaned_text.split()),
                            'Character_Count': len(cleaned_text),
                            'Extraction_Date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                    
                    df = pd.DataFrame(structured_data)
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        label="📊 Download as Structured CSV",
                        data=csv_data,
                        file_name=f"{file_name}_structured.csv",
                        mime="text/csv",
                        key=f"download_csv_{file_name}"
                    )
                
                else:  # Single image
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.image(data['image'], caption=f"🖼️ Image: {file_name}", use_container_width=True)
                    with col2:
                        if st.button(f"👁️ Preview Text", key=f"preview_img_{file_name}"):
                            st.session_state.show_popup = {
                                'text': clean_extracted_text(data['text']),
                                'title': f"{file_name} - Extracted Text",
                                'key': f"img_{file_name}"
                            }
                            st.rerun()
                        
                        # Download button
                        cleaned_text = clean_extracted_text(data['text'])
                        st.download_button(
                            label="📥 Download as TXT",
                            data=cleaned_text,
                            file_name=f"{file_name}_extracted.txt",
                            mime="text/plain",
                            key=f"download_{file_name}"
                        )
                        
                        # Statistics
                        word_count = len(cleaned_text.split())
                        char_count = len(cleaned_text)
                        st.markdown(f'<div class="info-box"><strong>📈 Statistics:</strong> {word_count} words, {char_count} characters</div>', unsafe_allow_html=True)
                        
                        # Structured CSV download
                        structured_data = {
                            'File': file_name,
                            'Text': cleaned_text,
                            'Word_Count': word_count,
                            'Character_Count': char_count,
                            'Extraction_Date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        df = pd.DataFrame([structured_data])
                        csv_data = df.to_csv(index=False)
                        st.download_button(
                            label="📊 Download as Structured CSV",
                            data=csv_data,
                            file_name=f"{file_name}_structured.csv",
                            mime="text/csv",
                            key=f"download_csv_img_{file_name}"
                        )
    
    elif not uploaded_files:
        st.markdown('<div class="info-box">📋 Please upload an image or PDF to extract text</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()