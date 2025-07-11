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
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import io
from fastapi import HTTPException
from PyPDF2 import PdfReader
from db import db_counter


user_uploads = {}  # {user_id: file_count}
UPLOAD_LIMIT = 50

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        #logging.FileHandler('ocr_errors.log'),
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv()

class APIResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: str

class PageData(BaseModel):
    page: int
    text: str
    word_count: int
    character_count: int

class ExtractionResult(BaseModel):
    type: str  # 'image' or 'pdf'
    text: Optional[str] = None
    pages: Optional[List[PageData]] = None
    processing_time: float

def configure_gemini():
    """Configure and return Gemini model"""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required")
    
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.5-pro')

def count_pdf_pages(file_obj):
    try:
        reader = PdfReader(file_obj)
        return len(reader.pages)
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return 0

def can_upload(file_type, file_obj):
    current = db_counter.get_current_count()

    if file_type == "pdf":
        pages = count_pdf_pages(file_obj)
        if current + pages > UPLOAD_LIMIT:
            return False, f"Upload limit exceeded. You have {UPLOAD_LIMIT - current} remaining.", 0
        return True, "", pages
    else:
        if current + 1 > UPLOAD_LIMIT:
            return False, f"Upload limit exceeded. You have {UPLOAD_LIMIT - current} remaining.", 0
        return True, "", 1

def update_upload_count(count):
    return db_counter.increment_counter(count)

def preprocess_image(image: Image.Image) -> Optional[Image.Image]:
    """
    Preprocess image for better OCR results
    
    Args:
        image: PIL Image object
        
    Returns:
        Preprocessed PIL Image object or None if error
    """
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
        elif image.mode == 'L':
            # Convert grayscale to RGB
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        
        # Enhance contrast
        img_array = cv2.convertScaleAbs(img_array, alpha=1.5, beta=0)
        return Image.fromarray(img_array)
        
    except Exception as e:
        logging.error(f"Image preprocessing error: {str(e)}")
        return None

def convert_pdf_to_images(pdf_bytes: bytes) -> List[Image.Image]:
    """
    Convert PDF bytes to list of PIL Images
    
    Args:
        pdf_bytes: PDF file content as bytes
        
    Returns:
        List of PIL Image objects
    """
    try:
        images = pdf2image.convert_from_bytes(pdf_bytes, dpi=300, fmt='RGB')
        return images
    except Exception as e:
        logging.error(f"PDF conversion error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error converting PDF: {str(e)}")

def clean_extracted_text(text: str) -> str:
    """
    Clean the extracted text by removing HTML tags and fixing formatting issues
    
    Args:
        text: Raw extracted text
        
    Returns:
        Cleaned text
    """
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

def extract_text_from_image(model, image: Image.Image) -> str:
    """
    Extract text from image using Gemini API
    
    Args:
        model: Gemini model instance
        image: PIL Image object
        
    Returns:
        Extracted text
    """
    try:
        prompt = """
You are an expert medical document analyst specializing in transcribing handwritten and printed clinical notes, prescriptions, and lab reports. Your goal is to accurately extract and format all textual information from the provided image, even if the handwriting is highly illegible or the document is poorly scanned. Note that several fields containing personal identifiable information (PII) may be redacted with a black marker. The handwriting in some areas may be difficult to read, and interpretations should be noted with `(unsure)` or `(illegible)`. Follow these instructions:

1. **Image Orientation Correction**: Assess the image orientation. If it is tilted, rotated, or not upright, virtually correct it to an upright position before text analysis to ensure accuracy.
2. **Robust Text Extraction with Confidence Scoring**: Extract all discernible text. For terms or phrases with low confidence due to illegibility, append `(unsure)` immediately after the term. For completely illegible sections, note `(illegible)` with an estimated word count if possible (e.g., `(illegible, ~3 words)`).
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
        raise HTTPException(status_code=500, detail=f"Error extracting text: {str(e)}")

def create_structured_data(result: Dict[str, Any], filename: str) -> List[Dict[str, Any]]:
    """
    Create structured data for CSV export
    
    Args:
        result: Extraction result dictionary
        filename: Original filename
        
    Returns:
        List of structured data dictionaries
    """
    structured_data = []
    
    if result['type'] == 'pdf':
        for page_data in result['pages']:
            structured_data.append({
                'File': filename,
                'Page': page_data['page'],
                'Text': page_data['text'],
                'Word_Count': page_data['word_count'],
                'Character_Count': page_data['character_count'],
                'Extraction_Date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
    else:
        structured_data.append({
            'File': filename,
            'Page': 1,
            'Text': result['text'],
            'Word_Count': len(result['text'].split()),
            'Character_Count': len(result['text']),
            'Extraction_Date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    return structured_data

async def process_uploaded_file(
    model, 
    file_content: bytes, 
    filename: str, 
    content_type: str
) -> Dict[str, Any]:
    """
    Process uploaded file and extract text
    
    Args:
        model: Gemini model instance
        file_content: File content as bytes
        filename: Original filename
        content_type: MIME type of the file
        
    Returns:
        Dictionary containing extraction results
    """
    start_time = datetime.now()
    
    try:
        if content_type == "application/pdf":
            # Process PDF
            images = convert_pdf_to_images(file_content)
            if not images:
                raise HTTPException(status_code=400, detail="No pages found in PDF")
            
            pages_data = []
            for page_idx, image in enumerate(images):
                # Preprocess image
                processed_image = preprocess_image(image)
                if not processed_image:
                    logging.warning(f"Failed to preprocess page {page_idx + 1}")
                    continue
                
                # Extract text
                extracted_text = extract_text_from_image(model, processed_image)
                if extracted_text:
                    pages_data.append({
                        'page': page_idx + 1,
                        'text': extracted_text,
                        'word_count': len(extracted_text.split()),
                        'character_count': len(extracted_text)
                    })
            
            if not pages_data:
                raise HTTPException(status_code=400, detail="No text could be extracted from PDF")
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'type': 'pdf',
                'pages': pages_data,
                'total_pages': len(pages_data),
                'processing_time': processing_time
            }
            
        else:
            # Process image
            try:
                image = Image.open(io.BytesIO(file_content))
                
                # Ensure image is in RGB mode
                if image.mode != 'RGB':
                    if image.mode == 'RGBA':
                        background = Image.new('RGB', image.size, (255, 255, 255))
                        background.paste(image, mask=image.split()[-1])
                        image = background
                    else:
                        image = image.convert('RGB')
                
                # Preprocess image
                processed_image = preprocess_image(image)
                if not processed_image:
                    raise HTTPException(status_code=400, detail="Failed to preprocess image")
                
                # Extract text
                extracted_text = extract_text_from_image(model, processed_image)
                if not extracted_text:
                    raise HTTPException(status_code=400, detail="No text could be extracted from image")
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                return {
                    'type': 'image',
                    'text': extracted_text,
                    'word_count': len(extracted_text.split()),
                    'character_count': len(extracted_text),
                    'processing_time': processing_time
                }
                
            except Exception as e:
                logging.error(f"Error processing image: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")
                
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

def validate_file_size(file_content: bytes, max_size_mb: int = 50) -> bool:
    """
    Validate file size
    
    Args:
        file_content: File content as bytes
        max_size_mb: Maximum allowed size in MB
        
    Returns:
        True if valid, False otherwise
    """
    file_size_mb = len(file_content) / (1024 * 1024)
    return file_size_mb <= max_size_mb

def get_file_info(file_content: bytes, filename: str, content_type: str) -> Dict[str, Any]:
    """
    Get file information
    
    Args:
        file_content: File content as bytes
        filename: Original filename
        content_type: MIME type
        
    Returns:
        Dictionary with file information
    """
    file_size_mb = len(file_content) / (1024 * 1024)
    
    return {
        'filename': filename,
        'content_type': content_type,
        'file_size_mb': round(file_size_mb, 2),
        'file_size_bytes': len(file_content)
    }