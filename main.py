from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import List, Optional
import os
from datetime import datetime
import tempfile
import json
from app1 import (
    configure_gemini,
    process_uploaded_file,
    extract_text_from_image,
    convert_pdf_to_images,
    preprocess_image,
    clean_extracted_text,
    create_structured_data,
    APIResponse,
    can_upload,          # Add this
    update_upload_count, # Add this
    UPLOAD_LIMIT 
)

app = FastAPI(
    title="MediScan API",
    description="Extract and enhance text from medical documents using Gemini AI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini model on startup
model = None

@app.on_event("startup")
async def startup_event():
    global model
    try:
        model = configure_gemini()
        print("✅ Gemini model initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing Gemini model: {str(e)}")
        raise

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "MediScan API - Medical Document Text Extraction",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "extract_text": "/extract-text",
            "extract_text_advanced": "/extract-text-advanced",
            "health_check": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "gemini_model": "initialized" if model else "not_initialized"
    }

@app.post("/extract-text")
async def extract_text_endpoint(file: UploadFile = File(...)):
    try:
        # 1. First check upload limit
        can_upload_result, message, count = can_upload(file.content_type, file.file)
        if not can_upload_result:
            raise HTTPException(status_code=400, detail=message)
        
        # Validate file type
        allowed_types = ['image/png', 'image/jpeg', 'image/jpg', 'application/pdf']
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed types: {', '.join(allowed_types)}"
            )
        
        # Read file content
        file_content = await file.read()
        if not file_content:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        # Process the file
        result = await process_uploaded_file(model, file_content, file.filename, file.content_type)
        
        # 2. Only update counter if processing was successful
        update_upload_count(count)
        
        return APIResponse(
            success=True,
            message="Text extraction completed successfully",
            data=result,
            timestamp=datetime.now().isoformat()
        ).dict()
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/extract-text-advanced")
async def extract_text_advanced_endpoint(
    file: UploadFile = File(...),
    include_metadata: bool = True,
    include_statistics: bool = True,
    output_format: str = "json"
):
    """
    Advanced text extraction with additional options
    
    Args:
        file: Image (PNG, JPG, JPEG) or PDF file
        include_metadata: Whether to include file metadata
        include_statistics: Whether to include text statistics
        output_format: Output format (json, csv, txt)
        
    Returns:
        JSON response with extracted text and optional metadata/statistics
    """
    try:
        # First check upload limit
        can_upload_result, message, count = can_upload(file.content_type, file.file)
        if not can_upload_result:
            raise HTTPException(status_code=400, detail=message)

        # Validate file type
        allowed_types = ['image/png', 'image/jpeg', 'image/jpg', 'application/pdf']
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed types: {', '.join(allowed_types)}"
            )
        
        # Validate output format
        if output_format not in ['json', 'csv', 'txt']:
            raise HTTPException(
                status_code=400,
                detail="Invalid output format. Allowed: json, csv, txt"
            )
        
        # Read file content
        file_content = await file.read()
        if not file_content:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        # Process the file
        result = await process_uploaded_file(model, file_content, file.filename, file.content_type)
        
        # Update counter only after successful processing
        update_upload_count(count)
        
        # Add metadata if requested
        if include_metadata:
            result['metadata'] = {
                'filename': file.filename,
                'file_size': len(file_content),
                'content_type': file.content_type,
                'processed_at': datetime.now().isoformat(),
                'units_consumed': count  # Show how many units this file used
            }
        
        # Add statistics if requested
        if include_statistics:
            if result['type'] == 'pdf':
                total_words = sum(len(page['text'].split()) for page in result['pages'])
                total_chars = sum(len(page['text']) for page in result['pages'])
                result['statistics'] = {
                    'total_pages': len(result['pages']),
                    'total_words': total_words,
                    'total_characters': total_chars,
                    'average_words_per_page': total_words / len(result['pages']) if result['pages'] else 0
                }
            else:
                result['statistics'] = {
                    'word_count': len(result['text'].split()),
                    'character_count': len(result['text'])
                }
        
        # Handle different output formats
        if output_format == 'json':
            return APIResponse(
                success=True,
                message="Advanced text extraction completed successfully",
                data=result,
                timestamp=datetime.now().isoformat()
            ).dict()
        elif output_format == 'csv':
            # Convert to CSV format
            csv_data = create_structured_data(result, file.filename)
            
            # Create a temporary CSV file
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.csv', delete=False) as tmp_file:
                df = pd.DataFrame(csv_data)
                df.to_csv(tmp_file.name, index=False)
            
            # Return the file as a download response
            return FileResponse(
                tmp_file.name,
                media_type='text/csv',
                filename=f"{os.path.splitext(file.filename)[0]}_extracted.csv",
                background=BackgroundTask(lambda: os.unlink(tmp_file.name))
            )
            
        elif output_format == 'txt':
            # Convert to plain text format
            if result['type'] == 'pdf':
                text_content = ""
                for page in result['pages']:
                    text_content += f"=== PAGE {page['page']} ===\n\n"
                    text_content += page['text'] + "\n\n"
            else:
                text_content = result['text']
            
            # Create a temporary text file
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False) as tmp_file:
                tmp_file.write(text_content)
            
            # Return the file as a download response
            return FileResponse(
                tmp_file.name,
                media_type='text/plain',
                filename=f"{os.path.splitext(file.filename)[0]}_extracted.txt",
                background=BackgroundTask(lambda: os.unlink(tmp_file.name)))
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
        
@app.post("/batch-extract")
async def batch_extract_endpoint(files: List[UploadFile] = File(...)):
    """
    Extract text from multiple files in batch
    
    Args:
        files: List of image or PDF files
        
    Returns:
        JSON response with extracted text for each file
    """
    try:
        if len(files) > 10:  # Limit batch size
            raise HTTPException(
                status_code=400,
                detail="Batch size limit exceeded. Maximum 10 files allowed."
            )
        
        results = []
        total_units_consumed = 0
        
        for file in files:
            file_result = {
                "filename": file.filename,
                "success": False,
                "error": None,
                "data": None,
                "units_consumed": 0
            }
            
            try:
                # Check upload limit for this file
                can_upload_result, message, count = can_upload(file.content_type, file.file)
                if not can_upload_result:
                    file_result["error"] = message
                    results.append(file_result)
                    continue
                
                # Validate file type
                allowed_types = ['image/png', 'image/jpeg', 'image/jpg', 'application/pdf']
                if file.content_type not in allowed_types:
                    file_result["error"] = f"Unsupported file type: {file.content_type}"
                    results.append(file_result)
                    continue
                
                # Read file content
                file_content = await file.read()
                if not file_content:
                    file_result["error"] = "Empty file"
                    results.append(file_result)
                    continue
                
                # Process the file
                result = await process_uploaded_file(model, file_content, file.filename, file.content_type)
                
                # Mark as successful
                file_result.update({
                    "success": True,
                    "data": result,
                    "units_consumed": count
                })
                results.append(file_result)
                
                # Add to total count
                total_units_consumed += count
                
            except Exception as e:
                file_result["error"] = str(e)
                results.append(file_result)
                continue
        
        # Update counter with total successful count
        if total_units_consumed > 0:
            update_upload_count(total_units_consumed)
        
        # Calculate summary statistics
        successful = sum(1 for r in results if r['success'])
        total_units = sum(r['units_consumed'] for r in results)
        remaining_units = UPLOAD_LIMIT - db_counter.get_current_count()
        
        return APIResponse(
            success=True,
            message=f"Batch processing completed. {successful} out of {len(results)} files processed successfully. {total_units} units consumed.",
            data={
                "results": results,
                "summary": {
                    "total_files": len(results),
                    "successful_files": successful,
                    "failed_files": len(results) - successful,
                    "total_units_consumed": total_units,
                    "remaining_units": remaining_units
                }
            },
            timestamp=datetime.now().isoformat()
        ).dict()
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/supported-formats")
async def get_supported_formats():
    """Get list of supported file formats"""
    return {
        "supported_formats": {
            "images": ["PNG", "JPG", "JPEG"],
            "documents": ["PDF"]
        },
        "max_file_size": "50MB",
        "batch_limit": 10
    }

@app.get("/upload-count")
async def get_upload_count():
    """Get current upload count"""
    current_count = db_counter.get_current_count()
    return {
        "current_count": current_count,
        "remaining": UPLOAD_LIMIT - current_count,
        "limit": UPLOAD_LIMIT
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)

    #uvicorn main:app --host 0.0.0.0 --port 8000 --reload