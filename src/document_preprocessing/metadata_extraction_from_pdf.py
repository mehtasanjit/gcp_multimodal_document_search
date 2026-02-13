import os
import json
import hashlib
import time
import argparse
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try importing Google Gen AI SDK
try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

def generate_doc_id(filename):
    """Generates a safe document ID from the filename."""
    return hashlib.md5(filename.encode()).hexdigest()

def get_file_metadata(filepath):
    """Extracts basic file metadata using OS stats."""
    stat = os.stat(filepath)
    return {
        "size_bytes": stat.st_size,
        "created_timestamp": int(stat.st_ctime),
        "modified_timestamp": int(stat.st_mtime),
        "created_iso": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "modified_iso": datetime.fromtimestamp(stat.st_mtime).isoformat()
    }

def clean_filename_to_title(filename):
    """Converts a filename like 'some-report-v.2.0.pdf' to 'Some Report V.2.0'"""
    name = os.path.splitext(filename)[0]
    name = name.replace('-', ' ').replace('_', ' ')
    name = ' '.join(name.split())
    return name.title()

def infer_attributes(filepath, client, model_id, prompt_text):
    """Uses Gemini to extract attributes from the PDF."""
    if not HAS_GENAI:
        logger.error("google-genai library not installed. Skipping AI inference.")
        return None

    try:
        logger.info(f"AI Processing: {os.path.basename(filepath)}...")
        
        with open(filepath, "rb") as f:
            pdf_bytes = f.read()

        response = client.models.generate_content(
            model=model_id,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")
                    ]
                )
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                system_instruction=prompt_text
            )
        )
        
        if response.text:
            return json.loads(response.text)
        return None

    except Exception as e:
        logger.error(f"Error during AI inference for {filepath}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Generate NDJSON metadata for PDF documents in a folder.")
    parser.add_argument("folder", help="Path to the folder containing PDF documents")
    parser.add_argument("--gcs-base-uri", required=True, help="Base GCS URI (e.g. gs://bucket/path/to/docs)")
    parser.add_argument("--category", default="Technical Report", help="Default category for these documents")
    
    # AI Arguments
    parser.add_argument("--infer-ai-attributes", action="store_true", help="Enable AI-based attribute extraction")
    parser.add_argument("--project", help="GCP Project ID")
    parser.add_argument("--location", default="us-central1", help="GCP Location")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Model ID to use")
    parser.add_argument("--prompt-file", default="src/document_preprocessing/metadata_extraction_prompt.txt", help="Path to prompt file")

    args = parser.parse_args()
    
    target_dir = args.folder
    if not os.path.exists(target_dir):
        logger.error(f"Error: Directory '{target_dir}' does not exist.")
        return
        
    # Normalize GCS base URI
    gcs_base = args.gcs_base_uri.rstrip('/')

    output_file = os.path.join(target_dir, "metadata.jsonl")
    
    # Initialize GenAI Client if needed
    client = None
    prompt_text = ""
    
    if args.infer_ai_attributes:
        if not HAS_GENAI:
            logger.error("Error: --infer-ai-attributes requested but google-genai package is missing.")
            return
        if not args.project:
            logger.error("Error: --project is required when using AI inference.")
            return
            
        logger.info(f"Initializing GenAI Client for project {args.project}...")
        client = genai.Client(vertexai=True, project=args.project, location=args.location)
        
        if os.path.exists(args.prompt_file):
            with open(args.prompt_file, 'r') as f:
                prompt_text = f.read()
        else:
            logger.warning(f"Prompt file {args.prompt_file} not found. Using default.")
            prompt_text = "Extract metadata from this document as JSON."

    files = [f for f in os.listdir(target_dir) if f.lower().endswith('.pdf')]
    files.sort()
    
    logger.info(f"Found {len(files)} PDF files in {target_dir}")
    logger.info(f"Generating metadata to {output_file}...")
    
    # Track results
    records = []
    
    for filename in files:
        filepath = os.path.join(target_dir, filename)
        
        # 1. Basic Metadata
        file_meta = get_file_metadata(filepath)
        
        # 2. Derived Metadata
        doc_id = generate_doc_id(filename)
        title = clean_filename_to_title(filename)
        gcs_uri = f"{gcs_base}/{filename}"
        
        struct_data = {
            "title": title,
            "filename": filename,
            "category": args.category,
            "file_size": file_meta["size_bytes"],
            "upload_date": datetime.now().isoformat(),
            "source_local_path": os.path.abspath(filepath),
            "gcs_uri": gcs_uri
        }

        # 3. AI Inference
        if args.infer_ai_attributes and client:
            ai_data = infer_attributes(filepath, client, args.model, prompt_text)
            if ai_data:
                struct_data["ai_inferred_attributes"] = ai_data
            time.sleep(1) # Rate limit protection

        # 4. Construct Record
        record = {
            "id": doc_id,
            "structData": struct_data,
            "content": {
                "mimeType": "application/pdf",
                "uri": gcs_uri
            }
        }
        records.append(record)
        
        # Write incrementally (optional, but good for progress)
        # We write all at once at end to keep file clean or append? 
        # Let's write incrementally to a temp list and then dump
        
    with open(output_file, 'w') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')
            
    logger.info(f"Done. Metadata saved to {output_file}")

if __name__ == "__main__":
    main()
