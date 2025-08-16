#!/usr/bin/env python3
"""
Model downloader for OpenVoice
Downloads the necessary models for voice cloning functionality
"""

import os
import sys
import zipfile
import shutil
import requests
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model URLs from OpenVoice documentation
MODELS = {
    "checkpoints_v1": {
        "url": "https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_1226.zip",
        "extract_to": "/app/models/checkpoints"
    },
    "checkpoints_v2": {
        "url": "https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip", 
        "extract_to": "/app/models/checkpoints_v2"
    }
}

def download_file(url: str, local_path: str):
    """Download a file from URL to local path"""
    try:
        logger.info(f"Downloading {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Downloaded to {local_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False

def download_and_extract(url: str, extract_to: str):
    """Download and extract a model archive"""
    try:
        extract_path = Path(extract_to)
        extract_path.mkdir(parents=True, exist_ok=True)
        
        # Download to temp file
        temp_file = "/tmp/model_download.zip"
        if not download_file(url, temp_file):
            return False
        
        # Extract
        logger.info(f"Extracting to {extract_to}...")
        with zipfile.ZipFile(temp_file, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        
        # Clean up
        os.remove(temp_file)
        logger.info("Download and extraction completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to download/extract {url}: {e}")
        return False
    
    return True

def verify_models():
    """Verify that required models are present"""
    # Check for V1 models (prioritized for compatibility)
    v1_files = [
        "/app/models/checkpoints/base_speakers/EN/config.json",
        "/app/models/checkpoints/base_speakers/EN/checkpoint.pth",
        "/app/models/checkpoints/converter/config.json",
        "/app/models/checkpoints/converter/checkpoint.pth"
    ]
    
    v1_exists = all(Path(f).exists() for f in v1_files)
    
    if v1_exists:
        logger.info("OpenVoice V1 models found")
        return True
    
    # Check for V2 models
    v2_files = [
        "/app/models/checkpoints_v2/base_speakers/EN/config.json",
        "/app/models/checkpoints_v2/base_speakers/EN/checkpoint.pth",
        "/app/models/checkpoints_v2/converter/config.json", 
        "/app/models/checkpoints_v2/converter/checkpoint.pth"
    ]
    
    v2_exists = all(Path(f).exists() for f in v2_files)
    
    if v2_exists:
        logger.info("OpenVoice V2 models found")
        return True
    
    logger.warning("No OpenVoice models found")
    return False

def main():
    """Main function to download all models"""
    logger.info("Starting OpenVoice model download...")
    
    # Check if models already exist
    if verify_models():
        logger.info("Models already exist, skipping download")
        return True
    
    # Try to download V1 models first (more stable)
    logger.info("Downloading OpenVoice V1 models...")
    if download_and_extract(MODELS["checkpoints_v1"]["url"], MODELS["checkpoints_v1"]["extract_to"]):
        if verify_models():
            logger.info("OpenVoice V1 models downloaded and verified successfully!")
            return True
    
    # If V1 fails, try V2
    logger.info("V1 download failed, trying OpenVoice V2 models...")
    if download_and_extract(MODELS["checkpoints_v2"]["url"], MODELS["checkpoints_v2"]["extract_to"]):
        if verify_models():
            logger.info("OpenVoice V2 models downloaded and verified successfully!")
            return True
    
    logger.error("Failed to download any OpenVoice models")
    return False

if __name__ == "__main__":
    success = main()
    if not success:
        logger.warning("Model download failed, but continuing startup. Models can be downloaded manually.")
    sys.exit(0)  # Don't fail startup even if models fail to download
