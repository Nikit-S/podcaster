import os
import sys
import torch
from huggingface_hub import login
from pyannote.audio import Pipeline, Model
from faster_whisper import WhisperModel
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

def download_diarization_model(hf_token):
    """Download and cache the diarization model"""
    logger.info("📥 Downloading pyannote/speaker-diarization-3.1 model...")
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        logger.info("✅ Diarization model downloaded successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Error downloading diarization model: {str(e)}")
        return False

def download_segmentation_model(hf_token):
    """Download and cache the segmentation model correctly"""
    logger.info("📥 Downloading pyannote/segmentation-3.0 model...")
    try:
        # Правильный способ загрузки segmentation модели
        model = Model.from_pretrained(
            "pyannote/segmentation-3.0",
            use_auth_token=hf_token
        )
        logger.info("✅ Segmentation model downloaded successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Error downloading segmentation model: {str(e)}")
        return False

def download_whisper_model(model_size="medium"):
    """Download and cache the Whisper model"""
    logger.info(f"📥 Downloading Whisper {model_size} model...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "float32"
        
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        logger.info(f"✅ Whisper {model_size} model downloaded successfully (device: {device})")
        return True
    except Exception as e:
        logger.error(f"❌ Error downloading Whisper model: {str(e)}")
        return False

if __name__ == "__main__":
    hf_token = os.environ.get("HF_TOKEN")
    
    if not hf_token:
        logger.error("❌ HF_TOKEN environment variable is not set!")
        sys.exit(1)
    
    # Аутентификация
    try:
        login(token=hf_token, add_to_git_credential=False)
        logger.info("✅ Successfully logged in to Hugging Face Hub")
    except Exception as e:
        logger.error(f"❌ Failed to login to Hugging Face Hub: {str(e)}")
        sys.exit(1)
    
    success = True
    
    # Скачиваем модели
    if not download_diarization_model(hf_token):
        success = False
    
    # Скачиваем segmentation модель ПРАВИЛЬНЫМ способом
    if not download_segmentation_model(hf_token):
        success = False
    
    # Скачиваем Whisper модель
    model_size = os.environ.get("WHISPER_MODEL_SIZE", "medium")
    if not download_whisper_model(model_size):
        success = False
    
    if success:
        logger.info("\n🎉 All models downloaded successfully!")
        sys.exit(0)
    else:
        logger.error("\n❌ Some models failed to download. Build will fail.")
        sys.exit(1)