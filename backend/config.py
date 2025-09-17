# backend/config.py
import os
from typing import Optional

class Config:
    """Application configuration"""
    
    # Model settings
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "large-v3")
    DEVICE: str = os.getenv("DEVICE", "cuda")
    COMPUTE_TYPE: str = os.getenv("COMPUTE_TYPE", "float16")
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "16"))
    
    # Server settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    WS_PORT: int = int(os.getenv("WS_PORT", "9090"))
    
    # API Keys
    HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    
    # Paths
    MODEL_CACHE_DIR: str = os.getenv("MODEL_CACHE_DIR", "/models")
    AUDIO_TEMP_DIR: str = os.getenv("AUDIO_TEMP_DIR", "/tmp/audio")

config = Config()