# backend/config_updated.py
# Save this as backend/config.py to use gpt-4o-mini

import os
from typing import Optional

class Config:
    """Application configuration"""
    
    # Model settings - Updated for your requirements
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "large-v3")  # Using large-v3
    DEVICE: str = os.getenv("DEVICE", "cuda")
    COMPUTE_TYPE: str = os.getenv("COMPUTE_TYPE", "float16")
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "8"))  # Reduced for large-v3
    
    # Server settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    WS_PORT: int = int(os.getenv("WS_PORT", "9090"))
    
    # API Keys
    HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    
    # LLM Settings - Updated for gpt-4o-mini
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")  # Using gpt-4o-mini
    
    # Paths
    MODEL_CACHE_DIR: str = os.getenv("MODEL_CACHE_DIR", "/models")
    AUDIO_TEMP_DIR: str = os.getenv("AUDIO_TEMP_DIR", "/tmp/audio")
    
    # Features
    ENABLE_DIARIZATION: bool = os.getenv("ENABLE_DIARIZATION", "true").lower() == "true"
    ENABLE_VAD: bool = os.getenv("ENABLE_VAD", "true").lower() == "true"

config = Config()