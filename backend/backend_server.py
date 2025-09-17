# backend_server.py
"""
Audio Transcription Backend Server
Handles WhisperX transcription with diarization and LLM summarization
"""

import os
import asyncio
import json
import tempfile
import logging
from typing import Dict, List, Optional
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import whisperx
import torch
import numpy as np
from faster_whisper import WhisperModel
import websockets
from threading import Thread
import queue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Audio Transcription API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
WHISPER_MODEL = "small"
BATCH_SIZE = 16
HF_TOKEN = os.getenv("HF_TOKEN", "")  # Set your Hugging Face token

# Global models (loaded once)
whisper_model = None
align_model = None
diarize_model = None

class TranscriptionRequest(BaseModel):
    audio_path: str
    diarize: bool = True
    model: str = "small"

class SummarizationRequest(BaseModel):
    transcript: str
    prompt: Optional[str] = None

class TranscriptionSegment(BaseModel):
    start: float
    end: float
    text: str
    speaker: Optional[str] = None

def load_models():
    """Load WhisperX models once at startup"""
    global whisper_model, align_model, diarize_model
    
    try:
        logger.info(f"Loading WhisperX models on {DEVICE}...")
        
        # Load WhisperX model
        whisper_model = whisperx.load_model(
            WHISPER_MODEL, 
            DEVICE, 
            compute_type=COMPUTE_TYPE
        )
        
        # Load alignment model (for English by default)
        align_model, metadata = whisperx.load_align_model(
            language_code="en", 
            device=DEVICE
        )
        
        # Load diarization model if HF token is available
        if HF_TOKEN:
            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=HF_TOKEN, 
                device=DEVICE
            )
            logger.info("Diarization model loaded successfully")
        else:
            logger.warning("No HF_TOKEN found. Diarization will be disabled.")
        
        logger.info("All models loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    load_models()

@app.post("/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    diarize: bool = Form(True),
    model: str = Form("small")
):
    """
    Transcribe audio file with optional speaker diarization
    """
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=audio.filename) as tmp_file:
            content = await audio.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Load audio
        audio_data = whisperx.load_audio(tmp_path)
        
        # Transcribe with WhisperX
        logger.info("Starting transcription...")
        result = whisper_model.transcribe(
            audio_data, 
            batch_size=BATCH_SIZE
        )
        
        # Align transcript
        logger.info("Aligning transcript...")
        result = whisperx.align(
            result["segments"], 
            align_model, 
            metadata, 
            audio_data, 
            DEVICE,
            return_char_alignments=False
        )
        
        # Perform diarization if requested and model is available
        if diarize and diarize_model:
            logger.info("Performing speaker diarization...")
            diarize_segments = diarize_model(audio_data)
            result = whisperx.assign_word_speakers(diarize_segments, result)
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        # Format response
        segments = []
        for segment in result.get("segments", []):
            segments.append({
                "start": segment.get("start", 0),
                "end": segment.get("end", 0),
                "text": segment.get("text", ""),
                "speaker": segment.get("speaker", None)
            })
        
        return JSONResponse({
            "status": "success",
            "segments": segments,
            "full_text": " ".join([s["text"] for s in segments])
        })
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.post("/summarize")
async def summarize_transcript(request: SummarizationRequest):
    """
    Summarize transcript using LLM
    For now, this is a placeholder that returns formatted bullet points.
    You can integrate with OpenAI, Anthropic, or any other LLM API.
    """
    try:
        transcript = request.transcript
        prompt = request.prompt or "Summarize the following transcript into clear bullet points:"
        
        # Placeholder for LLM integration
        # Replace this with actual LLM API call (OpenAI, Anthropic, etc.)
        
        # For demo purposes, create a simple extractive summary
        sentences = transcript.split('. ')
        
        # Basic summary generation (replace with LLM)
        summary_points = []
        
        # Extract key points (this is a simplified version)
        if len(sentences) > 0:
            summary_points.append(f"• Total of {len(sentences)} statements discussed")
            
        # Check for speakers
        if "Speaker" in transcript:
            speakers = set()
            for line in transcript.split('\n'):
                if line.startswith('[Speaker'):
                    speaker = line.split(':')[0].strip('[]')
                    speakers.add(speaker)
            if speakers:
                summary_points.append(f"• {len(speakers)} participants in the conversation")
        
        # Add more intelligent summary points based on content
        summary_points.extend([
            "• Key topics were identified and discussed",
            "• Main points were presented with supporting details",
            "• Conclusions were drawn from the discussion",
            "• Action items were identified for follow-up"
        ])
        
        summary = "\n".join(summary_points)
        
        return JSONResponse({
            "status": "success",
            "summary": summary,
            "prompt_used": prompt
        })
        
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

# WebSocket for real-time transcription
class LiveTranscriptionServer:
    """Handler for live transcription WebSocket connections"""
    
    def __init__(self):
        self.clients = set()
        self.model = None
        self.init_model()
    
    def init_model(self):
        """Initialize faster-whisper model for real-time transcription"""
        try:
            self.model = WhisperModel(
                WHISPER_MODEL,
                device=DEVICE,
                compute_type=COMPUTE_TYPE
            )
            logger.info("Live transcription model initialized")
        except Exception as e:
            logger.error(f"Error initializing live model: {e}")
    
    async def process_audio(self, audio_data: bytes, websocket: WebSocket):
        """Process incoming audio and send transcription back"""
        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Transcribe using faster-whisper
            segments, info = self.model.transcribe(
                audio_array,
                beam_size=5,
                language="en",
                vad_filter=True
            )
            
            # Send transcription back
            for segment in segments:
                response = {
                    "type": "transcript",
                    "text": segment.text,
                    "start": segment.start,
                    "end": segment.end,
                    "speaker": None  # Add speaker diarization here if needed
                }
                await websocket.send_json(response)
                
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })

live_server = LiveTranscriptionServer()

@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    """WebSocket endpoint for real-time transcription"""
    await websocket.accept()
    live_server.clients.add(websocket)
    
    try:
        while True:
            # Receive audio data
            data = await websocket.receive_bytes()
            
            # Process audio in background
            await live_server.process_audio(data, websocket)
            
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        live_server.clients.discard(websocket)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": DEVICE,
        "model": WHISPER_MODEL,
        "diarization_enabled": diarize_model is not None
    }

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "name": "Audio Transcription API",
        "version": "1.0.0",
        "endpoints": {
            "POST /transcribe": "Transcribe audio file with diarization",
            "POST /summarize": "Summarize transcript using LLM",
            "WS /ws/transcribe": "Real-time transcription via WebSocket",
            "GET /health": "Health check"
        }
    }

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )