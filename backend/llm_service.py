# llm_service.py
"""
LLM Service for transcript summarization
Supports OpenAI, Anthropic, and local models
"""

import os
import logging
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

import openai
from langchain.llms import LlamaCpp
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

logger = logging.getLogger(__name__)

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    async def summarize(self, transcript: str, prompt: Optional[str] = None) -> str:
        """Summarize the transcript"""
        pass

class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider for summarization"""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    
    async def summarize(self, transcript: str, prompt: Optional[str] = None) -> str:
        """Summarize using OpenAI GPT"""
        
        default_prompt = """
        You are an expert at summarizing meeting transcripts and conversations.
        Please analyze the following transcript and provide a comprehensive summary with:
        
        1. **Key Topics Discussed**: Main subjects covered in the conversation
        2. **Important Points**: Critical information and insights shared
        3. **Decisions Made**: Any conclusions or decisions reached
        4. **Action Items**: Tasks or follow-ups identified
        5. **Participants**: Key contributors and their main points (if speaker labels present)
        
        Format the summary as clear, concise bullet points.
        
        Transcript:
        {transcript}
        
        Summary:
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional meeting summarizer."},
                    {"role": "user", "content": (prompt or default_prompt).format(transcript=transcript)}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

class LocalLLMProvider(LLMProvider):
    """Local LLM provider using LlamaCpp"""
    
    def __init__(self, model_path: str):
        """Initialize with local model"""
        try:
            self.llm = LlamaCpp(
                model_path=model_path,
                temperature=0.3,
                max_tokens=1000,
                n_ctx=4096,
                n_batch=8,
                n_gpu_layers=35  # Adjust based on your GPU
            )
        except Exception as e:
            logger.error(f"Error loading local model: {e}")
            raise
    
    async def summarize(self, transcript: str, prompt: Optional[str] = None) -> str:
        """Summarize using local LLM"""
        
        default_prompt = """
        Summarize the following transcript into bullet points covering:
        - Key topics discussed
        - Important decisions made
        - Action items identified
        - Main contributions by speakers
        
        Transcript: {transcript}
        
        Summary:
        """
        
        try:
            # Split long transcripts
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=200
            )
            
            if len(transcript) > 2000:
                docs = text_splitter.create_documents([transcript])
                chain = load_summarize_chain(
                    self.llm,
                    chain_type="map_reduce"
                )
                summary = chain.run(docs)
            else:
                summary = self.llm(
                    (prompt or default_prompt).format(transcript=transcript)
                )
            
            return summary
            
        except Exception as e:
            logger.error(f"Local LLM error: {e}")
            raise

class LLMService:
    """Main LLM service for managing different providers"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration"""
        self.provider_name = config.get("provider", "openai")
        self.provider = self._init_provider(config)
    
    def _init_provider(self, config: Dict[str, Any]) -> LLMProvider:
        """Initialize the appropriate provider"""
        
        if self.provider_name == "openai":
            api_key = config.get("openai_api_key")
            if not api_key:
                raise ValueError("OpenAI API key required")
            return OpenAIProvider(
                api_key=api_key,
                model=config.get("model", "gpt-3.5-turbo")
            )
        
        elif self.provider_name == "local":
            model_path = config.get("model_path")
            if not model_path:
                raise ValueError("Local model path required")
            return LocalLLMProvider(model_path=model_path)
        
        else:
            raise ValueError(f"Unknown provider: {self.provider_name}")
    
    async def summarize_transcript(
        self,
        transcript: str,
        prompt: Optional[str] = None,
        include_speaker_analysis: bool = True
    ) -> Dict[str, Any]:
        """
        Summarize transcript with optional speaker analysis
        """
        
        # Enhance prompt if speaker analysis requested
        if include_speaker_analysis and "[Speaker" in transcript:
            enhanced_prompt = prompt or ""
            enhanced_prompt += """
            
            Please also analyze speaker contributions:
            - Identify how many speakers participated
            - Note the main points made by each speaker
            - Highlight any agreements or disagreements
            """
            prompt = enhanced_prompt
        
        try:
            summary = await self.provider.summarize(transcript, prompt)
            
            # Extract speaker statistics if present
            speaker_stats = self._analyze_speakers(transcript)
            
            return {
                "summary": summary,
                "speaker_stats": speaker_stats,
                "provider": self.provider_name,
                "word_count": len(transcript.split()),
                "estimated_duration": len(transcript.split()) / 150  # Rough estimate: 150 words/min
            }
            
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            # Fallback to basic summary
            return {
                "summary": self._create_fallback_summary(transcript),
                "speaker_stats": self._analyze_speakers(transcript),
                "provider": "fallback",
                "error": str(e)
            }
    
    def _analyze_speakers(self, transcript: str) -> Dict[str, Any]:
        """Analyze speaker participation in transcript"""
        
        speakers = {}
        lines = transcript.split('\n')
        
        for line in lines:
            if line.startswith('[Speaker'):
                try:
                    speaker = line.split(':')[0].strip('[]')
                    if speaker not in speakers:
                        speakers[speaker] = {
                            "line_count": 0,
                            "word_count": 0,
                            "percentage": 0
                        }
                    
                    speakers[speaker]["line_count"] += 1
                    text = line.split(':', 1)[1] if ':' in line else ""
                    speakers[speaker]["word_count"] += len(text.split())
                    
                except Exception:
                    continue
        
        # Calculate percentages
        total_words = sum(s["word_count"] for s in speakers.values())
        if total_words > 0:
            for speaker in speakers:
                speakers[speaker]["percentage"] = round(
                    (speakers[speaker]["word_count"] / total_words) * 100, 1
                )
        
        return {
            "speaker_count": len(speakers),
            "speakers": speakers,
            "total_lines": len([l for l in lines if l.strip()])
        }
    
    def _create_fallback_summary(self, transcript: str) -> str:
        """Create a basic summary when LLM fails"""
        
        lines = transcript.split('\n')
        word_count = len(transcript.split())
        
        # Count speakers
        speakers = set()
        for line in lines:
            if line.startswith('[Speaker'):
                speaker = line.split(':')[0].strip('[]')
                speakers.add(speaker)
        
        summary = f"""
        • Total transcript length: {word_count} words
        • Number of participants: {len(speakers) if speakers else 'Unknown'}
        • Estimated duration: {word_count / 150:.1f} minutes
        
        • The transcript contains a conversation or presentation
        • Multiple topics may have been discussed
        • Please use an LLM service for detailed summarization
        """
        
        return summary.strip()

# Example usage function
async def create_llm_service(config_dict: Optional[Dict] = None) -> LLMService:
    """Factory function to create LLM service"""
    
    if config_dict is None:
        # Load from environment
        config_dict = {
            "provider": os.getenv("LLM_PROVIDER", "openai"),
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
            "model": os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
            "model_path": os.getenv("LOCAL_MODEL_PATH")
        }
    
    return LLMService(config_dict)