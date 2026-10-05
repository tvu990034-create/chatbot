"""
Cloud AI Chatbot - Enterprise Edition
Copyright (c) 2026. All rights reserved.

This software is proprietary and confidential. Unauthorized use,
distribution, or modification is strictly prohibited.

For licensing information, contact the development team.
"""

import logging
import asyncio
from typing import Optional
from openai import AsyncOpenAI
import os

logger = logging.getLogger(__name__)


class TextToSpeech:
    """
    Text-to-speech service using OpenAI's TTS API.
    """
    
    def __init__(self, enabled: bool = False, api_key: Optional[str] = None, model: str = "tts-1", voice: str = "alloy"):
        self.enabled = enabled
        self.model = model
        self.voice = voice
        self.client = None
        
        if enabled and api_key:
            self.client = AsyncOpenAI(api_key=api_key)
    
    async def synthesize(self, text: str) -> Optional[bytes]:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to convert to speech
            
        Returns:
            Audio data as bytes (MP3 format)
        """
        if not self.enabled or not self.client:
            logger.warning("TTS is disabled or not configured")
            return None
        
        try:
            response = await self.client.audio.speech.create(
                model=self.model,
                voice=self.voice,
                input=text
            )
            
            # Return audio content as bytes
            return response.content
        
        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
            return None
    
    async def synthesize_stream(self, text: str):
        """
        Synthesize speech from text and stream the audio.
        
        Args:
            text: Text to convert to speech
            
        Yields:
            Audio chunks
        """
        if not self.enabled or not self.client:
            logger.warning("TTS is disabled or not configured")
            return
        
        try:
            response = await self.client.audio.speech.create(
                model=self.model,
                voice=self.voice,
                input=text
            )
            
            # Stream audio content
            for chunk in response.iter_bytes():
                yield chunk
        
        except Exception as e:
            logger.error(f"TTS streaming error: {e}")
            yield b""
