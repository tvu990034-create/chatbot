"""
Cloud AI Chatbot - Enterprise Edition
Copyright (c) 2026. All rights reserved.

This software is proprietary and confidential. Unauthorized use,
distribution, or modification is strictly prohibited.

For licensing information, contact the development team.
"""

"""
Streaming utilities for buffering and chunking token streams
"""
from typing import Iterator, AsyncIterator, List


class BufferedStreamer:
    """Buffers tokens and yields them in chunks for better streaming performance"""
    
    def __init__(self, buffer_size: int = 4):
        """
        Initialize the buffered streamer.
        
        Args:
            buffer_size: Number of tokens to accumulate before yielding a chunk
        """
        self.buffer_size = buffer_size
        self.buffer: List[str] = []
    
    def stream(self, token_stream: Iterator[str]) -> Iterator[str]:
        """
        Wrap a token stream and buffer tokens into chunks.
        
        Args:
            token_stream: Iterator of individual tokens
            
        Yields:
            Chunks of buffered tokens (joined with spaces)
        """
        for token in token_stream:
            self.buffer.append(token)
            
            # Yield when buffer is full
            if len(self.buffer) >= self.buffer_size:
                chunk = " ".join(self.buffer)
                self.buffer = []
                yield chunk
        
        # Flush remaining buffer
        if self.buffer:
            chunk = " ".join(self.buffer)
            self.buffer = []
            yield chunk
    
    async def stream_async(self, token_stream: AsyncIterator[str]) -> AsyncIterator[str]:
        """
        Wrap an async token stream and buffer tokens into chunks.
        
        Args:
            token_stream: Async iterator of individual tokens
            
        Yields:
            Chunks of buffered tokens (joined with spaces)
        """
        async for token in token_stream:
            self.buffer.append(token)
            
            # Yield when buffer is full
            if len(self.buffer) >= self.buffer_size:
                chunk = " ".join(self.buffer)
                self.buffer = []
                yield chunk
        
        # Flush remaining buffer
        if self.buffer:
            chunk = " ".join(self.buffer)
            self.buffer = []
            yield chunk
    
    def reset(self):
        """Clear the buffer"""
        self.buffer = []
