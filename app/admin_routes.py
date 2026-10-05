"""
Cloud AI Chatbot - Enterprise Edition
Copyright (c) 2026. All rights reserved.

This software is proprietary and confidential. Unauthorized use,
distribution, or modification is strictly prohibited.

For licensing information, contact the development team.
"""

import os
import json
import uuid
import logging
from typing import List, Dict, Optional
from fastapi import HTTPException, Header, Request, APIRouter
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class FeedbackRequest(BaseModel):
    """Request model for submitting feedback."""
    session_id: str
    message_index: int
    rating: str  # "up" or "down"


class ShareRequest(BaseModel):
    """Request model for sharing a conversation."""
    session_id: str


def verify_admin_key(admin_key: str = Header(..., alias="X-Admin-Key")) -> str:
    """
    Verify admin API key for protected endpoints.
    
    Args:
        admin_key: Admin API key from header
        
    Returns:
        The admin key if valid
        
    Raises:
        HTTPException: If key is invalid
    """
    expected_key = os.getenv("ADMIN_API_KEY", "")
    if not expected_key:
        raise HTTPException(status_code=500, detail="Admin API key not configured")
    
    if admin_key != expected_key:
        raise HTTPException(status_code=403, detail="Invalid admin API key")
    
    return admin_key


def register_admin_routes(app, storage, rate_limiter):
    """
    Register admin and feedback routes to the FastAPI app.
    
    Args:
        app: FastAPI application instance
        storage: ConversationStore instance
        rate_limiter: RateLimiter instance
    """
    
    @app.post("/feedback", tags=["Feedback"], summary="Submit feedback on a response")
    async def submit_feedback(req: FeedbackRequest):
        """
        Submit user feedback (thumbs up/down) for a specific message.
        This is for data collection and future fine-tuning.
        """
        if req.rating not in ["up", "down"]:
            raise HTTPException(status_code=400, detail="Rating must be 'up' or 'down'")
        
        await storage.add_feedback(req.session_id, req.message_index, req.rating)
        return {"status": "success", "message": "Feedback recorded"}
    
    @app.get("/admin/sessions", tags=["Admin"], summary="List all sessions")
    async def list_sessions(admin_key: str = Header(..., alias="X-Admin-Key")):
        """
        List all conversation sessions with metadata.
        Requires admin API key.
        """
        verify_admin_key(admin_key)
        sessions = await storage.list_all_sessions()
        return {
            "sessions": [
                {
                    "session_id": s.session_id,
                    "title": s.title,
                    "created_at": s.created_at,
                    "last_activity": s.last_activity,
                    "message_count": s.message_count,
                    "system_prompt": s.system_prompt
                }
                for s in sessions
            ]
        }
    
    @app.get("/admin/sessions/{session_id}", tags=["Admin"], summary="Get session details")
    async def get_session(session_id: str, admin_key: str = Header(..., alias="X-Admin-Key")):
        """
        Get full conversation history for a specific session.
        Requires admin API key.
        """
        verify_admin_key(admin_key)
        
        session = storage.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        messages = storage.get_conversation_history(session_id)
        
        return {
            "session": {
                "session_id": session.session_id,
                "title": session.title,
                "created_at": session.created_at,
                "last_activity": session.last_activity,
                "message_count": session.message_count,
                "system_prompt": session.system_prompt
            },
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp,
                    "sources": msg.sources,
                    "rating": msg.rating
                }
                for msg in messages
            ]
        }
    
    @app.delete("/admin/sessions/{session_id}", tags=["Admin"], summary="Delete a session")
    async def delete_session(session_id: str, admin_key: str = Header(..., alias="X-Admin-Key")):
        """
        Delete a session and all its messages.
        Requires admin API key.
        """
        verify_admin_key(admin_key)
        
        deleted = await storage.delete_session(session_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Also reset rate limiter for this session
        rate_limiter.reset(session_id)
        
        return {"status": "success", "message": "Session deleted"}
    
    @app.get("/admin/stats", tags=["Admin"], summary="Get usage statistics")
    async def get_stats(admin_key: str = Header(..., alias="X-Admin-Key")):
        """
        Get usage statistics for the chatbot.
        Requires admin API key.
        """
        verify_admin_key(admin_key)
        
        stats = await storage.get_usage_stats()
        
        # Add rate limiter stats
        rate_limiter_stats = {
            "active_buckets": len(rate_limiter.buckets),
            "requests_per_minute": rate_limiter.requests_per_minute
        }
        
        return {
            "storage": stats,
            "rate_limiter": rate_limiter_stats
        }
    
    @app.post("/admin/sessions/{session_id}/title", tags=["Admin"], summary="Update session title")
    async def update_session_title(
        session_id: str,
        title: str,
        admin_key: str = Header(..., alias="X-Admin-Key")
    ):
        """
        Update the title of a session.
        Requires admin API key.
        """
        verify_admin_key(admin_key)
        
        session = storage.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        await storage.update_session_title(session_id, title)
        
        return {"status": "success", "message": "Title updated"}
    
    @app.post("/admin/sessions/{session_id}/system-prompt", tags=["Admin"], summary="Update session system prompt")
    async def update_session_system_prompt(
        session_id: str,
        system_prompt: str,
        admin_key: str = Header(..., alias="X-Admin-Key")
    ):
        """
        Update the custom system prompt for a session.
        Requires admin API key.
        """
        verify_admin_key(admin_key)
        
        session = storage.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        await storage.update_session_system_prompt(session_id, system_prompt)
        
        return {"status": "success", "message": "System prompt updated"}
    
    @app.post("/share", tags=["Sharing"], summary="Share a conversation")
    async def share_conversation(req: ShareRequest):
        """
        Generate a shareable link for a conversation.
        Returns a unique share ID and the conversation data.
        """
        session = storage.get_session(req.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        messages = storage.get_conversation_history(req.session_id)
        
        # Generate unique share ID
        share_id = str(uuid.uuid4())
        
        # Prepare conversation data for sharing
        conversation_data = {
            "share_id": share_id,
            "session": {
                "session_id": session.session_id,
                "title": session.title,
                "created_at": session.created_at,
                "message_count": session.message_count
            },
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp
                }
                for msg in messages
            ]
        }
        
        # In production, store this in the database with the share_id
        # For now, return the data directly
        
        return {
            "share_id": share_id,
            "share_url": f"/shared/{share_id}",
            "conversation": conversation_data
        }

