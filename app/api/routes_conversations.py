"""
Conversation management API routes for Phase 2
"""
import json
import asyncio
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel

from ..models import (
    User, Conversation, Message, ConversationCreate, ConversationResponse,
    ConversationWithMessages, MessageCreate, MessageResponse, ChatRequest,
    get_user_conversations, create_conversation, get_conversation_with_messages,
    create_message, delete_conversation, generate_conversation_title
)
from ..auth import get_current_user
from ..database import get_db

# Conversation router
conversations_router = APIRouter(prefix="/conversations", tags=["conversations"])

# In-memory cache for active conversation states
# This bridges the existing in-memory system with persistent storage
conversation_states = {}


class ConversationChatRequest(BaseModel):
    message: str
    seed: int = 42


@conversations_router.post("", response_model=ConversationResponse)
async def create_new_conversation(
    conversation_data: ConversationCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new conversation for the current user."""
    conversation = create_conversation(
        db=db,
        user_id=current_user.id,
        title=conversation_data.title
    )
    return conversation


@conversations_router.get("", response_model=List[ConversationResponse])
async def list_conversations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List all conversations for the current user."""
    conversations = get_user_conversations(db=db, user_id=current_user.id)
    return conversations


@conversations_router.get("/{conversation_id}", response_model=ConversationWithMessages)
async def get_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get a conversation with all its messages."""
    conversation = get_conversation_with_messages(
        db=db,
        conversation_id=conversation_id,
        user_id=current_user.id
    )
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    return conversation


@conversations_router.delete("/{conversation_id}")
async def delete_conversation_endpoint(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a conversation and all its messages."""
    success = delete_conversation(
        db=db,
        conversation_id=conversation_id,
        user_id=current_user.id
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    # Clean up in-memory state
    if conversation_id in conversation_states:
        del conversation_states[conversation_id]
    
    return {"message": "Conversation deleted successfully"}


@conversations_router.post("/{conversation_id}/chat")
async def chat_in_conversation(
    conversation_id: int,
    chat_request: ConversationChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Send a message in a conversation and stream the response."""
    # Verify conversation belongs to user
    conversation = get_conversation_with_messages(
        db=db,
        conversation_id=conversation_id,
        user_id=current_user.id
    )
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    # Get or create conversation state from cache
    cache_key = f"conv_{conversation_id}"
    if cache_key not in conversation_states:
        from identity import ConversationState
        conversation_states[cache_key] = ConversationState()
    
    state = conversation_states[cache_key]
    
    # Save user message to database
    user_message = MessageCreate(
        conversation_id=conversation_id,
        role="user",
        content=chat_request.message
    )
    create_message(db=db, message=user_message)
    
    # Generate title for new conversations (if this is the first message)
    if len(conversation.messages) == 0:
        title = generate_conversation_title(chat_request.message)
        conversation.title = title
        try:
            db.commit()
        except Exception as e:
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to update conversation title: {str(e)}"
            )
    
    # Get CloudChatApp instance (reuse existing one)
    if cloud_app is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat service is currently unavailable"
        )
    
    # Create streaming response generator
    async def stream_chat_response():
        assistant_response = ""
        retrieval_latency_ms = 0
        metrics = {}
        
        try:
            # Stream response from existing chat logic
            async for chunk in cloud_app.stream_chat(
                CloudChatRequest(
                    message=chat_request.message,
                    session_id=f"conv_{conversation_id}",
                    seed=chat_request.seed
                )
            ):
                if chunk.startswith("data: "):
                    data_str = chunk[6:]  # Remove "data: " prefix
                    
                    try:
                        data = json.loads(data_str)
                        
                        if data.get("type") == "token":
                            token_value = data.get("value", "")
                            assistant_response += token_value
                            yield chunk
                        
                        elif data.get("type") == "done":
                            metrics = data.get("metrics", {})
                            retrieval_latency_ms = data.get("retrieval_latency_ms", 0)
                            
                            # Save assistant response to database
                            assistant_message = MessageCreate(
                                conversation_id=conversation_id,
                                role="assistant",
                                content=assistant_response.strip()
                            )
                            create_message(db=db, message=assistant_message)
                            
                            yield chunk
                            break
                            
                    except json.JSONDecodeError:
                        # Pass through invalid JSON chunks
                        yield chunk
                else:
                    yield chunk
                    
        except Exception as e:
            # In case of error, still yield what we have
            error_chunk = f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            yield error_chunk
    
    return StreamingResponse(
        stream_chat_response(),
        media_type="text/event-stream"
    )


@conversations_router.post("/{conversation_id}/rename")
async def rename_conversation(
    conversation_id: int,
    new_title: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Rename a conversation."""
    conversation = get_conversation_with_messages(
        db=db,
        conversation_id=conversation_id,
        user_id=current_user.id
    )
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    try:
        conversation.title = new_title.strip()
        db.commit()
        return {"message": "Conversation renamed successfully"}
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to rename conversation: {str(e)}"
        )


@conversations_router.get("/{conversation_id}/export")
async def export_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Export a conversation as text."""
    conversation = get_conversation_with_messages(
        db=db,
        conversation_id=conversation_id,
        user_id=current_user.id
    )
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    # Generate text export
    export_text = f"Conversation: {conversation.title}\n"
    export_text += f"Created: {conversation.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
    export_text += f"Last Updated: {conversation.updated_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
    export_text += "=" * 50 + "\n\n"
    
    for message in conversation.messages:
        timestamp = message.created_at.strftime('%Y-%m-%d %H:%M:%S')
        export_text += f"[{timestamp}] {message.role.upper()}:\n"
        export_text += f"{message.content}\n\n"
    
    return {
        "filename": f"conversation_{conversation_id}_{conversation.title.replace(' ', '_')}.txt",
        "content": export_text
    }


@conversations_router.post("/{conversation_id}/clear-cache")
async def clear_conversation_cache(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Clear the in-memory cache for a conversation."""
    # Verify conversation belongs to user
    conversation = get_conversation_with_messages(
        db=db,
        conversation_id=conversation_id,
        user_id=current_user.id
    )
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    # Clear in-memory state
    cache_key = f"conv_{conversation_id}"
    if cache_key in conversation_states:
        del conversation_states[cache_key]
    
    return {"message": "Conversation cache cleared successfully"}


# Utility endpoints
@conversations_router.get("/stats/summary")
async def get_conversation_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get summary statistics for user's conversations."""
    conversations = get_user_conversations(db=db, user_id=current_user.id)
    
    total_messages = sum(len(conv.messages) for conv in conversations)
    total_conversations = len(conversations)
    
    # Calculate average messages per conversation
    avg_messages = total_messages / total_conversations if total_conversations > 0 else 0
    
    # Find most recent conversation
    most_recent = max(conversations, key=lambda c: c.updated_at) if conversations else None
    
    return {
        "total_conversations": total_conversations,
        "total_messages": total_messages,
        "average_messages_per_conversation": round(avg_messages, 2),
        "most_recent_conversation": {
            "id": most_recent.id,
            "title": most_recent.title,
            "updated_at": most_recent.updated_at
        } if most_recent else None
    }
