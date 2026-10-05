"""
Admin dashboard API routes for Phase 2
"""
import os
from typing import Dict, Any, List
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func, and_

from ..models import User, Conversation, Message, get_admin_stats
from ..auth import get_current_admin_user
from ..database import get_db
from ..cache import PreWarmedCache

# Admin router
admin_router = APIRouter(prefix="/admin", tags=["admin"])


def get_cache_statistics() -> dict:
    """Get cache statistics from the global cache instance."""
    try:
        # Import here to avoid circular imports
        from ..server import cloud_app
        if hasattr(cloud_app, 'cache') and isinstance(cloud_app.cache, PreWarmedCache):
            return {
                "cache_size": len(cloud_app.cache.exact),
                "hit_rate": cloud_app.cache.hits / (cloud_app.cache.hits + cloud_app.cache.misses) if (cloud_app.cache.hits + cloud_app.cache.misses) > 0 else 0.0,
                "hits": cloud_app.cache.hits,
                "misses": cloud_app.cache.misses,
                "hamming_threshold": cloud_app.cache.hamming_threshold
            }
    except Exception:
        pass
    return {
        "cache_size": 0,
        "hit_rate": 0.0,
        "hits": 0,
        "misses": 0,
        "hamming_threshold": 4
    }


@admin_router.get("/stats")
async def get_admin_statistics(
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Get comprehensive admin statistics."""
    # Basic counts
    stats = get_admin_stats(db)
    
    # Additional statistics
    total_users = db.query(User).count()
    active_users = db.query(User).filter(User.is_active == True).count()
    admin_users = db.query(User).filter(User.is_admin == True).count()
    
    # Recent activity (last 24 hours)
    yesterday = datetime.utcnow() - timedelta(days=1)
    recent_conversations = db.query(Conversation).filter(
        Conversation.created_at >= yesterday
    ).count()
    recent_messages = db.query(Message).filter(
        Message.created_at >= yesterday
    ).count()
    
    # User activity breakdown
    user_activity = (
        db.query(
            func.count(Message.id).label('message_count'),
            User.email
        )
        .join(Conversation, Message.conversation_id == Conversation.id)
        .join(User, Conversation.user_id == User.id)
        .filter(Message.created_at >= yesterday)
        .group_by(User.id, User.email)
        .order_by(func.count(Message.id).desc())
        .limit(10)
        .all()
    )
    
    # Cache statistics
    cache_stats = get_cache_statistics()
    
    return {
        **stats,
        "active_users": active_users,
        "admin_users": admin_users,
        "recent_conversations_24h": recent_conversations,
        "recent_messages_24h": recent_messages,
        "top_active_users_24h": [
            {"email": email, "message_count": message_count}
            for message_count, email in user_activity
        ],
        "cache": cache_stats
    }


@admin_router.get("/users")
async def list_users(
    skip: int = 0,
    limit: int = 50,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """List all users with pagination."""
    users = (
        db.query(User)
        .offset(skip)
        .limit(limit)
        .all()
    )
    
    total_users = db.query(User).count()
    
    return {
        "users": [
            {
                "id": user.id,
                "email": user.email,
                "created_at": user.created_at,
                "is_active": user.is_active,
                "is_admin": user.is_admin,
                "conversation_count": len(user.conversations)
            }
            for user in users
        ],
        "total": total_users,
        "skip": skip,
        "limit": limit
    }


@admin_router.get("/users/{user_id}")
async def get_user_details(
    user_id: int,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Get detailed information about a specific user."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # User's conversations
    conversations = (
        db.query(Conversation)
        .filter(Conversation.user_id == user_id)
        .order_by(Conversation.updated_at.desc())
        .limit(10)
        .all()
    )
    
    # Message statistics
    total_messages = (
        db.query(func.count(Message.id))
        .join(Conversation, Message.conversation_id == Conversation.id)
        .filter(Conversation.user_id == user_id)
        .scalar()
    )
    
    # Recent activity
    last_week = datetime.utcnow() - timedelta(days=7)
    recent_messages = (
        db.query(func.count(Message.id))
        .join(Conversation, Message.conversation_id == Conversation.id)
        .filter(and_(
            Conversation.user_id == user_id,
            Message.created_at >= last_week
        ))
        .scalar()
    )
    
    return {
        "user": {
            "id": user.id,
            "email": user.email,
            "created_at": user.created_at,
            "is_active": user.is_active,
            "is_admin": user.is_admin
        },
        "statistics": {
            "total_conversations": len(user.conversations),
            "total_messages": total_messages,
            "recent_messages_7d": recent_messages
        },
        "recent_conversations": [
            {
                "id": conv.id,
                "title": conv.title,
                "created_at": conv.created_at,
                "updated_at": conv.updated_at,
                "message_count": len(conv.messages)
            }
            for conv in conversations
        ]
    }


@admin_router.post("/users/{user_id}/toggle-active")
async def toggle_user_active_status(
    user_id: int,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Toggle a user's active status."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Prevent deactivating the last admin
    if user.is_admin and user.is_active:
        admin_count = db.query(User).filter(
            User.is_admin == True, User.is_active == True
        ).count()
        if admin_count <= 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot deactivate the last active admin user"
            )
    
    user.is_active = not user.is_active
    try:
        db.commit()
        return {
            "message": f"User {user.email} {'activated' if user.is_active else 'deactivated'}",
            "is_active": user.is_active
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update user status: {str(e)}"
        )


@admin_router.post("/users/{user_id}/toggle-admin")
async def toggle_user_admin_status(
    user_id: int,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Toggle a user's admin status."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Prevent removing admin from the last admin
    if user.is_admin:
        admin_count = db.query(User).filter(
            User.is_admin == True, User.is_active == True
        ).count()
        if admin_count <= 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot remove admin status from the last admin user"
            )
    
    user.is_admin = not user.is_admin
    try:
        db.commit()
        return {
            "message": f"User {user.email} {'granted' if user.is_admin else 'revoked'} admin privileges",
            "is_admin": user.is_admin
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update user admin status: {str(e)}"
        )


@admin_router.get("/conversations")
async def list_all_conversations(
    skip: int = 0,
    limit: int = 50,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """List all conversations with pagination."""
    conversations = (
        db.query(Conversation)
        .order_by(Conversation.updated_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    
    total_conversations = db.query(Conversation).count()
    
    return {
        "conversations": [
            {
                "id": conv.id,
                "title": conv.title,
                "user_email": conv.user.email,
                "created_at": conv.created_at,
                "updated_at": conv.updated_at,
                "message_count": len(conv.messages)
            }
            for conv in conversations
        ],
        "total": total_conversations,
        "skip": skip,
        "limit": limit
    }


@admin_router.get("/conversations/{conversation_id}")
async def get_conversation_details(
    conversation_id: int,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Get detailed information about a conversation."""
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id
    ).first()
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    return {
        "conversation": {
            "id": conversation.id,
            "title": conversation.title,
            "user_email": conversation.user.email,
            "created_at": conversation.created_at,
            "updated_at": conversation.updated_at
        },
        "messages": [
            {
                "id": msg.id,
                "role": msg.role,
                "content": msg.content,
                "created_at": msg.created_at
            }
            for msg in conversation.messages
        ]
    }


@admin_router.delete("/conversations/{conversation_id}")
async def delete_conversation_admin(
    conversation_id: int,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Delete a conversation (admin only)."""
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id
    ).first()
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    try:
        db.delete(conversation)
        db.commit()
        return {"message": "Conversation deleted successfully"}
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete conversation: {str(e)}"
        )


@admin_router.get("/system/health")
async def get_system_health(
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Get system health information."""
    # Database health
    try:
        from sqlalchemy import text
        db.execute(text("SELECT 1"))
        db_healthy = True
        db_error = None
    except Exception as e:
        db_healthy = False
        db_error = str(e)
    
    # Cache health
    cache_stats = get_cache_statistics()
    cache_healthy = cache_stats is not None
    
    # System info
    import psutil
    import platform
    
    import os
    # Use platform-independent path for disk usage
    disk_path = os.path.abspath('/') if os.name != 'nt' else 'C:\\'
    system_info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(),
        "memory_total": psutil.virtual_memory().total,
        "memory_available": psutil.virtual_memory().available,
        "disk_usage": psutil.disk_usage(disk_path).percent
    }
    
    return {
        "database": {
            "healthy": db_healthy,
            "error": db_error
        },
        "cache": {
            "healthy": cache_healthy,
            "stats": cache_stats
        },
        "system": system_info,
        "timestamp": datetime.utcnow().isoformat()
    }


@admin_router.post("/system/clear-cache")
async def clear_system_cache(
    current_user: User = Depends(get_current_admin_user)
):
    """Clear the system cache."""
    try:
        # Access the global cache instance
        from cloud_server import cloud_app
        cloud_app.cache.clear()
        
        return {"message": "System cache cleared successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear cache: {str(e)}"
        )




@admin_router.get("/metrics")
async def get_detailed_metrics(
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Get detailed system metrics for monitoring."""
    
    # Time-based statistics
    now = datetime.utcnow()
    last_hour = now - timedelta(hours=1)
    last_day = now - timedelta(days=1)
    last_week = now - timedelta(weeks=1)
    
    # Message volume over time
    messages_last_hour = (
        db.query(func.count(Message.id))
        .filter(Message.created_at >= last_hour)
        .scalar()
    )
    
    messages_last_day = (
        db.query(func.count(Message.id))
        .filter(Message.created_at >= last_day)
        .scalar()
    )
    
    messages_last_week = (
        db.query(func.count(Message.id))
        .filter(Message.created_at >= last_week)
        .scalar()
    )
    
    # Conversation creation over time
    conversations_last_hour = (
        db.query(func.count(Conversation.id))
        .filter(Conversation.created_at >= last_hour)
        .scalar()
    )
    
    conversations_last_day = (
        db.query(func.count(Conversation.id))
        .filter(Conversation.created_at >= last_day)
        .scalar()
    )
    
    conversations_last_week = (
        db.query(func.count(Conversation.id))
        .filter(Conversation.created_at >= last_week)
        .scalar()
    )
    
    # User registration over time
    users_last_day = (
        db.query(func.count(User.id))
        .filter(User.created_at >= last_day)
        .scalar()
    )
    
    users_last_week = (
        db.query(func.count(User.id))
        .filter(User.created_at >= last_week)
        .scalar()
    )
    
    return {
        "messages": {
            "last_hour": messages_last_hour,
            "last_day": messages_last_day,
            "last_week": messages_last_week
        },
        "conversations": {
            "last_hour": conversations_last_hour,
            "last_day": conversations_last_day,
            "last_week": conversations_last_week
        },
        "users": {
            "last_day": users_last_day,
            "last_week": users_last_week
        },
        "cache": get_cache_statistics(),
        "timestamp": now.isoformat()
    }
