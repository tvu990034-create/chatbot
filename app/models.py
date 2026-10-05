"""
Database models for Phase 2 - User authentication and conversation management
"""
from datetime import datetime
from typing import List, Optional, ClassVar
from sqlalchemy import (
    Column, Integer, String, DateTime, ForeignKey, Text, Boolean
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session, Mapped
from sqlalchemy.sql import func
from pydantic import BaseModel, EmailStr
from fastapi import HTTPException, status

Base = declarative_base()


# SQLAlchemy Models
class User(Base):
    __allow_unmapped__ = True
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    
    # Relationships
    conversations: ClassVar = relationship(
        "Conversation", back_populates="user", cascade="all, delete-orphan"
    )


class Conversation(Base):
    __allow_unmapped__ = True
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    user: ClassVar = relationship("User", back_populates="conversations")
    messages: ClassVar = relationship(
        "Message", back_populates="conversation", cascade="all, delete-orphan", order_by="Message.created_at"
    )


class Message(Base):
    __allow_unmapped__ = True
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False)
    role = Column(String, nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    conversation: ClassVar = relationship("Conversation", back_populates="messages")


# Pydantic Models for API
class UserBase(BaseModel):
    email: EmailStr


class UserCreate(UserBase):
    password: str


class UserResponse(UserBase):
    id: int
    created_at: datetime
    is_active: bool
    is_admin: bool
    
    class Config:
        from_attributes = True


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    email: Optional[str] = None


class ConversationBase(BaseModel):
    title: Optional[str] = None


class ConversationCreate(ConversationBase):
    pass


class ConversationResponse(ConversationBase):
    id: int
    user_id: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class ConversationWithMessages(ConversationResponse):
    messages: List["MessageResponse"]
    
    class Config:
        from_attributes = True


class MessageBase(BaseModel):
    role: str
    content: str


class MessageCreate(MessageBase):
    conversation_id: int


class MessageResponse(MessageBase):
    id: int
    conversation_id: int
    created_at: datetime
    
    class Config:
        from_attributes = True


class ChatRequest(BaseModel):
    message: str
    seed: int = 42


# Helper functions for conversation title generation
def generate_conversation_title(first_message: str, max_length: int = 50) -> str:
    """Generate a conversation title from the first user message."""
    # Remove common greetings and trim
    title = first_message.strip()
    
    # Remove common prefixes
    prefixes_to_remove = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    for prefix in prefixes_to_remove:
        if title.lower().startswith(prefix):
            title = title[len(prefix):].strip()
            if title.startswith((",", ".", ":", ";")):
                title = title[1:].strip()
    
    # Truncate if too long
    if len(title) > max_length:
        title = title[:max_length].rsplit(' ', 1)[0] + "..."
    
    # Fallback if title is empty after processing
    if not title:
        title = "New Conversation"
    
    return title


# Database helper functions
def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Get user by email."""
    return db.query(User).filter(User.email == email).first()


def create_user(db: Session, user: UserCreate) -> User:
    """Create a new user."""
    from auth import get_password_hash
    
    hashed_password = get_password_hash(user.password)
    db_user = User(
        email=user.email,
        hashed_password=hashed_password
    )
    db.add(db_user)
    try:
        db.commit()
        db.refresh(db_user)
        return db_user
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create user: {str(e)}"
        )


def get_user_conversations(db: Session, user_id: int) -> List[Conversation]:
    """Get all conversations for a user."""
    return db.query(Conversation).filter(Conversation.user_id == user_id).order_by(Conversation.updated_at.desc()).all()


def create_conversation(db: Session, user_id: int, title: Optional[str] = None) -> Conversation:
    """Create a new conversation."""
    if not title:
        title = "New Conversation"
    
    db_conversation = Conversation(
        user_id=user_id,
        title=title
    )
    db.add(db_conversation)
    try:
        db.commit()
        db.refresh(db_conversation)
        return db_conversation
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create conversation: {str(e)}"
        )


def get_conversation_with_messages(db: Session, conversation_id: int, user_id: int) -> Optional[Conversation]:
    """Get a conversation with all its messages, ensuring it belongs to the user."""
    try:
        return (
            db.query(Conversation)
            .filter(Conversation.id == conversation_id, Conversation.user_id == user_id)
            .first()
        )
    except Exception as e:
        print(f"Error retrieving conversation: {e}")
        return None


def create_message(db: Session, message: MessageCreate) -> Message:
    """Create a new message."""
    db_message = Message(
        conversation_id=message.conversation_id,
        role=message.role,
        content=message.content
    )
    db.add(db_message)
    try:
        db.commit()
        db.refresh(db_message)
        
        # Update conversation's updated_at timestamp
        db.query(Conversation).filter(Conversation.id == message.conversation_id).update({
            Conversation.updated_at: func.now()
        })
        db.commit()
        
        return db_message
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create message: {str(e)}"
        )


def delete_conversation(db: Session, conversation_id: int, user_id: int) -> bool:
    """Delete a conversation and its messages."""
    conversation = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id, Conversation.user_id == user_id)
        .first()
    )
    
    if conversation:
        try:
            db.delete(conversation)
            db.commit()
            return True
        except Exception as e:
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete conversation: {str(e)}"
            )
    return False


def get_admin_stats(db: Session) -> dict:
    """Get statistics for admin dashboard."""
    total_users = db.query(User).count()
    total_conversations = db.query(Conversation).count()
    total_messages = db.query(Message).count()
    
    # Average messages per conversation
    if total_conversations > 0:
        avg_messages = total_messages / total_conversations
    else:
        avg_messages = 0
    
    return {
        "total_users": total_users,
        "total_conversations": total_conversations,
        "total_messages": total_messages,
        "avg_messages_per_conversation": avg_messages or 0
    }
