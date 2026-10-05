"""
Cloud AI Chatbot - Enterprise Edition
Copyright (c) 2026. All rights reserved.

This software is proprietary and confidential. Unauthorized use,
distribution, or modification is strictly prohibited.

For licensing information, contact the development team.
"""

import logging
from typing import Dict, Optional
from dataclasses import dataclass, asdict
import json

logger = logging.getLogger(__name__)


@dataclass
class UserProfile:
    """User profile with preferences and custom instructions."""
    user_id: str
    preferences: Dict[str, str]  # e.g., {"language": "Spanish", "tone": "formal"}
    custom_instructions: Optional[str] = None  # Permanent instructions for all conversations
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class PersonalizationStore:
    """
    In-memory store for user preferences and custom instructions.
    For production, this should use a persistent database.
    """
    
    def __init__(self):
        self._profiles: Dict[str, UserProfile] = {}
    
    def get_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile."""
        return self._profiles.get(user_id)
    
    def set_preference(self, user_id: str, key: str, value: str) -> UserProfile:
        """Set a user preference."""
        profile = self._profiles.get(user_id)
        if profile is None:
            profile = UserProfile(user_id=user_id, preferences={})
            self._profiles[user_id] = profile
        
        profile.preferences[key] = value
        return profile
    
    def set_custom_instructions(self, user_id: str, instructions: str) -> UserProfile:
        """Set custom instructions for a user."""
        profile = self._profiles.get(user_id)
        if profile is None:
            profile = UserProfile(user_id=user_id, preferences={})
            self._profiles[user_id] = profile
        
        profile.custom_instructions = instructions
        return profile
    
    def get_system_prompt_additions(self, user_id: str) -> str:
        """
        Get system prompt additions from user preferences and custom instructions.
        Returns a string that can be appended to the system prompt.
        """
        profile = self._profiles.get(user_id)
        if not profile:
            return ""
        
        additions = []
        
        # Add custom instructions if present
        if profile.custom_instructions:
            additions.append(f"Custom Instructions: {profile.custom_instructions}")
        
        # Add preferences as natural language
        if profile.preferences:
            pref_text = ", ".join([f"{k}: {v}" for k, v in profile.preferences.items()])
            additions.append(f"User Preferences: {pref_text}")
        
        return "\n\n".join(additions)
    
    def parse_set_command(self, message: str, user_id: str) -> Optional[str]:
        """
        Parse a /set command to set preferences.
        Format: /set <key> <value>
        Example: /set language Spanish
        """
        if not message.startswith("/set "):
            return None
        
        parts = message[5:].strip().split(maxsplit=1)
        if len(parts) < 2:
            return "Usage: /set <key> <value>"
        
        key, value = parts[0], parts[1]
        self.set_preference(user_id, key, value)
        return f"Set preference: {key} = {value}"
    
    def parse_custom_instructions_command(self, message: str, user_id: str) -> Optional[str]:
        """
        Parse a /custom command to set custom instructions.
        Format: /custom <instructions>
        Example: /custom Always answer in a friendly tone
        """
        if not message.startswith("/custom "):
            return None
        
        instructions = message[8:].strip()
        self.set_custom_instructions(user_id, instructions)
        return f"Set custom instructions: {instructions}"
