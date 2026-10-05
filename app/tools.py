"""
Cloud AI Chatbot - Enterprise Edition
Copyright (c) 2026. All rights reserved.

This software is proprietary and confidential. Unauthorized use,
distribution, or modification is strictly prohibited.

For licensing information, contact the development team.
"""

import logging
import asyncio
import requests
from typing import Dict, Callable, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """Represents a tool/function call."""
    tool_name: str
    parameters: Dict[str, Any]
    result: Optional[str] = None


@dataclass
class ToolDefinition:
    """Definition of a tool that can be called."""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON schema for parameters
    handler: Callable[[Dict[str, Any]], str]


class ToolRegistry:
    """
    Registry for tool/function calling capabilities.
    """
    
    def __init__(self, enabled: bool = False, web_search_enabled: bool = False):
        """
        Initialize tool registry.
        
        Args:
            enabled: Whether tools are enabled
            web_search_enabled: Whether real-time web search is enabled
        """
        self.enabled = enabled
        self.web_search_enabled = web_search_enabled
        self.tools: Dict[str, ToolDefinition] = {}
        
        if self.enabled:
            self._register_default_tools()
    
    def _register_default_tools(self):
        """Register default tools."""
        # Real-time web search (if enabled)
        if self.web_search_enabled:
            self.register_tool(
                ToolDefinition(
                    name="web_search",
                    description="Search the web for real-time information",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"}
                        },
                        "required": ["query"]
                    },
                    handler=self._web_search_handler
                )
            )
        else:
            # Placeholder for web search
            self.register_tool(
                ToolDefinition(
                    name="web_search",
                    description="Search the web for information",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"}
                        },
                        "required": ["query"]
                    },
                    handler=lambda params: f"[Placeholder] Web search for: {params.get('query', '')}"
                )
            )
        
        # Placeholder for weather
        self.register_tool(
            ToolDefinition(
                name="get_weather",
                description="Get current weather for a location",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name or location"}
                    },
                    "required": ["location"]
                },
                handler=lambda params: f"[Placeholder] Weather for: {params.get('location', '')}"
            )
        )
        
        # Placeholder for calculator
        self.register_tool(
            ToolDefinition(
                name="calculator",
                description="Perform mathematical calculations",
                parameters={
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Mathematical expression to evaluate"}
                    },
                    "required": ["expression"]
                },
                handler=lambda params: f"[Placeholder] Calculating: {params.get('expression', '')}"
            )
        )
    
    def _web_search_handler(self, params: Dict[str, Any]) -> str:
        """
        Real-time web search using DuckDuckGo.
        
        Args:
            params: Dictionary with 'query' key
            
        Returns:
            Search results as formatted string
        """
        query = params.get("query", "")
        
        try:
            # Use DuckDuckGo instant answer API
            url = "https://api.duckduckgo.com/"
            response = requests.get(url, params={"q": query, "format": "json"}, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract abstract and related topics
                results = []
                
                if data.get("Abstract"):
                    results.append(f"Abstract: {data['Abstract']}")
                
                if data.get("AbstractText"):
                    results.append(f"Summary: {data['AbstractText']}")
                
                if data.get("AbstractSource"):
                    results.append(f"Source: {data['AbstractSource']}")
                
                if data.get("RelatedTopics"):
                    for topic in data["RelatedTopics"][:3]:  # Top 3 related topics
                        if isinstance(topic, dict) and topic.get("Text"):
                            results.append(f"- {topic['Text']}")
                
                if results:
                    return "\n".join(results)
                else:
                    return f"No results found for: {query}"
            else:
                return f"Search failed for: {query}"
        
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return f"Search error: {str(e)}"
    
    def register_tool(self, tool: ToolDefinition):
        """Register a new tool."""
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def list_tools(self) -> Dict[str, ToolDefinition]:
        """List all available tools."""
        return self.tools.copy()
    
    def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> ToolCall:
        """
        Execute a tool with given parameters.
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Parameters for the tool
            
        Returns:
            ToolCall with result
        """
        if not self.enabled:
            logger.warning(f"Tool execution attempted but tools are disabled: {tool_name}")
            return ToolCall(
                tool_name=tool_name,
                parameters=parameters,
                result="[Tools are disabled]"
            )
        
        tool = self.get_tool(tool_name)
        if not tool:
            logger.warning(f"Tool not found: {tool_name}")
            return ToolCall(
                tool_name=tool_name,
                parameters=parameters,
                result=f"[Tool not found: {tool_name}]"
            )
        
        try:
            result = tool.handler(parameters)
            logger.info(f"Executed tool {tool_name} with result: {result[:100]}...")
            return ToolCall(
                tool_name=tool_name,
                parameters=parameters,
                result=result
            )
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return ToolCall(
                tool_name=tool_name,
                parameters=parameters,
                result=f"[Error executing tool: {str(e)}]"
            )
    
    def detect_tool_trigger(self, query: str) -> Optional[str]:
        """
        Detect if a query should trigger a tool call.
        Simple pattern matching for MVP.
        
        Args:
            query: User query
            
        Returns:
            Tool name if triggered, None otherwise
        """
        if not self.enabled:
            return None
        
        query_lower = query.lower()
        
        # Simple pattern matching
        if "weather" in query_lower or "temperature" in query_lower:
            return "get_weather"
        elif "search" in query_lower or "look up" in query_lower or "find" in query_lower:
            return "web_search"
        elif "calculate" in query_lower or "compute" in query_lower or any(
            op in query_lower for op in ["+", "-", "*", "/", "="]
        ):
            return "calculator"
        
        return None
