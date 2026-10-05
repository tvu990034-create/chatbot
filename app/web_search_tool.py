"""
Web Search Tool Integration
Provides web search capabilities for the RAG system to retrieve real-time information.
"""

import os
import json
import requests
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class SearchResult:
    """Represents a single web search result."""
    title: str
    url: str
    snippet: str
    source: str


class WebSearchTool:
    """
    Web search tool that integrates with various search APIs.
    Supports multiple search providers and provides unified interface.
    """
    
    def __init__(self, api_key: Optional[str] = None, provider: str = "duckduckgo"):
        self.api_key = api_key or os.getenv("SEARCH_API_KEY")
        self.provider = provider
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        """
        Perform web search and return results.
        
        Args:
            query: Search query string
            num_results: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        if self.provider == "duckduckgo":
            return self._search_duckduckgo(query, num_results)
        elif self.provider == "google":
            return self._search_google(query, num_results)
        elif self.provider == "bing":
            return self._search_bing(query, num_results)
        else:
            return self._search_duckduckgo(query, num_results)
    
    def _search_duckduckgo(self, query: str, num_results: int) -> List[SearchResult]:
        """Search using DuckDuckGo (free, no API key required)."""
        try:
            url = "https://duckduckgo.com/html/"
            params = {
                'q': query,
                'kl': 'us-en'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            # Parse HTML response (simplified parsing)
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            results = []
            result_divs = soup.find_all('div', class_='result')
            
            for div in result_divs[:num_results]:
                title_tag = div.find('a', class_='result__a')
                snippet_tag = div.find('a', class_='result__snippet')
                
                if title_tag:
                    title = title_tag.get_text(strip=True)
                    url = title_tag.get('href', '')
                    snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
                    
                    results.append(SearchResult(
                        title=title,
                        url=url,
                        snippet=snippet,
                        source="duckduckgo"
                    ))
            
            return results
            
        except Exception as e:
            print(f"DuckDuckGo search error: {e}")
            return []
    
    def _search_google(self, query: str, num_results: int) -> List[SearchResult]:
        """Search using Google Custom Search API (requires API key)."""
        if not self.api_key:
            print("Google search requires API key")
            return []
        
        try:
            search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID", "017576662512468239146:omuauf_lfve")
            url = "https://www.googleapis.com/customsearch/v1"
            
            params = {
                'key': self.api_key,
                'cx': search_engine_id,
                'q': query,
                'num': num_results
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get('items', []):
                results.append(SearchResult(
                    title=item.get('title', ''),
                    url=item.get('link', ''),
                    snippet=item.get('snippet', ''),
                    source="google"
                ))
            
            return results
            
        except Exception as e:
            print(f"Google search error: {e}")
            return []
    
    def _search_bing(self, query: str, num_results: int) -> List[SearchResult]:
        """Search using Bing Search API (requires API key)."""
        if not self.api_key:
            print("Bing search requires API key")
            return []
        
        try:
            url = "https://api.bing.microsoft.com/v7.0/search"
            
            headers = {
                'Ocp-Apim-Subscription-Key': self.api_key
            }
            
            params = {
                'q': query,
                'count': num_results
            }
            
            response = self.session.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get('webPages', {}).get('value', []):
                results.append(SearchResult(
                    title=item.get('name', ''),
                    url=item.get('url', ''),
                    snippet=item.get('snippet', ''),
                    source="bing"
                ))
            
            return results
            
        except Exception as e:
            print(f"Bing search error: {e}")
            return []
    
    def get_search_context(self, query: str, num_results: int = 3) -> str:
        """
        Get formatted search context for RAG.
        
        Args:
            query: Search query
            num_results: Number of results to include
            
        Returns:
            Formatted string with search results
        """
        results = self.search(query, num_results)
        
        if not results:
            return ""
        
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(
                f"Web Search Result {i}:\n"
                f"Title: {result.title}\n"
                f"URL: {result.url}\n"
                f"Content: {result.snippet}\n"
            )
        
        return "\n".join(context_parts)


class HybridSearch:
    """
    Hybrid search combining local knowledge base with web search.
    Provides comprehensive retrieval from both sources.
    """
    
    def __init__(self, retrieval_pipeline, web_search_tool: Optional[WebSearchTool] = None):
        self.retrieval_pipeline = retrieval_pipeline
        self.web_search = web_search_tool or WebSearchTool()
        self.use_web_search = os.getenv("USE_WEB_SEARCH", "true").lower() == "true"
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve relevant documents from both local knowledge base and web.
        
        Args:
            query: Search query
            top_k: Number of results to return from each source
            
        Returns:
            Combined list of retrieved documents with metadata
        """
        results = []
        
        # Retrieve from local knowledge base
        local_results = self.retrieval_pipeline.retrieve(query)
        for doc_id, content, score in local_results[:top_k]:
            results.append({
                'content': content,
                'source': 'local',
                'doc_id': doc_id,
                'score': score,
                'type': 'knowledge_base'
            })
        
        # Retrieve from web search if enabled
        if self.use_web_search:
            web_results = self.web_search.search(query, top_k)
            for result in web_results:
                results.append({
                    'content': f"{result.title}: {result.snippet}",
                    'source': 'web',
                    'doc_id': result.url,
                    'score': 0.8,  # Default score for web results
                    'type': 'web_search',
                    'url': result.url
                })
        
        return results
    
    def get_enhanced_context(self, query: str, max_local: int = 3, max_web: int = 2) -> str:
        """
        Get enhanced context combining local and web sources.
        
        Args:
            query: Search query
            max_local: Maximum local results
            max_web: Maximum web results
            
        Returns:
            Formatted context string
        """
        results = self.retrieve(query, top_k=max(max_local, max_web))
        
        context_parts = []
        local_count = 0
        web_count = 0
        
        for result in results:
            if result['type'] == 'knowledge_base' and local_count < max_local:
                context_parts.append(
                    f"[Local Knowledge - {result['doc_id']}]\n"
                    f"{result['content']}\n"
                )
                local_count += 1
            elif result['type'] == 'web_search' and web_count < max_web:
                context_parts.append(
                    f"[Web Search - {result['source']}]\n"
                    f"Title: {result.get('title', 'N/A')}\n"
                    f"{result['content']}\n"
                )
                web_count += 1
        
        return "\n---\n".join(context_parts)
