"""Web search tool using a simple web search approach."""

import asyncio
import requests
from typing import List, Dict, Any
from bs4 import BeautifulSoup
import re
from .base_tool import BaseTool, ToolInput, ToolOutput


class WebSearchTool(BaseTool):
    """Tool for searching the web and extracting relevant information."""
    
    def __init__(self):
        super().__init__(
            name="web_search",
            description="Search the web for current information and news"
        )
        self.search_engines = [
            "https://duckduckgo.com/html/?q={}",
            # Add more search engines as needed
        ]
    
    async def execute(self, input_data: ToolInput) -> ToolOutput:
        """Execute web search for the given query."""
        try:
            query = input_data.query.strip()
            if not query:
                return self._create_error_output("Empty search query provided")
            
            # Perform search
            search_results = await self._search_web(query)
            
            if not search_results:
                return ToolOutput(
                    result="No relevant web results found for the query.",
                    source="web_search",
                    confidence=0.1,
                    metadata={"query": query, "results_count": 0}
                )
            
            # Format results
            formatted_result = self._format_search_results(search_results, query)
            
            return ToolOutput(
                result=formatted_result,
                source="web_search",
                confidence=0.8,
                metadata={
                    "query": query,
                    "results_count": len(search_results),
                    "sources": [r.get("url", "Unknown") for r in search_results[:3]]
                }
            )
            
        except Exception as e:
            self.logger.error(f"Web search error: {str(e)}")
            return self._create_error_output(f"Web search failed: {str(e)}")
    
    async def _search_web(self, query: str) -> List[Dict[str, Any]]:
        """Perform web search (simplified implementation)."""
        try:
            # This is a simplified implementation
            # In a real application, you'd use proper search APIs like Google Custom Search
            search_url = f"https://duckduckgo.com/html/?q={query.replace(' ', '+')}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(search_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Parse results (simplified)
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # Extract search result links and titles
            for result in soup.find_all('a', class_='result__a')[:5]:
                title = result.get_text(strip=True)
                url = result.get('href', '')
                if title and url:
                    results.append({
                        "title": title,
                        "url": url,
                        "snippet": title  # Simplified - in real implementation, extract snippets
                    })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Search execution error: {str(e)}")
            return []
    
    def _format_search_results(self, results: List[Dict[str, Any]], query: str) -> str:
        """Format search results into a readable response."""
        if not results:
            return f"No web results found for '{query}'"
        
        formatted = f"Web search results for '{query}':\n\n"
        
        for i, result in enumerate(results[:3], 1):
            title = result.get('title', 'No title')
            snippet = result.get('snippet', result.get('title', ''))
            url = result.get('url', '')
            
            formatted += f"{i}. {title}\n"
            if snippet and snippet != title:
                formatted += f"   {snippet}\n"
            if url:
                formatted += f"   Source: {url}\n"
            formatted += "\n"
        
        return formatted.strip()
    
    def is_relevant(self, query: str) -> bool:
        """Check if web search is relevant for the query."""
        web_keywords = [
            "news", "current", "recent", "latest", "today", "now",
            "what is", "who is", "when did", "how to", "search"
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in web_keywords)