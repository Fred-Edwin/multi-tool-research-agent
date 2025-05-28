"""Wikipedia tool for fetching encyclopedic information."""

import wikipedia
import asyncio
from typing import Optional, List
from .base_tool import BaseTool, ToolInput, ToolOutput


class WikipediaTool(BaseTool):
    """Tool for searching and retrieving information from Wikipedia."""
    
    def __init__(self):
        super().__init__(
            name="wikipedia",
            description="Search Wikipedia for factual and encyclopedic information"
        )
        # Set Wikipedia language (default: English)
        wikipedia.set_lang("en")
    
    async def execute(self, input_data: ToolInput) -> ToolOutput:
        """Execute Wikipedia search for the given query."""
        try:
            query = input_data.query.strip()
            if not query:
                return self._create_error_output("Empty Wikipedia query provided")
            
            # Search for relevant pages
            search_results = await self._search_wikipedia(query)
            if not search_results:
                return ToolOutput(
                    result=f"No Wikipedia articles found for '{query}'",
                    source="wikipedia",
                    confidence=0.1,
                    metadata={"query": query, "search_results": []}
                )
            
            # Get detailed information from the best match
            article_content = await self._get_article_content(search_results[0])
            if not article_content:
                return ToolOutput(
                    result=f"Could not retrieve Wikipedia content for '{query}'",
                    source="wikipedia",
                    confidence=0.2,
                    metadata={"query": query, "search_results": search_results}
                )
            
            # Format the result
            formatted_result = self._format_article_content(
                article_content, search_results[0], query
            )
            
            return ToolOutput(
                result=formatted_result,
                source="wikipedia",
                confidence=0.85,
                metadata={
                    "query": query,
                    "article_title": search_results[0],
                    "search_results": search_results[:3],
                    "url": f"https://en.wikipedia.org/wiki/{search_results[0].replace(' ', '_')}"
                }
            )
            
        except Exception as e:
            self.logger.error(f"Wikipedia tool error: {str(e)}")
            return self._create_error_output(f"Wikipedia search failed: {str(e)}")
    
    async def _search_wikipedia(self, query: str) -> List[str]:
        """Search Wikipedia for relevant articles."""
        try:
            # Run Wikipedia search in thread to avoid blocking
            loop = asyncio.get_event_loop()
            search_results = await loop.run_in_executor(
                None, lambda: wikipedia.search(query, results=5)
            )
            return search_results
            
        except wikipedia.exceptions.DisambiguationError as e:
            # If disambiguation page, return the options
            return e.options[:5]
        except Exception as e:
            self.logger.error(f"Wikipedia search error: {str(e)}")
            return []
    
    async def _get_article_content(self, title: str) -> Optional[str]:
        """Get the content of a Wikipedia article."""
        try:
            loop = asyncio.get_event_loop()
            
            # Get page summary (first few sentences)
            summary = await loop.run_in_executor(
                None, lambda: wikipedia.summary(title, sentences=3)
            )
            
            # Get full page for additional info if needed
            page = await loop.run_in_executor(
                None, lambda: wikipedia.page(title)
            )
            
            return {
                'title': page.title,
                'summary': summary,
                'url': page.url,
                'content': page.content[:1000] if len(page.content) > 1000 else page.content
            }
            
        except wikipedia.exceptions.DisambiguationError as e:
            # Try the first option from disambiguation
            try:
                return await self._get_article_content(e.options[0])
            except:
                return None
        except wikipedia.exceptions.PageError:
            self.logger.warning(f"Wikipedia page not found: {title}")
            return None
        except Exception as e:
            self.logger.error(f"Wikipedia content fetch error: {str(e)}")
            return None
    
    def _format_article_content(self, content: dict, title: str, query: str) -> str:
        """Format Wikipedia article content into readable text."""
        try:
            result = f"Wikipedia information for '{title}':\n\n"
            result += f"{content['summary']}\n\n"
            
            # Add additional relevant content if available
            if 'content' in content and content['content']:
                # Extract first paragraph from full content that's different from summary
                full_content = content['content']
                paragraphs = full_content.split('\n\n')
                
                for paragraph in paragraphs[:2]:
                    if len(paragraph) > 100 and paragraph not in content['summary']:
                        result += f"{paragraph[:300]}...\n\n"
                        break
            
            result += f"Source: {content['url']}"
            return result
            
        except Exception as e:
            self.logger.error(f"Wikipedia formatting error: {str(e)}")
            return f"Wikipedia article found for '{title}' but formatting failed."
    
    def is_relevant(self, query: str) -> bool:
        """Check if Wikipedia is relevant for the query."""
        wikipedia_keywords = [
            "who is", "what is", "who was", "what was", "history of",
            "biography", "definition", "meaning", "explain", "information about",
            "tell me about", "facts about", "born", "died", "founded",
            "invented", "discovered", "created"
        ]
        
        query_lower = query.lower()
        
        # Check for question words that often lead to encyclopedic searches
        question_words = ["who", "what", "when", "where", "why", "how"]
        has_question_word = any(word in query_lower for word in question_words)
        
        # Check for specific Wikipedia-relevant keywords
        has_wiki_keywords = any(keyword in query_lower for keyword in wikipedia_keywords)
        
        # Avoid if it's clearly a calculation, weather, or current news query
        avoid_keywords = ["calculate", "weather", "temperature", "current", "today", "now"]
        should_avoid = any(keyword in query_lower for keyword in avoid_keywords)
        
        return (has_question_word or has_wiki_keywords) and not should_avoid