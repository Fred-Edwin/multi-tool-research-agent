"""Response formatting utilities for the Research Agent."""

from typing import List, Dict, Any
from datetime import datetime


class ResponseFormatter:
    """Formats agent responses in a user-friendly manner."""
    
    def __init__(self):
        self.timestamp = datetime.now()
    
    def format_final_response(
        self,
        original_query: str,
        tool_results: List[Any],  # Changed from List[ToolOutput] to List[Any]
        synthesized_answer: str
    ) -> str:
        """
        Format the final response with all tool results and synthesis.
        
        Args:
            original_query: The original user query
            tool_results: List of results from various tools
            synthesized_answer: The synthesized final answer
            
        Returns:
            Formatted response string
        """
        response = f"# Research Results for: {original_query}\n\n"
        
        # Add synthesized answer first
        if synthesized_answer:
            response += f"## Summary\n{synthesized_answer}\n\n"
        
        # Add detailed results from each tool
        if tool_results:
            response += "## Detailed Information\n\n"
            
            for i, result in enumerate(tool_results, 1):
                if result.error:
                    continue  # Skip failed tools in detailed section
                
                tool_name = result.source.replace('_', ' ').title()
                response += f"### {i}. {tool_name} Results\n"
                response += f"{result.result}\n\n"
        
        # Add sources section
        sources = self._extract_sources(tool_results)
        if sources:
            response += "## Sources\n"
            for i, source in enumerate(sources, 1):
                response += f"{i}. {source}\n"
            response += "\n"
        
        # Add metadata
        response += self._format_metadata(tool_results)
        
        return response
    
    def format_error_response(self, query: str, error_message: str) -> str:
        """Format an error response."""
        return f"""# Error Processing Query: {query}

I encountered an error while processing your request:

**Error:** {error_message}

Please try rephrasing your question or check if all required services are available.
"""
    
    def format_partial_response(
        self,
        original_query: str,
        successful_results: List[Any],  # Changed from List[ToolOutput]
        failed_tools: List[str]
    ) -> str:
        """Format a partial response when some tools fail."""
        response = f"# Partial Results for: {original_query}\n\n"
        
        if successful_results:
            # Synthesize available results
            synthesized = self._synthesize_partial_results(successful_results)
            response += f"## Available Information\n{synthesized}\n\n"
            
            # Add detailed results
            response += "## Detailed Results\n\n"
            for i, result in enumerate(successful_results, 1):
                tool_name = result.source.replace('_', ' ').title()
                response += f"### {i}. {tool_name}\n{result.result}\n\n"
        
        # Note failed tools
        if failed_tools:
            response += "## Note\n"
            response += f"Some information sources were unavailable: {', '.join(failed_tools)}\n\n"
        
        return response
    
    def _extract_sources(self, tool_results: List[Any]) -> List[str]:
        """Extract and format sources from tool results."""
        sources = []
        
        for result in tool_results:
            if result.error:
                continue
                
            source_name = result.source.replace('_', ' ').title()
            
            # Try to get URL from metadata
            if result.metadata and 'url' in result.metadata:
                sources.append(f"{source_name}: {result.metadata['url']}")
            elif result.metadata and 'sources' in result.metadata:
                for source_url in result.metadata['sources'][:2]:  # Limit to 2 sources per tool
                    sources.append(f"{source_name}: {source_url}")
            else:
                sources.append(f"{source_name}")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_sources = []
        for source in sources:
            if source not in seen:
                seen.add(source)
                unique_sources.append(source)
        
        return unique_sources
    
    def _format_metadata(self, tool_results: List[Any]) -> str:
        """Format metadata information."""
        metadata_info = []
        
        # Count successful vs failed tools
        successful = sum(1 for r in tool_results if not r.error)
        failed = sum(1 for r in tool_results if r.error)
        
        metadata_info.append(f"**Tools Used:** {successful} successful, {failed} failed")
        
        # Add confidence information
        if successful > 0:
            avg_confidence = sum(r.confidence for r in tool_results if not r.error) / successful
            metadata_info.append(f"**Average Confidence:** {avg_confidence:.1%}")
        
        # Add timestamp
        metadata_info.append(f"**Generated:** {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        
        return "---\n" + " | ".join(metadata_info) + "\n"
    
    def _synthesize_partial_results(self, results: List[Any]) -> str:
        """Create a synthesis from partial results."""
        if not results:
            return "No information could be retrieved at this time."
        
        # Simple synthesis - in a real implementation, you might use an LLM here
        synthesis = "Based on the available information:\n\n"
        
        for result in results:
            if not result.error and result.result:
                # Extract key points from each result
                lines = result.result.split('\n')
                key_info = lines[0] if lines else result.result[:100]
                synthesis += f"• {key_info}\n"
        
        return synthesis.strip()
    
    def format_tool_selection_info(self, selected_tools: List[str], query: str) -> str:
        """Format information about tool selection."""
        if not selected_tools:
            return "No tools selected for this query."
        
        info = f"**Selected Tools for '{query}':**\n"
        tool_descriptions = {
            'web_search': 'Web Search - for current information and news',
            'calculator': 'Calculator - for mathematical calculations',
            'weather': 'Weather - for weather information',
            'wikipedia': 'Wikipedia - for factual/encyclopedic information'
        }
        
        for tool in selected_tools:
            description = tool_descriptions.get(tool, f'{tool} - specialized tool')
            info += f"• {description}\n"
        
        return info