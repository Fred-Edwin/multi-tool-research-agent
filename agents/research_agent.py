"""Main research agent implementation."""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from config.settings import settings
from tools import WebSearchTool, CalculatorTool, WeatherTool, WikipediaTool, ToolOutput
from utils import QueryParser, ResponseFormatter

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format=settings.log_format
)
logger = logging.getLogger(__name__)


class AgentState(BaseModel):
    """State model for the research agent."""
    original_query: str
    parsed_query: Dict[str, Any] = {}
    sub_queries: List[str] = []
    selected_tools: List[str] = []
    tool_results: List[ToolOutput] = []
    final_answer: str = ""
    error_message: str = ""


class ResearchAgent:
    """Main research agent that coordinates multiple tools to answer complex queries."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens
        )
        
        # Initialize tools
        self.tools = {
            'web_search': WebSearchTool(),
            'calculator': CalculatorTool(),
            'weather': WeatherTool(),
            'wikipedia': WikipediaTool()
        }
        
        # Initialize utilities
        self.query_parser = QueryParser()
        self.response_formatter = ResponseFormatter()
        
        self.logger = logger
    
    async def process_query(self, query: str) -> str:
        """
        Process a user query and return a comprehensive answer.
        
        Args:
            query: The user's question or request
            
        Returns:
            Formatted response string
        """
        state = AgentState(original_query=query)
        
        try:
            # Step 1: Parse and analyze the query
            self.logger.info(f"Processing query: {query}")
            state = await self._parse_query(state)
            
            # Step 2: Select appropriate tools
            state = await self._select_tools(state)
            
            # Step 3: Execute tools
            state = await self._execute_tools(state)
            
            # Step 4: Synthesize results
            state = await self._synthesize_results(state)
            
            # Step 5: Format final response
            return self._format_response(state)
            
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            state.error_message = str(e)
            return self.response_formatter.format_error_response(query, str(e))
    
    async def _parse_query(self, state: AgentState) -> AgentState:
        """Parse the user query into actionable components."""
        try:
            parsed = await self.query_parser.parse_query(state.original_query)
            state.parsed_query = parsed
            state.sub_queries = parsed.get('sub_queries', [state.original_query])
            
            self.logger.info(f"Parsed query into {len(state.sub_queries)} sub-queries")
            return state
            
        except Exception as e:
            self.logger.error(f"Query parsing failed: {str(e)}")
            # Fallback: treat as single query
            state.sub_queries = [state.original_query]
            state.parsed_query = {'complexity': 'low', 'required_tools': []}
            return state
    
    async def _select_tools(self, state: AgentState) -> AgentState:
        """Select appropriate tools based on query analysis."""
        try:
            selected_tools = set()
            
            # Use required tools from parser
            if state.parsed_query.get('required_tools'):
                selected_tools.update(state.parsed_query['required_tools'])
            
            # Also check each sub-query against tool relevance
            for sub_query in state.sub_queries:
                for tool_name, tool in self.tools.items():
                    if tool.is_relevant(sub_query):
                        selected_tools.add(tool_name)
            
            # Ensure we have at least one tool
            if not selected_tools:
                # Default tools for general queries
                if any(keyword in state.original_query.lower() for keyword in ['what', 'who', 'when', 'where']):
                    selected_tools.add('wikipedia')
                selected_tools.add('web_search')  # Fallback
            
            state.selected_tools = list(selected_tools)
            self.logger.info(f"Selected tools: {state.selected_tools}")
            return state
            
        except Exception as e:
            self.logger.error(f"Tool selection failed: {str(e)}")
            state.selected_tools = ['web_search']  # Safe fallback
            return state
    
    async def _execute_tools(self, state: AgentState) -> AgentState:
        """Execute selected tools with appropriate queries."""
        try:
            tasks = []
            
            # Create tasks for each tool
            for tool_name in state.selected_tools:
                if tool_name in self.tools:
                    tool = self.tools[tool_name]
                    
                    # Determine best query for this tool
                    best_query = self._select_query_for_tool(
                        tool_name, state.original_query, state.sub_queries
                    )
                    
                    # Create async task
                    task = self._execute_single_tool(tool, best_query)
                    tasks.append(task)
                    
                    self.logger.info(f"Queuing {tool_name} with query: {best_query}")
            
            # Execute all tools concurrently
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for result in results:
                    if isinstance(result, Exception):
                        self.logger.error(f"Tool execution error: {str(result)}")
                    elif isinstance(result, ToolOutput):
                        state.tool_results.append(result)
            
            self.logger.info(f"Executed {len(state.tool_results)} tools successfully")
            return state
            
        except Exception as e:
            self.logger.error(f"Tool execution failed: {str(e)}")
            return state
    
    async def _execute_single_tool(self, tool, query: str) -> ToolOutput:
        """Execute a single tool with timeout and retry logic."""
        from tools.base_tool import ToolInput
        
        max_retries = settings.max_retries
        timeout = settings.tool_timeout
        
        for attempt in range(max_retries):
            try:
                # Execute with timeout
                result = await asyncio.wait_for(
                    tool.execute(ToolInput(query=query)),
                    timeout=timeout
                )
                
                if result.error:
                    self.logger.warning(f"{tool.name} returned error: {result.error}")
                else:
                    self.logger.info(f"{tool.name} executed successfully")
                
                return result
                
            except asyncio.TimeoutError:
                self.logger.warning(f"{tool.name} timed out (attempt {attempt + 1})")
                if attempt == max_retries - 1:
                    return tool._create_error_output(f"Tool timed out after {timeout}s")
                
            except Exception as e:
                self.logger.error(f"{tool.name} error on attempt {attempt + 1}: {str(e)}")
                if attempt == max_retries - 1:
                    return tool._create_error_output(f"Tool failed: {str(e)}")
                
                # Wait before retry
                await asyncio.sleep(1)
    
    def _select_query_for_tool(self, tool_name: str, original_query: str, sub_queries: List[str]) -> str:
        """Select the most appropriate query for a specific tool."""
        # Tool-specific logic for query selection
        tool_preferences = {
            'calculator': ['calculate', 'compute', 'math', '%', 'interest', 'equation'],
            'weather': ['weather', 'temperature', 'climate'],
            'wikipedia': ['who is', 'what is', 'history', 'biography', 'definition'],
            'web_search': ['current', 'recent', 'news', 'latest', 'today']
        }
        
        preferred_keywords = tool_preferences.get(tool_name, [])
        
        # Check sub-queries first
        for sub_query in sub_queries:
            if any(keyword in sub_query.lower() for keyword in preferred_keywords):
                return sub_query
        
        # Check original query
        if any(keyword in original_query.lower() for keyword in preferred_keywords):
            return original_query
        
        # Return the first sub-query or original query as fallback
        return sub_queries[0] if sub_queries else original_query
    
    async def _synthesize_results(self, state: AgentState) -> AgentState:
        """Synthesize results from multiple tools into a coherent answer."""
        try:
            if not state.tool_results:
                state.final_answer = "I wasn't able to find information to answer your question."
                return state
            
            # Filter successful results
            successful_results = [r for r in state.tool_results if not r.error]
            
            if not successful_results:
                state.final_answer = "All information sources encountered errors. Please try again later."
                return state
            
            # Use LLM to synthesize results
            synthesis_prompt = self._create_synthesis_prompt(
                state.original_query, successful_results
            )
            
            messages = [
                SystemMessage(content="You are a research assistant. Synthesize the provided information into a comprehensive, accurate answer."),
                HumanMessage(content=synthesis_prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            state.final_answer = response.content
            
            self.logger.info("Successfully synthesized results")
            return state
            
        except Exception as e:
            self.logger.error(f"Result synthesis failed: {str(e)}")
            # Fallback: combine results without LLM synthesis
            state.final_answer = self._simple_synthesis(state.tool_results)
            return state
    
    def _create_synthesis_prompt(self, query: str, results: List[ToolOutput]) -> str:
        """Create a prompt for result synthesis."""
        prompt = f"Original question: {query}\n\nInformation gathered:\n\n"
        
        for i, result in enumerate(results, 1):
            source_name = result.source.replace('_', ' ').title()
            prompt += f"{i}. From {source_name}:\n{result.result}\n\n"
        
        prompt += """Please synthesize this information into a comprehensive answer that:
1. Directly answers the original question
2. Combines relevant information from all sources
3. Is clear and well-organized
4. Mentions when information comes from specific sources
5. Notes any contradictions or uncertainties

Answer:"""
        
        return prompt
    
    def _simple_synthesis(self, results: List[ToolOutput]) -> str:
        """Simple fallback synthesis without LLM."""
        successful_results = [r for r in results if not r.error]
        
        if not successful_results:
            return "No information could be retrieved."
        
        synthesis = "Based on the available information:\n\n"
        
        for result in successful_results:
            source_name = result.source.replace('_', ' ').title()
            synthesis += f"**{source_name}:** {result.result}\n\n"
        
        return synthesis.strip()
    
    def _format_response(self, state: AgentState) -> str:
        """Format the final response for the user."""
        try:
            if state.error_message:
                return self.response_formatter.format_error_response(
                    state.original_query, state.error_message
                )
            
            # Check if we have any successful results
            successful_results = [r for r in state.tool_results if not r.error]
            failed_tools = [r.source for r in state.tool_results if r.error]
            
            if successful_results:
                return self.response_formatter.format_final_response(
                    state.original_query,
                    successful_results,
                    state.final_answer
                )
            else:
                return self.response_formatter.format_error_response(
                    state.original_query,
                    "All information sources failed to provide results."
                )
                
        except Exception as e:
            self.logger.error(f"Response formatting failed: {str(e)}")
            return f"# Error\n\nSorry, I encountered an error while formatting the response to your query: {state.original_query}"