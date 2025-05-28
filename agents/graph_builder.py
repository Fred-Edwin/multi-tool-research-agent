"""LangGraph workflow builder for the Research Agent."""

from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .research_agent import ResearchAgent, AgentState


class ResearchAgentGraph:
    """Builds and manages the LangGraph workflow for the Research Agent."""
    
    def __init__(self):
        self.agent = ResearchAgent()
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        # Create the state graph
        workflow = StateGraph(AgentState)
        
        # Add nodes for each step in the research process
        workflow.add_node("parse_query", self._parse_query_node)
        workflow.add_node("select_tools", self._select_tools_node)
        workflow.add_node("execute_tools", self._execute_tools_node)
        workflow.add_node("synthesize_results", self._synthesize_results_node)
        workflow.add_node("format_response", self._format_response_node)
        
        # Define the workflow edges
        workflow.set_entry_point("parse_query")
        workflow.add_edge("parse_query", "select_tools")
        workflow.add_edge("select_tools", "execute_tools")
        workflow.add_edge("execute_tools", "synthesize_results")
        workflow.add_edge("synthesize_results", "format_response")
        workflow.add_edge("format_response", END)
        
        # Add memory for conversation state
        memory = MemorySaver()
        
        # Compile the graph
        return workflow.compile(checkpointer=memory)
    
    async def _parse_query_node(self, state: AgentState) -> Dict[str, Any]:
        """Node for query parsing."""
        updated_state = await self.agent._parse_query(state)
        return updated_state.dict()
    
    async def _select_tools_node(self, state: AgentState) -> Dict[str, Any]:
        """Node for tool selection."""
        if isinstance(state, dict):
            state = AgentState(**state)
        
        updated_state = await self.agent._select_tools(state)
        return updated_state.dict()
    
    async def _execute_tools_node(self, state: AgentState) -> Dict[str, Any]:
        """Node for tool execution."""
        if isinstance(state, dict):
            state = AgentState(**state)
        
        updated_state = await self.agent._execute_tools(state)
        return updated_state.dict()
    
    async def _synthesize_results_node(self, state: AgentState) -> Dict[str, Any]:
        """Node for result synthesis."""
        if isinstance(state, dict):
            state = AgentState(**state)
        
        updated_state = await self.agent._synthesize_results(state)
        return updated_state.dict()
    
    async def _format_response_node(self, state: AgentState) -> Dict[str, Any]:
        """Node for response formatting."""
        if isinstance(state, dict):
            state = AgentState(**state)
        
        # The formatting is handled in the main process method
        # This node just passes through the state
        return state.dict()
    
    async def run(self, query: str, config: Dict[str, Any] = None) -> str:
        """
        Run the research agent workflow.
        
        Args:
            query: The user query to process
            config: Optional configuration for the workflow
            
        Returns:
            The formatted response string
        """
        if config is None:
            config = {"configurable": {"thread_id": "default"}}
        
        # Initialize state
        initial_state = AgentState(original_query=query)
        
        try:
            # Run the workflow
            result = await self.graph.ainvoke(
                initial_state.dict(),
                config=config
            )
            
            # Convert result back to AgentState and format response
            final_state = AgentState(**result)
            return self.agent._format_response(final_state)
            
        except Exception as e:
            self.agent.logger.error(f"Graph execution failed: {str(e)}")
            return self.agent.response_formatter.format_error_response(
                query, f"Workflow execution failed: {str(e)}"
            )
    
    async def stream_run(self, query: str, config: Dict[str, Any] = None):
        """
        Stream the research agent workflow execution.
        
        Args:
            query: The user query to process
            config: Optional configuration for the workflow
            
        Yields:
            Intermediate states and final result
        """
        if config is None:
            config = {"configurable": {"thread_id": "default"}}
        
        initial_state = AgentState(original_query=query)
        
        try:
            async for chunk in self.graph.astream(
                initial_state.dict(),
                config=config
            ):
                yield chunk
                
        except Exception as e:
            self.agent.logger.error(f"Graph streaming failed: {str(e)}")
            error_response = self.agent.response_formatter.format_error_response(
                query, f"Workflow streaming failed: {str(e)}"
            )
            yield {"error": error_response}


# Convenience function to create and run the graph
async def run_research_agent(query: str) -> str:
    """
    Convenience function to run the research agent.
    
    Args:
        query: The user query
        
    Returns:
        The research agent's response
    """
    graph = ResearchAgentGraph()
    return await graph.run(query)