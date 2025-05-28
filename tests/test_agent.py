"""Tests for the research agent."""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

from agents import ResearchAgent, AgentState
from tools.base_tool import ToolOutput


class TestResearchAgent:
    """Test cases for the Research Agent."""
    
    @pytest.fixture
    def agent(self):
        """Create a research agent for testing."""
        return ResearchAgent()
    
    @pytest.fixture
    def sample_state(self):
        """Create a sample agent state for testing."""
        return AgentState(
            original_query="What is the weather in Tokyo?",
            sub_queries=["weather in Tokyo"],
            selected_tools=["weather"]
        )
    
    def test_agent_initialization(self, agent):
        """Test that the agent initializes correctly."""
        assert agent.llm is not None
        assert len(agent.tools) == 4
        assert 'web_search' in agent.tools
        assert 'calculator' in agent.tools
        assert 'weather' in agent.tools
        assert 'wikipedia' in agent.tools
        assert agent.query_parser is not None
        assert agent.response_formatter is not None
    
    @pytest.mark.asyncio
    async def test_parse_query(self, agent, sample_state):
        """Test query parsing functionality."""
        with patch.object(agent.query_parser, 'parse_query') as mock_parse:
            mock_parse.return_value = {
                'sub_queries': ['test query'],
                'required_tools': ['web_search'],
                'complexity': 'low'
            }
            
            result = await agent._parse_query(sample_state)
            
            assert result.sub_queries == ['test query']
            assert result.parsed_query['required_tools'] == ['web_search']
            mock_parse.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_select_tools(self, agent, sample_state):
        """Test tool selection functionality."""
        sample_state.parsed_query = {'required_tools': ['weather', 'web_search']}
        
        result = await agent._select_tools(sample_state)
        
        assert 'weather' in result.selected_tools
        assert len(result.selected_tools) >= 1
    
    def test_select_query_for_tool(self, agent):
        """Test query selection for specific tools."""
        original_query = "What's the weather in Tokyo and calculate 2+2?"
        sub_queries = ["weather in Tokyo", "calculate 2+2"]
        
        # Test weather tool query selection
        weather_query = agent._select_query_for_tool('weather', original_query, sub_queries)
        assert "weather" in weather_query.lower()
        
        # Test calculator tool query selection
        calc_query = agent._select_query_for_tool('calculator', original_query, sub_queries)
        assert "calculate" in calc_query.lower() or "2+2" in calc_query
    
    @pytest.mark.asyncio
    async def test_execute_tools_success(self, agent, sample_state):
        """Test successful tool execution."""
        sample_state.selected_tools = ['calculator']
        sample_state.sub_queries = ['calculate 2+2']
        
        # Mock the calculator tool
        mock_result = ToolOutput(
            result="2 + 2 = 4",
            source="calculator",
            confidence=0.95
        )
        
        with patch.object(agent.tools['calculator'], 'execute', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = mock_result
            
            result = await agent._execute_tools(sample_state)
            
            assert len(result.tool_results) == 1
            assert result.tool_results[0].result == "2 + 2 = 4"
            assert result.tool_results[0].source == "calculator"
    
    @pytest.mark.asyncio
    async def test_execute_tools_with_error(self, agent, sample_state):
        """Test tool execution with errors."""
        sample_state.selected_tools = ['calculator']
        sample_state.sub_queries = ['invalid calculation']
        
        # Mock tool returning error
        mock_result = ToolOutput(
            result="",
            source="calculator",
            confidence=0.0,
            error="Invalid expression"
        )
        
        with patch.object(agent.tools['calculator'], 'execute', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = mock_result
            
            result = await agent._execute_tools(sample_state)
            
            assert len(result.tool_results) == 1
            assert result.tool_results[0].error == "Invalid expression"
    
    @pytest.mark.asyncio
    async def test_synthesize_results(self, agent, sample_state):
        """Test result synthesis."""
        # Mock successful tool results
        sample_state.tool_results = [
            ToolOutput(
                result="Tokyo weather: 25°C, sunny",
                source="weather",
                confidence=0.9
            ),
            ToolOutput(
                result="Tokyo is the capital of Japan",
                source="wikipedia",
                confidence=0.85
            )
        ]
        
        with patch.object(agent.llm, 'ainvoke', new_callable=AsyncMock) as mock_llm:
            mock_response = MagicMock()
            mock_response.content = "Tokyo is currently 25°C and sunny. It's the capital of Japan."
            mock_llm.return_value = mock_response
            
            result = await agent._synthesize_results(sample_state)
            
            assert result.final_answer == "Tokyo is currently 25°C and sunny. It's the capital of Japan."
            mock_llm.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_synthesize_results_fallback(self, agent, sample_state):
        """Test synthesis fallback when LLM fails."""
        sample_state.tool_results = [
            ToolOutput(
                result="Test result",
                source="test_tool",
                confidence=0.8
            )
        ]
        
        with patch.object(agent.llm, 'ainvoke', side_effect=Exception("LLM error")):
            result = await agent._synthesize_results(sample_state)
            
            # Should use simple synthesis
            assert "Test result" in result.final_answer
            assert "Test Tool" in result.final_answer
    
    def test_create_synthesis_prompt(self, agent):
        """Test synthesis prompt creation."""
        query = "Test query"
        results = [
            ToolOutput(result="Result 1", source="tool1", confidence=0.8),
            ToolOutput(result="Result 2", source="tool2", confidence=0.9)
        ]
        
        prompt = agent._create_synthesis_prompt(query, results)
        
        assert "Test query" in prompt
        assert "Result 1" in prompt
        assert "Result 2" in prompt
        assert "Tool1" in prompt
        assert "Tool2" in prompt
    
    def test_simple_synthesis(self, agent):
        """Test simple synthesis without LLM."""
        results = [
            ToolOutput(result="Result 1", source="tool1", confidence=0.8),
            ToolOutput(result="Result 2", source="tool2", confidence=0.9),
            ToolOutput(result="", source="failed_tool", confidence=0.0, error="Failed")
        ]
        
        synthesis = agent._simple_synthesis(results)
        
        assert "Result 1" in synthesis
        assert "Result 2" in synthesis
        assert "Tool1" in synthesis
        assert "Tool2" in synthesis
        # Should not include failed tool results
        assert "Failed" not in synthesis
    
    def test_format_response_success(self, agent):
        """Test response formatting for successful execution."""
        state = AgentState(
            original_query="Test query",
            final_answer="Test answer",
            tool_results=[
                ToolOutput(result="Result 1", source="tool1", confidence=0.8),
                ToolOutput(result="Result 2", source="tool2", confidence=0.9)
            ]
        )
        
        with patch.object(agent.response_formatter, 'format_final_response') as mock_format:
            mock_format.return_value = "Formatted response"
            
            result = agent._format_response(state)
            
            assert result == "Formatted response"
            mock_format.assert_called_once_with(
                "Test query",
                state.tool_results,
                "Test answer"
            )
    
    def test_format_response_error(self, agent):
        """Test response formatting for error cases."""
        state = AgentState(
            original_query="Test query",
            error_message="Test error"
        )
        
        with patch.object(agent.response_formatter, 'format_error_response') as mock_format:
            mock_format.return_value = "Error response"
            
            result = agent._format_response(state)
            
            assert result == "Error response"
            mock_format.assert_called_once_with("Test query", "Test error")
    
    @pytest.mark.asyncio
    async def test_process_query_end_to_end(self, agent):
        """Test the complete query processing workflow."""
        query = "What is 2 + 2?"
        
        # Mock all the dependencies
        with patch.object(agent.query_parser, 'parse_query') as mock_parse, \
             patch.object(agent.tools['calculator'], 'execute', new_callable=AsyncMock) as mock_calc, \
             patch.object(agent.llm, 'ainvoke', new_callable=AsyncMock) as mock_llm, \
             patch.object(agent.response_formatter, 'format_final_response') as mock_format:
            
            # Setup mocks
            mock_parse.return_value = {
                'sub_queries': ['calculate 2 + 2'],
                'required_tools': ['calculator'],
                'complexity': 'low'
            }
            
            mock_calc.return_value = ToolOutput(
                result="2 + 2 = 4",
                source="calculator",
                confidence=0.95
            )
            
            mock_llm_response = MagicMock()
            mock_llm_response.content = "The answer is 4."
            mock_llm.return_value = mock_llm_response
            
            mock_format.return_value = "Final formatted response"
            
            # Execute
            result = await agent.process_query(query)
            
            # Verify
            assert result == "Final formatted response"
            mock_parse.assert_called_once_with(query)
            mock_calc.assert_called_once()
            mock_llm.assert_called_once()
            mock_format.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_query_with_exception(self, agent):
        """Test query processing with exception handling."""
        query = "Test query"
        
        with patch.object(agent.query_parser, 'parse_query', side_effect=Exception("Parse error")):
            result = await agent.process_query(query)
            
            # Should return error response
            assert "Error" in result or "error" in result
            assert "Parse error" in result


class TestAgentState:
    """Test cases for the AgentState model."""
    
    def test_agent_state_creation(self):
        """Test creating an AgentState instance."""
        state = AgentState(original_query="Test query")
        
        assert state.original_query == "Test query"
        assert state.parsed_query == {}
        assert state.sub_queries == []
        assert state.selected_tools == []
        assert state.tool_results == []
        assert state.final_answer == ""
        assert state.error_message == ""
    
    def test_agent_state_with_data(self):
        """Test creating an AgentState with full data."""
        tool_result = ToolOutput(
            result="Test result",
            source="test_tool",
            confidence=0.8
        )
        
        state = AgentState(
            original_query="Test query",
            parsed_query={"complexity": "low"},
            sub_queries=["sub query 1"],
            selected_tools=["tool1"],
            tool_results=[tool_result],
            final_answer="Final answer",
            error_message=""
        )
        
        assert state.original_query == "Test query"
        assert state.parsed_query["complexity"] == "low"
        assert len(state.sub_queries) == 1
        assert len(state.selected_tools) == 1
        assert len(state.tool_results) == 1
        assert state.final_answer == "Final answer"


# Integration tests
class TestAgentIntegration:
    """Integration tests for the research agent."""
    
    @pytest.mark.asyncio
    async def test_calculator_integration(self):
        """Test integration with calculator tool."""
        agent = ResearchAgent()
        
        # Mock only the LLM synthesis to avoid API calls
        with patch.object(agent.llm, 'ainvoke', new_callable=AsyncMock) as mock_llm:
            mock_response = MagicMock()
            mock_response.content = "The calculation result is 4."
            mock_llm.return_value = mock_response
            
            result = await agent.process_query("calculate 2 + 2")
            
            # Should contain calculation result
            assert "4" in result
            assert "calculate" in result.lower() or "calculation" in result.lower()
    
    @pytest.mark.asyncio
    async def test_multiple_tools_integration(self):
        """Test integration with multiple tools."""
        agent = ResearchAgent()
        
        # This test requires mocking multiple tools to avoid API calls
        with patch.object(agent.tools['calculator'], 'execute', new_callable=AsyncMock) as mock_calc, \
             patch.object(agent.tools['weather'], 'execute', new_callable=AsyncMock) as mock_weather, \
             patch.object(agent.llm, 'ainvoke', new_callable=AsyncMock) as mock_llm:
            
            mock_calc.return_value = ToolOutput(
                result="2 + 2 = 4",
                source="calculator",
                confidence=0.95
            )
            
            mock_weather.return_value = ToolOutput(
                result="Tokyo: 25°C, sunny",
                source="weather",
                confidence=0.9
            )
            
            mock_response = MagicMock()
            mock_response.content = "The answer is 4 and Tokyo is 25°C."
            mock_llm.return_value = mock_response
            
            result = await agent.process_query("Calculate 2+2 and tell me Tokyo weather")
            
            # Should contain results from both tools
            assert isinstance(result, str)
            assert len(result) > 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])