"""Tests for the research agent tools."""

import pytest
import asyncio
from unittest.mock import patch, MagicMock

from tools import CalculatorTool, WeatherTool, WikipediaTool, WebSearchTool
from tools.base_tool import ToolInput


class TestCalculatorTool:
    """Test cases for the Calculator tool."""
    
    @pytest.fixture
    def calculator(self):
        return CalculatorTool()
    
    @pytest.mark.asyncio
    async def test_simple_calculation(self, calculator):
        """Test basic arithmetic calculation."""
        input_data = ToolInput(query="calculate 15 + 25")
        result = await calculator.execute(input_data)
        
        assert result.error is None
        assert "40" in result.result
        assert result.confidence > 0.9
    
    @pytest.mark.asyncio
    async def test_percentage_calculation(self, calculator):
        """Test percentage calculation."""
        input_data = ToolInput(query="what is 20% of 150")
        result = await calculator.execute(input_data)
        
        assert result.error is None
        assert "30" in result.result
        assert result.source == "calculator"
    
    @pytest.mark.asyncio
    async def test_compound_interest(self, calculator):
        """Test compound interest calculation."""
        input_data = ToolInput(query="compound interest on $1000 at 5% for 2 years")
        result = await calculator.execute(input_data)
        
        assert result.error is None
        assert result.confidence > 0.9
        # Should be approximately 1102.50
        assert "1102" in result.result or "1103" in result.result
    
    def test_is_relevant(self, calculator):
        """Test relevance detection."""
        assert calculator.is_relevant("calculate 2 + 2")
        assert calculator.is_relevant("what is 15% of 200")
        assert calculator.is_relevant("compound interest on $5000")
        assert not calculator.is_relevant("weather in Tokyo")
        assert not calculator.is_relevant("who is Einstein")
    
    @pytest.mark.asyncio
    async def test_invalid_expression(self, calculator):
        """Test handling of invalid expressions."""
        input_data = ToolInput(query="calculate invalid expression")
        result = await calculator.execute(input_data)
        
        assert result.error is not None
        assert result.confidence == 0.0


class TestWeatherTool:
    """Test cases for the Weather tool."""
    
    @pytest.fixture
    def weather_tool(self):
        return WeatherTool()
    
    def test_extract_location(self, weather_tool):
        """Test location extraction from queries."""
        assert "tokyo" in weather_tool._extract_location("weather in Tokyo").lower()
        assert "new york" in weather_tool._extract_location("How's the weather in New York?").lower()
        assert "london" in weather_tool._extract_location("London weather").lower()
    
    def test_is_relevant(self, weather_tool):
        """Test relevance detection."""
        assert weather_tool.is_relevant("weather in Tokyo")
        assert weather_tool.is_relevant("temperature in London")
        assert weather_tool.is_relevant("how hot is it in Paris")
        assert not weather_tool.is_relevant("calculate 2 + 2")
        assert not weather_tool.is_relevant("who is Einstein")
    
    @pytest.mark.asyncio
    async def test_no_location(self, weather_tool):
        """Test handling when no location is found."""
        input_data = ToolInput(query="weather")
        result = await weather_tool.execute(input_data)
        
        assert result.error is not None
        assert "location" in result.error.lower()


class TestWikipediaTool:
    """Test cases for the Wikipedia tool."""
    
    @pytest.fixture
    def wikipedia_tool(self):
        return WikipediaTool()
    
    def test_is_relevant(self, wikipedia_tool):
        """Test relevance detection."""
        assert wikipedia_tool.is_relevant("who is Albert Einstein")
        assert wikipedia_tool.is_relevant("what is quantum physics")
        assert wikipedia_tool.is_relevant("history of the internet")
        assert not wikipedia_tool.is_relevant("weather in Tokyo")
        assert not wikipedia_tool.is_relevant("calculate 2 + 2")
    
    @pytest.mark.asyncio
    @patch('wikipedia.search')
    @patch('wikipedia.summary')
    @patch('wikipedia.page')
    async def test_successful_search(self, mock_page, mock_summary, mock_search, wikipedia_tool):
        """Test successful Wikipedia search."""
        # Mock Wikipedia responses
        mock_search.return_value = ["Albert Einstein"]
        mock_summary.return_value = "Albert Einstein was a theoretical physicist..."
        
        mock_page_obj = MagicMock()
        mock_page_obj.title = "Albert Einstein"
        mock_page_obj.url = "https://en.wikipedia.org/wiki/Albert_Einstein"
        mock_page_obj.content = "Albert Einstein was a German-born theoretical physicist..."
        mock_page.return_value = mock_page_obj
        
        input_data = ToolInput(query="who is Albert Einstein")
        result = await wikipedia_tool.execute(input_data)
        
        assert result.error is None
        assert "Albert Einstein" in result.result
        assert result.confidence > 0.8
        assert result.source == "wikipedia"


class TestWebSearchTool:
    """Test cases for the Web Search tool."""
    
    @pytest.fixture
    def web_search_tool(self):
        return WebSearchTool()
    
    def test_is_relevant(self, web_search_tool):
        """Test relevance detection."""
        assert web_search_tool.is_relevant("latest news about AI")
        assert web_search_tool.is_relevant("current events today")
        assert web_search_tool.is_relevant("recent developments in technology")
        assert web_search_tool.is_relevant("what is happening now")
    
    @pytest.mark.asyncio
    async def test_empty_query(self, web_search_tool):
        """Test handling of empty query."""
        input_data = ToolInput(query="")
        result = await web_search_tool.execute(input_data)
        
        assert result.error is not None
        assert "empty" in result.error.lower()
    
    def test_format_search_results(self, web_search_tool):
        """Test formatting of search results."""
        mock_results = [
            {"title": "Test Title 1", "url": "http://example1.com", "snippet": "Test snippet 1"},
            {"title": "Test Title 2", "url": "http://example2.com", "snippet": "Test snippet 2"}
        ]
        
        formatted = web_search_tool._format_search_results(mock_results, "test query")
        
        assert "Test Title 1" in formatted
        assert "Test Title 2" in formatted
        assert "http://example1.com" in formatted
        assert "test query" in formatted


# Integration test
@pytest.mark.asyncio
async def test_tool_integration():
    """Test that all tools can be initialized and basic functionality works."""
    tools = [
        CalculatorTool(),
        WeatherTool(),
        WikipediaTool(),
        WebSearchTool()
    ]
    
    # Test that all tools have required methods
    for tool in tools:
        assert hasattr(tool, 'execute')
        assert hasattr(tool, 'is_relevant')
        assert hasattr(tool, 'name')
        assert hasattr(tool, 'description')
        
        # Test with empty query (should handle gracefully)
        input_data = ToolInput(query="")
        result = await tool.execute(input_data)
        assert hasattr(result, 'error')
        assert hasattr(result, 'result')
        assert hasattr(result, 'source')
        assert hasattr(result, 'confidence')


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])