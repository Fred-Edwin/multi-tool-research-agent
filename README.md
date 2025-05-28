# Multi-Tool Research Agent

A sophisticated AI agent built with LangGraph that can intelligently use multiple tools to answer complex research queries. The agent breaks down complex questions, selects appropriate tools, and synthesizes comprehensive answers from multiple information sources.

## 🌟 Features

- **Multi-Tool Integration**: Web search, calculator, weather data, and Wikipedia
- **Intelligent Query Parsing**: Breaks complex queries into actionable sub-tasks
- **Concurrent Execution**: Runs multiple tools simultaneously for faster results
- **Result Synthesis**: Combines information from multiple sources into coherent answers
- **Error Handling**: Graceful handling of tool failures with partial results
- **Interactive Mode**: Chat-like interface for continuous research sessions

## 🛠️ Tools Available

1. **Web Search Tool**: Current information and news from the web
2. **Calculator Tool**: Mathematical calculations including compound interest
3. **Weather Tool**: Real-time weather information for any location
4. **Wikipedia Tool**: Factual and encyclopedic information

## 📋 Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Weather API key (from OpenWeatherMap)

## 🚀 Installation

1. **Clone or create the project directory:**
   ```bash
   mkdir multi-tool-research-agent
   cd multi-tool-research-agent
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   WEATHER_API_KEY=your_weather_api_key_here
   ```

4. **Get API Keys:**
   - **OpenAI API Key**: Get from [OpenAI Platform](https://platform.openai.com/api-keys)
   - **Weather API Key**: Get free key from [OpenWeatherMap](https://openweathermap.org/api)

## 💻 Usage

### Command Line Interface

**Single Query:**
```bash
python main.py "What's the weather like in Tokyo and how does it compare to New York?"
```

**Interactive Mode:**
```bash
python main.py --interactive
```

**Verbose Logging:**
```bash
python main.py --verbose "Calculate compound interest on $10000 at 5% for 10 years"
```

### Python Integration

```python
import asyncio
from agents import run_research_agent

async def main():
    result = await run_research_agent("Who won the 2023 Nobel Prize in Physics?")
    print(result)

asyncio.run(main())
```

## 📝 Example Queries

### 🧮 Mathematical Calculations
- "Calculate 15% of 2,500"
- "What's the compound interest on $5,000 at 3% annually for 5 years?"
- "Convert 100 kilometers to miles"

### 🌤️ Weather Information
- "What's the current weather in London?"
- "Compare the temperature in Tokyo and New York"
- "How's the climate in Sydney right now?"

### 📚 Factual Information
- "Who is Elon Musk and what companies did he found?"
- "What is quantum computing and how does it work?"
- "Tell me about the history of artificial intelligence"

### 🔍 Current Information
- "Latest developments in renewable energy"
- "Recent news about space exploration"
- "Current trends in artificial intelligence"

### 🔄 Complex Multi-Tool Queries
- "What's the weather in Paris, calculate a 20% tip on €150, and tell me about the Eiffel Tower"
- "Compare the GDP of Japan and Germany, and what's the current exchange rate?"
- "Calculate my mortgage payment for $300,000 at 4% for 30 years and tell me about housing market trends"

## 🏗️ Project Structure

```
research_agent/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .env.example             # Environment variables template
├── .gitignore              # Git ignore rules
├── main.py                 # Main entry point
├── config/
│   └── settings.py         # Configuration management
├── agents/
│   ├── __init__.py
│   ├── research_agent.py   # Main agent logic
│   └── graph_builder.py    # LangGraph workflow
├── tools/
│   ├── __init__.py
│   ├── base_tool.py        # Tool interface
│   ├── web_search.py       # Web search tool
│   ├── calculator.py       # Calculator tool
│   ├── weather.py          # Weather tool
│   └── wikipedia.py        # Wikipedia tool
├── utils/
│   ├── __init__.py
│   ├── query_parser.py     # Query analysis
│   └── response_formatter.py # Response formatting
└── tests/
    ├── __init__.py
    ├── test_tools.py       # Tool tests
    └── test_agent.py       # Agent tests
```

## 🧪 Testing

Run the test suite:
```bash
pytest tests/ -v
```

Run specific test files:
```bash
pytest tests/test_tools.py -v
pytest tests/test_agent.py -v
```

## ⚙️ Configuration

Edit `config/settings.py` to customize:

- **OpenAI Model**: Change the model (default: gpt-4)
- **Temperature**: Adjust response creativity (default: 0.7)
- **Max Tokens**: Control response length (default: 1000)
- **Tool Timeout**: Set tool execution timeout (default: 30s)
- **Max Retries**: Tool retry attempts (default: 3)

## 📊 Architecture

The agent follows a clean, modular architecture:

1. **Query Parser**: Analyzes queries and breaks them into sub-tasks
2. **Tool Selector**: Chooses appropriate tools based on query content
3. **Tool Executor**: Runs selected tools concurrently with error handling
4. **Result Synthesizer**: Uses LLM to combine results into coherent answers
5. **Response Formatter**: Formats final output with sources and metadata

## 🔧 Development

### Adding New Tools

1. Create a new tool class inheriting from `BaseTool`
2. Implement the `execute()` and `is_relevant()` methods
3. Add the tool to the `ResearchAgent.tools` dictionary
4. Update tool selection logic if needed

Example:
```python
from tools.base_tool import BaseTool, ToolInput, ToolOutput

class MyNewTool(BaseTool):
    def __init__(self):
        super().__init__("my_tool", "Description of my tool")
    
    async def execute(self, input_data: ToolInput) -> ToolOutput:
        # Tool implementation
        pass
    
    def is_relevant(self, query: str) -> bool:
        # Relevance logic
        return "my_keyword" in query.lower()
```

### Code Quality

The project follows best practices:
- **Type Hints**: Full type annotations throughout
- **Error Handling**: Comprehensive exception handling
- **Logging**: Structured logging for debugging
- **Testing**: Unit and integration tests
- **Documentation**: Detailed docstrings

Format code:
```bash
black .
flake8 .
```

## 🚦 Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure your `.env` file has valid API keys
2. **Import Errors**: Check that all dependencies are installed
3. **Tool Timeouts**: Increase timeout values in `config/settings.py`
4. **Rate Limits**: Some APIs have rate limits; the agent includes retry logic

### Debug Mode

Enable verbose logging:
```bash
python main.py --verbose "your query here"
```

## 📈 Performance

- **Concurrent Execution**: Tools run in parallel for faster results
- **Caching**: Consider adding caching for repeated queries
- **Rate Limiting**: Built-in retry logic handles API rate limits
- **Timeout Handling**: Prevents hanging on slow tools

## 🛣️ Future Enhancements

- **Memory**: Add conversation memory for follow-up questions
- **More Tools**: PDF reader, database queries, image analysis
- **Caching**: Redis-based caching for repeated queries
- **Web Interface**: Flask/FastAPI web interface
- **Streaming**: Real-time streaming of intermediate results
- **Tool Marketplace**: Plugin system for community tools

## 📜 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📞 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section
2. Review the test files for usage examples
3. Open an issue on the project repository

---

**Happy Researching! 🔍✨**