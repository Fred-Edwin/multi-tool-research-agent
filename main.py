"""Main entry point for the Multi-Tool Research Agent."""

import asyncio
import argparse
import sys
from typing import Optional

from agents import run_research_agent
from config.settings import settings


async def main():
    """Main function to run the research agent."""
    parser = argparse.ArgumentParser(
        description="Multi-Tool Research Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "What's the weather like in Tokyo?"
  python main.py "Calculate compound interest on $10000 at 5% for 10 years"
  python main.py "Who won the 2023 Nobel Prize in Physics?"
  python main.py "Compare the population of Tokyo and New York"
        """
    )
    
    parser.add_argument(
        "query",
        nargs="?",
        help="The research query to process"
    )
    
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate configuration
    try:
        settings.validate_required_keys()
    except ValueError as e:
        print(f"Configuration Error: {e}")
        print("Please check your .env file and ensure all required API keys are set.")
        return 1
    
    if args.interactive:
        return await run_interactive_mode()
    elif args.query:
        return await run_single_query(args.query)
    else:
        parser.print_help()
        return 1


async def run_single_query(query: str) -> int:
    """Run a single query and print the result."""
    try:
        print(f"Processing query: {query}")
        print("-" * 50)
        
        result = await run_research_agent(query)
        print(result)
        
        return 0
        
    except KeyboardInterrupt:
        print("\nQuery interrupted by user.")
        return 1
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1


async def run_interactive_mode() -> int:
    """Run the agent in interactive mode."""
    print("🔍 Multi-Tool Research Agent")
    print("=" * 40)
    print("Ask me anything! I can search the web, do calculations, check weather, and more.")
    print("Type 'quit', 'exit', or 'q' to stop.")
    print("Type 'help' for example queries.")
    print()
    
    while True:
        try:
            # Get user input
            query = input("👤 You: ").strip()
            
            if not query:
                continue
            
            # Check for exit commands
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            # Check for help
            if query.lower() in ['help', 'h']:
                print_help()
                continue
            
            # Process the query
            print("\n🤖 Research Agent: Thinking...")
            print("-" * 40)
            
            result = await run_research_agent(query)
            print(result)
            print("\n" + "=" * 40 + "\n")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            print("Please try again with a different query.\n")
    
    return 0


def print_help():
    """Print help information with example queries."""
    help_text = """
📚 Example Queries:

🌍 General Information:
  • "Who is Elon Musk?"
  • "What is quantum computing?"
  • "Tell me about the history of the internet"

🧮 Calculations:
  • "Calculate 15% of 2500"
  • "What's the compound interest on $5000 at 3% for 5 years?"
  • "Convert 100 miles to kilometers"

🌤️  Weather:
  • "What's the weather in London?"
  • "How's the temperature in Tokyo?"
  • "Weather forecast for New York"

🔍 Current Information:
  • "Latest news about artificial intelligence"
  • "Current stock price of Apple"
  • "Recent developments in renewable energy"

🔀 Complex Queries:
  • "Compare the GDP of Japan and Germany"
  • "What's the weather like in Paris and how does it compare to the average?"
  • "Calculate the tip on a $67 meal and tell me about tipping culture"

💡 Tips:
  • Be specific in your questions
  • I can handle multiple parts in one query
  • Ask follow-up questions for more details
"""
    print(help_text)


if __name__ == "__main__":
    # Handle different Python versions
    if sys.version_info >= (3, 7):
        exit_code = asyncio.run(main())
    else:
        loop = asyncio.get_event_loop()
        exit_code = loop.run_until_complete(main())
    
    sys.exit(exit_code)