"""
Main entry point for the Stock Price LLM Agent
Provides both interactive and single-query modes.
"""

import argparse
import sys

from agent import StockPriceAgent


def print_welcome():
    """Print welcome message for interactive mode."""
    print("Stock Price Agent - Interactive Mode")
    print("=" * 50)
    print("Ask me about stock prices!")
    print("\nExamples:")
    print("  - What's the Apple stock price for the last week?")
    print("  - Get Tesla stock from Jan 1 to Jan 15, 2024")
    print("  - Show me Microsoft stock prices")
    print("\nType 'quit' or 'exit' to stop.\n")


def interactive_mode(agent: StockPriceAgent):
    """
    Run the agent in interactive mode.
    
    Args:
        agent: Initialized StockPriceAgent instance
    """
    print_welcome()
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            print("\nAgent: ", end="", flush=True)
            response = agent.query(user_input)
            print(f"{response}\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


def single_query_mode(agent: StockPriceAgent, query: str):
    """
    Run the agent with a single query.
    
    Args:
        agent: Initialized StockPriceAgent instance
        query: User query string
    """
    print(f"Query: {query}\n")
    try:
        response = agent.query(query)
        print(f"Response:\n{response}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def main():
    """Main function to run the agent."""
    parser = argparse.ArgumentParser(
        description="Stock Price LLM Agent - Get stock prices using natural language",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Interactive mode with OpenAI:
    python main.py
    
  Interactive mode with Ollama:
    python main.py --provider ollama
    
  Single query with specific model:
    python main.py --provider openai --model gpt-4 --query "What's AAPL stock price last week?"
    
  Single query with Ollama:
    python main.py --provider ollama --model llama3.1 --query "Get Tesla stock prices"
        """
    )
    
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "ollama"],
        default="openai",
        help="LLM provider to use (default: openai)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specific model name (optional, uses provider defaults)"
    )
    
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Single query to run (optional, otherwise interactive mode)"
    )
    
    args = parser.parse_args()
    
    # Print initialization message
    provider_display = args.provider.upper()
    model_display = f" ({args.model})" if args.model else ""
    print(f"Initializing Stock Price Agent with {provider_display}{model_display}...")
    
    try:
        # Create the agent
        agent = StockPriceAgent(provider=args.provider, model=args.model)
        print("Agent ready!\n")
        
        # Run in appropriate mode
        if args.query:
            single_query_mode(agent, args.query)
        else:
            interactive_mode(agent)
            
    except Exception as e:
        print(f"Error initializing agent: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you've installed all requirements: pip install -r requirements.txt")
        print("2. For OpenAI: Set OPENAI_API_KEY environment variable")
        print("3. For Ollama: Make sure Ollama is running and the model is installed")
        sys.exit(1)


if __name__ == "__main__":
    main()
