"""
Examples of using the Stock Price LLM Agent
"""

from agent import StockPriceAgent


def example_basic_usage():
    """Basic usage with OpenAI"""
    print("=" * 60)
    print("Example 1: Basic Usage with OpenAI")
    print("=" * 60)
    
    agent = StockPriceAgent(provider="openai")
    
    queries = [
        "What's the Apple stock price for the last week?",
        "Get Tesla stock from January 1 to January 15, 2024",
        "Show me Microsoft stock prices"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 60)
        response = agent.query(query)
        print(f"Response:\n{response}\n")


def example_ollama_usage():
    """Using Ollama for local inference"""
    print("=" * 60)
    print("Example 2: Using Ollama (Local LLM)")
    print("=" * 60)
    
    agent = StockPriceAgent(provider="ollama", model="llama3.1")
    
    query = "What was Amazon's stock price in the last 5 days?"
    print(f"\nQuery: {query}")
    print("-" * 60)
    response = agent.query(query)
    print(f"Response:\n{response}\n")


def example_specific_dates():
    """Queries with specific date ranges"""
    print("=" * 60)
    print("Example 3: Specific Date Ranges")
    print("=" * 60)
    
    agent = StockPriceAgent(provider="openai")
    
    queries = [
        "Get Google stock from December 1, 2023 to December 31, 2023",
        "What was Netflix stock price on January 15, 2024?",
        "Show me Nvidia stock for the first week of February 2024"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 60)
        response = agent.query(query)
        print(f"Response:\n{response}\n")


def example_company_names():
    """Using company names instead of symbols"""
    print("=" * 60)
    print("Example 4: Using Company Names")
    print("=" * 60)
    
    agent = StockPriceAgent(provider="openai")
    
    queries = [
        "What's Microsoft's stock price this week?",
        "Show me Facebook's stock for the last month",
        "Get Amazon stock prices from last week"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 60)
        response = agent.query(query)
        print(f"Response:\n{response}\n")


def example_comparison_query():
    """Asking for analysis or comparison"""
    print("=" * 60)
    print("Example 5: Analysis and Comparison")
    print("=" * 60)
    
    agent = StockPriceAgent(provider="openai")
    
    query = "How did Tesla stock perform in the first two weeks of January 2024?"
    print(f"\nQuery: {query}")
    print("-" * 60)
    response = agent.query(query)
    print(f"Response:\n{response}\n")


def example_error_handling():
    """Examples of error handling"""
    print("=" * 60)
    print("Example 6: Error Handling")
    print("=" * 60)
    
    agent = StockPriceAgent(provider="openai")
    
    # Invalid symbol
    query = "Get stock price for INVALIDXYZ from last week"
    print(f"\nQuery: {query}")
    print("-" * 60)
    response = agent.query(query)
    print(f"Response:\n{response}\n")


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("STOCK PRICE LLM AGENT - EXAMPLES")
    print("=" * 60 + "\n")
    
    try:
        # Run examples
        example_basic_usage()
        
        # Uncomment to run Ollama example (requires Ollama installation)
        # example_ollama_usage()
        
        example_specific_dates()
        example_company_names()
        example_comparison_query()
        example_error_handling()
        
        print("\n" + "=" * 60)
        print("All examples completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("\nMake sure you have:")
        print("1. Installed all requirements: pip install -r requirements.txt")
        print("2. Set OPENAI_API_KEY environment variable (for OpenAI examples)")
        print("3. Installed and started Ollama (for Ollama examples)")


if __name__ == "__main__":
    main()
