"""
Stock Price Agent
Main agent class that combines the LLM, tools, and graph.
"""

from typing import Literal
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from tools import ALL_TOOLS
from graph import create_graph, get_system_message


def create_llm(provider: Literal["openai", "ollama"] = "openai", model: str = None):
    """
    Create an LLM instance based on the provider.
    
    Args:
        provider: Either "openai" or "ollama"
        model: Model name (optional, uses defaults if not provided)
    
    Returns:
        LLM instance bound with tools
    """
    if provider == "openai":
        model_name = model or "gpt-4o-mini"
        llm = ChatOpenAI(model=model_name, temperature=0)
    elif provider == "ollama":
        model_name = model or "llama3.1"
        llm = ChatOllama(model=model_name, temperature=0)
    else:
        raise ValueError(f"Unknown provider: {provider}. Choose 'openai' or 'ollama'")
    
    return llm.bind_tools(ALL_TOOLS)


class StockPriceAgent:
    """Main agent class for stock price queries."""
    
    def __init__(self, provider: Literal["openai", "ollama"] = "openai", model: str = None):
        """
        Initialize the stock price agent.
        
        Args:
            provider: LLM provider ("openai" or "ollama")
            model: Specific model name (optional)
        """
        self.provider = provider
        self.model = model
        self.llm = create_llm(provider, model)
        self.graph = create_graph(self.llm)
        self.system_message = get_system_message()
    
    def query(self, user_input: str) -> str:
        """
        Process a user query about stock prices.
        
        Args:
            user_input: Natural language query about stock prices
            
        Returns:
            Response with stock price information
        """
        # Create initial state with system message and user query
        initial_state = {
            "messages": [
                HumanMessage(content=self.system_message),
                HumanMessage(content=user_input)
            ]
        }
        
        # Run the graph
        result = self.graph.invoke(initial_state)
        
        # Extract the final response
        final_message = result["messages"][-1]
        return final_message.content
    
    def stream_query(self, user_input: str):
        """
        Process a user query with streaming response.
        
        Args:
            user_input: Natural language query about stock prices
            
        Yields:
            Response chunks as they are generated
        """
        # Create initial state with system message and user query
        initial_state = {
            "messages": [
                HumanMessage(content=self.system_message),
                HumanMessage(content=user_input)
            ]
        }
        
        # Stream the graph execution
        for event in self.graph.stream(initial_state):
            # Extract messages from events
            for value in event.values():
                if "messages" in value:
                    for message in value["messages"]:
                        if hasattr(message, 'content') and message.content:
                            yield message.content
