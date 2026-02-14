"""
LangGraph Workflow
Defines the agent workflow using LangGraph for orchestrating tool calls.
"""

from typing import Annotated, TypedDict, Literal
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from tools import ALL_TOOLS


# Define the agent state
class AgentState(TypedDict):
    """State of the agent containing conversation messages."""
    messages: Annotated[list, add_messages]


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """
    Determine whether to continue with tool calls or end the workflow.
    
    Args:
        state: Current agent state
        
    Returns:
        "tools" if the LLM made a tool call, "end" otherwise
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the LLM makes a tool call, continue to tools node
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    # Otherwise, end the graph
    return "end"


def call_model(state: AgentState, llm):
    """
    Call the LLM with the current state.
    
    Args:
        state: Current agent state
        llm: Language model instance (bound with tools)
        
    Returns:
        Dictionary with updated messages
    """
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


def create_graph(llm):
    """
    Create the LangGraph workflow.
    
    The workflow consists of:
    1. Agent node - calls the LLM to decide on actions
    2. Tools node - executes any tool calls made by the LLM
    3. Conditional routing - decides whether to call tools or end
    
    Args:
        llm: Language model instance (should be bound with tools)
        
    Returns:
        Compiled LangGraph workflow
    """
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", lambda state: call_model(state, llm))
    workflow.add_node("tools", ToolNode(ALL_TOOLS))
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    
    # Add edge from tools back to agent
    workflow.add_edge("tools", "agent")
    
    # Compile the graph
    return workflow.compile()


def get_system_message() -> str:
    """
    Get the system message that guides the LLM's behavior.
    
    Returns:
        System message string
    """
    return """You are a helpful stock price assistant. 
When users ask about stock prices, you should:
1. Extract the stock symbol or company name from their query
2. Determine the date range (start and end dates)
3. If no end date is mentioned, use today's date
4. If no start date is mentioned, use 7 days ago (last week)
5. Use the get_stock_price tool to fetch the data
6. Present the results in a clear, formatted way

For date formats, always use YYYY-MM-DD format when calling the tool.
If a company name is given instead of a symbol, convert it to the appropriate stock symbol.

Common stock symbols:
- Apple: AAPL
- Microsoft: MSFT
- Google/Alphabet: GOOGL
- Amazon: AMZN
- Tesla: TSLA
- Meta/Facebook: META
- Netflix: NFLX
- Nvidia: NVDA
- Adobe: ADBE
- Salesforce: CRM
- Intel: INTC
- AMD: AMD
- Oracle: ORCL
- IBM: IBM
- Cisco: CSCO
"""
