# Architecture Documentation

## Overview

The Stock Price LLM Agent is built with a modular architecture that separates concerns into distinct components. This design makes the codebase maintainable, testable, and extensible.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                         main.py                             │
│  (CLI Interface - Argument parsing and user interaction)    │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                        agent.py                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │            StockPriceAgent Class                     │   │
│  │  - Manages LLM initialization                        │   │
│  │  - Orchestrates graph execution                      │   │
│  │  - Handles query processing                          │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────┬────────────────────────────┬─────────────────┘
               │                            │
               ▼                            ▼
┌──────────────────────────┐    ┌──────────────────────────┐
│       graph.py           │    │       tools.py           │
│                          │    │                          │
│  - AgentState            │    │  - get_stock_price()     │
│  - create_graph()        │    │  - ALL_TOOLS list        │
│  - should_continue()     │    │  - yfinance integration  │
│  - call_model()          │    │                          │
│  - get_system_message()  │    │                          │
│                          │    │                          │
│  LangGraph Workflow:     │    │  Tool Functions:         │
│  ┌────────────────────┐  │    │  ┌────────────────────┐  │
│  │   Agent Node      │  │    │  │  Fetch stock data  │  │
│  │   (LLM decides)   │  │    │  │  via yfinance API  │  │
│  └─────────┬──────────┘  │    │  │                    │  │
│            │             │    │  │  Return JSON with  │  │
│            ▼             │    │  │  OHLC + Volume     │  │
│  ┌────────────────────┐  │    │  └────────────────────┘  │
│  │  Should Continue? │  │    │                          │
│  │  (Conditional)    │  │    │                          │
│  └─────┬──────┬───────┘  │    │                          │
│        │      │          │    │                          │
│     Tools    End         │    │                          │
│        │                 │    │                          │
│        ▼                 │    │                          │
│  ┌────────────────────┐  │    │                          │
│  │   Tools Node      │  │    │                          │
│  │   (Execute tool)  │◄─┼────┼──────────────────────────┤
│  └─────────┬──────────┘  │    │                          │
│            │             │    │                          │
│            └─────────────┼────┤                          │
│          (Loop back)     │    │                          │
└──────────────────────────┘    └──────────────────────────┘
```

## Component Breakdown

### 1. `main.py` - Entry Point

**Responsibilities:**
- Parse command-line arguments
- Handle interactive vs. single-query modes
- Provide user-friendly error messages
- Initialize and run the agent

**Key Functions:**
- `main()`: Entry point with argument parsing
- `interactive_mode()`: Run continuous Q&A loop
- `single_query_mode()`: Process one query and exit

### 2. `agent.py` - Agent Orchestration

**Responsibilities:**
- Initialize the LLM (OpenAI or Ollama)
- Create and manage the LangGraph workflow
- Process user queries through the graph
- Handle system messages and prompting

**Key Classes:**
- `StockPriceAgent`: Main agent class that ties everything together

**Key Functions:**
- `create_llm()`: Factory function for LLM creation
- `query()`: Process a user query through the workflow
- `stream_query()`: Stream responses (future enhancement)

### 3. `graph.py` - Workflow Definition

**Responsibilities:**
- Define the LangGraph state and workflow
- Implement conditional routing logic
- Manage the agent's decision-making flow
- Provide system prompting

**Key Components:**
- `AgentState`: TypedDict defining the conversation state
- `create_graph()`: Builds the LangGraph workflow
- `should_continue()`: Decides whether to call tools or end
- `call_model()`: Invokes the LLM with current state
- `get_system_message()`: Returns the system prompt

**Workflow Steps:**
1. Agent receives user query
2. LLM analyzes and decides if tools are needed
3. If tools needed → execute tools → return to agent
4. If no tools needed → end and return response

### 4. `tools.py` - Tool Definitions

**Responsibilities:**
- Define all tools available to the agent
- Implement business logic for each tool
- Handle external API integrations (yfinance)
- Format and return tool results

**Key Components:**
- `get_stock_price()`: Decorated tool function using `@tool`
- `ALL_TOOLS`: List of all available tools

**Tool Features:**
- Input validation
- Error handling
- Data formatting (JSON output)
- Summary statistics calculation

## Data Flow

### Query Processing Flow

```
User Input
    │
    ▼
main.py receives input
    │
    ▼
agent.py creates initial state with:
    - System message
    - User query
    │
    ▼
graph.py workflow executes:
    │
    ├─► Agent Node (LLM processes query)
    │   │
    │   ▼
    │   LLM decides: need tools?
    │   │
    │   ├─► YES → Tools Node
    │   │        │
    │   │        ▼
    │   │   tools.py executes get_stock_price()
    │   │        │
    │   │        ▼
    │   │   yfinance fetches data
    │   │        │
    │   │        ▼
    │   │   Return formatted JSON
    │   │        │
    │   │        ▼
    │   └────────┘ (Loop back to Agent Node)
    │
    └─► NO → End workflow
         │
         ▼
    Return final response
    │
    ▼
agent.py extracts response
    │
    ▼
main.py displays to user
```

## State Management

The agent uses LangGraph's state management to track:
- **Messages**: Full conversation history including:
  - System messages
  - User queries
  - Assistant responses
  - Tool calls
  - Tool results

The state is immutable and updated through reducers, ensuring:
- Reproducibility
- Easy debugging
- Conversation history tracking

## LLM Integration

### OpenAI
- Uses `ChatOpenAI` from `langchain_openai`
- Default model: `gpt-4o-mini`
- Requires `OPENAI_API_KEY` environment variable

### Ollama
- Uses `ChatOllama` from `langchain_ollama`
- Default model: `llama3.1`
- Runs locally, no API key needed
- Requires Ollama to be installed and running

Both LLMs are bound with tools using `.bind_tools(ALL_TOOLS)`, enabling function calling.

## Tool Execution

Tools are executed using LangGraph's `ToolNode`:
1. LLM generates a tool call with parameters
2. `ToolNode` validates and executes the tool
3. Tool result is added to the conversation
4. Flow returns to Agent node for processing

## Extensibility

### Adding New Tools

1. Define tool in `tools.py`:
```python
@tool
def new_tool(param1: str, param2: int) -> str:
    """Tool description"""
    # Implementation
    return result

# Add to ALL_TOOLS
ALL_TOOLS = [get_stock_price, new_tool]
```

2. Update system message in `graph.py` if needed

3. The graph automatically incorporates the new tool

### Adding New LLM Providers

1. Modify `create_llm()` in `agent.py`:
```python
elif provider == "anthropic":
    from langchain_anthropic import ChatAnthropic
    llm = ChatAnthropic(model=model_name, temperature=0)
```

2. Update CLI in `main.py` to accept new provider

### Customizing the Workflow

Modify `create_graph()` in `graph.py` to:
- Add new nodes (e.g., validation, formatting)
- Change routing logic
- Add parallel tool execution
- Implement custom state reducers

## Error Handling

Each layer handles errors appropriately:

- **tools.py**: Returns error JSON if tool execution fails
- **agent.py**: Catches initialization errors
- **main.py**: Provides user-friendly error messages and troubleshooting

## Testing Strategy

- **Unit Tests**: Test individual tools in isolation
- **Integration Tests**: Test graph execution
- **End-to-End Tests**: Test full agent workflows

Example test structure:
```python
# Test tool
result = get_stock_price.invoke({
    "symbol": "AAPL",
    "start_date": "2024-01-01",
    "end_date": "2024-01-15"
})

# Test graph
from graph import create_graph
from agent import create_llm

llm = create_llm("openai")
graph = create_graph(llm)
result = graph.invoke(initial_state)
```

## Performance Considerations

- **Caching**: LangChain provides automatic caching of LLM responses
- **Streaming**: Agent supports streaming for real-time responses
- **Rate Limiting**: Handled by LLM providers
- **Tool Optimization**: yfinance data is fetched only when needed

## Security

- **API Keys**: Stored in environment variables, never hardcoded
- **Input Validation**: Tools validate all inputs
- **Error Messages**: Don't expose sensitive information
- **Dependencies**: Regular updates via `requirements.txt`

## Future Enhancements

Potential improvements to the architecture:

1. **Memory**: Add conversation memory across sessions
2. **Caching**: Cache stock price data to reduce API calls
3. **Async**: Implement async tool execution for better performance
4. **Multi-tool**: Execute multiple tools in parallel
5. **Validation**: Add input validation node before tools
6. **Monitoring**: Add logging and telemetry
