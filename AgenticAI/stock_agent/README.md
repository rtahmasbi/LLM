# Stock Price LLM Agent 📈

A LangGraph-based AI agent that uses natural language processing to fetch and analyze stock price data. The agent intelligently extracts stock symbols, date ranges from user queries, and retrieves historical price data using the yfinance API.

## Features

- 🤖 **Natural Language Interface**: Ask about stock prices in plain English
- 🔄 **LangGraph Orchestration**: Sophisticated workflow management with conditional routing
- 🎯 **Smart Date Parsing**: Automatically handles missing dates (defaults to last week if no start date, today if no end date)
- 🔌 **Multiple LLM Providers**: Support for both OpenAI and Ollama (local models)
- 📊 **Comprehensive Data**: Fetches Open, High, Low, Close prices and trading volume
- 📈 **Summary Statistics**: Provides aggregated insights (average, max, min, total volume)

## Architecture

The system uses LangGraph to create an agentic workflow:

```
User Query → Agent (LLM) → Tool Call Decision → get_stock_price Tool → Response
                ↑                                           ↓
                └───────────────────────────────────────────┘
```

### Components

1. **Agent Node**: LLM processes user query and decides on tool usage
2. **Tool Node**: Executes the `get_stock_price` function
3. **Conditional Routing**: Determines whether to call tools or end the workflow
4. **State Management**: Maintains conversation context and message history

## Installation

### Prerequisites

- Python 3.8 or higher
- For OpenAI: OpenAI API key
- For Ollama: Ollama installed locally

### Setup

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:

For **OpenAI**:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

For **Ollama**:
- Install Ollama from https://ollama.ai
- Pull a model: `ollama pull llama3.1`

## Usage

### Interactive Mode

**Using OpenAI (default):**
```bash
python main.py
```

**Using Ollama:**
```bash
python main.py --provider ollama
```

**Using a specific model:**
```bash
# OpenAI with GPT-4
python main.py --provider openai --model gpt-4

# Ollama with specific model
python main.py --provider ollama --model mistral
```

### Single Query Mode

```bash
python main.py --query "What's the Apple stock price for the last week?"

python main.py --provider ollama --query "Get Tesla stock from Jan 1 to Jan 15, 2024"
```

### Example Queries

The agent understands various natural language queries:

- "What's the Apple stock price for the last week?"
- "Get Tesla stock from January 1 to January 15, 2024"
- "Show me Microsoft stock prices from last month"
- "GOOGL stock price today"
- "What was Amazon's stock doing in December 2023?"
- "Compare Netflix stock prices over the last 2 weeks"

## Python API Usage

You can also use the agent programmatically:

```python
from agent import StockPriceAgent

# Initialize with OpenAI
agent = StockPriceAgent(provider="openai")

# Or with Ollama
agent = StockPriceAgent(provider="ollama", model="llama3.1")

# Query
response = agent.query("What's the Apple stock price for the last week?")
print(response)
```

### Using Individual Modules

```python
# Import specific components
from tools import get_stock_price
from graph import create_graph, get_system_message
from agent import create_llm

# Use the tool directly
result = get_stock_price.invoke({
    "symbol": "AAPL",
    "start_date": "2024-01-01",
    "end_date": "2024-01-15"
})
print(result)

# Create custom LLM configuration
llm = create_llm(provider="openai", model="gpt-4")

# Create custom graph
graph = create_graph(llm)
```

## Tool Details

### `get_stock_price` Tool

**Parameters:**
- `symbol` (str): Stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'TSLA')
- `start_date` (str): Start date in YYYY-MM-DD format
- `end_date` (str): End date in YYYY-MM-DD format

**Returns:**
JSON with:
- Individual daily price data (open, high, low, close, volume)
- Summary statistics (average close, max high, min low, total volume)

**Example Response:**
```json
{
  "symbol": "AAPL",
  "start_date": "2024-01-01",
  "end_date": "2024-01-15",
  "data": [
    {
      "date": "2024-01-02",
      "open": 187.15,
      "high": 188.44,
      "low": 183.89,
      "close": 185.64,
      "volume": 82488800
    },
    ...
  ],
  "summary": {
    "total_days": 10,
    "avg_close": 185.23,
    "max_high": 190.12,
    "min_low": 180.45,
    "total_volume": 724892100
  }
}
```

## How It Works

### Date Handling

The LLM intelligently processes date information:

1. **No dates specified**: Uses last 7 days (last week) to today
2. **Only start date**: Uses start date to today
3. **Only end date**: Uses 7 days before end date to end date
4. **Both dates**: Uses the specified range

### Symbol Recognition

The agent can recognize:
- Explicit ticker symbols (AAPL, TSLA, MSFT)
- Company names (Apple, Tesla, Microsoft)
- Common variations (Google → GOOGL, Facebook → META)

### LangGraph Workflow

1. **Agent receives user query** with system instructions
2. **LLM analyzes** the query and extracts:
   - Stock symbol
   - Date range
3. **LLM calls** `get_stock_price` tool with extracted parameters
4. **Tool executes**, fetching data from yfinance
5. **Results returned** to LLM
6. **LLM formats** the response in natural language
7. **User receives** formatted answer with stock data

## Environment Variables

Create a `.env` file (optional) for configuration:

```env
# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key-here

# Ollama Configuration (if using Ollama)
OLLAMA_HOST=http://localhost:11434
```

## Common Stock Symbols

| Company | Symbol |
|---------|--------|
| Apple | AAPL |
| Microsoft | MSFT |
| Google/Alphabet | GOOGL |
| Amazon | AMZN |
| Tesla | TSLA |
| Meta/Facebook | META |
| Netflix | NFLX |
| Nvidia | NVDA |

## Troubleshooting

### OpenAI Issues

**Error: "OpenAI API key not found"**
- Make sure you've set the `OPENAI_API_KEY` environment variable
- Or create a `.env` file with your API key

### Ollama Issues

**Error: "Could not connect to Ollama"**
- Make sure Ollama is installed and running: `ollama serve`
- Check that the model is pulled: `ollama pull llama3.1`

**Error: "Model not found"**
- Pull the model first: `ollama pull <model-name>`
- Use `ollama list` to see available models

### Stock Data Issues

**Error: "No data found for symbol"**
- Verify the stock symbol is correct
- Check if the market was open during the requested date range
- Try a different date range

## Project Structure

```
stock-price-llm-tool/
├── main.py                # Main entry point and CLI
├── agent.py               # StockPriceAgent class
├── graph.py               # LangGraph workflow definition
├── tools.py               # Tool definitions (get_stock_price)
├── stock_price_agent.py   # Backward compatibility wrapper
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── QUICKSTART.md         # Quick start guide
├── .env.example          # Environment variable template
├── examples.py           # Usage examples
└── test_setup.py         # Setup verification script
```

### Module Descriptions

- **`main.py`**: Command-line interface for running the agent in interactive or single-query mode
- **`agent.py`**: Contains the `StockPriceAgent` class that orchestrates the LLM and tools
- **`graph.py`**: Defines the LangGraph workflow with nodes and conditional routing
- **`tools.py`**: Implements the `get_stock_price` tool that fetches stock data from yfinance
- **`stock_price_agent.py`**: Backward compatibility - imports from modular structure

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

MIT License - feel free to use this project for your own purposes.

## Acknowledgments

- Built with [LangGraph](https://github.com/langchain-ai/langgraph) by LangChain
- Stock data provided by [yfinance](https://github.com/ranaroussi/yfinance)
- Supports OpenAI and Ollama LLM providers
