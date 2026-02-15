Agent Development Kit (ADK)

```sh
pip install google-adk
adk create my_agent
```

```txt
my_agent/
    agent.py      # main agent code
    .env          # API keys or project IDs
    __init__.py
```


in the `agent.py`
```py
from google.adk.agents.llm_agent import Agent

# Mock tool implementation
def get_current_time(city: str) -> dict:
    """Returns the current time in a specified city."""
    return {"status": "success", "city": city, "time": "10:30 AM"}

root_agent = Agent(
    model='gemini-3-flash-preview',
    name='root_agent',
    description="Tells the current time in a specified city.",
    instruction="You are a helpful assistant that tells the current time in cities. Use the 'get_current_time' tool for this purpose.",
    tools=[get_current_time],
)

```


# API key
```sh
echo 'GOOGLE_API_KEY="YOUR_API_KEY"' > .env
```

or in the `.env`
```
GOOGLE_GENAI_USE_VERTEXAI=FALSE
GOOGLE_API_KEY=PASTE_YOUR_ACTUAL_API_KEY_HERE
```

# run
```sh
adk run my_agent
```

# run with web interface
```sh
adk web --port 8000
```

http://localhost:8000



# advanced topics
- Setup Runner and Session Service
- Define Tools for Sub-Agents
- Adding Memory and Personalization with Session State
- Initialize New Session Service and State
- Adding Safety - Input Guardrail with before_model_callback
