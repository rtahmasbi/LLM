
import datetime
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain.messages import SystemMessage, HumanMessage

from langchain_openai import ChatOpenAI

# Initialize the model
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)



@tool
def get_current_time(format: str) -> str:
    """Returns the current time in the specified format."""
    return datetime.datetime.now().strftime(format)


@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

@tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"Weather in {location}: Sunny, 72°F"


# Create a list of available tools
tools = [get_current_time, search, get_weather]

system_prompt = "You are a helpful assistant that uses tools to answer questions."

agent = create_agent(
    model,
    tools,
    system_prompt=system_prompt,
)


agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)



content=[
    {
        "type": "text",
        "text": "You are an AI assistant tasked with analyzing literary works.",
    },
    {
        "type": "text",
        "text": "<the entire contents of 'Pride and Prejudice'>",
        "cache_control": {"type": "ephemeral"}
    }
]
system_prompt = SystemMessage(content=content)
human_prompt =  [HumanMessage("Analyze the major themes in 'Pride and Prejudice'.")]

result = literary_agent.invoke({"messages": human_prompt})



# dynamic_prompt
from langchain.agents.middleware import dynamic_prompt, ModelRequest

class Context(TypedDict):
    user_role: str

@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    """Generate system prompt based on user role."""
    user_role = request.runtime.context.get("user_role", "user")
    base_prompt = "You are a helpful assistant."

    if user_role == "expert":
        return f"{base_prompt} Provide detailed technical responses."
    elif user_role == "beginner":
        return f"{base_prompt} Explain concepts simply and avoid jargon."

    return base_prompt

agent = create_agent(
    model="gpt-4.1",
    tools=[web_search],
    middleware=[user_role_prompt],
    context_schema=Context
)

# The system prompt will be set dynamically based on context
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Explain machine learning"}]},
    context={"user_role": "expert"}
)



from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage

model = init_chat_model("gpt-5-nano")

system_msg = SystemMessage("You are a helpful assistant.")
human_msg = HumanMessage("Hello, how are you?")

# Use with chat models
messages = [system_msg, human_msg]
response = model.invoke(messages)  # Returns AIMessage
