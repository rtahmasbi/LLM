from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

@tool
def fill_field(field_name: str, value: str) -> str:
    """Useful for when you need to fill information"""
    return f"Filled '{field_name}' with '{value}'"

@tool
def get_options(field_name: str) -> list[str]:
    """Tool to get dynamic options (simulates hidden fields appearing)"""
    dynamic_options = {
        "country": ["USA", "Canada", "UK"],
        "state": ["California", "Texas", "New York"],
        "city": ["San Francisco", "Austin", "New York City"]
    }
    return dynamic_options.get(field_name, [])

tools = [fill_field, get_options]


agent = create_react_agent(
    tools=tools,
    model="gpt-4",   # can also use "gpt-3.5-turbo"
)

task = """
I want to fill the form:
- Country: USA
- State: California
- City: San Francisco
If a field has multiple options, pick the first one.
"""

inputs = {"messages": [("user", task)]}


result = agent.invoke(inputs)
print("Agent Result:\n", result)

for r in result["messages"]:
    print("-"*80, r.type)
    print(r)



