

import datetime

from langchain.agents import create_agent
from typing import Literal

from langchain_core.tools import tool
from langchain import hub

from langchain_openai import ChatOpenAI, OpenAIEmbeddings



from langchain_utils.cretae_embaddings import create_new_embadding
from langchain_utils.langgraph_react_agent import get_react_agent


model = ChatOpenAI(model="gpt-4o", temperature=0)







@tool
def get_anwers_about_me(query: str):
    """Useful for when you need to answer questions about me. Use my resume to anwers the qustion"""
    llm = ChatOpenAI(temperature=0, model="gpt-4-0125-preview", streaming=False)
    prompt = hub.pull("hwchase17/openai-tools-agent")
    prompt.messages[0].prompt.template = "you are help asssistent for Mohammed Ashraf.\
    you need to anwers the questions asked in the job application form using the tools you have.\
    incuding my contact details if needed."

    agent = create_openai_tools_agent(llm, [retriever_tool], prompt)
    agent_executor = AgentExecutor(agent=agent, tools=[retriever_tool])
    response = agent_executor.invoke({"input":query})
    return response


@tool
def get_current_time(format: str) -> str:
    """Returns the current time in the specified format."""
    return datetime.datetime.now().strftime(format)


# Create a list of available tools
tools = [get_current_time]



system_prompt = "You are a helpful assistant that uses tools to answer questions."

agent = create_agent(
    llm,
    tools,
    system_prompt=system_prompt,
)

from langchain.agents import AgentExecutor

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
