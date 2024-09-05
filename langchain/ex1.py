"""
https://python.langchain.com/v0.1/docs/get_started/quickstart/


pip install langchain,langchain_community
pip install langchain-openai



export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_API_KEY="..."


"""
# STEP1: pick LLM

## llama2
from langchain_community.llms import Ollama
llm = Ollama(model="llama2")

llm.invoke("how can langsmith help with testing?")


## ChatOpenAI

from langchain_openai import ChatOpenAI

llm = ChatOpenAI()
llm.invoke("What is the best way to learn LLM?")


# STEP2: make a prompt

from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a world class technical documentation writer."),
    ("user", "{input}")
])

chain = prompt | llm 

chain.invoke({"input": "how can langsmith help with testing?"})


