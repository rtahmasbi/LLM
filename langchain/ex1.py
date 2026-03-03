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
