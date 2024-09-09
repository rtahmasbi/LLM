# API Key
For `langchain`, you need `LANGCHAIN_API_KEY`.

Go to  https://smith.langchain.com/ and get one for free.


# doc
https://python.langchain.com/v0.1/docs/get_started/quickstart/


# Env
```sh

pip install langchain, langchain_community
pip install langchain-openai

export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_API_KEY="..."

```

# main modules
```py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

```

