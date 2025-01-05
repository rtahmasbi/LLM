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


# faiss
https://python.langchain.com/docs/integrations/vectorstores/faiss/

```py
from langchain_community.vectorstores import FAISS

index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)


results = vector_store.similarity_search(
    "LangChain provides abstractions to make working with LLMs easy",
    k=2,
    filter={"source": "tweet"},
)

retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 1})
retriever.invoke("Stealing from the bank is a crime", filter={"source": "news"})

```


# langchain_community LLMs
https://github.com/langchain-ai/langchain/tree/master/libs/community/langchain_community/llms




# a great exampel of prompts for graph_qa
https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/chains/graph_qa/prompts.py


Check these files as well:
- https://github.com/rtahmasbi/LLM/blob/main/graph/rdf_example2.py
- https://github.com/langchain-ai/langchain/blob/master/docs/docs/tutorials/graph.ipynb




https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/chains/graph_qa/base.py

https://github.com/langchain-ai/langchain/blob/41b7a5169d3b6bdb0949a409397707eb69b3cd07/libs/community/tests/integration_tests/chains/test_graph_database_sparql.py#L9


https://github.com/langchain-ai/langchain/blob/41b7a5169d3b6bdb0949a409397707eb69b3cd07/libs/community/langchain_community/chains/graph_qa/sparql.py#L25
