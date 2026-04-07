
# My examples
- [simple examples with langchain](simple_examples_with_langchain.md)
- [RAG with langchain](pdf_qa/)
- [RAG example 2](rag_example2/)


# Info
https://python.langchain.com/v0.2/docs/tutorials/rag/


- Loading data with a Document Loader
- Chunking the indexed data with a Text Splitter to make it more easily usable by a model
- Embedding the data and storing the data in a `vectorstore`
- Retrieving the previously stored chunks in response to incoming questions
- Generating an answer using the retrieved chunks as context


For `vectorstore` check:

https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html



# Metrics for RAG pattern
## RAGAS
https://github.com/explodinggradients/ragas/blob/main/src/ragas/metrics/_faithfulness.py

Faithfulness and Answer Relevancy


https://github.com/NirDiamant/RAG_Techniques/blob/main/evaluation/evaluation_deep_eval.ipynb



# Ref
https://github.com/NirDiamant/RAG_Techniques


# chunking strategies
- more info at [chunking_strategies.md](chunking_strategies.md)


# reranking strategy
- CohereRerank
- FlashrankRerank
- CrossEncoderReranker
- LLMListwiseRerank
- Colbert
- Jina
- more info at [reranking_strategies.md](reranking_strategies.md)


# retrievers
https://python.langchain.com/docs/integrations/retrievers/

A retriever is an interface that returns documents given an unstructured query.

## arxiv
https://python.langchain.com/docs/integrations/retrievers/arxiv/

## tavily - search engine
https://python.langchain.com/docs/integrations/retrievers/tavily/


## wikipedia
https://python.langchain.com/docs/integrations/retrievers/wikipedia/


https://www.mediawiki.org/wiki/Extension:TextExtracts#API

https://en.wikipedia.org/w/api.php?action=help&modules=query%2Bextracts


https://github.com/goldsmith/Wikipedia


https://pypi.org/project/wikipedia/

