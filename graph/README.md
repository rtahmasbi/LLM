# Q&A chain over a graph database
https://python.langchain.com/v0.1/docs/use_cases/graph/quickstart/

At a high-level, the steps of most graph chains are:

1. Convert question to a graph database query: Model converts user input to a graph database query (e.g. Cypher).
2. Execute graph database query: Execute the graph database query.
3. Answer the question: Model responds to user input using the query results.



# Constructing knowledge graphs
https://python.langchain.com/v0.1/docs/use_cases/graph/constructing/

main module
```py
from langchain_experimental.graph_transformers import LLMGraphTransformer
```
