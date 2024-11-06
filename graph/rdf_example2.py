
# https://python.langchain.com/docs/integrations/graphs/rdflib_sparql/

"""
pip install rdflib


https://www.wikidata.org/wiki/Special:EntityData/Q42.ttl

"""


from langchain.chains import GraphSparqlQAChain
from langchain_community.graphs import RdfGraph
from langchain_openai import ChatOpenAI


graph = RdfGraph(
    source_file="/home/rt/GITHUB/LLM/graph/g1.ttl",
    standard="rdfs",
    local_copy="test.ttl",
)


graph.load_schema()
print(graph.get_schema)


chain = GraphSparqlQAChain.from_llm(
    ChatOpenAI(temperature=0), graph=graph, verbose=True, allow_dangerous_requests=True
)

chain.run("What is Tim Berners-Lee's work homepage?")


chain = GraphSparqlQAChain.from_llm(
    ChatOpenAI(temperature=0), graph=graph, verbose=True, allow_dangerous_requests=True, return_sparql_query=True
)
chain.run("What is Tim Berners-Lee's work homepage?")


query = """PREFIX foaf: <http://xmlns.com/foaf/0.1/>
SELECT ?homepage
WHERE {
    ?person foaf:name "Tim Berners-Lee" .
    ?person foaf:workplaceHomepage ?homepage .
}
"""
graph.query(query)




query = """
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

SELECT ?homepage ?kk
WHERE {
    ?person foaf:name ?kk .
    ?person foaf:workplaceHomepage ?homepage .
}
"""
graph.query(query)






chain = GraphSparqlQAChain.from_llm(
    ChatOpenAI(temperature=0), graph=graph, verbose=True, allow_dangerous_requests=True
)

chain.invoke({"query": "What is the list of stocks in p1?"})


query = """
PREFIX pred: <tifin://predicator/>
SELECT ?stock
WHERE {
    ?portfolio pred:list_stock ?stock .
    ?portfolio pred:name "portfolio1" .
}
"""
graph.query(query)

chain.run("Create a portfolio named p3.")



chain.invoke({"query": "Create a portfolio named p3."})


query = """
PREFIX pred: <tifin://predicator/>
SELECT ?stock
WHERE {
    ?portfolio pred:list_stock ?stock .
}
"""
graph.query(query)


graph.graph.all_nodes() 
list(graph.graph.predicates())


https://github.com/langchain-ai/langchain/blob/41b7a5169d3b6bdb0949a409397707eb69b3cd07/libs/community/langchain_community/chains/graph_qa/sparql.py#L25

