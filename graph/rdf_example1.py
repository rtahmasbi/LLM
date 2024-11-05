# https://python.langchain.com/docs/integrations/graphs/rdflib_sparql/
"""
pip install rdflib



"""


from langchain.chains import GraphSparqlQAChain
from langchain_community.graphs import RdfGraph
from langchain_openai import ChatOpenAI


graph = RdfGraph(
    source_file="http://www.w3.org/People/Berners-Lee/card",
    standard="rdf",
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


