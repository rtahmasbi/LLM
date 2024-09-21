pip install pysqlite3-binary


https://gist.github.com/defulmere/8b9695e415a44271061cc8e272f3c300

```py
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

```



https://python.langchain.com/v0.2/docs/tutorials/pdf_qa/


```sh
python -m venv lang
source lang/bin/activate

pip install -qU pypdf langchain_community
pip install -qU langchain-openai
pip install langchain_chroma langchain_openai
pip install pysqlite3-binary
```


```py

from langchain_community.document_loaders import PyPDFLoader

file_path = "Earnings_Presentation.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

print(len(docs))

print(docs[0].page_content[0:100])
print(docs[0].metadata)



import getpass
import os


from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")


__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')



from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()



from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

results = rag_chain.invoke({"input": "What was Nike's revenue in 2023?"})

results



print(results["context"][0].page_content)



print(results["context"][0].metadata)

```





# Five Levels of Chunking Strategies in RAG
https://medium.com/@anuragmishra_27746/five-levels-of-chunking-strategies-in-rag-notes-from-gregs-video-7b735895694d

(pdf)

with gitbub
https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb


## Level 1 : Fixed Size Chunking
character_text_splitter

https://python.langchain.com/docs/how_to/character_text_splitter/

```py
text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)
```

It breaks down the text into chunks of a specified number of characters, regardless of their content or structure.


## Level 2: Recursive Chunking
recursive_text_splitter

https://python.langchain.com/docs/how_to/recursive_text_splitter/

```py
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

```
default separator list of ["\n\n", "\n", " ", ""] 

```py
text_splitter = RecursiveCharacterTextSplitter(
    separators=[
        "\n\n",
        "\n",
        " ",
        ".",
        ",",
        "\u200b",  # Zero-width space
        "\uff0c",  # Fullwidth comma
        "\u3001",  # Ideographic comma
        "\uff0e",  # Fullwidth full stop
        "\u3002",  # Ideographic full stop
        "",
    ],
    # Existing args
)

```


## Level 3 : Document Based Chunking
- Document with Markdown: Langchain provides `MarkdownTextSplitter`
- Document with Python/JS: Langchain provides `PythonCodeTextSplitter`
- Document with tables: <table> tags in HTML, CSV format separated by ';', etc.
- Document with images (Multi- Modal): `Unstructured.io` provides partition_pdf method to extract images from pdf document.


## Level 4: Semantic Chunking
Llamindex has SemanticSplitterNodeParse class

The hypothesis here is we can use embeddings of individual sentences to make more meaningful chunks. Basic idea is as follows:

- Split the documents into sentences based on separators(.,?,!)
- Index each sentence based on position.
- Group: Choose how many sentences to be on either side. Add a buffer of sentences on either side of our selected sentence.
- Calculate distance between group of sentences.
- Merge groups based on similarity i.e. keep similar sentences together.
- Split the sentences that are not similar.


For langChain:
https://python.langchain.com/docs/how_to/semantic-chunker/


```py
from langchain_experimental.text_splitter import SemanticChunker

# percentile
text_splitter = SemanticChunker(
    OpenAIEmbeddings(), breakpoint_threshold_type="percentile"
)

# Standard Deviation
text_splitter = SemanticChunker(
    OpenAIEmbeddings(), breakpoint_threshold_type="standard_deviation"
)

# Percentile
text_splitter = SemanticChunker(
    OpenAIEmbeddings(), breakpoint_threshold_type="percentile"
)

# Interquartile
# Gradient

```

## Level 5: Agentic Chunking
This chunking strategy explore the possibility to use LLM to determine how much and what text should be included in a chunk based on the context.




# very good
https://medium.com/the-ai-forum/semantic-chunking-for-rag-f4733025d5f5


# more ref
https://www.rungalileo.io/blog/mastering-rag-advanced-chunking-techniques-for-llm-applications


https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/proposition_chunking.ipynb

