
https://python.langchain.com/v0.2/docs/tutorials/rag/


- Loading data with a Document Loader
- Chunking the indexed data with a Text Splitter to make it more easily usable by a model
- Embedding the data and storing the data in a `vectorstore`
- Retrieving the previously stored chunks in response to incoming questions
- Generating an answer using the retrieved chunks as context


For `vectorstore` check:

https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html



```sh
pip install langchain langchain_community langchain_chroma

pip install -qU langchain-openai
```

```
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_API_KEY="..."
```



# Preview
```py
import getpass
import os

os.environ["OPENAI_API_KEY"] = getpass.getpass()

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")


import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load, chunk and index the contents of the blog.
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain.invoke("What is Task Decomposition?")


# cleanup
vectorstore.delete_collection()


```

# Detailed codes

```py
# STEP1: Indexing: Load
import bs4
from langchain_community.document_loaders import WebBaseLoader

# Only keep post title, headers, and content from the full HTML.
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()

len(docs[0].page_content)


# STEP2: Indexing: Split

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

len(all_splits)



# STEP3: Indexing: Store
from langchain_chroma import Chroma # Chroma class which is a vector store for handling various tasks.
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())


# STEP4: Retrieval and Generation: Retrieve
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
retrieved_docs = retriever.invoke("What are the approaches to Task Decomposition?")
len(retrieved_docs)

# STEP5: Retrieval and Generation: Generate
pip install -qU langchain-openai
import getpass
import os

os.environ["OPENAI_API_KEY"] = getpass.getpass()

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini")

from langchain import hub # LangChain prompt hub

prompt = hub.pull("rlm/rag-prompt")

example_messages = prompt.invoke(
    {"context": "filler context", "question": "filler question"}
).to_messages()

example_messages



from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

for chunk in rag_chain.stream("What is Task Decomposition?"):
    print(chunk, end="", flush=True)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
)


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

response = rag_chain.invoke({"input": "What is Task Decomposition?"})
print(response["answer"])


for document in response["context"]:
    print(document)
    print()

```


# Metrics for RAG pattern
## RAGAS
https://github.com/explodinggradients/ragas/blob/main/src/ragas/metrics/_faithfulness.py

Faithfulness and Answer Relevancy


https://github.com/NirDiamant/RAG_Techniques/blob/main/evaluation/evaluation_deep_eval.ipynb



# more info
https://github.com/NirDiamant/RAG_Techniques


# reranking strategy
- CohereRerank
- FlashrankRerank
- CrossEncoderReranker
- LLMListwiseRerank
- Colbert
- Jina

# retrievers
https://python.langchain.com/docs/integrations/retrievers/

A retriever is an interface that returns documents given an unstructured query.


## arxiv
https://python.langchain.com/docs/integrations/retrievers/arxiv/


pip install -qU langchain-community arxiv

```py

from langchain_community.retrievers import ArxivRetriever

retriever = ArxivRetriever(
    load_max_docs=2,
    get_ful_documents=True,
)

docs = retriever.invoke("1605.08386")

retriever.invoke("Rasool Tahmasbi")




from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the context provided.

Context: {context}

Question: {question}"""
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

chain.invoke("What is the ImageBind model?")


chain.invoke("Rasool Tahmasbi")



```


## tavily - search engine
https://python.langchain.com/docs/integrations/retrievers/tavily/


## wikipedia
https://python.langchain.com/docs/integrations/retrievers/wikipedia/


https://www.mediawiki.org/wiki/Extension:TextExtracts#API

https://en.wikipedia.org/w/api.php?action=help&modules=query%2Bextracts


https://github.com/goldsmith/Wikipedia?tab=readme-ov-file


https://pypi.org/project/wikipedia/


# q_proj, k_proj, v_proj, up_proj, down_proj
https://medium.com/@saratbhargava/mastering-llama-math-part-1-a-step-by-step-guide-to-counting-parameters-in-llama-2-b3d73bc3ae31

(pdf)

- Embedding Block
- Attention block (q_proj, k_proj, v_proj, o_proj)
- MLP Block (gate_proj, up_proj, down_proj)
- RMS Norm layers


```py

LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(32000, 5120)
    (layers): ModuleList(
      (0-39): 40 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=5120, out_features=5120, bias=False)
          (k_proj): Linear(in_features=5120, out_features=5120, bias=False)
          (v_proj): Linear(in_features=5120, out_features=5120, bias=False)
          (o_proj): Linear(in_features=5120, out_features=5120, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=5120, out_features=13824, bias=False)
          (up_proj): Linear(in_features=5120, out_features=13824, bias=False)
          (down_proj): Linear(in_features=13824, out_features=5120, bias=False)
          (act_fn): SiLUActivation()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
    )
    (norm): LlamaRMSNorm()
  )
  (lm_head): Linear(in_features=5120, out_features=32000, bias=False)
)

```

https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py

https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/te_llama/tutorial_accelerate_hf_llama_with_te.html
