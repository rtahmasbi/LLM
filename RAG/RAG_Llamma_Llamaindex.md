https://medium.com/@tiwarisaurabh7757/rag-app-using-llama-and-llama-index-c538f6b14a78

(I have pdf)


For vector index, we can use: `llama_index.core import VectorStoreIndex`



```sh
!pip install pypdf
!pip install transformers einops accelerate langchain bitsandbytes
!pip install sentence_transformers #Embedding
!pip install llama_index
!pip install llama-index-embeddings-langchain
!pip install llama-index-llms-huggingface
```


```py

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext  #Vector store index is for indexing the vector
from llama_index.llms.huggingface import HuggingFaceLLM



documents = SimpleDirectoryReader('/content/pdf').load_data()


system_prompt="""
You are a Q&A assistant. Your goal is to answer questions as
accurately as possible based on the instructions and context provided.
"""


!huggingface-cli login


import torch

llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.0, "do_sample": False},
    system_prompt=system_prompt,
    tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
    model_name="meta-llama/Llama-2-7b-chat-hf",
    device_map="auto",
    # loading model in 8bit for reducing memory
    model_kwargs={"torch_dtype": torch.float16 , "load_in_8bit":True}
)


from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.core import ServiceContext
from langchain.embeddings.huggingface import HuggingFaceEmbeddings


embed_model= HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


service_context=ServiceContext.from_defaults(
    chunk_size=1024,
    llm=llm,
    embed_model=embed_model
)
index=VectorStoreIndex.from_documents(documents,service_context=service_context)
query_engine=index.as_query_engine()
response=query_engine.query("what are the rights against exploitation?")
print(response)



```

