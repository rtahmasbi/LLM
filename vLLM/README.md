https://www.hopsworks.ai/dictionary/vllm

The attention mechanism allows LLMs to focus on relevant parts of the input sequence while generating output/response. Inside the attention mechanism, the attention scores for all input tokens need to be calculated. Existing systems store KV pairs in contiguous memory spaces, limiting memory sharing and leading to inefficient memory management.


PagedAttention is an attention algorithm inspired by the concept of paging in operating systems. It allows storing continuous KV pairs in non-contiguous memory space by partitioning the KV cache of each sequence into KV block tables.




## Install
```
pip install vllm
```

## Run server
```sh
python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B-Instruct
```


