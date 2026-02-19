https://www.hopsworks.ai/dictionary/vllm

The attention mechanism allows LLMs to focus on relevant parts of the input sequence while generating output/response. Inside the attention mechanism, the attention scores for all input tokens need to be calculated. Existing systems store KV pairs in contiguous memory spaces, limiting memory sharing and leading to inefficient memory management.


PagedAttention is an attention algorithm inspired by the concept of paging in operating systems. It allows storing continuous KV pairs in non-contiguous memory space by partitioning the KV cache of each sequence into KV block tables.




## Install
```
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/



# vllm needs python 3.10-3.12
conda create -n vllm_env python=3.10
conda activate vllm_env


pip install vllm
```

## Run server
```sh
python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B-Instruct
```

```sh
vllm serve Qwen/Qwen2.5-1.5B-Instruct
vllm serve Qwen/Qwen2.5-1.5B-Instruct --attention-backend FLASH_ATTN

curl http://localhost:8000/v1/models


curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'


curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"}
        ]
    }'
```



## run locally
```py

import torch
print(torch.cuda.is_available())
print(torch.version.cuda)

from vllm import LLM

#llm = LLM(model=..., revision=..., runner=..., trust_remote_code=True)
llm = LLM("meta-llama/Meta-Llama-3-8B-Instruct", device="cuda") # or "cpu"
llm = LLM("Qwen/Qwen2.5-1.5B-Instruct", device="cuda") # or "cpu"


# For generative models (runner=generate) only
output = llm.generate("Hello, my name is")
print(output)

# For pooling models (runner=pooling) only
output = llm.encode("Hello, my name is")
print(output)

```


