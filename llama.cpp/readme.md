
# Install
## Download pre-built binaries
```
wget https://github.com/ggml-org/llama.cpp/releases/download/b8495/llama-b8495-bin-ubuntu-x64.tar.gz
tar -xvzf llama-b8495-bin-ubuntu-x64.tar.gz
```

## Install llama.cpp using brew, nix or winget
https://github.com/ggml-org/llama.cpp/blob/master/docs/install.md


# run
`-m` for the local model, `-hf` to read from HF.
```sh
cd /home/ras/llama-b8495/
./llama-cli -hf ggml-org/gemma-3-1b-it-GGUF

./llama-cli -hf bartowski/Meta-Llama-3-8B-Instruct-GGUF
./llama-cli -hf Qwen/Qwen2.5-1.5B-Instruct-GGUF
./llama-cli -hf Qwen/Qwen2.5-7B-Instruct-GGUF

```


# local model repo
```
/home/ras/.cache/llama.cpp/
```


# completion-bash
```sh
build/bin/llama-cli --completion-bash > ~/.llama-completion.bash
source ~/.llama-completion.bash
```

# llama-server
```sh
llama-server -m model.gguf --port 8080
```

# llama-bench
```sh
./llama-bench -hf Qwen/Qwen2.5-1.5B-Instruct-GGUF

| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen2 1.5B Q4_K - Medium       |   1.04 GiB |     1.78 B | CPU        |       8 |           pp512 |        218.98 ± 0.81 |
| qwen2 1.5B Q4_K - Medium       |   1.04 GiB |     1.78 B | CPU        |       8 |           tg128 |         27.38 ± 0.00 |

build: 7cadbfce1 (8495)

```


# python

```py
# pip install llama-cpp-python

from llama_cpp import Llama

# Load model
llm = Llama(model_path="/home/ras/.cache/llama.cpp/ggml-org_gemma-3-1b-it-GGUF_gemma-3-1b-it-Q4_K_M.gguf")

# Generate text
prompt = "Write a short poem about a sunny day."
output = llm(prompt, max_tokens=100)
print(output['choices'][0]['text'])

```


