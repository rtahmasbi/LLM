
# Install
## Download pre-built binaries
```
wget https://github.com/ggml-org/llama.cpp/releases/download/b8495/llama-b8495-bin-ubuntu-x64.tar.gz
tar -xvzf llama-b8495-bin-ubuntu-x64.tar.gz
```

## Install llama.cpp using brew, nix or winget
https://github.com/ggml-org/llama.cpp/blob/master/docs/install.md


# run
```sh
cd /home/ras/llama-b8495/
./llama-cli -hf ggml-org/gemma-3-1b-it-GGUF

./llama-cli -hf bartowski/Meta-Llama-3-8B-Instruct-GGUF
```


# local model repo
```
/home/ras/.cache/llama.cpp/
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


