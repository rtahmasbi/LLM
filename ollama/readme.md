
# Install
## On linux
```sh
curl -fsSL https://ollama.com/install.sh | sh
```

## On Mac
```sh
brew install ollama
```


# Run a model
```sh
# text
ollama run qwen2.5
ollama run llama3
ollama run mistral
ollama run phi3:3.8b

# thinking, local
ollama run qwen3
ollama run qwen3:14b
ollama run deepseek-r1
ollama run deepseek-r1:8b
ollama run qwen3-vl:8b # vision+text
ollama run phi4-reasoning:14b

# thinking, cloud
ollama run kimi-k2.5:cloud
ollama run qwen3-vl:235b-cloud

# coding
ollama run qwen3-coder
ollama run codellama

```


# Applications
```sh
ollama launch claude --model llama3.1
ollama launch codex --model llama3.1
ollama launch opencode --model llama3.1
ollama launch openclaw --model llama3.1

# cloud:
ollama launch claude --model kimi-k2.5:cloud
ollama launch codex --model kimi-k2.5:cloud
ollama launch opencode --model kimi-k2.5:cloud
ollama launch openclaw --model kimi-k2.5:cloud

```


# Running Cloud models
```sh
ollama signin
```

# List models
```sh
curl https://ollama.com/api/tags | jq

```

# systemctl
```sh
ollama serve
sudo systemctl start ollama
sudo systemctl status ollama
```


# Path
On Linux:
```sh
/usr/share/ollama/.ollama/models
```


# vision
```sh
ollama run qwen3-vl:8b ./xx.jpg whats in this image?
```

