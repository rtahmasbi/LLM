
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


# more paramaters
```sh
ollama run qwen3 --think=false
ollama run qwen3 --format json
ollama run qwen3 --experimental-websearch
ollama run qwen3 --hidethinking
ollama show qwen3 --modelfile

think: true/false or high/medium/low for supported models (gpt-oss)


# during the run
/set parameter temperature 0.2
/show parameters
top_k, top_p, and num_ctx

/? set
Available Commands:
  /set parameter ...     Set a parameter
  /set system <string>   Set system message
  /set history           Enable history
  /set nohistory         Disable history
  /set wordwrap          Enable wordwrap
  /set nowordwrap        Disable wordwrap
  /set format json       Enable JSON mode
  /set noformat          Disable formatting
  /set verbose           Show LLM stats
  /set quiet             Disable LLM stats
  /set think             Enable thinking
  /set nothink           Disable thinking


 /? show
Available Commands:
  /show info         Show details for this model
  /show license      Show model license
  /show modelfile    Show Modelfile for this model
  /show parameters   Show parameters for this model
  /show system       Show system message
  /show template     Show prompt template



ollama show qwen3

/show modelfile

```


GPT models have `message.thinking` and `message.content`, DeepSeek and Qwen thinkg models have `<think>xx<\think>` tags.

`ollama show qwen3 --modelfile`

