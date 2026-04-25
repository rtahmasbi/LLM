
# GPT2
```py
import torch
from transformers import GPT2Tokenizer, GPT2Model

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SEQ_LEN = 1024            # GPT-2's hard context limit
model_name  = "gpt2"  # gpt2 | gpt2-medium | gpt2-large | gpt2-xl

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# GPT-2 has no pad token by default; use eos as pad
tokenizer.pad_token = tokenizer.eos_token

model = GPT2Model.from_pretrained(
    model_name,
    output_hidden_states=True,   # return all layer outputs
)

model.eval()
model.to(DEVICE)
sentences = ["A brown fax jumped into the water.", "You should study."]

with torch.no_grad():
    encoding = tokenizer(
        sentences,
        return_tensors      = "pt",
        padding             = True,
        truncation          = True,
        max_length          = MAX_SEQ_LEN,
        return_attention_mask = True,
    )
    input_ids      = encoding["input_ids"].to(DEVICE)       # (B, L), L is the max len of B
    attention_mask = encoding["attention_mask"].to(DEVICE)  # (B, L)
    # ── Forward pass
    outputs = model(
        input_ids      = input_ids,
        attention_mask = attention_mask,
    )


outputs.last_hidden_state
outputs.hidden_states

b = 0
real_mask = attention_mask[b].bool()  # (L,)
real_ids = input_ids[b][real_mask].tolist()
decoded  = [tokenizer.decode([tid]) for tid in real_ids]
# ['A', ' brown', ' fax', ' jumped', ' into', ' the', ' water', '.']

```


# TinyLlama
https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF

(max 3.5GB RAM)


# Mistral
https://huggingface.co/TheBloke/Mistral-7B-Claude-Chat-GGUF



# facebook/opt-350m
Open Pre-trained Transformer


https://huggingface.co/facebook/opt-350m

https://github.com/huggingface/


`pip install tf-keras`


```py
from transformers import pipeline
generator = pipeline('text-generation', model="facebook/opt-350m", do_sample=True, num_return_sequences=5)
# , do_sample=True makes the answer random
generator("What are we having for dinner?")


```

