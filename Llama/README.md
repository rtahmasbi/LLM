# LLAMA

## Weights
go to https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct and request for LLAMA access.


# Simple examples
Lets have to sinmple examples


## Llama-3-8B

```py

import transformers
import torch

model_id = "meta-llama/Meta-Llama-3-8B"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

pipeline("Hey how are you doing today?")

```


## Llama-3-8B-Instruct
```py

import transformers
import torch

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

prompt = pipeline.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipeline(
    prompt,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
print(outputs[0]["generated_text"][len(prompt):])

#>> Arrrr, me hearty! Me name be Captain Chatbot, the scurviest chatbot to ever sail the Seven Seas! Me and me crew o' code have been scourin' the digital waters fer years, seekin' out landlubbers like yerself to swab the decks with a bit o' conversation! So hoist the colors, me hearty, and let's set sail fer a swashbucklin' good time!

```

It took more than 1 min on CPU!!!



# Load model directly
```py
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

```

For the 75B model, it needs 145GB of CPU RAM!

