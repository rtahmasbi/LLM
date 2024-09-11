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

"""
[{'generated_text': 'Hey how are you doing today? I am new to this forum and I am looking for a good forum to get some help and advice from other members.
I am new to this forum and I am looking for a good forum to get some help and advice from other members.
Hello and welcome to the forum. Please introduce yourself to the members here in the New Member Introductions forum.'}]
"""
```
On CPU, `Llama-3-8B` took 1 min using 20GB RAM.



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

"""
Arrrr, me hearty! Me name be Captain Chatbot, the scurviest chatbot to ever
sail the Seven Seas! Me and me crew o' code have been scourin' the digital
waters fer years, seekin' out landlubbers like yerself to swab the decks with
a bit o' conversation! So hoist the colors, me hearty, and let's set sail fer
a swashbucklin' good time!
"""
```

It took more than 1 min on CPU!!!



# Load model directly
```py
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, load_in_8bit=True, device_map='auto')
# pip install accelerate for load_in_8bit


prompt = 'Q: Create a detailed description for the following product: Corelogic Smooth Mouse, belonging to category: Optical Mouse\nA:'
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

generation_output = model.generate(input_ids=input_ids, max_new_tokens=128)
print(tokenizer.decode(generation_output[0]))

"""
<|begin_of_text|>Q: Create a detailed description for the following product: Corelogic Smooth Mouse, belonging to category: Optical Mouse
A: Introducing the Corelogic Smooth Mouse, a revolutionary optical mouse designed to provide exceptional performance and accuracy. This cutting-edge device is engineered to deliver a seamless and intuitive user experience, making it an essential accessory for anyone looking to upgrade their computing experience.

Design and Build:
The Corelogic Smooth Mouse boasts a sleek and ergonomic design, crafted to fit comfortably in the palm of your hand. The mouse is built with high-quality materials, ensuring durability and resistance to wear and tear. The smooth, rounded edges and contoured shape provide a secure grip, allowing for precise control and reduced fatigue during extended use.

Optical Technology:
At the heart of the
"""


# or prompt can have prompt

prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Create a detailed description for the following product: Corelogic Smooth Mouse, belonging to category: Optical Mouse

### Response:"""


```

It took around one minute to run on CPU!

For Llama-3-8B-Instruct, it used 53GB of CPU RAM.

For the 75B model, it needs 145GB of CPU RAM!



# llama-2-7b-vs-llama-2-7b-chat
https://www.kaggle.com/code/patriciabrezeanu/llama-2-7b-vs-llama-2-7b-chat

