# OPENAI_API_KEY
Go to
https://platform.openai.com/account/api-keys
and get your API_KEY.


```sh
export OPENAI_API_KEY=...
```

# Simple examples

## example 1: chat.completions
```py

from openai import OpenAI
client = OpenAI()

#model_id = "gpt-4o-mini"
model_id = "gpt-3.5-turbo"

response = client.chat.completions.create(
  model=model_id,
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who won the world series in 2020?"},
    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
    {"role": "user", "content": "Where was it played?"}
  ]
)

response.choices[0].message.content
# 'The 2020 World Series was played at Globe Life Field in Arlington, Texas.'

```

## example 2:

```py

from openai import OpenAI
client = OpenAI()

text = "Heap buffer overflow in libwebp in Google Chrome prior to 116.0.5845.187 and libwebp 1.3.2 allowed a remote attacker to perform an out of bounds memory write via a crafted HTML page. (Chromium security severity: Critical)"

response = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a cybersecurity assistant."},
    {"role": "user", "content": f"what are the valnarable libraries or packeges in this statement: {text}"}
  ]
)

response.choices[0].message.content
'The vulnerable library mentioned in the statement is libwebp. The vulnerability specifically relates to a heap buffer overflow in libwebp, which impacts Google Chrome versions prior to 116.0.5845.187 and libwebp 1.3.2. This vulnerability allows a remote attacker to perform an out-of-bounds memory write via a crafted HTML page. This issue has been classified as having a critical severity by the Chromium security team.'

```



## get list of all OpenAI models
```py

import os
from openai import OpenAI
client = OpenAI()

models = client.models.list()
for m in models:
   print(m)

```


## calling API using curl

```sh
curl https://api.openai.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
     "model": "gpt-3.5-turbo",
     "messages": [{"role": "user", "content": "Say this is a test!"}],
     "temperature": 0.7
   }'

```




# fine-tuning
https://platform.openai.com/docs/guides/fine-tuning


You need to prepare data and upload it to OpenAI, they will take care of the rest.

## format
The conversational chat format is required to fine-tune gpt-4o-mini and gpt-3.5-turbo:
```
{"messages": [{"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."}, {"role": "user", "content": "What's the capital of France?"}, {"role": "assistant", "content": "Paris, as if everyone doesn't know that already."}]}
{"messages": [{"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."}, {"role": "user", "content": "Who wrote 'Romeo and Juliet'?"}, {"role": "assistant", "content": "Oh, just some guy named William Shakespeare. Ever heard of him?"}]}
{"messages": [{"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."}, {"role": "user", "content": "How far is the Moon from Earth?"}, {"role": "assistant", "content": "Around 384,400 kilometers. Give or take a few, like that really matters."}]}
```

For babbage-002 and davinci-002, you can follow the prompt completion pair format as shown below:
```
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
```


Multi-turn chat examples:
```
{"messages": [{"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."}, {"role": "user", "content": "What's the capital of France?"}, {"role": "assistant", "content": "Paris", "weight": 0}, {"role": "user", "content": "Can you be more sarcastic?"}, {"role": "assistant", "content": "Paris, as if everyone doesn't know that already.", "weight": 1}]}
{"messages": [{"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."}, {"role": "user", "content": "Who wrote 'Romeo and Juliet'?"}, {"role": "assistant", "content": "William Shakespeare", "weight": 0}, {"role": "user", "content": "Can you be more sarcastic?"}, {"role": "assistant", "content": "Oh, just some guy named William Shakespeare. Ever heard of him?", "weight": 1}]}
{"messages": [{"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."}, {"role": "user", "content": "How far is the Moon from Earth?"}, {"role": "assistant", "content": "384,400 kilometers", "weight": 0}, {"role": "user", "content": "Can you be more sarcastic?"}, {"role": "assistant", "content": "Around 384,400 kilometers. Give or take a few, like that really matters.", "weight": 1}]}
```


```py
from openai import OpenAI
client = OpenAI()

client.files.create(
  file=open("mydata.jsonl", "rb"),
  purpose="fine-tune"
)

```
and
```py
from openai import OpenAI
client = OpenAI()

client.fine_tuning.jobs.create(
  training_file="file-abc123", 
  model="gpt-4o-mini"
)
```



# pricing
https://openai.com/pricing


# images.generate()
It will return the url for the generated image.

## generate image
```py

from openai import OpenAI
client = OpenAI()

response = client.images.generate(
  model="dall-e-3",
  prompt="a white cat at a green pool",
  size="1024x1024",
  quality="standard",
  n=1,
)

image_url = response.data[0].url
print(image_url)

```


## adding mask
```py
from openai import OpenAI
client = OpenAI()

response = client.images.edit((
  model="dall-e-2",
  image=open("sunlit_lounge.png", "rb"),
  mask=open("mask.png", "rb"),
  prompt="A sunlit indoor lounge area with a pool containing a flamingo",
  n=1,
  size="1024x1024"
)
image_url = response.data[0].url

```