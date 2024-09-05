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



# pricing
https://openai.com/pricing


# client.images.generate()
It will return teh url for the generated image.

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