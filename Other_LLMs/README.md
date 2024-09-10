# TinyLlama
https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF

(max 3.5GB RAM)


# Mistral
https://huggingface.co/TheBloke/Mistral-7B-Claude-Chat-GGUF



# facebook/opt-350m
https://huggingface.co/facebook/opt-350m

https://github.com/huggingface/


`pip install tf-keras`


```py
from transformers import pipeline
generator = pipeline('text-generation', model="facebook/opt-350m", do_sample=True, num_return_sequences=5)
# , do_sample=True makes the answer random
generator("What are we having for dinner?")


```

