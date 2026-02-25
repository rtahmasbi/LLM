
https://huggingface.co/Qwen/Qwen3-0.6B

```py
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
messages = [
    {"role": "user", "content": "Who are you?"},
]
inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))




# In this mode, the model will not generate any think content and will not include a <think>...</think> block.
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
)

print(text)
# '<|im_start|>user\nGive me a short introduction to large language model.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'

```


# Advanced Usage: Switching Between Thinking and Non-Thinking Modes via User Input:
For API compatibility, when enable_thinking=True, regardless of whether the user uses /think or /no_think,
the model will always output a block wrapped in <think>...</think>.
However, the content inside this block may be empty if thinking is disabled.
When enable_thinking=False, the soft switches are not valid. Regardless of any /think or /no_think tags
input by the user, the model will not generate think content and will not include a <think>...</think> block.

```py
user_input_2 = "Then, how many r's in blueberries? /no_think"
user_input_3 = "Really? /think"
```



https://qwen.readthedocs.io/en/latest/getting_started/quickstart.html

Qwen3-Instruct-2507 supports only non-thinking mode and does not generate <think></think> blocks in its output. Different from Qwen3-2504, specifying enable_thinking=False is no longer required or supported.


