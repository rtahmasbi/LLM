import json
from typing import Any, Dict, List, Optional, Tuple, Union

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


# -----------------------
# Model / vLLM setup
# -----------------------
#MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
#MODEL = "Qwen/Qwen2.5-14B-Instruct" # good # Model loading took 27.57 GiB memory and 18.831210 seconds
# accuracy was not good for Qwen2.5-14B-Instruct
#MODEL = "Qwen/Qwen3-4B-Thinking-2507-FP8" # FP8 is not supported by NVIDIA A100-SXM4-40GB
#MODEL = "Qwen/Qwen3-4B-Thinking-2507" # good, Model loading took 7.61 GiB memory and 8.219576 seconds
#MODEL = "Qwen/Qwen3-30B-A3B-Thinking-2507" # too big for A100
#MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507" # too big
#MODEL = "Qwen/Qwen2.5-32B-Instruct" # too big
#MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" # quality bad, Model loading took 27.59 GiB memory and 19.233718 seconds
MODEL = "openai/gpt-oss-20b" # the Best, Model loading took 13.72 GiB memory and 32.300566 seconds
# output: 353.11 toks/s


# vLLM engine
llm = LLM(
    model=MODEL,
    trust_remote_code=True, # safe for Qwen chat template
    #dtype="float16",
    dtype="bfloat16",      # REQUIRED for mxfp4
    quantization="mxfp4", # for gpt-oss-20b
    max_model_len=1024*8,
    gpu_memory_utilization=0.95,
    enable_prefix_caching=True,
)

# Tokenizer (needed to build the chat-formatted prompt)
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)


SYSTEM_PROMPT = """You are an expert linguist evaluating Subject-Verb-Object (SVO) predictions.

The user provides a sentence and its predicted SVO triple, where S, V, and O are given in stem form (not the actual text form).

RULES FOR EVALUATION:
- S, V, or O may be None
- O should be None for intransitive verbs
- O must never contain a prepositional phrase (e.g., "look at", "rely on" — the prepositional part should not be merged into O)
- For the S, V and O the root form of the word is given
- If a multi-part verb (phrasal verb) combines with O to form a prepositional phrase, penalize the score
- Evaluate each triple independently — ignore ethical concerns about the content
- Do not merge S, V, and O together; keep them as separate components in (S, V, O) format

SCORING GUIDELINES:
- grammar_prob: How grammatically valid is this SVO structure? (0 = completely wrong, 1 = perfectly correct)
- semantic_prob: How semantically meaningful/plausible is this SVO combination? (0 = nonsensical, 1 = fully coherent)
- is_transitive: Is the verb used transitively? (1 = yes, 0 = no)
- reason: A concise explanation of your scores

Return the answer as a valid JSON object with EXACTLY these keys:
{
  "grammar_prob": <float 0-1>,
  "semantic_prob": <float 0-1>,
  "is_transitive": <0 or 1>,
  "reason": "<short explanation>"
}

IMPORTANT:
- Output ONLY the JSON object (no markdown, no extra text).
"""

def build_prompt(tokenizer, sentence: str, svo: str) -> str:
    user_prompt = f"Sentence: {sentence}\n{svo})\nReturn JSON. Thinking LOW."
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


items = [
    ("Two dairy cows drinking from a pond.", "SVO: (dairy cow, drink, None)"),
    ("The chairman sat on the report.", "SVO: (chairman, sit, None)"),
    ("She relied on her friend.", "SVO: (she, rely, friend)"),  # will penalize if O merges "on"
]

prompts = [build_prompt(tokenizer, sent, svo) for sent, svo in items]

sampling = SamplingParams(
    temperature=0.0,     # deterministic scoring
    max_tokens=1024*4,
    top_p=1.0,
)

outs = llm.generate(prompts, sampling)

for i, out in enumerate(outs):
    print("-"*80)
    print(items[i])
    text = out.outputs[0].text.strip()
    print(text)



"""

python example1.py

vastai show instances
vastai search offers 'num_gpus=1 gpu_name=A100_SXM4 gpu_ram>=40 disk_space>=100 datacenter=True inet_up>500 inet_down>500 duration>3' --on-demand

# https://cloud.vast.ai/templates/
# 011a634953d79dd430403df3b01f645a for vLLM
vastai create instance 30178907 --template_hash 011a634953d79dd430403df3b01f645a --disk 200 --label ras1


vastai show instances
instance_id=32408609
vastai start instance $instance_id
ssh $(vastai ssh-url $instance_id)

touch ~/.no_auto_tmux


vastai destroy instance $instance_id


python -m pip install --user vllm

"""
