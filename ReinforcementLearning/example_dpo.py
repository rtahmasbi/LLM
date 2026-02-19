
"""
DPO - Direct Preference Optimization

You can find more examples here: https://github.com/huggingface/trl/blob/main/trl/scripts/dpo.py


Loss functions
Given the preference data, we can fit a binary classifier according to the Bradley-Terry model and in fact the DPO authors propose the sigmoid loss on the normalized likelihood via the logsigmoid to fit a logistic regression.
- The RSO authors propose to use a hinge loss on the normalized likelihood
- The IPO authors provide a deeper theoretical understanding of the DPO algorithms and identify an issue with overfitting and propose an alternative loss which can be used via the loss_type="ipo" argument to the trainer.


sigmoid
robust
exo_pair
hinge
ipo
bco_pair
sppo_hard
nca_pair
aot_unpaired
aot
apo_zero
apo_down
discopop
sft

['sigmoid', 'hinge', 'ipo', 'exo_pair', 'nca_pair', 'robust', 'bco_pair', 'sppo_hard',
'aot', 'aot_unpaired', 'discopop', 'apo_zero', 'apo_down', 'sft']

"""


from trl import DPOConfig, DPOTrainer

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-3-8b"
model_name = "meta-llama/Llama-3.2-1B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Dataset: prompt, chosen, rejected
train_dataset = load_dataset("Anthropic/hh-rlhf", split="train")
# train_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

"""
train_dataset
Dataset({
    features: ['chosen', 'rejected'],
    num_rows: 160800
})

"""


training_args = DPOConfig(
    output_dir="ras1",
    bf16=False, # if ValueError: Your setup doesn't support bf16/gpu.
    fp16=False, # if ValueError: Your setup doesn't support bf16/gpu.
)

trainer = DPOTrainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=train_dataset
    beta=0.1,  # RL temperature
)


trainer.train()

