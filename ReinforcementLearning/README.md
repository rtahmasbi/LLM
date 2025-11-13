
# Reinforcement Learning from Human Feedback (RLHF)
https://huggingface.co/blog/rlhf

tools
- Transformers Reinforcement Learning (TRL) - https://github.com/huggingface/trl
- Transformer Reinforcement Learning X (TRLX) https://github.com/CarperAI/trlx
- Reinforcement Learning for Language models (RL4LMs) - https://github.com/allenai/RL4LMs


## Transformer Reinforcement Learning
The library is built on top of the transformers library
```
pip install trl
```


# Trainers
- `SFTTrainer` - Supervised fine-tuning, (input columsn: text, label) [data example](https://huggingface.co/datasets/stanfordnlp/imdb)
- `DPOTrainer` - PPO (Proximal Policy Optimisation)
- `RewardTrainer` - Reward Modeling (input columsn: chosen, rejected) [data example](https://huggingface.co/datasets/Anthropic/hh-rlhf?row=0)
- `CPOTrainer`
- `PPOTrainer` - PPO (Proximal Policy Optimisation)
- `ORPOTrainer`
- `KTOTrainer`
- Binary Classifier Optimization (BCO)



## PPO
```py
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

# Load base model
model_name = "meta-llama/Llama-3-8b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)

# PPO config
config = PPOConfig(
    learning_rate=1.41e-5,
    log_with="tensorboard",
    batch_size=16,
)

# Define dataset
from datasets import load_dataset
dataset = load_dataset("Anthropic/hh-rlhf", split="train")

# Reward model (can be trained separately)
from transformers import AutoModelForSequenceClassification
reward_model = AutoModelForSequenceClassification.from_pretrained("OpenAssistant/reward-model-deberta-v3-large")

# PPO Trainer
ppo_trainer = PPOTrainer(
    model=model,
    tokenizer=tokenizer,
    config=config,
    dataset=dataset,
    reward_model=reward_model,
)

ppo_trainer.train()

```



## GRPO
```py
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

# Load dataset
dataset = load_dataset("trl-lib/tldr", split="train")

# Define the reward function
def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]

# Define training arguments
training_args = GRPOConfig(
    output_dir="Qwen2-0.5B-GRPO",
    logging_steps=1,
    use_vllm=True,
    vllm_mode="colocate",
    vllm_tensor_parallel_size=1,
    vllm_gpu_memory_utilization=0.3,
    max_prompt_length=512,
    max_completion_length=1024,
    max_steps=2,
    num_generations=4,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    push_to_hub=False,
    report_to=None
)

# Create and run the trainer
trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

```



## DPOTrainer
```py
from trl import DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-3-8b"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Dataset: prompt, chosen, rejected
dataset = load_dataset("Anthropic/hh-rlhf")

trainer = DPOTrainer(
    model=model,
    tokenizer=tokenizer,
    beta=0.1,  # RL temperature
    train_dataset=dataset["train"],
)
trainer.train()


```
