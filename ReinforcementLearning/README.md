
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
SFTTrainer uses the standard LM next-token cross-entropy loss:
$$
L=−\sum_t \​log P_\thta​(y_t​|y<t​)
$$

```json
{
    "prompt": "### Question: ...",
    "response": "### Answer: ..."
}
```



- `DPOTrainer` - Direct Preference Optimization
```json
{
    "prompt": "...",
    "chosen": "...",   // preferred answer
    "rejected": "..."  // dispreferred answer
}
```
DPO fine-tunes a model by increasing the likelihood of the chosen output relative to rejected output.
$$
A_\theta(x, c) = \log \pi_\theta(c \mid x) - \log \pi_{\text{ref}}(c \mid x),
A_\theta(x, r) = \log \pi_\theta(r \mid x) - \log \pi_{\text{ref}}(r \mid x).

\mathcal{L}_{\text{DPO}} = - \log \sigma\!\left( \beta \left[ A_\theta(x, c) - A_\theta(x, r) \right] \right)
$$



- `RewardTrainer` - Reward Modeling (input columsn: chosen, rejected) [data example](https://huggingface.co/datasets/Anthropic/hh-rlhf?row=0)
- `CPOTrainer`
- `PPOTrainer` - PPO (Proximal Policy Optimisation)
- `ORPOTrainer`
- `KTOTrainer`
- Binary Classifier Optimization (BCO)
- Group Relative Policy Optimization



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



```py
from trl import GRPOTrainer

trainer = GRPOTrainer(
    model="meta-llama/Llama-3-8b-instruct",
    ref_model="meta-llama/Llama-3-8b-instruct",
    dataset=dataset,
    reward_model="OpenAssistant/reward-model-deberta-v3-large"
)
trainer.train()

```


## DPO - Direct Preference Optimization
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

# references
https://github.com/norhum/reinforcement-learning-from-scratch/blob/main/README.md
Reinforcement Learning From Scratch
- Multi-Armed Bandits (MAB): (Epsilon-Greedy strategy, Upper Confidence Bound)
- Value-Based Methods: (Q-Values, Q-Learning, SARSA)
- Deep Reinforcement Learning: (Deep Q-Networks - DQN)
- Policy Gradient Methods: (Monte Carlo policy gradients)
- Actor-Critic Methods: (Advantage Actor-Critic (A2C) algorithm)
