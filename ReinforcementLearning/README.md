
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
## `SFTTrainer` Supervised fine-tuning
[data example](https://huggingface.co/datasets/stanfordnlp/imdb)

SFTTrainer uses the standard LM next-token cross-entropy loss:
$\mathcal{L} = -\sum_{t=1}^{T} \log \pi_\theta (y_t \mid y_{<t})$


```json
{
    "prompt": "### Question: ...",
    "response": "### Answer: ..."
}
```



## `DPOTrainer` - Direct Preference Optimization
```json
{
    "prompt": "...",
    "chosen": "...",   # preferred answer
    "rejected": "..."  # dispreferred answer
}
```
DPO fine-tunes a model by increasing the likelihood of the chosen output relative to rejected output.
```math
A_\theta(x, c) = \log \pi_\theta(c \mid x) - \log \pi_{\text{ref}}(c \mid x),
```

```math
A_\theta(x, r) = \log \pi_\theta(r \mid x) - \log \pi_{\text{ref}}(r \mid x).
```

```math
\mathcal{L}_{\text{DPO}} = - \log \sigma\!\left( \beta \left[ A_\theta(x, c) - A_\theta(x, r) \right] \right)
```



## `RewardTrainer` - Reward Modeling
[data example](https://huggingface.co/datasets/Anthropic/hh-rlhf?row=0)
```json
{
    "prompt": "...",
    "chosen": "...",      # preferred output
    "rejected": "..."     # dispreferred output
}
```

`RewardTrainer` trains a reward model used for RLHF or DPO.
The input is pairwise preference data, similar to DPO but the model outputs a scalar reward, not a token distribution.

$$
r_\theta(x, c) = \text{RM}_\theta(x, c),
$$

$$
r_\theta(x, r) = \text{RM}_\theta(x, r).
$$

$$
\mathcal{L}_{\text{RM}} = -
\log \sigma \left(r_\theta(x, c) - r_\theta(x, r)\right),
$$



## `CPOTrainer`
## `PPOTrainer` - PPO (Proximal Policy Optimisation)
PPO uses a clipped policy gradient objective + KL reward shaping.

Policy ratio:

$$
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}.
$$

Clipped PPO Objective:

$$
\mathcal{L}_{\text{PPO}}
=
\mathbb{E}_t \left[
\min\Big(
r_t(\theta) A_t,\;
\text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) A_t
\Big)
\right].
$$


KL Penalty (used in RLHF):
$$
r_t^{\text{RLHF}}
=
r_t^{\text{RM}}
-
\beta \, 
D_{\mathrm{KL}}
\!\left(
\pi_\theta(\cdot \mid s_t)
\;\|\;
\pi_{\text{ref}}(\cdot \mid s_t)
\right).
$$


Generalized Advantage Estimation (GAE):
$$
A_t
=
\sum_{l=0}^{\infty}
(\gamma \lambda)^l
\left(
\delta_{t+l}
\right),
\qquad
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t).
$$


Full PPO Loss (Policy + Value + Entropy):
$$
\mathcal{L}
=
\mathcal{L}_{\text{PPO}}
+
c_v \, (V_\theta(s_t) - R_t)^2
-
c_e \, H(\pi_\theta(\cdot \mid s_t)).
$$





## `ORPOTrainer`
## `KTOTrainer`
## Binary Classifier Optimization (BCO)
## Group Relative Policy Optimization


# Codes
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
