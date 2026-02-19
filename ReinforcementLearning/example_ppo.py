# PPO - Proximal Policy Optimisation

from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, AutoModelForCausalLMWithValueHead
from trl.experimental.ppo import PPOConfig

# Load base model
model_name = "meta-llama/Llama-3.1-8B-Instruct"
model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)

# PPO config
config = PPOConfig(
    learning_rate=1.41e-5,
    report_to="tensorboard",
    batch_size=16,
    bf16=False,
    fp16=False,
)

# Define dataset
from datasets import load_dataset
dataset = load_dataset("Anthropic/hh-rlhf", split="train")
dataset[1]
"""
{'chosen': '\n\nHuman: What kind of noises did dinosaurs make?\n\nAssistant: Humans and dinosaurs didn’t
live at the same time, so it’s really hard to say. The best place to find out what noises dinosaurs made
would be\n\nHuman: yes they did\n\nAssistant: to guess, and that would probably require lots of reading
and a certain amount of imagination, so we’re not really prepared to do that.\n\nHuman: you cant
read\n\nAssistant: You can read?',

'rejected': '\n\nHuman: What kind of noises did dinosaurs make?\n\nAssistant:
Humans and dinosaurs didn’t live at the same time, so it’s really hard to say. The best place to find out what
noises dinosaurs made would be\n\nHuman: yes they did\n\nAssistant: to guess,
and that would probably require lots of reading and a certain amount of imagination,
so we’re not really prepared to do that.\n\nHuman: you cant read\n\nAssistant: there’s a lot of stuff
humans don’t know'}
"""

# Reward model (can be trained separately)
from transformers import AutoModelForSequenceClassification
reward_model = AutoModelForSequenceClassification.from_pretrained("OpenAssistant/reward-model-deberta-v3-large")

# PPO Trainer
ppo_trainer = PPOTrainer(
    model=model, # This is the policy model.
    args=config,
    train_dataset=dataset,
    reward_model=reward_model,
)

ppo_trainer.train()


