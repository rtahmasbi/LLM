
# Reinforcement Learning from Human Feedback (RLHF)
https://huggingface.co/blog/rlhf

tools
- Transformers Reinforcement Learning (TRL) - https://github.com/huggingface/trl
- Transformer Reinforcement Learning X (TRLX) https://github.com/CarperAI/trlx
- Reinforcement Learning for Language models (RL4LMs) - https://github.com/allenai/RL4LMs


```
pip install trl
```

# Trainers
- SFTTrainer - Supervised fine-tuning, (input columsn: text, label) [data example](https://huggingface.co/datasets/stanfordnlp/imdb)
- DPOTrainer - PPO (Proximal Policy Optimisation)
- RewardTrainer - Reward Modeling (input columsn: chosen, rejected) [data example](https://huggingface.co/datasets/Anthropic/hh-rlhf?row=0)
- CPOTrainer
- PPOTrainer - PPO (Proximal Policy Optimisation)
