
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
## `SFTTrainer` - Supervised fine-tuning
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



**Loss functions**
Given the preference data, we can fit a binary classifier according to the Bradley-Terry model and in fact the DPO authors propose the sigmoid loss on the normalized likelihood via the logsigmoid to fit a logistic regression.
- The RSO authors propose to use a hinge loss on the normalized likelihood
- The IPO authors provide a deeper theoretical understanding of the DPO algorithms and identify an issue with overfitting and propose an alternative loss which can be used via the loss_type="ipo" argument to the trainer.

- sigmoid
- robust
- exo_pair
- hinge
- ipo
- bco_pair
- sppo_hard
- nca_pair
- aot_unpaired
- aot
- apo_zero
- apo_down
- discopop
- sft



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

## `PPOTrainer` - Proximal Policy Optimisation
PPO uses a clipped policy gradient objective + KL reward shaping.

The clip function is defined as: $$clip(x,a,b)$$ is $$a$$ if $$x<a$$, is $$b$$ if $$x>b$$ else, $$x$$.


Policy ratio:

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}.$$

Clipped PPO Objective:

$$\mathcal{L}_{\text{PPO}} = \mathbb{E}_t [\min( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) A_t )].$$


KL Penalty (used in RLHF):

$$r_t^{\text{RLHF}} = r_t^{\text{RM}} - \beta \, D_{\mathrm{KL}}
(
\pi_\theta(\cdot \mid s_t)
\;\|\;
\pi_{\text{ref}}(\cdot \mid s_t) ).
$$


Generalized Advantage Estimation (GAE):

$$A_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l (\delta_{t+l}), \qquad
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t).
$$


Full PPO Loss (Policy + Value + Entropy):

$$\mathcal{L} = \mathcal{L}_{\text{PPO}} + c_v \, (V_\theta(s_t) - R_t)^2 - c_e \, H(\pi_\theta(\cdot \mid s_t)).
$$




## `GRPOTrainer`
## `ORPOTrainer`
## `KTOTrainer`
## Binary Classifier Optimization (BCO)
## Group Relative Policy Optimization
## `RLOOTrainer` - Reinforce Leave One Out




# Papers
## Group Relative Policy Optimization (GRPO) - 2025
https://www.nature.com/articles/s41586-025-09422-z.pdf


## Reinforcement Learning with Verifiable Rewards (RLVR) - 2025
https://arxiv.org/pdf/2506.14245


## Decoupled Clip and Dynamic sAmpling Policy Optimization (DAPO) - 2025
https://arxiv.org/pdf/2503.14476


## Identity Preference Optimisation (IPO)
https://arxiv.org/pdf/2502.16182v1

## Kahneman-Tversky Optimisation (KTO)
https://arxiv.org/pdf/2402.01306




# references
https://github.com/norhum/reinforcement-learning-from-scratch/blob/main/README.md
Reinforcement Learning From Scratch
- Multi-Armed Bandits (MAB): (Epsilon-Greedy strategy, Upper Confidence Bound)
- Value-Based Methods: (Q-Values, Q-Learning, SARSA)
- Deep Reinforcement Learning: (Deep Q-Networks - DQN)
- Policy Gradient Methods: (Monte Carlo policy gradients)
- Actor-Critic Methods: (Advantage Actor-Critic (A2C) algorithm)
