from datasets import load_dataset
from trl import RewardConfig, RewardTrainer
import torch

train_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")
"""
Dataset({
    features: ['chosen', 'rejected', 'score_chosen', 'score_rejected'],
    num_rows: 62135
})

"""

training_args = RewardConfig(
    model_init_kwargs={"dtype": torch.bfloat16},
    bf16=False, # if ValueError: Your setup doesn't support bf16/gpu.
    fp16=False, # if ValueError: Your setup doesn't support bf16/gpu.
)

trainer = RewardTrainer(
    args=training_args,
    model="Qwen/Qwen3-0.6B",
    train_dataset=train_dataset,
)

trainer.train()

