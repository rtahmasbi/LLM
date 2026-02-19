
# pip install math-verify

from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from trl.rewards import accuracy_reward

from trl.import_utils import is_vllm_available

train_dataset = load_dataset("trl-lib/DeepMath-103K", split="train")
train_dataset
"""
Dataset({
    features: ['prompt', 'solution'],
    num_rows: 97870
})
"""

training_args = GRPOConfig(
    push_to_hub=False,
    report_to=None,
    bf16=False, # if ValueError: Your setup doesn't support bf16/gpu.
    fp16=False, # if ValueError: Your setup doesn't support bf16/gpu.
)


trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=accuracy_reward,
    train_dataset=train_dataset,
    args=training_args,
)

trainer.train()


"""
is_vllm_available()
<python-input-7>:1: UserWarning: TRL currently supports vLLM versions: 0.10.2, 0.11.0, 0.11.1, 0.11.2, 0.12.0. You have version 0.15.1 installed. We recommend installing a supported version to avoid compatibility issues.
True

"""