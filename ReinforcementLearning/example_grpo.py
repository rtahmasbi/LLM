
## Group Relative Policy Optimization (GRPO)

from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

# Load dataset
train_dataset = load_dataset("trl-lib/tldr", split="train")
"""
Dataset({
    features: ['prompt', 'completion'],
    num_rows: 116722
})


"""

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
    max_completion_length=1024,
    max_steps=2,
    num_generations=4,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    push_to_hub=False,
    report_to=None,
    bf16=False, # if ValueError: Your setup doesn't support bf16/gpu.
    fp16=False, # if ValueError: Your setup doesn't support bf16/gpu.
)

# Create and run the trainer
trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()



# Or define another model as reward_model:

from trl import GRPOTrainer

trainer = GRPOTrainer(
    model="meta-llama/Llama-3-8b-instruct",
    ref_model="meta-llama/Llama-3-8b-instruct",
    dataset=train_dataset,
    reward_model="OpenAssistant/reward-model-deberta-v3-large",
)

trainer.train()


