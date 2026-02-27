"""
RL Training for Qwen3:xb Thinking Model — with GSM8K Support
=============================================================
Uses GRPO (Group Relative Policy Optimization) — the same RL algorithm
used to train DeepSeek-R1 and Qwen3 thinking models.

Algorithm overview:
  1. Sample G responses per prompt (group sampling)
  2. Score each response with a reward function
  3. Normalize rewards within the group → advantages
  4. Compute policy gradient loss (REINFORCE with baseline)
  5. Add KL penalty against a frozen reference model
  6. Backprop + update

Dependencies:
    pip install torch transformers datasets tqdm accelerate bitsandbytes

For 8B model, recommended: GPU with ≥24GB VRAM (or use 4-bit quant).

# important
- We have a policy model (the model that we are going to train) and the ref_model, which we dont train.
- Samples are taken from the policy model.

"""

import os
import re
import json
import math
import time
import random
import argparse
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset as hf_load_dataset
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GRPOConfig:
    # Model
    model_name: str = "Qwen/Qwen3-8B"          # HuggingFace model ID
    load_in_4bit: bool = True                    # Use 4-bit quantization (saves VRAM)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Training
    learning_rate: float = 1e-6
    num_epochs: int = 3
    batch_size: int = 2                          # prompts per update step
    group_size: int = 8                          # G: responses sampled per prompt
    max_new_tokens: int = 512
    max_prompt_len: int = 512
    
    # GRPO
    kl_coeff: float = 0.04                       # β: KL penalty weight
    clip_eps: float = 0.2                        # PPO-style clipping epsilon
    temperature: float = 0.8                     # sampling temperature
    top_p: float = 0.95
    
    # Logging
    log_every: int = 1
    save_every: int = 50
    output_dir: str = "./qwen_rl_checkpoints"
    
    # Thinking mode
    enable_thinking: bool = True                 # Qwen3 /think mode
    
    # Dataset
    dataset_source: str = "gsm8k"               # "gsm8k" | "json" | "builtin"
    dataset_path: Optional[str] = None          # path for "json" source
    gsm8k_train_split: str = "train"            # GSM8K train split
    gsm8k_test_split: str = "test"              # GSM8K eval split
    max_train_samples: Optional[int] = None     # cap training set size (None = all)
    max_eval_samples: Optional[int] = 200       # cap eval set size for speed


# ──────────────────────────────────────────────────────────────────────────────
# Dataset Loading
# ──────────────────────────────────────────────────────────────────────────────

BUILTIN_DATASET = [
    {"prompt": "What is 17 x 43?",                                                  "answer": "731"},
    {"prompt": "What is the square root of 144?",                                   "answer": "12"},
    {"prompt": "Solve: 2x + 5 = 13. What is x?",                                    "answer": "4"},
    {"prompt": "What is 15% of 200?",                                               "answer": "30"},
    {"prompt": "If a train travels 60 mph for 2.5 hours, how far does it go?",      "answer": "150"},
    {"prompt": "What is the capital of France?",                                    "answer": "Paris"},
    {"prompt": "How many days are in a leap year?",                                 "answer": "366"},
    {"prompt": "What is 7 factorial (7!)?",                                         "answer": "5040"},
    {"prompt": "Solve: x² = 81. What is x (positive)?",                             "answer": "9"},
    {"prompt": "What is 3^5?",                                                      "answer": "243"},
]


def extract_gsm8k_answer(raw_answer: str) -> str:
    """
    GSM8K answers look like:
        'Janet sells ... \n#### 42'
    Extract the numeric part after ####, strip commas.
    """
    if "####" in raw_answer:
        return raw_answer.split("####")[-1].strip().replace(",", "")
    # Fallback: grab the last number in the string
    nums = re.findall(r"-?\d+(?:\.\d+)?", raw_answer.replace(",", ""))
    return nums[-1] if nums else raw_answer.strip()


def load_gsm8k(split: str = "train", max_samples: Optional[int] = None) -> list[dict]:
    """
    Load GSM8K from HuggingFace and format for the trainer.
    Returns list of {"prompt": ..., "answer": ...} dicts.
    """
    print(f"📥 Loading GSM8K ({split} split)...")
    ds = hf_load_dataset("openai/gsm8k", "main", split=split)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    formatted = [
        {
            "prompt": item["question"],
            "answer": extract_gsm8k_answer(item["answer"]),
        }
        for item in ds
    ]
    print(f"   ✓ Loaded {len(formatted)} GSM8K samples.")
    return formatted


def load_dataset_for_config(config: GRPOConfig) -> tuple[list[dict], list[dict]]:
    """
    Load train + eval datasets based on config.dataset_source.
    Returns (train_data, eval_data).
    """
    if config.dataset_source == "gsm8k":
        train_data = load_gsm8k(config.gsm8k_train_split, config.max_train_samples)
        eval_data  = load_gsm8k(config.gsm8k_test_split,  config.max_eval_samples)
        return train_data, eval_data
    elif config.dataset_source == "json":
        assert config.dataset_path, "dataset_path must be set when dataset_source='json'"
        with open(config.dataset_path) as f:
            data = json.load(f)
        split = int(0.9 * len(data))
        return data[:split], data[split:]
    else:  # builtin
        return BUILTIN_DATASET, BUILTIN_DATASET


# ──────────────────────────────────────────────────────────────────────────────
# Reward Functions
# ──────────────────────────────────────────────────────────────────────────────

def extract_final_number(text: str) -> Optional[str]:
    """
    Extract the last number from text (strips commas, handles negatives/decimals).
    Used to compare model output to numeric GSM8K ground truth.
    """
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    clean = clean.replace(",", "")
    nums  = re.findall(r"-?\d+(?:\.\d+)?", clean)
    return nums[-1] if nums else None


def reward_gsm8k(response: str, ground_truth: str) -> float:
    """
    GSM8K-specific reward: extract the last number from the response
    and compare to ground truth. Returns 1.0 if correct, else 0.0.
    """
    predicted = extract_final_number(response)
    if predicted is None:
        return 0.0
    try:
        return 1.0 if float(predicted) == float(ground_truth.replace(",", "")) else 0.0
    except ValueError:
        return 0.0


def reward_format_quality(response: str) -> float:
    """
    Format reward: bonus for having a proper <think> block followed by an answer.
    Returns 0.0-1.0.
    """
    has_think  = bool(re.search(r"<think>.+?</think>", response, re.DOTALL))
    has_answer = (
        len(response.split("</think>")[-1].strip()) > 10
        if "</think>" in response else False
    )
    return 0.5 * has_think + 0.5 * has_answer


def reward_length_penalty(response: str, target_len: int = 300, sigma: float = 200) -> float:
    """
    Gaussian reward centered on target response length (chars/4 ≈ tokens).
    Penalizes both too-short and too-long responses.
    """
    length = len(response) / 4
    return math.exp(-((length - target_len) ** 2) / (2 * sigma ** 2))


def combined_reward_gsm8k(
    response: str,
    ground_truth: str,
    weights: tuple = (0.7, 0.2, 0.1),
) -> float:
    """
    Weighted reward tailored for GSM8K:
      0.7 x numeric correctness
      0.2 x thinking format quality
      0.1 x response length
    """
    r_correct = reward_gsm8k(response, ground_truth)
    r_format  = reward_format_quality(response)
    r_length  = reward_length_penalty(response)
    return weights[0] * r_correct + weights[1] * r_format + weights[2] * r_length


# ──────────────────────────────────────────────────────────────────────────────
# Model Utilities
# ──────────────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(config: GRPOConfig):
    """Load model with optional 4-bit quantization."""
    print(f"Loading model: {config.model_name}")
    print(f"Device: {config.device} | 4-bit quant: {config.load_in_4bit}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, trust_remote_code=True
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    bnb_config = None
    if config.load_in_4bit and config.device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if not config.load_in_4bit else None,
        device_map="auto" if config.device == "cuda" else None,
        trust_remote_code=True,
    )
    if config.device == "cpu":
        model = model.to(config.device)
    model.train()
    return model, tokenizer


# Rasool: I did not see any diff in the model_predict using enable_thinking
def build_prompt(question: str, enable_thinking: bool = True) -> str:
    """Format prompt with Qwen3 chat template + optional thinking mode."""
    think_tag = "/think" if enable_thinking else "/no_think"
    return (
        f"<|im_start|>system\nYou are a helpful assistant. {think_tag}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


@torch.no_grad()
def generate_group(
    model,
    tokenizer,
    prompt: str,
    config: GRPOConfig,
) -> list[str]:
    """
    Sample G responses for a single prompt.
    Returns list of decoded response strings (length G).
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=config.max_prompt_len,
        truncation=True,
    ).to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    responses  = []
    for idx in range(config.group_size):
        #print("idx:", idx)
        output = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            do_sample=True,
            temperature=config.temperature,
            top_p=config.top_p,
            pad_token_id=tokenizer.pad_token_id,
        )
        gen_ids  = output[0, prompt_len:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        responses.append(gen_text)
    #
    return responses


def compute_log_probs(
    model,
    tokenizer,
    prompt: str,
    response: str,
    config: GRPOConfig,
) -> torch.Tensor:
    """
    Compute sum of token log-probs for (prompt, response) under `model`.
    Returns a scalar tensor.
    """
    full_text  = prompt + response
    inputs     = tokenizer(
        full_text,
        return_tensors="pt",
        max_length=config.max_prompt_len + config.max_new_tokens,
        truncation=True,
    ).to(model.device)
    
    prompt_len = tokenizer(
        prompt, return_tensors="pt"
    ).input_ids.shape[1]
    
    outputs   = model(**inputs)
    logits    = outputs.logits[0, prompt_len - 1:-1]   # [T, vocab], shifted
    targets   = inputs["input_ids"][0, prompt_len:]    # [T]
    
    log_probs = F.log_softmax(logits, dim=-1)
    token_lps = log_probs[range(len(targets)), targets]
    return token_lps.sum()


# ──────────────────────────────────────────────────────────────────────────────
# GRPO Loss
# ──────────────────────────────────────────────────────────────────────────────

def grpo_loss(
    policy_log_probs: torch.Tensor,   # [G]
    ref_log_probs: torch.Tensor,      # [G]
    rewards: torch.Tensor,            # [G]
    config: GRPOConfig,
) -> tuple[torch.Tensor, dict]:
    """
    GRPO objective:
        L = -E[ A_i * clip(r_i, 1-ε, 1+ε) ] + β * KL(π || π_ref)

    where:
        r_i  = exp(log π - log π_ref)
        A_i  = (R_i - mean(R)) / std(R)   ← group-normalized advantage
    """
    # Normalize rewards → advantages
    mean_r     = rewards.mean()
    std_r      = rewards.std() + 1e-8
    advantages = (rewards - mean_r) / std_r                          # [G]
    
    # Probability ratio π / π_ref
    log_ratios = policy_log_probs - ref_log_probs.detach()           # [G]
    ratios     = log_ratios.exp()
    
    # Clipped surrogate
    surr1       = ratios * advantages
    surr2       = ratios.clamp(1 - config.clip_eps, 1 + config.clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # KL penalty
    kl   = (ratios - 1 - log_ratios).mean()
    loss = policy_loss + config.kl_coeff * kl
    
    return loss, {
        "policy_loss": policy_loss.item(),
        "kl":          kl.item(),
        "mean_reward": mean_r.item(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────────

class GRPOTrainer:
    def __init__(
        self,
        config: GRPOConfig,
        reward_fn: Optional[Callable[[str, str], float]] = None,
    ):
        self.config = config

        # Auto-select reward function based on dataset source
        if reward_fn is not None:
            self.reward_fn = reward_fn
        elif config.dataset_source == "gsm8k":
            self.reward_fn = combined_reward_gsm8k
            print("🎯 Using GSM8K numeric reward function.")
        else:
            self.reward_fn = combined_reward_gsm8k

        # Load datasets
        self.train_data, self.eval_data = load_dataset_for_config(config)

        # Load policy model
        self.model, self.tokenizer = load_model_and_tokenizer(config)
        self.model.config.use_cache = False # for memory

        # Frozen reference model
        print("Creating frozen reference model copy...")
        self.ref_model = deepcopy(self.model).eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

        # Optimizer + cosine LR schedule
        trainable      = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable, lr=config.learning_rate, weight_decay=0.01)
        total_steps    = config.num_epochs * math.ceil(len(self.train_data) / config.batch_size)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps, eta_min=1e-8)

        self.global_step = 0
        self.history: list[dict] = []

        os.makedirs(config.output_dir, exist_ok=True)
        print(
            f"\n✅ Trainer ready.\n"
            f"   Dataset source : {config.dataset_source}\n"
            f"   Train samples  : {len(self.train_data)}\n"
            f"   Eval  samples  : {len(self.eval_data)}\n"
            f"   Total steps    : {total_steps}\n"
        )

    # ── Training Step ──────────────────────────────────────────────────────────

    def train_step(self, batch: list[dict]) -> dict:
        """Run one GRPO update on a mini-batch of prompts."""
        batch_loss    = torch.tensor(0.0, device=self.config.device)
        batch_metrics = {"policy_loss": 0.0, "kl": 0.0, "mean_reward": 0.0}
        n             = 0

        for sample in batch:
            question = sample["prompt"]
            answer   = sample["answer"]
            prompt   = build_prompt(question, self.config.enable_thinking)

            # 1. Sample G responses
            responses = generate_group(self.model, self.tokenizer, prompt, self.config)

            # 2. Score responses
            rewards = torch.tensor(
                [self.reward_fn(r, answer) for r in responses],
                dtype=torch.float32,
                device=self.config.device,
            )

            # 3. Compute log-probs under policy and ref
            policy_lps = torch.stack([
                compute_log_probs(self.model,     self.tokenizer, prompt, r, self.config)
                for r in responses
            ])
            ref_lps = torch.stack([
                compute_log_probs(self.ref_model, self.tokenizer, prompt, r, self.config)
                for r in responses
            ])

            # 4. GRPO loss
            loss, metrics = grpo_loss(policy_lps, ref_lps, rewards, self.config)
            batch_loss    = batch_loss + loss

            for k in batch_metrics:
                batch_metrics[k] += metrics[k]
            n += 1

        batch_loss = batch_loss / max(n, 1)

        # 5. Backprop
        self.optimizer.zero_grad()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], max_norm=1.0
        )
        self.optimizer.step()
        self.scheduler.step()

        return {
            "loss": batch_loss.item(),
            "lr":   self.scheduler.get_last_lr()[0],
            **{k: v / max(n, 1) for k, v in batch_metrics.items()},
        }

    # ── Training Loop ──────────────────────────────────────────────────────────

    def train(self):
        print("=" * 60)
        print(f"  GRPO RL Training — {self.config.model_name} Thinking Model + GSM8K")
        print("=" * 60)

        for epoch in range(1, self.config.num_epochs + 1):
            print(f"\n{'─'*60}")
            print(f"  Epoch {epoch}/{self.config.num_epochs}")
            print(f"{'─'*60}")

            random.shuffle(self.train_data)
            batches = [
                self.train_data[i:i + self.config.batch_size]
                for i in range(0, len(self.train_data), self.config.batch_size)
            ]

            epoch_bar = tqdm(batches, desc=f"Epoch {epoch}", unit="batch")

            for batch in epoch_bar:
                t0      = time.time()
                metrics = self.train_step(batch)
                elapsed = time.time() - t0

                self.global_step   += 1
                metrics["epoch"]    = epoch
                metrics["step"]     = self.global_step
                metrics["time_s"]   = round(elapsed, 2)
                self.history.append(metrics)

                if self.global_step % self.config.log_every == 0:
                    epoch_bar.set_postfix({
                        "loss":   f"{metrics['loss']:.4f}",
                        "reward": f"{metrics['mean_reward']:.3f}",
                        "kl":     f"{metrics['kl']:.4f}",
                        "lr":     f"{metrics['lr']:.2e}",
                    })

                if self.global_step % self.config.save_every == 0:
                    self.save_checkpoint()

        print("\n✅ Training complete!")
        self.save_checkpoint(final=True)
        self.save_history()

    # ── Checkpointing ──────────────────────────────────────────────────────────

    def save_checkpoint(self, final: bool = False):
        tag  = "final" if final else f"step_{self.global_step}"
        path = f"{self.config.output_dir}/{tag}"
        print(f"\n💾 Saving checkpoint → {path}")
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def save_history(self):
        path = f"{self.config.output_dir}/training_history.json"
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"📊 Training history saved → {path}")

    # ── Evaluation ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, data: Optional[list[dict]] = None) -> dict:
        """
        Run greedy decoding on eval set and report:
          - avg_reward  (weighted composite)
          - accuracy    (pure numeric correctness for GSM8K)
        """
        self.model.eval()
        eval_data = data or self.eval_data
        rewards, correct = [], []

        for sample in tqdm(eval_data, desc="Evaluating"):
            question = sample["prompt"]
            answer   = sample["answer"]
            prompt   = build_prompt(question, self.config.enable_thinking)

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=self.config.max_prompt_len,
                truncation=True,
            ).to(self.model.device)

            output = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            prompt_len = inputs["input_ids"].shape[1]
            response   = self.tokenizer.decode(output[0, prompt_len:], skip_special_tokens=True)

            rewards.append(self.reward_fn(response, answer))
            correct.append(reward_gsm8k(response, answer))

        self.model.train()

        avg_reward = sum(rewards) / len(rewards)
        accuracy   = sum(correct) / len(correct)

        print(f"\n📈 Eval Results ({len(eval_data)} samples):")
        print(f"   Avg Reward : {avg_reward:.4f}")
        print(f"   Accuracy   : {accuracy:.2%}  ← numeric exact match")

        return {"avg_reward": avg_reward, "accuracy": accuracy, "rewards": rewards}


# ──────────────────────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRPO RL Training with GSM8K")
    
    parser.add_argument("--model",             default="Qwen/Qwen3-0.6B",  help="HuggingFace model ID")
    parser.add_argument("--dataset_source",    default="gsm8k",
                        choices=["gsm8k", "json", "builtin"],             help="Dataset to use")
    parser.add_argument("--dataset_path",      default=None,              help="Path to JSON dataset (if source=json)")
    parser.add_argument("--max_train_samples", type=int, default=None,    help="Cap training set (None = all 7473)")
    parser.add_argument("--max_eval_samples",  type=int, default=200,     help="Cap eval set size")
    parser.add_argument("--epochs",            type=int,   default=3,     help="Number of epochs")
    parser.add_argument("--lr",                type=float, default=1e-6,  help="Learning rate")
    parser.add_argument("--group_size",        type=int,   default=4,     help="G: samples per prompt")
    parser.add_argument("--batch_size",        type=int,   default=2,     help="Prompts per update")
    parser.add_argument("--kl",                type=float, default=0.04,  help="KL penalty coefficient")
    parser.add_argument("--max_new_tokens",    type=int,   default=512,   help="Max tokens to generate")
    parser.add_argument("--no_4bit",           action="store_true",       help="Disable 4-bit quant")
    parser.add_argument("--no_think",          action="store_true",       help="Disable Qwen3 thinking mode")
    parser.add_argument("--output_dir",        default="./qwen_rl_checkpoints")
    parser.add_argument("--eval_only",         action="store_true",       help="Skip training, just eval")
    args = parser.parse_args()
    
    config = GRPOConfig(
        model_name         = args.model,
        num_epochs         = args.epochs,
        learning_rate      = args.lr,
        group_size         = args.group_size,
        batch_size         = args.batch_size,
        kl_coeff           = args.kl,
        max_new_tokens     = args.max_new_tokens,
        load_in_4bit       = not args.no_4bit,
        enable_thinking    = not args.no_think,
        output_dir         = args.output_dir,
        dataset_source     = args.dataset_source,
        dataset_path       = args.dataset_path,
        max_train_samples  = args.max_train_samples,
        max_eval_samples   = args.max_eval_samples,
    )
    
    trainer = GRPOTrainer(config=config)
    
    if args.eval_only:
        trainer.evaluate()
    else:
        trainer.train()
        trainer.evaluate()


"""
python qwen3_to_thinking_train.py
python qwen3_to_thinking_train.py --model Qwen/Qwen3-4B
python qwen3_to_thinking_train.py --model Qwen/Qwen3-0.6B

# if memory error (for small GPU's)
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
or
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


huggingface-cli upload xxxx/qwen3-0.6B-gsm8k-grpo ./qwen_rl_checkpoints/final

I COULD RUN IT ON vastai

"""
