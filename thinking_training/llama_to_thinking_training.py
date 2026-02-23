
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType

# -------------------------------
# 1. Load LLaMA 3.1 Instruct
# -------------------------------
model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    dtype=torch.float16,
    low_cpu_mem_usage=True
)

# -------------------------------
# 2. LoRA Configuration
# -------------------------------
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)

# -------------------------------
# 3. Load GSM8K dataset
# -------------------------------
dataset = load_dataset("gsm8k", "main")

# Format dataset: combine question + step-by-step answer
def format_gsm8k(example):
    prompt = f"Instruction: Solve the following math problem step by step.\nInput: {example['question']}\nOutput: {example['answer']}"
    return {"text": prompt}

dataset = dataset.map(format_gsm8k)

# Tokenize function
def tokenize(batch):
    tokens = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized = dataset.map(tokenize, batched=True, remove_columns=["question","answer","text"])

# -------------------------------
# 4. Training arguments
# -------------------------------
training_args = TrainingArguments(
    output_dir="./lora-llama3-gsm8k",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    fp16=True,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_steps=50,
    evaluation_strategy="epoch",
    save_total_limit=2,
)

# -------------------------------
# 5. Trainer
# -------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
)

# -------------------------------
# 6. Train
# -------------------------------
trainer.train()

# -------------------------------
# 7. Save LoRA weights
# -------------------------------
model.save_pretrained("./lora-llama3-gsm8k")
tokenizer.save_pretrained("./lora-llama3-gsm8k")



"""

On 1 x A100 40GB or 80GB:
- Tokenization + preprocessing: negligible (<5 min)
- Per epoch training: ~20-30 minutes
- 3 epochs total: ~1-1.5 hours

"""