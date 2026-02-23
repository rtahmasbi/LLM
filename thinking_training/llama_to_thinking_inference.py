import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# -------------------------------
# Load the base LLaMA 3.1 and LoRA weights
# -------------------------------
base_model_name = "meta-llama/Llama-3.1-8B-Instruct"
lora_model_path = "./lora-llama3-gsm8k"

tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    dtype=torch.float16,
    low_cpu_mem_usage=True
)

# Apply LoRA weights
model = PeftModel.from_pretrained(model, lora_model_path)
model.eval()

# -------------------------------
# Inference function with step-by-step reasoning
# -------------------------------
def solve_problem(problem: str, max_new_tokens=256):
    prompt = f"Instruction: Solve the following math problem step by step.\nInput: {problem}\nOutput:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            do_sample=False,  # deterministic reasoning
            pad_token_id=tokenizer.eos_token_id
        )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Strip prompt from output
    answer = answer[len(prompt):].strip()
    return answer

# -------------------------------
# Example usage
# -------------------------------
problems = [
    "234 + 567",
    "If I have 3 apples and buy 4 more, how many apples do I have in total?",
    "A train travels 60 miles in 1 hour. How far will it travel in 2.5 hours?"
]

for p in problems:
    print(f"Problem: {p}")
    print("Answer:", solve_problem(p))
    print("-"*50)


