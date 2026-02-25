# function and imports comes from the `qwen3_to_thinking_train.py`


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


reward_fn = combined_reward_gsm8k
train_data, eval_data = load_dataset_for_config(config)

# Load policy model
model, tokenizer = load_model_and_tokenizer(config)

# Frozen reference model
print("Creating frozen reference model copy...")
ref_model = deepcopy(model).eval()
for p in ref_model.parameters():
    p.requires_grad_(False)

# Optimizer + cosine LR schedule
trainable      = [p for p in model.parameters() if p.requires_grad]
optimizer = AdamW(trainable, lr=config.learning_rate, weight_decay=0.01)
total_steps    = config.num_epochs * math.ceil(len(train_data) / config.batch_size)
scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-8)

global_step = 0
history: list[dict] = []


print(
    f"\n Trainer ready.\n"
    f"   Dataset source : {config.dataset_source}\n"
    f"   Train samples  : {len(train_data)}\n"
    f"   Eval  samples  : {len(eval_data)}\n"
    f"   Total steps    : {total_steps}\n"
)



def train_step(batch: list[dict]) -> dict:
    """Run one GRPO update on a mini-batch of prompts."""
    batch_loss    = torch.tensor(0.0, device=config.device)
    batch_metrics = {"policy_loss": 0.0, "kl": 0.0, "mean_reward": 0.0}
    n             = 0
    
    for sample in batch:
        question = sample["prompt"]
        answer   = sample["answer"]
        prompt   = build_prompt(question, config.enable_thinking)
        
        # 1. Sample G responses
        responses = generate_group(model, tokenizer, prompt, config)
        
        # 2. Score responses
        rewards = torch.tensor(
            [reward_fn(r, answer) for r in responses],
            dtype=torch.float32,
            device=config.device,
        )
        
        # 3. Compute log-probs under policy and ref
        policy_lps = torch.stack([
            compute_log_probs(model, tokenizer, prompt, r, config)
            for r in responses
        ])
        ref_lps = torch.stack([
            compute_log_probs(ref_model, tokenizer, prompt, r, config)
            for r in responses
        ])
        
        # 4. GRPO loss
        loss, metrics = grpo_loss(policy_lps, ref_lps, rewards, config)
        batch_loss    = batch_loss + loss
        
        for k in batch_metrics:
            batch_metrics[k] += metrics[k]
        n += 1
    
    batch_loss = batch_loss / max(n, 1)
    
    # 5. Backprop
    optimizer.zero_grad()
    batch_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], max_norm=1.0
    )
    optimizer.step()
    scheduler.step()
    
    return {
        "loss": batch_loss.item(),
        "lr":   scheduler.get_last_lr()[0],
        **{k: v / max(n, 1) for k, v in batch_metrics.items()},
    }


##

for epoch in range(1, config.num_epochs + 1):
    print(f"\n{'─'*60}")
    print(f"  Epoch {epoch}/{config.num_epochs}")
    print(f"{'─'*60}")
    
    random.shuffle(train_data)
    batches = [
        train_data[i:i + config.batch_size]
        for i in range(0, len(train_data), config.batch_size)
    ]

len(batches)

batch = batches[0]
t0      = time.time()
metrics = train_step(batch)
elapsed = time.time() - t0

sample = batch[0]
question = sample["prompt"]
answer   = sample["answer"]
prompt   = build_prompt(question, config.enable_thinking)

# 1. Sample G responses
responses = generate_group(model, tokenizer, prompt, config)


prompt2 = """
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Marly has ten $20 bills, eight $10 bills, and four $5 bills. If she wants to change her bills to $100 bills, how many pieces of $100 bills will she have?<|im_end|>
<|im_start|>assistant
<think>

</think>

"""
responses2 = generate_group(model, tokenizer, prompt2, config)
print(responses2[0])





# https://huggingface.co/Qwen/Qwen3-0.6B

# In this mode, the model will not generate any think content and will not include a <think>...</think> block.
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
)

print(text)
# '<|im_start|>user\nGive me a short introduction to large language model.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'




# Advanced Usage: Switching Between Thinking and Non-Thinking Modes via User Input:

user_input_2 = "Then, how many r's in blueberries? /no_think"
user_input_3 = "Really? /think"
"""
For API compatibility, when enable_thinking=True, regardless of whether the user uses /think or /no_think,
the model will always output a block wrapped in <think>...</think>.
However, the content inside this block may be empty if thinking is disabled.
When enable_thinking=False, the soft switches are not valid. Regardless of any /think or /no_think tags
input by the user, the model will not generate think content and will not include a <think>...</think> block.
"""


# https://qwen.readthedocs.io/en/latest/getting_started/quickstart.html
# Qwen3-Instruct-2507 supports only non-thinking mode and does not generate <think></think> blocks in its output. Different from Qwen3-2504, specifying enable_thinking=False is no longer required or supported.
