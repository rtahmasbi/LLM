# LLM

# install
pip install peft



# How to fine-tune LLaMA 2 using SFT, LORA

https://blog.accubits.com/how-to-fine-tune-llama-2-using-sft-lora/

```py

from typing import List
​
import fire
import torch
import transformers
from datasets import load_dataset
​
​
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
    PrefixTuningConfig,
    TaskType
)
from transformers import LlamaForCausalLM, LlamaTokenizer
​
​
​
def train(
    # model/data params
    base_model: str = "", 
    data_path: str = "",
    output_dir: str = "",
​
    micro_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    val_set_size: int = 2000,
    
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ]
):
​
    device_map = "auto"
​
​
    # Step 1: Load the model and tokenizer
​
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        # load_in_8bit=True, # Add this for using int8
        torch_dtype=torch.float16,
        device_map=device_map,
    )
​
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = 0
​
      Add this for training LoRA
​
      config = LoraConfig(
          r=lora_r,
          lora_alpha=lora_alpha,
          target_modules=lora_target_modules,
          lora_dropout=lora_dropout,
          bias="none",
          task_type="CAUSAL_LM",
      )
      model = get_peft_model(model, config)
​
      model = prepare_model_for_int8_training(model) # Add this for using int8
​
​
    # Step 2: Load the data
​
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)
    
    # Step 3: Tokenize the data
​
    def tokenize(data):
        source_ids = tokenizer.encode(data['input'])
        target_ids = tokenizer.encode(data['output'])
​
        input_ids = source_ids + target_ids + [tokenizer.eos_token_id]
        labels = [-100] * len(source_ids) + target_ids + [tokenizer.eos_token_id]
​
        return {
            "input_ids": input_ids,
            "labels": labels
        }
​
    #split thte data to train/val set
    train_val = data["train"].train_test_split(
        test_size=val_set_size, shuffle=False, seed=42
    )
    train_data = (
        train_val["train"].shuffle().map(tokenize)
    )
    val_data = (
        train_val["test"].shuffle().map(tokenize)
        
    )
​
    # Step 4: Initiate the trainer
​
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=200,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=3
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
​
    trainer.train()
​
​
    # Step 5: save the model
    model.save_pretrained(output_dir)
​
​
​
if __name__ == "__main__":
    fire.Fire(train)

```



# LoRA
https://www.databricks.com/blog/efficient-fine-tuning-lora-guide-llms
databricks provides evaluation before qnd after LoRA

# LoRA
https://medium.com/@nischal.345/customizing-large-language-models-fine-tuning-and-retrieval-augmented-generation-ab619b846535

good to check

# LoRA on Colab
https://towardsdatascience.com/fine-tune-your-own-llama-2-model-in-a-colab-notebook-df9823a04a32


################################################################################
################################################################################
################################################################################
# more
https://github.com/Lightning-AI/litgpt


https://www.mercity.ai/blog-post/guide-to-fine-tuning-llms-with-lora-and-qlora


https://lightning.ai/pages/community/lora-insights/



https://huggingface.co/docs/peft/en/index





https://huggingface.co/docs/peft/en/index


https://huggingface.co/docs/peft/en/quicktour

################################################################################
################################################################################
################################################################################
# LoraConfig
## TaskType
TaskType.CAUSAL_LM
TaskType.QUESTION_ANS
TaskType.SEQ_CLS
TaskType.mro()               
TaskType.FEATURE_EXTRACTION
TaskType.SEQ_2_SEQ_LM
TaskType.TOKEN_CLS           


- Causal language models are frequently used for text generation.
- Masked language modeling predicts a masked token in a sequence, and the model can attend to tokens bidirectionally. This means the model has full access to the tokens on the left and right. Masked language modeling is great for tasks that require a good contextual understanding of an entire sequence. BERT is an example of a masked language model.




## target_modules
model = AutoModelForCausalLM.from_pretrained("some-model-checkpoint")
print(model)
>
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(51200, 4096, padding_idx=0)
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
          (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
          (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
          (act_fn): SiLUActivation()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
    )
    (norm): LlamaRMSNorm()
  )
  (lm_head): Linear(in_features=4096, out_features=51200, bias=False)
)


-- If only targeting attention blocks of the model
target_modules = ["q_proj", "v_proj"]

-- If targeting all linear layers
target_modules = ['q_proj','k_proj','v_proj','o_proj','gate_proj','down_proj', 'up_proj','lm_head']




################################################################################
################################################################################
################################################################################
# trainer
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
trainer = Seq2SeqTrainer(...)
trainer.train()
trainer.model.save_pretrained(peft_model_id)
tokenizer.save_pretrained(peft_model_id)

## trainer.xx
from datasets import load_from_disk
ds = load_from_disk("path")
train_dataset=ds,


################################################################################
################################################################################
################################################################################
# facebook/opt-350m
https://huggingface.co/facebook/opt-350m

https://github.com/huggingface/


pip install tf-keras

```py
from transformers import pipeline
generator = pipeline('text-generation', model="facebook/opt-350m", do_sample=True, num_return_sequences=5)
# , do_sample=True makes teh answer random
generator("What are we having for dinner?")




```



################################################################################
################################################################################
################################################################################
# datasets
from datasets import list_datasets
list_datasets()
>
['acronym_identification',
 'ade_corpus_v2',
 'adversarial_qa',
 'aeslc',
 'afrikaans_ner_corpus',
 'ag_news',
 ...
]


## load local datasset
from datasets import load_dataset
ds = load_dataset('csv', data_files='path/to/local/my_dataset.csv')

from datasets import load_dataset
ds = load_dataset('json', data_files='path/to/local/my_dataset.json')

from datasets import load_dataset
ds = load_dataset('path/to/local/loading_script/loading_script.py', split='train')


from datasets import load_from_disk
ds = load_from_disk('path/to/dataset/directory')




# LoRA SEQ_2_SEQ_LM fine tune
https://heidloff.net/article/fine-tuning-llm-lora-small-gpu/

