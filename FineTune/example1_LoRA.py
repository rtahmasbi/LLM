import pandas as pd
from datasets import load_dataset
from datasets import Dataset

#Load the dataset from the HuggingFace Hub
rd_ds = load_dataset("xiyuez/red-dot-design-award-product-description")

#Convert to pandas dataframe for convenient processing
rd_df = pd.DataFrame(rd_ds['train'])

#Combine the two attributes into an instruction string
rd_df['instruction'] = 'Create a detailed description for the following product: '+ rd_df['product']+', belonging to category: '+ rd_df['category']

rd_df = rd_df[['instruction', 'description']]

#Get a 5000 sample subset for fine-tuning purposes
rd_df_sample = rd_df.sample(n=5000, random_state=42)

#Define template and format data into the template for supervised fine-tuning
template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:


### Response:\n"""

rd_df_sample['prompt'] = rd_df_sample["instruction"].apply(lambda x: template.format(x))
rd_df_sample.rename(columns={'description': 'response'}, inplace=True)
rd_df_sample['response'] = rd_df_sample['response'] + "\n### End"
rd_df_sample = rd_df_sample[['prompt', 'response']]

rd_df['text'] = rd_df["prompt"] + rd_df["response"]
rd_df.drop(columns=['prompt', 'response'], inplace=True)



# Testing model performance before fine-tuning

import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

model_path = 'openlm-research/open_llama_3b_v2'
tokenizer = LlamaTokenizer.from_pretrained(model_path)
# pip install accelerate for load_in_8bit
model = LlamaForCausalLM.from_pretrained(model_path, load_in_8bit=True, device_map='auto')

#Pass in a prompt and infer with the model
prompt = 'Q: Create a detailed description for the following product: Corelogic Smooth Mouse, belonging to category: Optical Mouse\nA:'
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

generation_output = model.generate(input_ids=input_ids, max_new_tokens=128)

print(tokenizer.decode(generation_output[0]))



prompt= """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Create a detailed description for the following product: Corelogic Smooth Mouse, belonging to category: Optical Mouse

### Response:"""
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

generation_output = model.generate(input_ids=input_ids, max_new_tokens=128)

print(tokenizer.decode(generation_output[0]))



# peft
from peft import LoraConfig

#If only targeting attention blocks of the model
target_modules = ["q_proj", "v_proj"]

#If targeting all linear layers
target_modules = ['q_proj','k_proj','v_proj','o_proj','gate_proj','down_proj','up_proj','lm_head']

lora_config = LoraConfig(
r=16,
target_modules = target_modules,
lora_alpha=8,
lora_dropout=0.05,
bias="none",
task_type="CAUSAL_LM",
)


import re
model_modules = str(model.modules)
pattern = r'\((\w+)\): Linear'
linear_layer_names = re.findall(pattern, model_modules)

names = []
# Print the names of the Linear layers
for name in linear_layer_names:
    names.append(name)
target_modules = list(set(names))


# Tuning the finetuning with LoRA
trainer = SFTTrainer(
model,
train_dataset=dataset['train'],
eval_dataset = dataset['test'],
dataset_text_field="text",
max_seq_length=256,
args=training_args,
)

# Initiate the training process
with mlflow.start_run(run_name= ‘run_name_of_choice’):
trainer.train()



model = get_peft_model(model, lora_config)
model.print_trainable_parameters()



peft_model = PeftModel.from_pretrained(model, adapter_location)

merged_model = peft_model.merge_and_unload()



