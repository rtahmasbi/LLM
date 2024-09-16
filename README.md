# LLM
I have covered these topics. You can go to each topic and see examples there.

## [langchain](langchain/)
- LangChain - [langchain](langchain/)
- LangSmith - production-grade LLM applications
- LangGraph - [LangGraph](graph/)
- LangServe - server API


## [Fine Tune](FineTune/)
- LoRA
- QLoRA
- [axolotl](FineTune)

## [RAG](RAG/)
- langchain_chroma
- vectorstore

## [OpenAI](OpenAI/)

## [Llama](Llama/)

## [Other LLM's](Other_LLMs/)
- TinyLlama
- Mistral
- facebook/opt-350m


## [Reinforcement Learning](RLHF/)
- [Reinforcement Learning from Human Feedback (RLHF)](RLHF/)
- Reinforcement learning from AI feedback (RLAIF)


## [vLLM](vLLM/)
- Optimizing KV pairs
- `PagedAttention` algorithm allows storing continuous KV pairs in non-contiguous memory space

## [Graph](graph/)
- query from graph database, such as neo4j with cypher
- building the knowledge graphs with `LLMGraphTransformer`


## Standard Set of Metrics for Evaluating LLMs

## LLM-as-a-Judge




# datasets
```py
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
```


## load local datasset
```py
from datasets import load_dataset
ds = load_dataset('csv', data_files='path/to/local/my_dataset.csv')

from datasets import load_dataset
ds = load_dataset('json', data_files='path/to/local/my_dataset.json')

from datasets import load_dataset
ds = load_dataset('path/to/local/loading_script/loading_script.py', split='train')


from datasets import load_from_disk
ds = load_from_disk('path/to/dataset/directory')
```




# perplexity
Perplexity evaluates a language model's ability to predict the next word or character based on the context of previous words or characters. A lower perplexity score indicates better predictive performance.

https://klu.ai/glossary/perplexity




# some useful links
https://github.com/Barnacle-ai/awesome-llm-list

https://github.com/horseee/Awesome-Efficient-LLM

https://github.com/rmovva/LLM-publication-patterns-public

https://github.com/HenryHZY/Awesome-Multimodal-LLM

https://github.com/DefTruth/Awesome-LLM-Inference


## TensorFlow-Examples
https://github.com/aymericdamien/TensorFlow-Examples/tree/master



# Standard Set of Metrics for Evaluating LLMs
https://www.linkedin.com/pulse/evaluating-large-language-models-llms-standard-set-metrics-biswas-ecjlc/

- Perplexity - perplexity = 2^(-log P(w1,w2,...,wn)/n), where P(w1,w2,...,wn) is the probability of the test set and n is the number of words in the test set.
- Accuracy
- F1-score
- ROUGE score - based on the concept of n-grams - [link](https://github.com/google-research/google-research/tree/master/rouge)
- BLEU score - based on the n-gram overlap
- METEOR score - It combines both precision and recall
- Question Answering Metrics
- Sentiment Analysis Metrics
- Named Entity Recognition Metrics
- Contextualized Word Embeddings
- BERTScore


While BLEU and ROUGE assess text similarity by analyzing matching n-gram statistics between the generated text and the reference text, BERTScore assesses similarity in the embedding space by assigning a score that reflects how closely the generated text aligns with the reference text in that space.


# Judging LLM-as-a-Judge
https://arxiv.org/pdf/2306.05685


# cloud GPU
https://www.runpod.io/console/console/pods



# Useful packages

## accelerate
https://huggingface.co/docs/accelerate/index

https://github.com/huggingface/accelerate

```sh
torchrun \ # python -m torch.distributed.run 
    --nproc_per_node 2 \
    --nnodes 2 \
    --rdzv_id 2299 \ # A unique job id 
    --rdzv_backend c10d \
    --rdzv_endpoint master_node_ip_address:29500 \
    ./nlp_example.py

```


```py
+ from accelerate import Accelerator
+ accelerator = Accelerator()

+ model, optimizer, training_dataloader, scheduler = accelerator.prepare(
+     model, optimizer, training_dataloader, scheduler
+ )

  for batch in training_dataloader:
      optimizer.zero_grad()
      inputs, targets = batch
      inputs = inputs.to(device)
      targets = targets.to(device)
      outputs = model(inputs)
      loss = loss_function(outputs, targets)
+     accelerator.backward(loss)
      optimizer.step()
      scheduler.step()
```


## bitsandbytes
The bitsandbytes library is a lightweight Python wrapper around CUDA custom functions, in particular 8-bit optimizers, matrix multiplication (LLM.int8()), and 8 & 4-bit quantization functions.

The library includes quantization primitives for 8-bit & 4-bit operations, through bitsandbytes.nn.Linear8bitLt and bitsandbytes.nn.Linear4bit and 8-bit optimizers through bitsandbytes.optim module.

https://huggingface.co/docs/bitsandbytes/main/en/index

https://github.com/bitsandbytes-foundation/bitsandbytes


## DeepSpeed
https://github.com/microsoft/DeepSpeed

DeepSpeed is a deep learning optimization library that makes distributed training easy, efficient, and effective. 10x Larger Models. 10x Faster Training.

DeepSpeed uses Accelerate.

https://deepspeed.readthedocs.io/en/latest/zero3.html

- [Transformers with DeepSpeed](https://huggingface.co/docs/transformers/main/main_classes/deepspeed)
- [Accelerate with DeepSpeed](https://huggingface.co/docs/accelerate/usage_guides/deepspeed)

DeepSpeed:
1. Optimizer state partitioning (ZeRO stage 1)
2. Gradient partitioning (ZeRO stage 2)
3. Parameter partitioning (ZeRO stage 3)
4. Custom mixed precision training handling
5. A range of fast CUDA-extension-based optimizers
6. ZeRO-Offload to CPU and Disk/NVMe
7. Hierarchical partitioning of model parameters (ZeRO++)

```
accelerate launch my_script.py --args_to_my_script
```




# multi GPU
https://pytorch.org/docs/stable/distributed.html



## torch
```py
torch.distributed.launch
```

```py
+ import torch.multiprocessing as mp
+ from torch.utils.data.distributed import DistributedSampler
+ from torch.nn.parallel import DistributedDataParallel as DDP
+ from torch.distributed import init_process_group, destroy_process_group
+ import os
```


## tensorflow
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/6_Hardware/multigpu_training.ipynb
s

```py
tf.device('/gpu:%i' % i):
```



## with accelerate
https://huggingface.co/docs/trl/example_overview
```sh
accelerate launch --config_file=examples/accelerate_configs/multi_gpu.yaml --num_processes {NUM_GPUS} path_to_script.py --all_arguments_of_the_script
```

## with DeepSpeed
https://huggingface.co/docs/trl/example_overview
```sh
accelerate launch --config_file=examples/accelerate_configs/deepspeed_zero{1,2,3}.yaml --num_processes {NUM_GPUS} path_to_script.py --all_arguments_of_the_script
```




# companies

## crewai
https://www.crewai.com/

## autogen-studio
AI Agents

https://autogen-studio.com/


## groq
Groq is Fast AI Inference

https://console.groq.com/playground



## gradio
https://www.gradio.app/

Gradio is the fastest way to demo your machine learning model with a friendly web interface so that anyone can use it, anywhere!

```py
import gradio as gr

def greet(name):
    return "Hello " + name + "!"

demo = gr.Interface(fn=greet, inputs="text", outputs="text")
demo.launch()   
```



# train LLM from scartch time
https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat

GPU SKUs	OPT-1.3B	OPT-6.7B	OPT-13.2B	OPT-30B	OPT-66B	OPT-175B
1x V100 32G	1.8 days					
1x A6000 48G	1.1 days	5.6 days				
1x A100 40G	15.4 hrs	3.4 days				
1x A100 80G	11.7 hrs	1.7 days	4.9 days			
8x A100 40G	2 hrs	5.7 hrs	10.8 hrs	1.85 days		
8x A100 80G	1.4 hrs($45)	4.1 hrs ($132)	9 hrs ($290)	18 hrs ($580)	2.1 days ($1620)	
64x A100 80G	31 minutes	51 minutes	1.25 hrs ($320)	4 hrs ($1024)	7.5 hrs ($1920)	20 hrs ($5120)


# arcprize
https://arcprize.org/arc

ARC Prize is a $1,000,000+ public competition to beat and open source a solution to the ARC-AGI benchmark.


Measuring task-specific skill is not a good proxy for intelligence.

Skill is heavily influenced by prior knowledge and experience: unlimited priors or unlimited training data allows developers to "buy" levels of skill for a system. This masks a system's own generalization power.

Intelligence lies in broad or general-purpose abilities; it is marked by skill-acquisition and generalization, rather than skill itself.

AGI is a system that can efficiently acquire new skills outside of its training data.



