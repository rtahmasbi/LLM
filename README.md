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
- [axolotl](https://github.com/axolotl-ai-cloud/axolotl)

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
- PagedAttention

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



https://www.crewai.com/

https://autogen-studio.com/



# Standard Set of Metrics for Evaluating LLMs
https://www.linkedin.com/pulse/evaluating-large-language-models-llms-standard-set-metrics-biswas-ecjlc/

- Perplexity
- Accuracy
- F1-score
- ROUGE score
- BLEU score
- METEOR score
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


# gradio
https://www.gradio.app/

Gradio is the fastest way to demo your machine learning model with a friendly web interface so that anyone can use it, anywhere!

```py
import gradio as gr

def greet(name):
    return "Hello " + name + "!"

demo = gr.Interface(fn=greet, inputs="text", outputs="text")
demo.launch()   
```



