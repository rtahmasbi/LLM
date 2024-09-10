# LLM
I have covered these topics. You can go to each topic and see examples there.

## [langchain](langchain/)
- LangChain
- LangSmith - production-grade LLM applications
- LangGraph
- LangServe - server API


## [Fine Tune](FineTune/)
- LoRA
- QLoRA

## [RAG](RAG/)

## [OpenAI](OpenAI/)

## [Llama](Llama/)

## [Other LLM's](Other_LLMs/)
- TinyLlama
- Mistral

## [Reinforcement Learning from Human Feedback (RLHF)](RLHF/)


## [vLLM](vLLM/)

## [Graph](graph/)
- query from graph database, such as neo4j with cypher
- building the knowledge graphs with `LLMGraphTransformer`






# facebook/opt-350m
https://huggingface.co/facebook/opt-350m

https://github.com/huggingface/


`pip install tf-keras`


```py
from transformers import pipeline
generator = pipeline('text-generation', model="facebook/opt-350m", do_sample=True, num_return_sequences=5)
# , do_sample=True makes the answer random
generator("What are we having for dinner?")


```



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



# Reinforcement learning from AI feedback (RLAIF)


