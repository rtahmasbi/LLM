# huggingface
all the models are here:

https://huggingface.co/docs/transformers/index

list of models are here:

https://huggingface.co/docs/transformers/tasks/language_modeling

https://huggingface.co/docs/transformers/tasks/token_classification

https://huggingface.co/docs/transformers/tasks/masked_language_modeling

https://huggingface.co/docs/transformers/tasks/summarization



--open llm leaderboard
https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard

https://huggingface.co/spaces/optimum/llm-perf-leaderboard



## location
l /home/rt/.cache/huggingface/hub/



## gguf
GPT-Generated Unified Format

https://huggingface.co/docs/hub/en/gguf

https://huggingface.co/docs/hub/en/gguf-llamacpp

https://github.com/ggerganov/ggml/blob/master/docs/gguf.md


## task_summary
https://huggingface.co/docs/transformers/task_summary



### Natural language processing
```py
classifier = pipeline(task="sentiment-analysis")
classifier = pipeline(task="ner")
question_answerer = pipeline(task="question-answering")
summarizer = pipeline(task="summarization") # default BART
translator = pipeline(task="translation", model="t5-small")  # default BART
#language modeling:
generator = pipeline(task="text-generation") # --> causal, default gpt2
fill_mask = pipeline(task="fill-mask")       # --> masked

# BART for summarization and translation that use an encoder-decoder

```


### Computer vision
- Image classification
- Object detection
- Image segmentation
- Depth estimation

### Audio
- Audio classification
- Automatic speech recognition



### Multimodal
Multimodal tasks require a model to process multiple data modalities (text, image, audio, video) to solve a particular problem.

Document question answering




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

