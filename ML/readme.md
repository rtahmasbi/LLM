
# help_torch
[ML/help_torch.md](ML/help_torch.md)


# help_tensorflow
[ML/help_tensorflow.md](ML/help_tensorflow.md)


# NNP (Next Number Prediction)
I have a notebook for NNP, which is based on the minGPT here [ML/MinGPT.ipynb](ML/MinGPT.ipynb)


# GPT models from scratch

https://github.com/saqib1707/gpt2-from-scratch/blob/master/src/model.py
The use this dataset: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu


https://github.com/karpathy/nanoGPT
```
54GB in huggingface .cache dir, about 8M documents (8,013,769)
dataset = load_dataset("openwebtext", num_proc=num_proc_load_dataset)

or

https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
# train.bin has 301,966 tokens
# val.bin has 36,059 tokens
```

https://github.com/karpathy/minGPT


https://github.com/graykode/gpt-2-Pytorch/blob/master/GPT2/model.py

https://github.com/openai/gpt-2


# References
## general
[Understanding Deep Learning](https://udlbook.github.io/udlbook/)


## tree parsing
- [Generative Pretrained Structured Transformers: Unsupervised Syntactic Language Models at Scale](https://arxiv.org/pdf/2403.08293)
A syntactic language model (SLM) incrementally generates a sentence with its syntactic tree in a left-to-right manner. We present Generative Pretrained Structured Transformers (GPST), an unsupervised SLM at scale capable  of being pre-trained from scratch on raw texts with high parallelism.
[github](https://github.com/ant-research/StructuredLM_RTDT)


- [Unsupervised Morphological Tree Tokenizer](https://aclanthology.org/2025.findings-acl.1146.pdf)

- [Sneaking Syntax into Transformer Language Models with Tree Regularization](https://aclanthology.org/2025.naacl-long.407.pdf)


