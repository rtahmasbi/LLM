
# Tree Parsing


## Tree Regularization - 2025 - Stanford
- [Sneaking Syntax into Transformer Language Models with Tree Regularization](https://aclanthology.org/2025.naacl-long.407.pdf)
- [GitHub](https://github.com/ananjan-nandi-9/tree_regularization)

**Summary**

While SLMs often improve data efficiency and syntactic generalization, they have several drawbacks: They often involve additional parameters to model syntax, constrain the attention mechanisms of the underlying model, or involve more complex and slower inference methodologies.

In this work, we instead devise a new differentiable loss function that softly injects syntactic
inductive biases into a given circuit of the transformer: TREEREG. TREEREG is simply added as a regularizer to the LM loss during training. Crucially, an LM trained with TREEREG is completely indistinguishable from a standard LM in both architecture and inference mechanism.

SCIN: Span Contextual Independence Score

**Important**: They use **Berkeley Neural Parser** to make the parse trees. So there is no tree structure prediction on the fly. We can assume that Berkeley Neural Parser acs as a prior info, they can reconstruct the parsed tree in Algorithm 2.



## GPST - 2024
- [Generative Pretrained Structured Transformers: Unsupervised Syntactic Language Models at Scale](https://arxiv.org/pdf/2403.08293)
- [github](https://github.com/ant-research/StructuredLM_RTDT)

A syntactic language model (SLM) incrementally generates a sentence with its syntactic tree in a left-to-right manner. We present Generative Pretrained Structured Transformers (GPST), an unsupervised SLM at scale capable  of being pre-trained from scratch on raw texts with high parallelism.

- (Fruit (flies (like a banana))) or ((Fruit flies) (like (a banana)))



## Unsupervised Morphological Tree Tokenizer - 2024
- [Unsupervised Morphological Tree Tokenizer](https://aclanthology.org/2025.findings-acl.1146.pdf)



## DIORA - 2019
Unsupervised Latent Tree Induction with Deep Inside-Outside Recursive Autoencoders
- It uses CKY
- [Paper](https://aclanthology.org/N19-1116.pdf)
- [GitHub](https://github.com/iesl/diora)


# Constituency Parsing with a Self-Attentive Encoder - 2018
**Berkeley Neural Parser**
- [Paper](https://arxiv.org/pdf/1805.01052)
- [GitHub](https://github.com/nikitakit/self-attentive-parser)
- [Spacy](https://spacy.io/universe/project/self-attentive-parser)
- [pip benepar 0.2.0](https://pypi.org/project/benepar/)



## A Systematic Assessment of Syntactic Generalization in Neural Language Models - 2005
It is for Assessment: SG score and Perplexity

- [Paper](https://arxiv.org/pdf/2005.03692)
- [Github](https://github.com/cpllab/syntactic-generalization)

**Summary**
- Syntactic Generalization (SG) score
- Perplexity: Standard language models are trained to predict the next token given a context of previous tokens. Language models are typically assessed by their perplexity, the inverse geometric mean of the joint probability of words `w_1, . . . , w_N` in a held-out test corpus `C`.
- In probabilistic language models, these garden-path disambiguation effects are well captured by word negative log probabilities, or SURPRISALS (Hale, 2001): `S(w|C) = − log_2 p(w|C)`, which are independently well-established to predict human incremental processing difficulty over several orders of magnitude in word probability.
- training data: Brown Laboratory for Linguistic Information Processing 1987-89 Corpus Release 1 (BLLIP; Charniak et al., 2000). The corpora are sampled such that the training set of each corpus is a proper subset of each larger corpus. We call these four corpora:
    - BLLIP-XS (40K sentences, 1M tokens)
    - BLLIP-SM (200K sentences, 5M tokens)
    - BLLIPMD (600K sentences, 14M tokens)
    - BLLIP-LG (2M sentences, 42M tokens)



## CYK algorithm - 1961
- context-free grammars
- O(n^3 |G|)

