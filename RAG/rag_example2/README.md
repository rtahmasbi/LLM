# PDF RAG Pipeline

A modular Retrieval-Augmented Generation (RAG) pipeline for PDF documents.  
Supports **OpenAI** and **Claude (Anthropic)** as interchangeable backends with multiple chunking strategies, FAISS vector search, cross-encoder reranking, and built-in evaluation metrics.

## Features

- PDF text extraction via PyMuPDF
- 4 chunking strategies: Fixed, Sentence, Recursive, Semantic
- OpenAI or Claude LLM + embeddings (drop-in swap)
- FAISS vector store with cosine similarity
- Cross-encoder reranking
- 4 evaluation metrics: faithfulness, answer relevancy, context recall, context precision

## Setup

### 1. Clone and create conda environment

```bash
conda env create -f environment.yml
conda activate rag-pipeline
```

> **GPU users:** replace `cpuonly` with `pytorch-cuda=12.1` (or match your CUDA version) in `environment.yml` before creating the env.

### Alternative: pip install

**macOS (Apple Silicon M1/M2/M3 — MPS backend)**
```bash
pip install torch>=2.4          # MPS support is built-in, no extra index needed
pip install -r requirements.txt
```

**macOS (Intel — CPU only)**
```bash
pip install -r requirements.txt  # torch>=2.4 included, CPU only
```

**Linux/Windows CPU**
```bash
pip install -r requirements.txt
```

**Linux/Windows GPU (CUDA)**
```bash
pip install torch>=2.4 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

> **Note:** `torch>=2.4` is required by `sentence-transformers==5.3.0`. NumPy is pinned to `1.26.4` for compatibility with compiled extensions (faiss, sentence-transformers).
>
> On Apple Silicon, PyTorch uses the **MPS (Metal Performance Shaders)** backend instead of CUDA. The cross-encoder reranker will automatically use MPS if available, falling back to CPU.

### 2. Set environment variables

```bash
# OpenAI provider
export OPENAI_API_KEY=your_openai_key

# Claude provider
export ANTHROPIC_API_KEY=your_anthropic_key
export VOYAGE_API_KEY=your_voyage_key
```

## Usage

### One-liner (full pipeline)

```python
from rag_pipeline import RAGPipeline, LLMProvider, ChunkStrategy

pipeline = RAGPipeline(
    provider=LLMProvider.OPENAI,
    chunk_strategy=ChunkStrategy.RECURSIVE,
    chunk_size=512,
    chunk_overlap=64,
    top_k=5,
    rerank=True,
)

result = pipeline.run("Investment-Case-For-Disruptive-Innovation.pdf", "What are the main findings?")
print(result["answer"])
```

### With evaluation

```python
result = pipeline.run(
    "Investment-Case-For-Disruptive-Innovation.pdf",
    "What are the main findings?",
    ground_truth="The paper concludes that ...",
)
# result["eval"] contains faithfulness, answer_relevancy, context_recall, context_precision
```

### Step-by-step

```python
from rag_pipeline import RAGPipeline, LLMProvider, ChunkStrategy

pipeline = RAGPipeline(
    provider=LLMProvider.CLAUDE,
    chunk_strategy=ChunkStrategy.RECURSIVE,
    chunk_size=512,
    chunk_overlap=64,
    top_k=5,
    rerank=True,
)

pipeline.load_pdf("Investment-Case-For-Disruptive-Innovation.pdf")
pipeline.chunk_document()
pipeline.build_index()

retrieved = pipeline.retrieve("How is the field of Robotics anticipated to evolve with the advancements in AI?")
result = pipeline.answer("How is the field of Robotics anticipated to evolve with the advancements in AI?")
```

## Parameters

| Parameter | Default | Options |
|---|---|---|
| `provider` | `OPENAI` | `OPENAI`, `CLAUDE` |
| `chunk_strategy` | `RECURSIVE` | `FIXED`, `SENTENCE`, `RECURSIVE`, `SEMANTIC` |
| `chunk_size` | `512` | any int (chars) |
| `chunk_overlap` | `64` | any int (chars) |
| `top_k` | `5` | any int |
| `rerank` | `True` | `True`, `False` |
| `llm_model` | provider default | any valid model name |
| `embed_model` | provider default | any valid model name |

### Provider defaults

| Provider | LLM | Embeddings |
|---|---|---|
| OpenAI | `gpt-4o-mini` | `text-embedding-3-small` |
| Claude | `claude-sonnet-4-6` | `voyage-3` |

## Reranking

Reranking is enabled by default (`rerank=True`) and re-scores retrieved chunks after FAISS search using a cross-encoder. Disable with `rerank=False` to remove the `sentence-transformers` dependency.

### Current default
`cross-encoder/ms-marco-MiniLM-L-6-v2` via `sentence-transformers`

### Alternatives

| Option | Package | Notes |
|---|---|---|
| **Voyage AI Rerank** | `voyageai` (already a dep) | Best fit for Claude provider — no extra dependency |
| **Cohere Rerank** | `cohere` | Hosted API, highest quality, costs money |
| **FlashRank** | `flashrank` | Lightweight local cross-encoder, faster than `sentence-transformers` |
| **BGE Reranker** | `FlagEmbedding` | Strong open-source model (`BAAI/bge-reranker-v2-m3`) |
| **Embedding similarity** | — | Re-score using cosine similarity, no new dependency |

**Recommendation by provider:**
- OpenAI provider → embedding similarity (no extra dep) or FlashRank
- Claude provider → Voyage AI rerank (already in stack)
- Best quality → Cohere Rerank API
- Local/offline → FlashRank or BGE Reranker

## Evaluation Metrics

| Metric | Description |
|---|---|
| `faithfulness` | Fraction of answer sentences supported by retrieved context |
| `answer_relevancy` | F1 token overlap between answer and question + ground truth |
| `context_recall` | Ground-truth key tokens found in retrieved context |
| `context_precision` | Fraction of retrieved chunks relevant to the question |

## Project Structure

```
rag_example2/
├── rag_pipeline.py     # RAGPipeline class
├── example_usage.py    # Usage examples
├── environment.yml     # Conda environment
├── requirements.txt    # Pip dependencies
├── Investment-Case-For-Disruptive-Innovation.pdf # pdf example file
└── README.md

```
