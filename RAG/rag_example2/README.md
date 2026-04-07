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

### 1. macOS (Apple Silicon M1/M2/M3 or Intel)

```bash
conda env create -f environment-mac.yml
conda activate rag-pipeline
```

> `faiss-cpu` is installed via conda-forge (not pip) to share the same OpenMP runtime as PyTorch. Installing via pip causes a segfault on macOS.

**pip alternative (macOS)**
```bash
conda install -c conda-forge faiss-cpu   # faiss must come from conda
pip install -r requirements-mac.txt
```

### 2. Linux (CPU)

```bash
conda env create -f environment-linux-cpu.yml
conda activate rag-pipeline
```

**pip alternative (Linux CPU)**
```bash
pip install -r requirements-linux-cpu.txt
```

### 3. Linux (GPU — CUDA)

```bash
conda env create -f environment-linux-gpu.yml
conda activate rag-pipeline
```

**pip alternative (Linux GPU)**
```bash
# install torch with CUDA wheel first
pip install torch>=2.4 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-linux-gpu.txt
```

> Adjust `pytorch-cuda=12.1` / `faiss-gpu-cu12` to match your CUDA version:
> - CUDA 11.x → `pytorch-cuda=11.8` and `faiss-gpu-cu11`
> - CUDA 12.x → `pytorch-cuda=12.1` and `faiss-gpu-cu12`

Check your CUDA version with:
```bash
nvidia-smi
```

### Known macOS issues

| Error | Fix |
|---|---|
| `Segmentation fault: 11` | `faiss-cpu` installed via pip — reinstall via conda-forge |
| `symbol not found: _mkl_blas_caxpy` | conda-forge installed MKL variant — run: `conda install -c conda-forge "faiss-cpu=*=*openblas*"` |
| `numpy.dtype size changed` / system sklearn loaded | sklearn missing from conda env — run: `conda install -c conda-forge scikit-learn` |
| `OMP: Error #15: libomp already initialized` | Set `KMP_DUPLICATE_LIB_OK=TRUE` or fix faiss install |
| `bitsandbytes` warning / `NoneType` error | `pip uninstall bitsandbytes -y` |
| `ImportError: PeftMixedModel` | Downgrade: `pip install sentence-transformers==3.3.1` |

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
├── rag_pipeline.py               # RAGPipeline class
├── example_usage.py              # Usage examples
├── environment-mac.yml           # Conda env — macOS
├── environment-linux-cpu.yml     # Conda env — Linux CPU
├── environment-linux-gpu.yml     # Conda env — Linux GPU (CUDA)
├── requirements-mac.txt          # Pip deps — macOS
├── requirements-linux-cpu.txt    # Pip deps — Linux CPU
├── requirements-linux-gpu.txt    # Pip deps — Linux GPU (CUDA)
├── Investment-Case-For-Disruptive-Innovation.pdf
└── README.md
```
