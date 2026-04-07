# RAG Pipeline — Project Overview

## Goal
A Python RAG pipeline for PDF documents, supporting OpenAI and Claude (Anthropic) as interchangeable backends.

## Files
- `rag_pipeline.py` — main `RAGPipeline` class
- `example_usage.py` — usage examples
- `requirements.txt` — dependencies

## Design

### LLM & Embeddings
- Provider selected via `LLMProvider.OPENAI` or `LLMProvider.CLAUDE`
- OpenAI: `gpt-4o-mini` + `text-embedding-3-small`
- Claude: `claude-sonnet-4-6` + `voyage-3` (Voyage AI embeddings)
- Override with `llm_model=` / `embed_model=` constructor params

### Chunking strategies (`ChunkStrategy`)
- `FIXED` — fixed character window with overlap
- `SENTENCE` — NLTK sentence tokenizer, fills up to `chunk_size`
- `RECURSIVE` — recursive split on `\n\n` → `\n` → `.` → ` ` (LangChain-style)
- `SEMANTIC` — embeds sentences, groups by cosine similarity threshold (0.85)

### Vector Store
- FAISS `IndexFlatIP` with L2-normalised vectors (cosine similarity)

### Reranking
- `cross-encoder/ms-marco-MiniLM-L-6-v2` via `sentence-transformers`
- Enabled with `rerank=True` (default)

### Evaluation metrics
- `faithfulness` — fraction of answer sentences supported by retrieved context
- `answer_relevancy` — F1 token overlap: answer vs. question + ground truth
- `context_recall` — ground-truth key tokens found in context
- `context_precision` — fraction of retrieved chunks relevant to the question

## Environment Variables
- `OPENAI_API_KEY` — required for OpenAI provider
- `ANTHROPIC_API_KEY` — required for Claude provider
- `VOYAGE_API_KEY` — required for Claude provider (Voyage embeddings)

## Key Parameters
| Param | Default | Description |
|---|---|---|
| `provider` | `OPENAI` | LLM + embedding backend |
| `chunk_strategy` | `RECURSIVE` | Chunking method |
| `chunk_size` | `512` | Target chars per chunk |
| `chunk_overlap` | `64` | Overlap chars between chunks |
| `top_k` | `5` | Chunks retrieved per query |
| `rerank` | `True` | Enable cross-encoder reranking |
