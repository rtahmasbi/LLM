"""
Quick usage examples for RAGPipeline.
"""
from rag_pipeline import RAGPipeline, LLMProvider, ChunkStrategy

PDF_PATH = "Investment-Case-For-Disruptive-Innovation.pdf"
QUERY    = "What are the main findings of this paper?"
GROUND_TRUTH = "The paper concludes that ..."   # optional, for evaluation

# ── Example 1: OpenAI + recursive chunking (default) ─────────────────────────
pipeline = RAGPipeline(
    provider=LLMProvider.OPENAI,
    chunk_strategy=ChunkStrategy.RECURSIVE,
    chunk_size=512,
    chunk_overlap=64,
    top_k=5,
    rerank=True,
)
result = pipeline.run(PDF_PATH, QUERY, ground_truth=GROUND_TRUTH)
print(result["answer"])

# ── Example 2: Claude + semantic chunking ─────────────────────────────────────
pipeline2 = RAGPipeline(
    provider=LLMProvider.CLAUDE,
    chunk_strategy=ChunkStrategy.SEMANTIC,
    top_k=5,
    rerank=True,
)
result2 = pipeline2.run(PDF_PATH, QUERY)
print(result2["answer"])

# ── Example 3: step-by-step (more control) ───────────────────────────────────
pipeline3 = RAGPipeline(
    provider=LLMProvider.OPENAI,
    chunk_strategy=ChunkStrategy.SENTENCE,
    chunk_size=400,
    chunk_overlap=50,
    top_k=4,
    rerank=False,
)
pipeline3.load_pdf(PDF_PATH)
chunks = pipeline3.chunk_document()
print(f"Total chunks: {len(chunks)}")
pipeline3.build_index()

retrieved = pipeline3.retrieve(QUERY)
for r in retrieved:
    print(f"Page {r.chunk.page} | score={r.score:.4f} | {r.chunk.text[:120]}")

answer_dict = pipeline3.answer(QUERY)
print(answer_dict["answer"])
