"""
RAG Pipeline for PDF documents.
Supports OpenAI and Claude (Anthropic) for LLM and embeddings.
Vector store: FAISS
Chunking: fixed-size, sentence-based, semantic, recursive
Reranking: cross-encoder
Evaluation: faithfulness, answer relevancy, context recall, context precision
"""

import os
import re
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional

import fitz  # PyMuPDF
import numpy as np
import faiss

# ── provider / strategy enums ─────────────────────────────────────────────────

class LLMProvider(str, Enum):
    OPENAI = "openai"
    CLAUDE = "claude"

class ChunkStrategy(str, Enum):
    FIXED = "fixed"           # fixed token/char size
    SENTENCE = "sentence"     # split on sentence boundaries
    RECURSIVE = "recursive"   # recursive character splitter
    SEMANTIC = "semantic"     # embedding-based semantic grouping


# ── data containers ───────────────────────────────────────────────────────────

@dataclass
class Chunk:
    text: str
    page: int
    index: int
    embedding: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


@dataclass
class EvalResult:
    faithfulness: float
    answer_relevancy: float
    context_recall: float
    context_precision: float

    def __str__(self) -> str:
        return (
            f"Faithfulness:       {self.faithfulness:.3f}\n"
            f"Answer Relevancy:   {self.answer_relevancy:.3f}\n"
            f"Context Recall:     {self.context_recall:.3f}\n"
            f"Context Precision:  {self.context_precision:.3f}"
        )


# ── main pipeline ─────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    End-to-end RAG pipeline for PDF files.

    Parameters
    ----------
    provider        : LLMProvider.OPENAI | LLMProvider.CLAUDE
    chunk_strategy  : ChunkStrategy member
    chunk_size      : target character length per chunk (for FIXED / RECURSIVE)
    chunk_overlap   : overlap in characters between consecutive chunks
    top_k           : number of chunks retrieved per query
    rerank          : enable cross-encoder reranking
    embed_model     : override default embedding model name
    llm_model       : override default LLM model name
    """

    # ── defaults per provider ──────────────────────────────────────────────
    _DEFAULTS = {
        LLMProvider.OPENAI: {
            "embed_model": "text-embedding-3-small",
            "llm_model":   "gpt-4o-mini",
        },
        LLMProvider.CLAUDE: {
            "embed_model": "voyage-3",          # Anthropic recommends Voyage embeddings
            "llm_model":   "claude-sonnet-4-6",
        },
    }

    def __init__(
        self,
        provider: LLMProvider = LLMProvider.OPENAI,
        chunk_strategy: ChunkStrategy = ChunkStrategy.RECURSIVE,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        top_k: int = 5,
        rerank: bool = True,
        embed_model: Optional[str] = None,
        llm_model: Optional[str] = None,
    ) -> None:
        self.provider = LLMProvider(provider)
        self.chunk_strategy = ChunkStrategy(chunk_strategy)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.rerank = rerank

        defaults = self._DEFAULTS[self.provider]
        self.embed_model = embed_model or defaults["embed_model"]
        self.llm_model   = llm_model   or defaults["llm_model"]

        self._chunks: list[Chunk] = []
        self._index: Optional[faiss.IndexFlatIP] = None
        self._dim: Optional[int] = None

        self._llm_client   = self._build_llm_client()
        self._embed_client = self._build_embed_client()
        self._reranker     = self._build_reranker() if rerank else None

    # ── client builders ────────────────────────────────────────────────────

    def _build_llm_client(self):
        if self.provider == LLMProvider.OPENAI:
            from openai import OpenAI
            return OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        else:
            import anthropic
            return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def _build_embed_client(self):
        if self.provider == LLMProvider.OPENAI:
            from openai import OpenAI
            return OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        else:
            # Voyage AI client (Anthropic-recommended embeddings)
            import voyageai
            return voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])

    def _build_reranker(self):
        from sentence_transformers import CrossEncoder
        return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # ── PDF extraction ─────────────────────────────────────────────────────

    def load_pdf(self, path: str) -> str:
        """Extract full text from a PDF, preserving page metadata."""
        doc = fitz.open(path)
        pages: list[tuple[int, str]] = []
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            if text:
                pages.append((page_num, text))
        self._raw_pages = pages
        full_text = "\n\n".join(t for _, t in pages)
        return full_text

    # ── chunking strategies ────────────────────────────────────────────────

    def _chunk_fixed(self, text: str, page: int) -> list[Chunk]:
        chunks, i, idx = [], 0, 0
        while i < len(text):
            end = min(i + self.chunk_size, len(text))
            chunks.append(Chunk(text=text[i:end], page=page, index=idx))
            idx += 1
            i += self.chunk_size - self.chunk_overlap
        return chunks

    def _chunk_sentence(self, text: str, page: int) -> list[Chunk]:
        import nltk
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
        sentences = nltk.sent_tokenize(text)
        chunks, buf, idx = [], [], 0
        buf_len = 0
        for sent in sentences:
            if buf_len + len(sent) > self.chunk_size and buf:
                chunks.append(Chunk(text=" ".join(buf), page=page, index=idx))
                idx += 1
                # keep overlap sentences
                overlap_buf: list[str] = []
                overlap_len = 0
                for s in reversed(buf):
                    if overlap_len + len(s) <= self.chunk_overlap:
                        overlap_buf.insert(0, s)
                        overlap_len += len(s)
                    else:
                        break
                buf, buf_len = overlap_buf, overlap_len
            buf.append(sent)
            buf_len += len(sent)
        if buf:
            chunks.append(Chunk(text=" ".join(buf), page=page, index=idx))
        return chunks

    def _chunk_recursive(self, text: str, page: int) -> list[Chunk]:
        separators = ["\n\n", "\n", ". ", " ", ""]
        return self._recursive_split(text, separators, page, idx_offset=0)

    def _recursive_split(
        self, text: str, separators: list[str], page: int, idx_offset: int
    ) -> list[Chunk]:
        if len(text) <= self.chunk_size:
            return [Chunk(text=text, page=page, index=idx_offset)]
        sep = separators[0] if separators else ""
        splits = text.split(sep) if sep else list(text)
        chunks: list[Chunk] = []
        buf, buf_len, idx = [], 0, idx_offset
        for split in splits:
            if buf_len + len(split) + len(sep) > self.chunk_size and buf:
                chunk_text = sep.join(buf)
                if len(chunk_text) > self.chunk_size and len(separators) > 1:
                    sub = self._recursive_split(chunk_text, separators[1:], page, idx)
                    chunks.extend(sub)
                    idx += len(sub)
                else:
                    chunks.append(Chunk(text=chunk_text, page=page, index=idx))
                    idx += 1
                # overlap
                overlap_buf, overlap_len = [], 0
                for s in reversed(buf):
                    if overlap_len + len(s) <= self.chunk_overlap:
                        overlap_buf.insert(0, s)
                        overlap_len += len(s)
                    else:
                        break
                buf, buf_len = overlap_buf, overlap_len
            buf.append(split)
            buf_len += len(split) + len(sep)
        if buf:
            chunks.append(Chunk(text=sep.join(buf), page=page, index=idx))
        return chunks

    def _chunk_semantic(self, text: str, page: int) -> list[Chunk]:
        """Group sentences by embedding similarity (cosine threshold)."""
        import nltk
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
        sentences = nltk.sent_tokenize(text)
        if not sentences:
            return []
        embeds = np.array(self._embed_texts(sentences))
        # normalise for cosine similarity
        norms = np.linalg.norm(embeds, axis=1, keepdims=True) + 1e-10
        embeds_norm = embeds / norms

        threshold = 0.85
        groups: list[list[str]] = [[sentences[0]]]
        for i in range(1, len(sentences)):
            sim = float(embeds_norm[i] @ embeds_norm[i - 1])
            if sim >= threshold:
                groups[-1].append(sentences[i])
            else:
                groups.append([sentences[i]])

        chunks = []
        for idx, group in enumerate(groups):
            chunks.append(Chunk(text=" ".join(group), page=page, index=idx))
        return chunks

    def chunk_document(self) -> list[Chunk]:
        """Apply the selected chunking strategy to all pages."""
        all_chunks: list[Chunk] = []
        for page_num, page_text in self._raw_pages:
            if self.chunk_strategy == ChunkStrategy.FIXED:
                page_chunks = self._chunk_fixed(page_text, page_num)
            elif self.chunk_strategy == ChunkStrategy.SENTENCE:
                page_chunks = self._chunk_sentence(page_text, page_num)
            elif self.chunk_strategy == ChunkStrategy.RECURSIVE:
                page_chunks = self._chunk_recursive(page_text, page_num)
            elif self.chunk_strategy == ChunkStrategy.SEMANTIC:
                page_chunks = self._chunk_semantic(page_text, page_num)
            else:
                raise ValueError(f"Unknown strategy: {self.chunk_strategy}")
            all_chunks.extend(page_chunks)
        self._chunks = all_chunks
        return all_chunks

    # ── embeddings ─────────────────────────────────────────────────────────

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        if self.provider == LLMProvider.OPENAI:
            response = self._embed_client.embeddings.create(
                model=self.embed_model, input=texts
            )
            return [item.embedding for item in response.data]
        else:
            result = self._embed_client.embed(texts, model=self.embed_model)
            return result.embeddings

    def build_index(self) -> None:
        """Embed all chunks and build a FAISS inner-product index."""
        texts = [c.text for c in self._chunks]
        embeddings = self._embed_texts(texts)
        matrix = np.array(embeddings, dtype=np.float32)
        # L2-normalise → inner product == cosine similarity
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
        matrix /= norms
        self._dim = matrix.shape[1]
        self._index = faiss.IndexFlatIP(self._dim)
        self._index.add(matrix)
        for chunk, emb in zip(self._chunks, embeddings):
            chunk.embedding = np.array(emb, dtype=np.float32)

    # ── retrieval & reranking ──────────────────────────────────────────────

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        q_emb = np.array(self._embed_texts([query])[0], dtype=np.float32)
        q_emb /= np.linalg.norm(q_emb) + 1e-10
        q_emb = q_emb.reshape(1, -1)

        scores, indices = self._index.search(q_emb, self.top_k)
        results = [
            RetrievedChunk(chunk=self._chunks[idx], score=float(score))
            for score, idx in zip(scores[0], indices[0])
            if idx != -1
        ]

        if self.rerank and self._reranker and results:
            pairs = [(query, r.chunk.text) for r in results]
            rerank_scores = self._reranker.predict(pairs)
            for r, s in zip(results, rerank_scores):
                r.score = float(s)
            results.sort(key=lambda x: x.score, reverse=True)

        return results

    # ── LLM answer generation ──────────────────────────────────────────────

    def _build_prompt(self, query: str, context_chunks: list[RetrievedChunk]) -> str:
        context = "\n\n---\n\n".join(
            f"[Page {r.chunk.page}]\n{r.chunk.text}" for r in context_chunks
        )
        return (
            "You are a helpful assistant. Answer the question using ONLY the context "
            "provided below. If the answer cannot be found in the context, say so.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\nAnswer:"
        )

    def answer(self, query: str) -> dict:
        """Retrieve context and generate an answer. Returns a result dict."""
        retrieved = self.retrieve(query)
        prompt = self._build_prompt(query, retrieved)

        if self.provider == LLMProvider.OPENAI:
            response = self._llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            answer_text = response.choices[0].message.content
        else:
            response = self._llm_client.messages.create(
                model=self.llm_model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            answer_text = response.content[0].text

        return {
            "question": query,
            "answer": answer_text,
            "sources": [
                {"page": r.chunk.page, "score": round(r.score, 4), "text": r.chunk.text[:200]}
                for r in retrieved
            ],
        }

    # ── evaluation ─────────────────────────────────────────────────────────

    def evaluate(
        self,
        question: str,
        answer: str,
        ground_truth: str,
        retrieved: list[RetrievedChunk],
    ) -> EvalResult:
        """
        Compute four RAG-oriented metrics without external frameworks.

        faithfulness       - how well the answer is supported by retrieved context
        answer_relevancy   - n-gram overlap between answer and question + ground truth
        context_recall     - how much of the ground truth is covered by context
        context_precision  - fraction of retrieved chunks that are relevant
        """
        context_texts = [r.chunk.text for r in retrieved]

        faithfulness     = self._score_faithfulness(answer, context_texts)
        answer_relevancy = self._score_answer_relevancy(answer, question, ground_truth)
        context_recall   = self._score_context_recall(ground_truth, context_texts)
        context_precision = self._score_context_precision(question, retrieved)

        return EvalResult(
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            context_recall=context_recall,
            context_precision=context_precision,
        )

    def _score_faithfulness(self, answer: str, context_texts: list[str]) -> float:
        """Sentence-level: fraction of answer sentences supported by any context chunk."""
        import nltk
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
        sentences = nltk.sent_tokenize(answer)
        if not sentences:
            return 0.0
        combined_context = " ".join(context_texts).lower()
        supported = sum(
            1 for s in sentences
            if any(w in combined_context for w in s.lower().split() if len(w) > 4)
        )
        return supported / len(sentences)

    def _score_answer_relevancy(
        self, answer: str, question: str, ground_truth: str
    ) -> float:
        """F1 token overlap between answer and (question + ground_truth)."""
        def tokenize(text: str) -> set[str]:
            return {w.lower() for w in re.findall(r"\w+", text) if len(w) > 3}
        ref_tokens = tokenize(question + " " + ground_truth)
        ans_tokens = tokenize(answer)
        if not ans_tokens or not ref_tokens:
            return 0.0
        precision = len(ans_tokens & ref_tokens) / len(ans_tokens)
        recall    = len(ans_tokens & ref_tokens) / len(ref_tokens)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def _score_context_recall(
        self, ground_truth: str, context_texts: list[str]
    ) -> float:
        """Fraction of ground-truth key tokens found in the retrieved context."""
        def tokenize(text: str) -> set[str]:
            return {w.lower() for w in re.findall(r"\w+", text) if len(w) > 4}
        gt_tokens = tokenize(ground_truth)
        if not gt_tokens:
            return 0.0
        ctx_tokens = tokenize(" ".join(context_texts))
        return len(gt_tokens & ctx_tokens) / len(gt_tokens)

    def _score_context_precision(
        self, question: str, retrieved: list[RetrievedChunk]
    ) -> float:
        """
        Fraction of retrieved chunks whose text shares meaningful tokens with question.
        Simple proxy for precision@k.
        """
        def tokenize(text: str) -> set[str]:
            return {w.lower() for w in re.findall(r"\w+", text) if len(w) > 4}
        q_tokens = tokenize(question)
        if not q_tokens or not retrieved:
            return 0.0
        relevant = sum(
            1 for r in retrieved if tokenize(r.chunk.text) & q_tokens
        )
        return relevant / len(retrieved)

    # ── convenience: full pipeline ─────────────────────────────────────────

    def run(self, pdf_path: str, query: str, ground_truth: str = "") -> dict:
        """
        Full pipeline: load → chunk → embed → retrieve → answer → (optionally) evaluate.

        Returns the answer dict, with an 'eval' key if ground_truth is provided.
        """
        print(f"[1/4] Loading PDF: {pdf_path}")
        self.load_pdf(pdf_path)

        print(f"[2/4] Chunking ({self.chunk_strategy.value}, size={self.chunk_size}) ...")
        chunks = self.chunk_document()
        print(f"      {len(chunks)} chunks created.")

        print("[3/4] Embedding & building FAISS index ...")
        self.build_index()

        print("[4/4] Retrieving and generating answer ...")
        result = self.answer(query)

        if ground_truth:
            retrieved = self.retrieve(query)
            eval_result = self.evaluate(query, result["answer"], ground_truth, retrieved)
            result["eval"] = {
                "faithfulness":      eval_result.faithfulness,
                "answer_relevancy":  eval_result.answer_relevancy,
                "context_recall":    eval_result.context_recall,
                "context_precision": eval_result.context_precision,
            }
            print("\nEvaluation:\n" + str(eval_result))

        return result
