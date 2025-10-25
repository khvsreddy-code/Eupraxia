"""RAG manager with graceful fallback.

This module tries to use sentence-transformers + faiss for good quality
retrieval. If those libraries aren't installed or fail to import (common on
lightweight developer machines), we expose a compatible, lightweight
fallback RAGManager that uses simple token-overlap scoring. The API is the
same: build_index(docs), retrieve(query, k) -> list of (idx, score, doc), and
get_docs().
"""
from typing import List, Tuple


try:
    # Preferred, higher-quality implementation
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np


    class RAGManager:
        def __init__(self, embed_model_name: str = 'thenlper/gte-small'):
            # Note: model download can be slow the first time
            self.embed_model = SentenceTransformer(embed_model_name)
            self.index = None
            self.docs: List[str] = []
            self.dimension = None

        def build_index(self, docs: List[str]):
            """Build a FAISS index from the provided list of document strings."""
            self.docs = docs
            embeddings = self.embed_model.encode(docs)
            embeddings = np.array(embeddings).astype('float32')
            self.dimension = embeddings.shape[1]
            faiss.normalize_L2(embeddings)
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index.add(embeddings)

        def retrieve(self, query: str, k: int = 3) -> List[Tuple[int, float, str]]:
            """Return top-k (idx, score, doc) by inner-product on normalized embs."""
            if self.index is None:
                return []
            q_emb = self.embed_model.encode([query]).astype('float32')
            faiss.normalize_L2(q_emb)
            D, I = self.index.search(q_emb, k)
            results = []
            for idx, score in zip(I[0], D[0]):
                if idx < 0 or idx >= len(self.docs):
                    continue
                results.append((int(idx), float(score), self.docs[idx]))
            return results

        def get_docs(self):
            return self.docs


except Exception as _e:
    # Lightweight fallback implementation: token-overlap scorer. No external
    # ML dependencies required. Good enough for local demos and tests.
    class RAGManager:
        def __init__(self, embed_model_name: str = None):
            self.docs: List[str] = []

        def build_index(self, docs: List[str]):
            # store original docs lowercased for simple matching
            self.docs = [d.strip() for d in docs]

        def _score(self, query: str, doc: str) -> float:
            q_tokens = set(query.lower().split())
            d_tokens = set(doc.lower().split())
            if not q_tokens or not d_tokens:
                return 0.0
            common = q_tokens.intersection(d_tokens)
            # score normalized by query length to prioritize relevance
            return len(common) / max(1, len(q_tokens))

        def retrieve(self, query: str, k: int = 3) -> List[Tuple[int, float, str]]:
            scores = []
            for i, d in enumerate(self.docs):
                s = self._score(query, d)
                if s > 0:
                    scores.append((i, float(s), d))
            # sort descending by score
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[:k]

        def get_docs(self):
            return self.docs
