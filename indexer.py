import os
import json
import numpy as np
import faiss
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from config import EMBEDDING_MODEL_NAME, INDEX_FILE_PATH, METADATA_FILE_PATH

class ContractIndex:
    def __init__(self):
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.dimension = self.embedder.get_sentence_embedding_dimension()
        self.faiss_index: Optional[faiss.IndexFlatIP] = None
        self.chunk_metadata: List[Dict[str, Any]] = []
        self.bm25_index: Optional[BM25Okapi] = None
        self.documents: List[Dict[str, Any]] = []
        self.known_hashes: set = set()

    def is_ready(self) -> bool:
        return self.faiss_index is not None and self.faiss_index.ntotal > 0

    def add_chunks(self, new_chunk_records: List[Dict[str, Any]]):
        if not new_chunk_records:
            return

        texts = [c["text"] for c in new_chunk_records]
        embeddings = self.embedder.encode(texts, convert_to_numpy=True)
        
        # Cosine normalization
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        normalized_vectors = (embeddings / norms).astype(np.float32)

        if self.faiss_index is None:
            self.faiss_index = faiss.IndexFlatIP(self.dimension)

        self.faiss_index.add(normalized_vectors)
        self.chunk_metadata.extend(new_chunk_records)

        # Build BM25 Index over all corpus texts
        corpus_texts = [c["text"].lower().split() for c in self.chunk_metadata]
        self.bm25_index = BM25Okapi(corpus_texts)

    def search_semantic(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        if not self.is_ready():
            return []
        q_vec = self.embedder.encode([query], convert_to_numpy=True)
        q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-10)
        
        distances, indices = self.faiss_index.search(q_norm.astype(np.float32), min(top_k, self.faiss_index.ntotal))
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:
                results.append((self.chunk_metadata[idx]["chunk_id"], float(dist)))
        return results

    def search_keyword(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        if not self.bm25_index or not self.chunk_metadata:
            return []
        tokenized_query = query.lower().split()
        scores = self.bm25_index.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        max_score = max(scores) if max(scores) > 0 else 1.0
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                normalized_score = float(scores[idx] / max_score)
                results.append((self.chunk_metadata[idx]["chunk_id"], normalized_score))
        return results

    def save_to_disk(self):
        os.makedirs(os.path.dirname(INDEX_FILE_PATH), exist_ok=True)
        if self.faiss_index:
            faiss.write_index(self.faiss_index, INDEX_FILE_PATH)
        with open(METADATA_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump({"documents": self.documents, "chunks": self.chunk_metadata}, f, indent=2)

    def chunk_lookup(self) -> Dict[str, Dict[str, Any]]:
        return {c["chunk_id"]: c for c in self.chunk_metadata}