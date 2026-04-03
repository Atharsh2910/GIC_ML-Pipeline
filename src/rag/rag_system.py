"""
RAG: embeddings + Chroma (local) or Pinecone (optional). Retrieval feeds agents + optional LLM prompts.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import numpy as np

import chromadb

import sys

sys.path.append("..")
from config.agent_config import VECTOR_STORE_CONFIG


def _pinecone_flat_metadata(meta: Dict[str, Any], text: str, category: str) -> Dict[str, Any]:
    """Pinecone metadata: strings, ints, floats, bools only; store chunk text for retrieval."""
    out: Dict[str, Any] = {"category": str(category), "text": str(text)[:12000]}
    for k, v in (meta or {}).items():
        if k == "text":
            continue
        if isinstance(v, bool):
            out[str(k)] = v
        elif isinstance(v, (int, float, str)):
            out[str(k)] = v
        elif v is not None:
            out[str(k)] = str(v)[:512]
    return out


class EmbeddingGenerator:
    """Sentence-transformer embeddings (same dim as Chroma collection)."""

    def __init__(self, model_name: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or VECTOR_STORE_CONFIG["embedding_model"]
        self.model = SentenceTransformer(self.model_name)

    def generate(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=False)

    def generate_single(self, text: str) -> np.ndarray:
        return self.model.encode([text], show_progress_bar=False)[0]


class VectorStore:
    """Chroma persistent store by default; Pinecone optional with env + API key."""

    def __init__(self, provider: str | None = None):
        self.config = VECTOR_STORE_CONFIG
        self.provider = (
            provider
            or os.environ.get("VECTOR_STORE_PROVIDER", "").strip().lower()
            or self.config["provider"]
        )
        self.embedder = EmbeddingGenerator()
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self) -> None:
        if self.provider == "chromadb":
            persist_dir = self.config["chromadb"]["persist_directory"]
            os.makedirs(persist_dir, exist_ok=True)
            self.client = chromadb.PersistentClient(path=persist_dir)
            name = self.config["collection_name"]
            self.collection = self.client.get_or_create_collection(name=name)
        elif self.provider == "pinecone":
            self._initialize_pinecone()
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _initialize_pinecone(self) -> None:
        from pinecone import Pinecone

        api_key = os.getenv("PINECONE_API_KEY", "").strip()
        if not api_key or api_key.startswith("http"):
            raise RuntimeError(
                "PINECONE_API_KEY must be your Pinecone API key (secret), not the host URL. "
                "Put the index URL in PINECONE_HOST instead."
            )
        pc = Pinecone(api_key=api_key)
        host = (os.getenv("PINECONE_HOST") or os.getenv("PINECONE_INDEX_HOST") or "").strip().rstrip("/")
        index_name = os.getenv("PINECONE_INDEX_NAME", self.config["pinecone"].get("index_name", "gigshield-index"))
        # Serverless / new console: connect with host URL
        if host:
            self._pinecone_index = pc.Index(host=host)
        else:
            self._pinecone_index = pc.Index(index_name)
        self.collection = None

    def add_documents(self, documents: List[Dict[str, Any]], category: str) -> None:
        if not documents:
            return
        texts = [doc["text"] for doc in documents]
        embeddings = self.embedder.generate(texts)
        ids = [doc["id"] for doc in documents]
        metadatas = []
        for doc in documents:
            md = dict(doc.get("metadata", {}))
            md["category"] = category
            metadatas.append(md)
        if self.provider == "chromadb":
            self.collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas,
            )
        elif self.provider == "pinecone":
            vectors = []
            for i, _id in enumerate(ids):
                flat = _pinecone_flat_metadata(metadatas[i], texts[i], category)
                vectors.append({"id": str(_id), "values": embeddings[i].tolist(), "metadata": flat})
            self._pinecone_index.upsert(vectors=vectors)

    def search(self, query: str, top_k: int = 5, category: str | None = None) -> List[Dict[str, Any]]:
        qemb = self.embedder.generate_single(query)
        if self.provider == "chromadb":
            where = {"category": category} if category else None
            results = self.collection.query(
                query_embeddings=[qemb.tolist()],
                n_results=top_k,
                where=where,
            )
            formatted: List[Dict[str, Any]] = []
            if not results["ids"] or not results["ids"][0]:
                return formatted
            for i in range(len(results["ids"][0])):
                formatted.append(
                    {
                        "id": results["ids"][0][i],
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i],
                    }
                )
            return formatted
        if self.provider == "pinecone":
            flt = {"category": {"$eq": category}} if category else None
            qkw: Dict[str, Any] = {
                "vector": qemb.tolist(),
                "top_k": top_k,
                "include_metadata": True,
            }
            if flt is not None:
                qkw["filter"] = flt
            ns = (os.getenv("PINECONE_NAMESPACE") or "").strip() or None
            if ns is not None:
                qkw["namespace"] = ns
            res = self._pinecone_index.query(**qkw)
            formatted: List[Dict[str, Any]] = []
            for m in res.get("matches") or []:
                meta = dict(m.get("metadata") or {})
                text = str(meta.pop("text", "") or "")
                score = float(m.get("score") or 0.0)
                formatted.append(
                    {
                        "id": m.get("id"),
                        "text": text,
                        "metadata": meta,
                        "similarity_score": score,
                    }
                )
            return formatted
        raise ValueError(f"Unsupported provider: {self.provider}")

    def delete_category(self, category: str) -> None:
        if self.provider == "chromadb":
            self.collection.delete(where={"category": category})
        elif self.provider == "pinecone":
            flt = {"category": {"$eq": category}}
            ns = (os.getenv("PINECONE_NAMESPACE") or "").strip() or None
            kw: Dict[str, Any] = {"filter": flt}
            if ns is not None:
                kw["namespace"] = ns
            self._pinecone_index.delete(**kw)


class RAGRetriever:
    """Retrieval for agents; filters by max L2 distance (Chroma)."""

    def __init__(self, vector_store: VectorStore | None = None):
        self.vector_store = vector_store or VectorStore()
        self.config = VECTOR_STORE_CONFIG["retrieval_config"]

    def retrieve_context(self, query: str, categories: List[str] | None = None) -> Dict[str, Any]:
        all_results: List[Dict[str, Any]] = []
        top_k = self.config.get("top_k", 5)
        max_d = self.config.get("max_l2_distance", 1.25)
        min_pc = self.config.get("pinecone_min_score", 0.25)

        if categories:
            for cat in categories:
                all_results.extend(self.vector_store.search(query=query, top_k=top_k, category=cat))
        else:
            all_results.extend(self.vector_store.search(query=query, top_k=top_k))

        if self.vector_store.provider == "pinecone":
            filtered = [r for r in all_results if r.get("similarity_score", 0) >= min_pc]
            filtered.sort(key=lambda x: -x.get("similarity_score", 0))
        else:
            filtered = [r for r in all_results if r.get("distance", 0) <= max_d]
            filtered.sort(key=lambda x: x["distance"])

        return {
            "query": query,
            "num_results": len(filtered),
            "results": filtered,
            "context_text": self._format_context(filtered),
        }

    def _format_context(self, results: List[Dict[str, Any]]) -> str:
        if not results:
            return "No relevant context found."
        parts = []
        for i, result in enumerate(results, 1):
            parts.append(f"[{i}] {result['text']}")
        return "\n\n".join(parts)

    def retrieve_for_fraud_detection(self, worker_data: Dict[str, Any]) -> Dict[str, Any]:
        q = (
            f"fraud gps spoofing {worker_data.get('gps_spoofing_score')} "
            f"movement {worker_data.get('movement_realism_score')} cluster {worker_data.get('coordinated_fraud_cluster_id')}"
        )
        return self.retrieve_context(query=q, categories=["fraud_cases"])

    def retrieve_for_claim_validation(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        q = (
            f"claim income loss {claim_data.get('income_loss_percentage')} "
            f"disruption {claim_data.get('disruption_type')} slab {claim_data.get('selected_slab')}"
        )
        return self.retrieve_context(query=q, categories=["historical_claims", "insurance_policies"])


def populate_knowledge_base(vector_store: VectorStore) -> None:
    """Load curated chunks (insurance_policies, fraud_cases, disruption_events, historical_claims, regional_data)."""
    from src.rag.knowledge_bundle import KNOWLEDGE_BY_CATEGORY

    for category, docs in KNOWLEDGE_BY_CATEGORY.items():
        vector_store.add_documents(docs, category)


if __name__ == "__main__":
    vs = VectorStore(provider="chromadb")
    populate_knowledge_base(vs)
    r = RAGRetriever(vs)
    print(r.retrieve_for_claim_validation({"income_loss_percentage": 0.35, "disruption_type": "rainfall", "selected_slab": "Slab 3 (100%)"})["context_text"])
