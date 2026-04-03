"""
LangChain LCEL chains on top of RAGRetriever (Chroma + embeddings).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.rag.rag_system import RAGRetriever


def _docs_from_rag(rag: RAGRetriever, query: str, categories: Optional[List[str]]) -> List[Document]:
    ctx = rag.retrieve_context(query=query, categories=categories)
    docs: List[Document] = []
    for r in ctx.get("results", []):
        text = r.get("text") or ""
        md = {k: v for k, v in (r.get("metadata") or {}).items() if isinstance(v, (str, int, float))}
        docs.append(Document(page_content=text, metadata=md))
    return docs


def _format_docs(docs: List[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs)


def make_retrieve_fn(rag: RAGRetriever, categories: Optional[List[str]] = None):
    def _fn(query: str) -> List[Document]:
        return _docs_from_rag(rag, query, categories)

    return RunnableLambda(_fn)


def build_rag_qa_chain(llm, rag: RAGRetriever, categories: Optional[List[str]] = None):
    """Grounded QA: single string input = user question (also used as retrieval query)."""
    retrieve = make_retrieve_fn(rag, categories)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a GigShield parametric insurance analyst. Answer ONLY using the context. "
                "If context is insufficient, state what is missing. Be concise.",
            ),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ]
    )
    return (
        RunnableParallel(
            context=retrieve | RunnableLambda(_format_docs),
            question=RunnablePassthrough(),
        )
        | prompt
        | llm
        | StrOutputParser()
    )


def build_fraud_reasoning_chain(llm, rag: RAGRetriever):
    """RAG over fraud_cases + worker JSON in one prompt."""
    retrieve = make_retrieve_fn(rag, ["fraud_cases"])
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You augment GigShield fraud detection with patterns from context. "
                "Reply with compact JSON only: {\"rationale\": str, \"escalate_review\": bool}",
            ),
            ("human", "Context:\n{context}\n\nWorker data JSON:\n{worker_json}\n"),
        ]
    )

    def _merge(input_: Dict[str, Any]) -> Dict[str, Any]:
        q = str(input_.get("query", "fraud patterns worker"))
        wj = input_.get("worker_json", "{}")
        if isinstance(wj, dict):
            wj = json.dumps(wj, default=str)
        docs = _docs_from_rag(rag, q, ["fraud_cases"])
        return {"context": _format_docs(docs), "worker_json": wj}

    return RunnableLambda(_merge) | prompt | llm | StrOutputParser()


def build_decision_chain(llm):
    """Structured decision from aggregated agent packet (string or dict serialized)."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "GigShield orchestrator. Output JSON only with keys: "
                "decision (auto_approve|auto_reject|manual_review), confidence (0-1), rationale (short string).",
            ),
            ("human", "Agent packet:\n{packet}\n"),
        ]
    )
    return RunnableLambda(lambda p: {"packet": p if isinstance(p, str) else json.dumps(p, default=str)}) | prompt | llm | StrOutputParser()


async def ainvoke_chain(chain, input_data: Any) -> str:
    if hasattr(chain, "ainvoke"):
        return await chain.ainvoke(input_data)
    return chain.invoke(input_data)
