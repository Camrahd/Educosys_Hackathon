# ragcore/rag/vectorstore.py
import os
from pathlib import Path
from typing import Iterable, Optional, Sequence

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from .embeddings import get_embeddings

# Persistent directory for Chroma DB
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", Path.cwd() / "chroma_db"))

def get_vectorstore(collection_name: str = "rag") -> Chroma:
    """
    Create/load a persistent Chroma vector store.
    Uses a stable collection_name to ensure the same collection is reused.
    """
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=collection_name,
        persist_directory=str(CHROMA_DIR),
        embedding_function=get_embeddings(),
    )

def upsert_documents(
    docs: Sequence[Document],
    ids: Optional[Iterable[str]] = None,
    collection_name: str = "rag",
) -> None:
    """
    Helper to add/update documents and persist the DB.
    If ids are provided and collide, Chroma will update those entries.
    """
    vs = get_vectorstore(collection_name=collection_name)
    if ids is not None:
        vs.add_documents(documents=list(docs), ids=list(ids))
    else:
        vs.add_documents(documents=list(docs))
    # Ensure data is flushed to disk
    vs.persist()