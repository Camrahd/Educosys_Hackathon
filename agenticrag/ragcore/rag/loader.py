# ragcore/rag/loader.py
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    Docx2txtLoader,
)

def load_document(path: str) -> List[Document]:
    p = Path(path)
    sfx = p.suffix.lower()

    if sfx == ".pdf":
        return PyPDFLoader(str(p)).load()
    if sfx in {".md", ".markdown"}:
        return UnstructuredMarkdownLoader(str(p)).load()
    if sfx == ".docx":
        return Docx2txtLoader(str(p)).load()
    # default: plain text
    return TextLoader(str(p), encoding="utf-8", autodetect_encoding=True).load()