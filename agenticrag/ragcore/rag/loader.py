# ragcore/rag/loader.py
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader

def load_document(path: str) -> List[Document]:
    """
    Load a file into a list[Document].
    - PDFs are loaded page-by-page via PyPDFLoader.
    - Other text files are loaded via TextLoader (utf-8 with 'ignore' errors).
    """
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(str(p))
        return loader.load()
    else:
        # Handles .txt, .md, .csv, etc. as plain text
        loader = TextLoader(str(p), encoding="utf-8", autodetect_encoding=True)
        return loader.load()