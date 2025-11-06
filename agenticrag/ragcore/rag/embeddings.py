# ragcore/rag/embeddings.py
# ragcore/rag/embeddings.py
import os

try:
    from django.conf import settings  # will work when Django is set up
except Exception:
    settings = None

from langchain_openai import OpenAIEmbeddings

def get_embeddings(model: str = "text-embedding-3-small") -> OpenAIEmbeddings:
    """
    Returns an OpenAIEmbeddings instance. Requires OPENAI_API_KEY.
    Checks Django settings first, then environment.
    """
    api_key = ""
    if settings and getattr(settings, "OPENAI_API_KEY", ""):
        api_key = settings.OPENAI_API_KEY
    else:
        api_key = os.getenv("OPENAI_API_KEY", "")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Put it in your .env (loaded by manage.py), "
            "your shell env, or Django settings."
        )

    return OpenAIEmbeddings(model=model, api_key=api_key)