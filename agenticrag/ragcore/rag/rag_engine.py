# ragcore/rag/rag_engine.py

import os
from typing import List, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from .vectorstore import get_vectorstore
from dotenv import load_dotenv
load_dotenv()

SYSTEM_PROMPT = """You are a helpful RAG assistant.
Use ONLY the provided context to answer the user's question.
If the answer is not in the context, say you don't know.

Context:
{context}
"""

def rag_answer(question: str) -> Tuple[str, List[str]]:
    """
    Run a RAG query and return (answer, citations).

    Returns:
        answer: str
        citations: list of source strings pulled from document metadata["source"]
    """
    vectordb = get_vectorstore()
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    # LLM
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Prompt with explicit {context} and {question}
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", "{question}")
    ])

    # Retrieve documents
    docs = retriever.invoke(question)  # list[Document]
    context_text = "\n\n".join(getattr(d, "page_content", "") for d in docs)

    # Format messages and call the model
    messages = prompt.format_messages(context=context_text, question=question)
    ai_msg = llm.invoke(messages)
    answer = getattr(ai_msg, "content", "") or ""

    # Collect unique sources
    seen = set()
    citations: List[str] = []
    for d in docs:
        src = (getattr(d, "metadata", {}) or {}).get("source", "unknown")
        if src not in seen:
            seen.add(src)
            citations.append(src)

    return answer, citations