# ragcore/rag/rag_engine.py

import os
from typing import List, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from .vectorstore import get_vectorstore
from dotenv import load_dotenv
from .views import get_answer_prompt
load_dotenv()

SYSTEM_PROMPT = """You are a helpful RAG assistant.
Use ONLY the provided context to answer the user's question.
If the answer is not in the context, say you don't know.

Context:
{context}
"""
def rag_answer(question: str, collection_name: str = "rag") -> Tuple[str, List[str]]:
    """
    Answer a question using the specified Chroma collection.
    Returns: (answer, citations)
    """
    vectordb = get_vectorstore(collection_name=collection_name)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", "{question}")
    ])

    # Retrieve docs and stuff into the prompt
    docs = retriever.invoke(question)
    context_text = "\n\n".join(getattr(d, "page_content", "") for d in docs)
    messages = prompt.format_messages(context=context_text, question=question)
    ai_msg = llm.invoke(messages)
    answer = getattr(ai_msg, "content", "") or ""

    # Collect sources
    seen, citations = set(), []
    for d in docs:
        src = (getattr(d, "metadata", {}) or {}).get("source", "unknown")
        if src not in seen:
            seen.add(src)
            citations.append(src)

    return answer, citations

def get_rag_answer_saranya(question: str, collection_name: str = "rag") -> Tuple[str, List[str]]:
    vector_store =  get_vectorstore(collection_name)

    # Step 2: Retrieve top-k most relevant chunks
    k = 5  # number of chunks to retrieve
    results = vector_store.similarity_search(question, k=k)

    # Step 3: Prepare context text from retrieved chunks
    context = "\n\n".join([doc.page_content for doc in results])

    # Step 4: Create a response-generation prompt
    answer_prompt = get_answer_prompt(question, context)

    # Step 5: Generate answer using LLM
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.3,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    answer_msg = llm.invoke([("system", "You are a helpful assistant."), ("user", answer_prompt)])
    answer_text = (getattr(answer_msg, "content", "") or "").strip()

    # Step 6: Return or send back the final answer
    return {
        "question": question,
        "collection_used": collection_name,
        "answer": answer_text,
        "retrieved_chunks": [doc.page_content for doc in results]
    }