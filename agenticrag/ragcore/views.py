import json
import os
from pathlib import Path
import uuid
from datetime import datetime
from typing import List

from django.conf import settings
from django.http import JsonResponse, HttpRequest
from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.csrf import csrf_exempt

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma

from .rag.loader import load_document
from .rag.vectorstore import get_vectorstore
from .models import RAG, RAGFile
from .rag.embeddings import get_embeddings


# ------------------------ Helpers ------------------------

def chunk_docs(docs, chunk_size=1200, chunk_overlap=160):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)


def call_llm(prompt: str) -> dict:
    """Calls the LLM to produce JSON metadata; returns a dict (best-effort)."""
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    msg = llm.invoke([("system", "Return ONLY valid JSON."), ("user", prompt)])
    text = (getattr(msg, "content", "") or "").strip()
    try:
        return json.loads(text)
    except Exception:
        return {
            "title": "Document",
            "summary": text,
            "topics": [],
            "keywords": [],
            "document_type": "unknown",
            "main_subject": "unknown",
        }


def generate_file_metadata(filename: str, chunks: List[str], filetype: str):
    total_chunks = len(chunks)
    if total_chunks <= 10:
        metadata_chunks = chunks
    else:
        idx = [
            0, 1, 2, 3, 4, 5,
            total_chunks // 4,
            total_chunks // 2,
            3 * total_chunks // 4,
            max(total_chunks - 5, 0),
            max(total_chunks - 4, 0),
            max(total_chunks - 3, 0),
            max(total_chunks - 2, 0),
            max(total_chunks - 1, 0),
        ]
        metadata_chunks = [chunks[i] for i in idx if 0 <= i < total_chunks]

    sample_text = "\n\n---\n\n".join(metadata_chunks)
    prompt = f"""
Analyze this document and extract metadata.

Filename: {filename}
Type: {filetype}
Total chunks: {total_chunks}

Sample:
{sample_text}

Respond ONLY with JSON:
{{
  "title": "descriptive title",
  "summary": "2-3 sentence summary",
  "topics": ["topic1","topic2","topic3"],
  "keywords": ["k1","k2","k3","k4","k5"],
  "document_type": "FAQ/spec/guide/report/manual/procedure/other",
  "main_subject": "primary subject"
}}
"""
    try:
        meta = call_llm(prompt)
        required = ["title", "summary", "topics", "keywords", "document_type", "main_subject"]
        if not all(k in meta for k in required):
            raise ValueError("missing fields")
        meta.update({
            "filename": filename,
            "filetype": filetype,
            "total_chunks": total_chunks,
            "generated_at": datetime.now().isoformat(),
        })
        return meta
    except Exception:
        return {
            "filename": filename,
            "filetype": filetype,
            "total_chunks": total_chunks,
            "title": filename,
            "summary": f"Document uploaded: {filename}",
            "topics": [],
            "keywords": [],
            "document_type": "unknown",
            "main_subject": "unknown",
            "generated_at": datetime.now().isoformat(),
        }


# ------------------------ Pages ------------------------

def index(request: HttpRequest):
    """Create a new RAG + ingest uploaded files; then redirect to its chat page."""
    if request.method == "POST":
        rag_name = (request.POST.get("ragName") or "").strip()
        rag_description = (request.POST.get("ragDescription") or "").strip()
        files = request.FILES.getlist("files")

        if not rag_name or not files:
            return render(request, "index.html", {"error": "Name and files required."})

        # Create the RAG (slug auto-set in model.save)
        rag_inst = RAG.objects.create(name=rag_name, description=rag_description)

        # Vectorstore collection per RAG (isolated)
        collection = f"rag_{rag_inst.slug}"
        

        # Save + ingest each file
        for f in files:
            # Save via FileField: stored at MEDIA_ROOT/rag_files/<name>
            unique_id = uuid.uuid4().hex  # or str(uuid.uuid4()) if you want dashes
            vectordb = get_vectorstore(collection_name=f'{collection}_{unique_id}')
            rag_file = RAGFile.objects.create(rag=rag_inst, file=f, collection_name=f'{collection}_{unique_id}')

            # Parse from disk path
            docs = load_document(rag_file.file.path)
            for d in docs:
                d.metadata["source"] = os.path.basename(rag_file.file.name)

            # Chunk + embed + persist
            chunks = chunk_docs(docs)
            vectordb.add_documents(chunks)
            vectordb.persist()

            # Metadata for UI
            chunks_text = [c.page_content for c in chunks[:200]]  # cap sample
            ext = os.path.splitext(rag_file.file.name)[1].lower()
            rag_file.metadata = generate_file_metadata(
                os.path.basename(rag_file.file.name),
                chunks_text,
                ext,
            )
            rag_file.processed = True
            rag_file.vectorstore_path = str(getattr(vectordb, "_persist_directory", "")) or None
            rag_file.save()

        return redirect("rag_detail", slug=rag_inst.slug)

    return render(request, "index.html")


def rags_list(request: HttpRequest):
    """List all RAGs."""
    rags = RAG.objects.order_by("-id")
    return render(request, "rags_list.html", {"rags": rags})


def rag_detail(request: HttpRequest, slug: str):
    """Chat UI for a single RAG (by slug)."""
    rag = get_object_or_404(RAG, slug=slug)
    files = rag.files.order_by("id")
    return render(request, "rag_chat.html", {"rag": rag, "files": files})

def generate_rag_collection_selection_prompt(question, rag_file_name_meta_data_dict):
    return f"""
You are a reasoning assistant that decides which document collection in a vector database 
is most relevant to answer a user's query.

User query:
\"\"\"{question}\"\"\"

Below are available collections and their associated metadata:

{json.dumps(rag_file_name_meta_data_dict, indent=2)}

Your task:
1. Analyze the user query.
2. Based on metadata, select the most relevant collection(s) for retrieval.
3. Return your answer as a JSON object with a key "selected_collections" containing a list 
   of collection names. Do not include explanations or additional text.

Example output:
{{"selected_collections": ["collection_abc123"]}}
"""

def get_answer_prompt(question: str, context: str) -> str:
    return f"""
    You are an expert assistant. Use the following retrieved context to answer the user's question.

    Context:
    \"\"\"{context}\"\"\"

    User question:
    \"\"\"{question}\"\"\"

    Instructions:
    - If the answer is directly supported by the context, use it.
    - If the context doesn’t contain enough information, say “I don’t have enough data to answer confidently.”
    - Be clear and concise.

    Return only the final answer text.
    """

@csrf_exempt
def ask_question(request: HttpRequest, slug: str):
    """POST {question} -> answer using the specific RAG (collection rag_<slug>)."""
    try:
        data = json.loads(request.body or "{}")
    except Exception:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    if RAG.objects.filter(slug=slug).count() == 0:
        #no such rag exists, redirect to list of rags
        return JsonResponse({"error": "RAG not found"}, status=404)
    question = (data.get("question") or "").strip()
    if not question:
        return JsonResponse({"error": "Question missing"}, status=400)
    
    rag_inst = RAG.objects.get(slug=slug)
    rag_files = RAGFile.objects.filter(rag=rag_inst)
    if rag_files.count() == 0:
        #no files ingested for this rag, redirect to list of rags
        return JsonResponse({"error": "No files ingested for this RAG"}, status=404)
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=400)
    rag_file_name_meta_data_dict = {}
    for rag_file in rag_files:
        rag_file_name_meta_data_dict[os.path.basename(rag_file.file.name)] = {
                                                                                'metadata': rag_file.metadata,
                                                                                'collection_name': rag_file.collection_name
                                                                             }
    
    #make LLM call to decide the file to be considered based on the user query
    retriever_selection_prompt = generate_rag_collection_selection_prompt(question, rag_file_name_meta_data_dict)
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    msg = llm.invoke([("system", "Return ONLY valid JSON."), ("user", retriever_selection_prompt)])
    text = (getattr(msg, "content", "") or "").strip()  
    try:
        selection_response = json.loads(text)
        selected_collections = selection_response.get("selected_collections", [])
        if not selected_collections:
            raise ValueError("No collections selected")
        # For simplicity, consider only the first selected collection
        collection_name = selected_collections[0]
    except Exception:
        # Fallback to default collection if parsing fails
        collection_name = f"rag_{slug}"
    print(f'Collection chosen: {collection_name}')
    from .rag.rag_engine import rag_answer, get_rag_answer
    #option1
    answer, citations = rag_answer(question, collection_name=collection_name)
    print(answer, citations)
    #option2
    response  = get_rag_answer(question, collection_name=collection_name)
    print(f"Second response: {response}")
    return JsonResponse({"answer": answer, "citations": citations, 'response': response})

from django.views.decorators.http import require_GET

@require_GET
def check_rag_name(request: HttpRequest):
    name = (request.GET.get("name") or "").strip()
    exists = False
    if name:
        exists = RAG.objects.filter(name__iexact=name).exists()
    return JsonResponse({"exists": exists})