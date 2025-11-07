from django.shortcuts import render

# Create your views here.
import os
import uuid
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import json
from .rag.loader import load_document
from .rag.vectorstore import get_vectorstore
from .models import RAG, RAGFile

from datetime import datetime
from typing import List

async def generate_file_metadata(filename: str, chunks: List[str], filetype: str):
    total_chunks = len(chunks)

    """If smaller chunks -> no need to extract metadata dynamically"""
    if total_chunks <= 10:
        # Use all chunks to extract metadata
        metadata_chunks = chunks
    else:
        """Triggers when we have larger chunks for larger docs"""
        # Beginning, middle, and end
        indices = [
            0, 1, 2, 3, 4, 5,  # First 5 chunks
            total_chunks // 4,  # Quarter of chunks
            total_chunks // 2,  # Middle of chunks
            3 * total_chunks // 4,  # Three quarters
            total_chunks - 5, total_chunks - 4, total_chunks - 3, total_chunks - 2, total_chunks - 1  # Last 5 chunks
        ]
        metadata_chunks = [chunks[i] for i in indices if i < total_chunks]

    # Combine chunks
    metadata_text = "\n\n---\n\n".join(metadata_chunks)

    # Generate metadata prompt
    prompt = f"""
    Analyze this document and extract metadata.
    Filename: {filename}
    Type: {filetype}
    Total chunks: {total_chunks}

    Document sample (from beginning, middle, and end):
    {metadata_text}

    Generate metadata in JSON format:
    {{
        "title": "descriptive title",
        "summary": "2-3 sentence summary of document content and purpose",
        "topics": ["main topic 1", "main topic 2", "main topic 3"],
        "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
        "document_type": "FAQ/specification/guide/report/manual/procedure/other",
        "main_subject": "primary subject matter"
    }}

    IMPORTANT:
    - Extract topics from the content (e.g., "quality management", "procedures") specific to the document uploaded.
    - Extract keywords people would search for (e.g., "owner", "approver", "permissions", "workflow") specific to the document uploaded.
    - Be comprehensive — include all important terms mentioned.
    - Look at table content, text content, and captions.
    Respond ONLY with valid JSON.
    """

    try:
        metadata = await call_llm(prompt)
        required_fields = ['title', 'summary', 'topics', 'keywords', 'document_type', 'main_subject']

        if not all(field in metadata for field in required_fields):
            raise ValueError("Missing required metadata fields")

        # Add extra fields
        metadata['filename'] = filename
        metadata['filetype'] = filetype
        metadata['total_chunks'] = total_chunks
        metadata['generated_at'] = datetime.now().isoformat()

        return metadata

    except Exception as e:
        # Fallback -> return basic metadata
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
            "generated_at": datetime.now().isoformat()
        }


def index(request):
    if request.method == "POST":
        rag_name = request.POST.get("ragName", "").strip()
        files = request.FILES.getlist("files")
        rag_description = request.POST.get("ragDescription", "").strip()
        # Here you can handle the RAG creation logic    
        rag_inst = RAG.objects.create(name=rag_name, description=rag_description)
        for file in files:
            RAGFile.objects.create(rag=rag_inst, file=file)
    return render(request, "index.html")



def upload_files(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST method required"}, status=400)

    files = request.FILES.getlist("files")
    if not files:
        return JsonResponse({"error": "No files uploaded"}, status=400)

    upload_dir = os.path.join(settings.BASE_DIR, "media")
    os.makedirs(upload_dir, exist_ok=True)

    vectordb = get_vectorstore()

    count = 0
    for f in files:
        filename = f"{uuid.uuid4()}_{f.name}"
        filepath = os.path.join(upload_dir, filename)

        # Store file
        with open(filepath, "wb+") as dest:
            for chunk in f.chunks():
                dest.write(chunk)

        # Load + parse the file
        docs = load_document(filepath)

        # Add metadata
        for d in docs:
            d.metadata["source"] = filename

        # Store embeddings into vector DB
        vectordb.add_documents(docs)
        vectordb.persist()

        count += 1

    return JsonResponse({"indexed_count": count})


@csrf_exempt
def ask_question(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=400)

    data = json.loads(request.body)
    question = data.get("question", "")

    if not question:
        return JsonResponse({"error": "Question missing"}, status=400)

    from .rag.rag_engine import rag_answer
    answer, citations = rag_answer(question)

    return JsonResponse({
        "answer": answer,
        "citations": citations
    })
