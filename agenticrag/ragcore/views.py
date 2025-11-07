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
