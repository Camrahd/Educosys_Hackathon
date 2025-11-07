from django.db import models
from django.contrib.postgres.fields import JSONField  # for metadata

class RAG(models.Model):
    """
    Represents a single RAG setup (e.g., Finance Knowledge Base).
    """
    name = models.CharField(max_length=255, unique=True)
    description = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    # Optional fields for advanced tracking
    vectorstore_path = models.CharField(max_length=500, blank=True, null=True)
    embedding_model = models.CharField(max_length=200, default="models/embedding-001")

    def __str__(self):
        return self.name


class RAGFile(models.Model):
    """
    Represents a single file associated with a RAG.
    """
    rag = models.ForeignKey(RAG, on_delete=models.CASCADE, related_name="files")
    file = models.FileField(upload_to="rag_files/")
    metadata = models.JSONField(default=dict, blank=True)  # dynamic metadata storage

    # Auto tracking
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed = models.BooleanField(default=False)
    vectorstore_path = models.CharField(max_length=500, blank=True, null=True)

    def __str__(self):
        return f"{self.file.name} ({self.rag.name})"
