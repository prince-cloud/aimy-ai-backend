from django.db import models
import uuid
from accounts.models import CustomUser
from django.core.validators import FileExtensionValidator
import hashlib
import os


class Document(models.Model):
    """Model to store uploaded documents with metadata"""

    user = models.ForeignKey(
        CustomUser,
        related_name="documents",
        on_delete=models.CASCADE,
    )
    title = models.CharField(max_length=255)
    file = models.FileField(
        upload_to="documents/",
        validators=[FileExtensionValidator(allowed_extensions=["pdf", "docx", "txt"])],
    )
    file_hash = models.CharField(
        max_length=64, unique=True, help_text="SHA256 hash of file content"
    )
    file_size = models.BigIntegerField(help_text="File size in bytes")
    file_type = models.CharField(max_length=10, help_text="File extension")

    # Processing status
    is_processed = models.BooleanField(
        default=False, help_text="Whether document has been processed and indexed"
    )
    processing_error = models.TextField(
        blank=True, null=True, help_text="Error message if processing failed"
    )

    # Vector store info
    index_name = models.CharField(
        max_length=255, unique=True, help_text="Redis index name for this document"
    )

    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    processed_at = models.DateTimeField(null=True, blank=True)
    uud = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.title} - {self.user.email}"

    def save(self, *args, **kwargs):
        if not self.file_hash:
            self.file_hash = self._calculate_file_hash()
        if not self.file_size:
            self.file_size = self.file.size if self.file else 0
        if not self.file_type:
            self.file_type = (
                os.path.splitext(self.file.name)[1].lower() if self.file else ""
            )
        if not self.index_name:
            self.index_name = f"doc_{self.file_hash[:16]}"
        super().save(*args, **kwargs)

    def _calculate_file_hash(self):
        """Calculate SHA256 hash of file content"""
        if not self.file:
            return ""

        hash_sha256 = hashlib.sha256()
        for chunk in self.file.chunks():
            hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def get_file_path(self):
        """Get absolute file path"""
        if not self.file:
            return None
        try:
            return self.file.path
        except Exception as e:
            # Log the error but don't raise
            from loguru import logger

            logger.error(f"Error getting file path for document {self.id}: {str(e)}")
            return None

    def is_file_accessible(self):
        """Check if the file is accessible"""
        file_path = self.get_file_path()
        if not file_path:
            return False
        return os.path.exists(file_path) and os.path.isfile(file_path)


class DocumentChunk(models.Model):
    """Model to store document chunks for vector search"""

    document = models.ForeignKey(
        Document,
        related_name="chunks",
        on_delete=models.CASCADE,
    )
    content = models.TextField(help_text="Text content of the chunk")
    chunk_index = models.IntegerField(help_text="Order of chunk in document")
    page_number = models.IntegerField(
        null=True, blank=True, help_text="Page number if applicable"
    )

    # Vector store metadata
    vector_id = models.CharField(
        max_length=255, unique=True, help_text="Vector store ID"
    )

    created_at = models.DateTimeField(auto_now_add=True)
    uud = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)

    class Meta:
        ordering = ["document", "chunk_index"]
        unique_together = ["document", "chunk_index"]

    def __str__(self):
        return f"Chunk {self.chunk_index} - {self.document.title}"


class ChatSession(models.Model):
    """Model to store chat sessions"""

    user = models.ForeignKey(
        CustomUser,
        related_name="chat_sessions",
        on_delete=models.CASCADE,
    )
    document = models.ForeignKey(
        Document,
        related_name="chat_sessions",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        help_text="Document being discussed (optional for general chat)",
    )
    title = models.CharField(
        max_length=255, blank=True, help_text="Auto-generated or user-provided title"
    )
    is_active = models.BooleanField(
        default=True, help_text="Whether session is still active"
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    last_message_at = models.DateTimeField(null=True, blank=True)
    uud = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)

    class Meta:
        ordering = ["-updated_at"]

    def __str__(self):
        return f"Chat: {self.title or 'Untitled'} - {self.user.email}"

    def get_message_count(self):
        return self.messages.count()


class ChatMessage(models.Model):
    """Model to store individual chat messages"""

    MESSAGE_TYPES = [
        ("user", "User Message"),
        ("assistant", "Assistant Response"),
        ("system", "System Message"),
    ]

    session = models.ForeignKey(
        ChatSession,
        related_name="messages",
        on_delete=models.CASCADE,
    )
    message_type = models.CharField(
        max_length=10, choices=MESSAGE_TYPES, default="user"
    )
    content = models.TextField()

    # For assistant messages, store context info
    sources = models.JSONField(
        default=list, blank=True, help_text="List of source documents/chunks used"
    )
    confidence_score = models.FloatField(
        null=True, blank=True, help_text="Confidence score of the response"
    )

    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    tokens_used = models.IntegerField(
        null=True, blank=True, help_text="Number of tokens used in this message"
    )
    uud = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self):
        return f"{self.message_type}: {self.content[:50]}..."


class ProcessingQueue(models.Model):
    """Model to track document processing queue"""

    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("processing", "Processing"),
        ("completed", "Completed"),
        ("failed", "Failed"),
    ]

    document = models.OneToOneField(
        Document, on_delete=models.CASCADE, related_name="processing_queue"
    )
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")
    error_message = models.TextField(blank=True, null=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    uud = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"Processing: {self.document.title} - {self.status}"


class GenericFile(models.Model):
    """Model for admin-uploaded generic files for general knowledge"""

    FILE_TYPE_CHOICES = [
        ("pdf", "PDF"),
        ("docx", "Word Document"),
        ("txt", "Text File"),
        ("md", "Markdown"),
        ("json", "JSON"),
        ("csv", "CSV"),
    ]

    title = models.CharField(max_length=255)
    description = models.TextField(blank=True, null=True)
    file = models.FileField(
        upload_to="generic_files/",
        validators=[
            FileExtensionValidator(
                allowed_extensions=["pdf", "docx", "txt", "md", "json", "csv"]
            )
        ],
    )
    file_type = models.CharField(max_length=10, choices=FILE_TYPE_CHOICES)
    file_size = models.BigIntegerField(default=0)
    file_hash = models.CharField(max_length=64, unique=True)
    is_processed = models.BooleanField(default=False)
    processing_error = models.TextField(blank=True, null=True)
    index_name = models.CharField(max_length=255, blank=True, null=True)
    processed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    uud = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)

    class Meta:
        db_table = "aimy_generic_file"
        verbose_name = "Generic File"
        verbose_name_plural = "Generic Files"
        ordering = ["-created_at"]

    def __str__(self):
        return self.title

    def save(self, *args, **kwargs):
        if not self.file_hash:
            self.file_hash = self._calculate_file_hash()
        if not self.file_size:
            self.file_size = self.file.size if self.file else 0
        if not self.file_type:
            self.file_type = (
                os.path.splitext(self.file.name)[1].lower().replace(".", "")
                if self.file
                else ""
            )
        if not self.index_name:
            self.index_name = f"generic_{self.file_hash[:16]}"
        super().save(*args, **kwargs)

    def _calculate_file_hash(self):
        """Calculate SHA256 hash of file content"""
        if not self.file:
            return ""

        hash_sha256 = hashlib.sha256()
        for chunk in self.file.chunks():
            hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def get_file_path(self):
        """Get absolute file path"""
        if not self.file:
            return None
        try:
            return self.file.path
        except Exception as e:
            from loguru import logger

            logger.error(
                f"Error getting file path for generic file {self.id}: {str(e)}"
            )
            return None

    def is_file_accessible(self):
        """Check if the file is accessible"""
        file_path = self.get_file_path()
        if not file_path:
            return False
        return os.path.exists(file_path) and os.path.isfile(file_path)


class GenericFileChunk(models.Model):
    """Model to store generic file chunks for vector search"""

    generic_file = models.ForeignKey(
        GenericFile,
        related_name="chunks",
        on_delete=models.CASCADE,
    )
    content = models.TextField(help_text="Text content of the chunk")
    chunk_index = models.IntegerField(help_text="Order of chunk in file")
    page_number = models.IntegerField(
        null=True, blank=True, help_text="Page number if applicable"
    )

    # Vector store metadata
    vector_id = models.CharField(
        max_length=255, unique=True, help_text="Vector store ID"
    )

    created_at = models.DateTimeField(auto_now_add=True)
    uud = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)

    class Meta:
        ordering = ["generic_file", "chunk_index"]
        unique_together = ["generic_file", "chunk_index"]
        db_table = "aimy_generic_file_chunk"
        verbose_name = "Generic File Chunk"
        verbose_name_plural = "Generic File Chunks"

    def __str__(self):
        return f"Chunk {self.chunk_index} of {self.generic_file.title}"
