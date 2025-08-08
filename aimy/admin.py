from django.contrib import admin
from unfold.admin import ModelAdmin
from .models import Document, DocumentChunk, ChatSession, ChatMessage, ProcessingQueue, GenericFile, GenericFileChunk


@admin.register(Document)
class DocumentAdmin(ModelAdmin):
    list_display = [
        "title",
        "user",
        "file_type",
        "file_size",
        "is_processed",
        "created_at",
        "processed_at",
    ]
    list_filter = [
        "is_processed",
        "file_type",
        "created_at",
        "processed_at",
        "user__email",
    ]
    search_fields = ["title", "user__email", "user__first_name", "user__last_name"]
    readonly_fields = [
        "file_hash",
        "file_size",
        "file_type",
        "index_name",
        "created_at",
        "updated_at",
        "processed_at",
    ]
    ordering = ["-created_at"]


@admin.register(DocumentChunk)
class DocumentChunkAdmin(ModelAdmin):
    list_display = [
        "document",
        "chunk_index",
        "page_number",
        "content_preview",
        "created_at",
    ]
    list_filter = ["document", "page_number", "created_at"]
    search_fields = ["content", "document__title"]
    readonly_fields = ["vector_id", "created_at"]
    ordering = ["document", "chunk_index"]

    def content_preview(self, obj):
        return obj.content[:100] + "..." if len(obj.content) > 100 else obj.content

    content_preview.short_description = "Content Preview"


@admin.register(ChatSession)
class ChatSessionAdmin(ModelAdmin):
    list_display = [
        "title",
        "user",
        "document",
        "message_count",
        "is_active",
        "created_at",
        "last_message_at",
    ]
    list_filter = [
        "is_active",
        "created_at",
        "last_message_at",
        "user__email",
        "document__title",
    ]
    search_fields = [
        "title",
        "user__email",
        "user__first_name",
        "user__last_name",
        "document__title",
    ]
    readonly_fields = ["created_at", "updated_at", "last_message_at"]
    ordering = ["-updated_at"]

    def message_count(self, obj):
        return obj.get_message_count()

    message_count.short_description = "Messages"


@admin.register(ChatMessage)
class ChatMessageAdmin(ModelAdmin):
    list_display = [
        "session",
        "message_type",
        "content_preview",
        "confidence_score",
        "tokens_used",
        "created_at",
    ]
    list_filter = [
        "message_type",
        "created_at",
        "session__user__email",
        "session__document__title",
    ]
    search_fields = ["content", "session__title", "session__user__email"]
    readonly_fields = ["created_at"]
    ordering = ["-created_at"]

    def content_preview(self, obj):
        return obj.content[:100] + "..." if len(obj.content) > 100 else obj.content

    content_preview.short_description = "Content Preview"


@admin.register(ProcessingQueue)
class ProcessingQueueAdmin(ModelAdmin):
    list_display = ["document", "status", "started_at", "completed_at", "created_at"]
    list_filter = [
        "status",
        "created_at",
        "started_at",
        "completed_at",
        "document__user__email",
    ]
    search_fields = ["document__title", "document__user__email", "error_message"]
    readonly_fields = ["created_at", "updated_at"]
    ordering = ["-created_at"]


@admin.register(GenericFile)
class GenericFileAdmin(ModelAdmin):
    list_display = ["title", "file_type", "file_size", "is_processed", "created_at"]
    list_filter = ["file_type", "is_processed", "created_at"]
    search_fields = ["title", "description"]
    readonly_fields = ["created_at", "updated_at", "processed_at"]
    ordering = ["-created_at"]


@admin.register(GenericFileChunk)
class GenericFileChunkAdmin(ModelAdmin):
    list_display = ["generic_file", "chunk_index", "page_number", "content_preview", "created_at"]
    list_filter = ["generic_file", "page_number", "created_at"]
    search_fields = ["content", "generic_file__title"]
    readonly_fields = ["vector_id", "created_at"]
    ordering = ["generic_file", "chunk_index"]

    def content_preview(self, obj):
        return obj.content[:100] + "..." if len(obj.content) > 100 else obj.content