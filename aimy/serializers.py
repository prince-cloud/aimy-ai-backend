from rest_framework import serializers
from .models import Document, DocumentChunk, ChatSession, ChatMessage, ProcessingQueue, GenericFile


class DocumentSerializer(serializers.ModelSerializer):
    """Serializer for Document model"""

    user_email = serializers.ReadOnlyField(source="user.email")
    file_url = serializers.SerializerMethodField()
    processing_status = serializers.SerializerMethodField()

    class Meta:
        model = Document
        fields = [
            "id",
            "title",
            "file",
            "file_url",
            "file_size",
            "file_type",
            "is_processed",
            "processing_error",
            "index_name",
            "user_email",
            "created_at",
            "updated_at",
            "processed_at",
            "processing_status",
        ]
        read_only_fields = [
            "file_hash",
            "file_size",
            "file_type",
            "is_processed",
            "processing_error",
            "index_name",
            "created_at",
            "updated_at",
            "processed_at",
        ]

    def get_file_url(self, obj):
        if obj.file:
            request = self.context.get("request")
            if request:
                return request.build_absolute_uri(obj.file.url)
        return None

    def get_processing_status(self, obj):
        if hasattr(obj, "processing_queue"):
            return obj.processing_queue.status
        return "not_queued"


class DocumentChunkSerializer(serializers.ModelSerializer):
    """Serializer for DocumentChunk model"""

    class Meta:
        model = DocumentChunk
        fields = [
            "id",
            "content",
            "chunk_index",
            "page_number",
            "vector_id",
            "created_at",
        ]
        read_only_fields = ["vector_id", "created_at"]


class ChatMessageSerializer(serializers.ModelSerializer):
    """Serializer for ChatMessage model"""

    class Meta:
        model = ChatMessage
        fields = [
            "id",
            "message_type",
            "content",
            "confidence_score",
            "created_at",
            "tokens_used",
        ]
        read_only_fields = ["confidence_score", "tokens_used"]


class ChatSessionSerializer(serializers.ModelSerializer):
    """Serializer for ChatSession model"""

    message_count = serializers.ReadOnlyField(source="get_message_count")
    document_title = serializers.ReadOnlyField(source="document.title")

    class Meta:
        model = ChatSession
        fields = [
            "id",
            "title",
            "document",
            "document_title",
            "is_active",
            "created_at",
            "updated_at",
            "last_message_at",
            "message_count",
        ]
        read_only_fields = ["created_at", "updated_at", "last_message_at"]


class ChatSessionCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating new chat sessions"""

    class Meta:
        model = ChatSession
        fields = ["title", "document"]


class ChatMessageCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating new chat messages"""

    class Meta:
        model = ChatMessage
        fields = ["content"]


class DocumentUploadSerializer(serializers.ModelSerializer):
    """Serializer for document upload"""

    class Meta:
        model = Document
        fields = ["title", "file"]

    def validate_file(self, value):
        """Validate uploaded file"""
        # Check file size (max 10MB)
        if value.size > 10 * 1024 * 1024:
            raise serializers.ValidationError("File size must be less than 10MB")

        # Check file type
        allowed_types = ["pdf", "docx", "txt"]
        file_extension = value.name.split(".")[-1].lower()
        if file_extension not in allowed_types:
            raise serializers.ValidationError(
                f"File type must be one of: {', '.join(allowed_types)}"
            )

        return value

    def create(self, validated_data):
        """Create document and add to processing queue"""
        user = self.context["request"].user
        validated_data["user"] = user

        # Check if document with same hash already exists BEFORE creating
        file = self.context["request"].FILES.get("file")
        if file:
            # Calculate file hash
            import hashlib

            hash_sha256 = hashlib.sha256()
            for chunk in file.chunks():
                hash_sha256.update(chunk)
            file_hash = hash_sha256.hexdigest()

            # Check if document with same hash already exists
            existing_doc = Document.objects.filter(
                file_hash=file_hash, user=user
            ).first()

            if existing_doc:
                return existing_doc

        # Create document only if it doesn't exist
        document = Document.objects.create(**validated_data)
        return document


class ProcessingQueueSerializer(serializers.ModelSerializer):
    """Serializer for ProcessingQueue model"""

    document_title = serializers.ReadOnlyField(source="document.title")

    class Meta:
        model = ProcessingQueue
        fields = [
            "id",
            "status",
            "error_message",
            "started_at",
            "completed_at",
            "created_at",
            "updated_at",
            "document_title",
        ]
        read_only_fields = ["started_at", "completed_at", "created_at", "updated_at"]


class AskQuestionSerializer(serializers.Serializer):
    """Serializer for asking questions"""

    question = serializers.CharField(max_length=1000)
    session_id = serializers.IntegerField(
        required=False, help_text="Chat session ID (optional)"
    )
    document_id = serializers.IntegerField(
        required=False, help_text="Document ID to search in (optional)"
    )
    max_results = serializers.IntegerField(default=5, min_value=1, max_value=20)
    temperature = serializers.FloatField(default=0.7, min_value=0.0, max_value=2.0)


class QuestionResponseSerializer(serializers.Serializer):
    """Serializer for question responses"""

    answer = serializers.CharField()
    confidence_score = serializers.FloatField()
    session_id = serializers.IntegerField()
    message_id = serializers.IntegerField()
    tokens_used = serializers.IntegerField()


class DocumentSearchSerializer(serializers.Serializer):
    """Serializer for document search"""

    query = serializers.CharField(max_length=500)
    document_id = serializers.IntegerField(required=False)
    max_results = serializers.IntegerField(default=5, min_value=1, max_value=20)


class SearchResultSerializer(serializers.Serializer):
    """Serializer for search results"""

    content = serializers.CharField()
    score = serializers.FloatField()
    document_title = serializers.CharField()
    chunk_index = serializers.IntegerField()
    page_number = serializers.IntegerField(required=False)


class GenericFileSerializer(serializers.ModelSerializer):
    """Serializer for GenericFile model"""
    
    file_url = serializers.SerializerMethodField()
    processing_status = serializers.SerializerMethodField()
    
    class Meta:
        model = GenericFile
        fields = [
            'id', 'title', 'description', 'file', 'file_url', 'file_size', 
            'file_type', 'is_processed', 'processing_error', 'index_name',
            'created_at', 'updated_at', 'processed_at', 'processing_status'
        ]
        read_only_fields = [
            'file_hash', 'file_size', 'file_type', 'is_processed',
            'processing_error', 'index_name', 'created_at', 'updated_at',
            'processed_at'
        ]
    
    def get_file_url(self, obj):
        if obj.file:
            request = self.context.get('request')
            if request:
                return request.build_absolute_uri(obj.file.url)
        return None
    
    def get_processing_status(self, obj):
        return 'completed' if obj.is_processed else 'failed' if obj.processing_error else 'pending'


class GenericFileUploadSerializer(serializers.ModelSerializer):
    """Serializer for generic file upload (admin only)"""
    
    class Meta:
        model = GenericFile
        fields = ['title', 'description', 'file']
    
    def validate_file(self, value):
        # Check file size (20MB limit for generic files)
        if value.size > 20 * 1024 * 1024:
            raise serializers.ValidationError("File size must be less than 20MB")
        
        # Check file type
        allowed_types = ['pdf', 'docx', 'txt', 'md', 'json', 'csv']
        file_extension = value.name.split('.')[-1].lower()
        if file_extension not in allowed_types:
            raise serializers.ValidationError(
                f"File type must be one of: {', '.join(allowed_types)}"
            )
        
        return value
    
    def create(self, validated_data):
        """Create generic file and process it"""
        # Check if file with same hash already exists
        file = self.context['request'].FILES.get('file')
        if file:
            import hashlib
            hash_sha256 = hashlib.sha256()
            for chunk in file.chunks():
                hash_sha256.update(chunk)
            file_hash = hash_sha256.hexdigest()
            
            # Check if file with same hash already exists
            existing_file = GenericFile.objects.filter(file_hash=file_hash).first()
            
            if existing_file:
                return existing_file
        
        # Create generic file
        generic_file = GenericFile.objects.create(**validated_data)
        
        # Process the file (in production, use Celery)
        try:
            from .services import DocumentProcessingService
            document_service = DocumentProcessingService()
            success = document_service.process_generic_file(generic_file)
            
            if not success:
                generic_file.processing_error = "Failed to process file"
                generic_file.save()
        except Exception as e:
            generic_file.processing_error = str(e)
            generic_file.save()
        
        return generic_file
