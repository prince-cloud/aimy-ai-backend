from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.parsers import MultiPartParser, FormParser
from django.utils import timezone

# Local imports
from .models import Document, DocumentChunk, ChatSession, ProcessingQueue, GenericFile
from .serializers import (
    DocumentSerializer,
    DocumentUploadSerializer,
    ChatSessionSerializer,
    ChatMessageSerializer,
    AskQuestionSerializer,
    ProcessingQueueSerializer,
    DocumentSearchSerializer,
    GenericFileSerializer,
    GenericFileUploadSerializer,
)
from .services import DocumentProcessingService, ChatService
from loguru import logger

# Initialize services
document_service = DocumentProcessingService()
chat_service = ChatService()


class DocumentUploadView(APIView):
    """API view for uploading documents"""

    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    serializer_class = DocumentUploadSerializer

    def post(self, request):
        """Upload a document"""
        try:
            serializer = DocumentUploadSerializer(
                data=request.data, context={"request": request}
            )

            if serializer.is_valid():
                # Create document (duplicate check handled in serializer)
                document = serializer.save()

                # Add to processing queue only if document is not already processed
                if not document.is_processed:
                    ProcessingQueue.objects.get_or_create(
                        document=document, defaults={"status": "pending"}
                    )

                # Process document asynchronously (in production, use Celery)
                try:
                    # Only process if document is not already processed
                    if not document.is_processed:
                        success = document_service.process_document(document)
                        if not success:
                            return Response(
                                {
                                    "status": "error",
                                    "message": "Failed to process document",
                                    "error": document.processing_error,
                                },
                                status=status.HTTP_400_BAD_REQUEST,
                            )
                    else:
                        logger.info(
                            f"Document {document.id} already processed, skipping processing"
                        )
                except Exception as e:
                    logger.error(f"Error processing document: {str(e)}")
                    # Update document with error
                    document.processing_error = str(e)
                    document.save()

                    # Update queue status
                    try:
                        queue_item = ProcessingQueue.objects.get(document=document)
                        queue_item.status = "failed"
                        queue_item.error_message = str(e)
                        queue_item.completed_at = timezone.now()
                        queue_item.save()
                    except ProcessingQueue.DoesNotExist:
                        pass

                    return Response(
                        {
                            "status": "error",
                            "message": "Error processing document",
                            "error": str(e),
                        },
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    )

                return Response(
                    {
                        "status": "success",
                        "message": "Document uploaded and processed successfully",
                        "document": DocumentSerializer(
                            document, context={"request": request}
                        ).data,
                    },
                    status=status.HTTP_201_CREATED,
                )

            return Response(
                {
                    "status": "error",
                    "message": "Invalid data",
                    "errors": serializer.errors,
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        except Exception as e:
            logger.error(f"Error uploading document: {str(e)}")
            return Response(
                {
                    "status": "error",
                    "message": "Internal server error",
                    "error": str(e),
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    def get(self, request):
        """Get user's documents"""
        try:
            documents = Document.objects.filter(user=request.user).order_by(
                "-created_at"
            )
            serializer = DocumentSerializer(
                documents, many=True, context={"request": request}
            )

            return Response(
                {"status": "success", "documents": serializer.data},
                status=status.HTTP_200_OK,
            )

        except Exception as e:
            logger.error(f"Error fetching documents: {str(e)}")
            return Response(
                {
                    "status": "error",
                    "message": "Error fetching documents",
                    "error": str(e),
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class DocumentDetailView(APIView):
    """API view for document details"""

    permission_classes = [IsAuthenticated]
    serializer_class = DocumentSerializer

    def get(self, request, document_id):
        """Get document details"""
        try:
            document = Document.objects.get(id=document_id, user=request.user)
            serializer = DocumentSerializer(document, context={"request": request})

            return Response(
                {"status": "success", "document": serializer.data},
                status=status.HTTP_200_OK,
            )

        except Document.DoesNotExist:
            return Response(
                {"status": "error", "message": "Document not found"},
                status=status.HTTP_404_NOT_FOUND,
            )
        except Exception as e:
            logger.error(f"Error fetching document: {str(e)}")
            return Response(
                {
                    "status": "error",
                    "message": "Error fetching document",
                    "error": str(e),
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    def delete(self, request, document_id):
        """Delete document"""
        try:
            document = Document.objects.get(id=document_id, user=request.user)
            document.delete()

            return Response(
                {"status": "success", "message": "Document deleted successfully"},
                status=status.HTTP_200_OK,
            )

        except Document.DoesNotExist:
            return Response(
                {"status": "error", "message": "Document not found"},
                status=status.HTTP_404_NOT_FOUND,
            )
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            return Response(
                {
                    "status": "error",
                    "message": "Error deleting document",
                    "error": str(e),
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class AskQuestionView(APIView):
    """API view for asking questions"""

    permission_classes = [IsAuthenticated]
    serializer_class = AskQuestionSerializer

    def post(self, request):
        """Ask a question"""
        try:
            serializer = AskQuestionSerializer(data=request.data)

            if serializer.is_valid():
                question = serializer.validated_data["question"]
                session_id = serializer.validated_data.get("session_id")
                document_id = serializer.validated_data.get("document_id")
                max_results = serializer.validated_data.get("max_results", 5)
                temperature = serializer.validated_data.get("temperature", 0.7)

                # Ask question using chat service
                response = chat_service.ask_question(
                    question=question,
                    user_id=request.user.id,
                    session_id=session_id,
                    document_id=document_id,
                    max_results=max_results,
                    temperature=temperature,
                )

                return Response(
                    {"status": "success", "data": response}, status=status.HTTP_200_OK
                )

            return Response(
                {
                    "status": "error",
                    "message": "Invalid data",
                    "errors": serializer.errors,
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        except Exception as e:
            logger.error(f"Error asking question: {str(e)}")
            return Response(
                {
                    "status": "error",
                    "message": "Error processing question",
                    "error": str(e),
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class ChatSessionView(APIView):
    """API view for chat sessions"""

    permission_classes = [IsAuthenticated]
    serializer_class = ChatSessionSerializer

    def get(self, request):
        """Get user's chat sessions"""
        try:
            sessions = ChatSession.objects.filter(user=request.user).order_by(
                "-updated_at"
            )
            serializer = ChatSessionSerializer(sessions, many=True)

            return Response(
                {"status": "success", "sessions": serializer.data},
                status=status.HTTP_200_OK,
            )

        except Exception as e:
            logger.error(f"Error fetching chat sessions: {str(e)}")
            return Response(
                {
                    "status": "error",
                    "message": "Error fetching chat sessions",
                    "error": str(e),
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    def post(self, request):
        """Create new chat session"""
        try:
            serializer = ChatSessionSerializer(data=request.data)

            if serializer.is_valid():
                session = serializer.save(user=request.user)

                return Response(
                    {
                        "status": "success",
                        "message": "Chat session created",
                        "session": ChatSessionSerializer(session).data,
                    },
                    status=status.HTTP_201_CREATED,
                )

            return Response(
                {
                    "status": "error",
                    "message": "Invalid data",
                    "errors": serializer.errors,
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        except Exception as e:
            logger.error(f"Error creating chat session: {str(e)}")
            return Response(
                {
                    "status": "error",
                    "message": "Error creating chat session",
                    "error": str(e),
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class ChatSessionDetailView(APIView):
    """API view for specific chat session"""

    permission_classes = [IsAuthenticated]
    serializer_class = ChatSessionSerializer

    def get(self, request, session_id):
        """Get chat session details"""
        try:
            session = ChatSession.objects.get(id=session_id, user=request.user)
            serializer = ChatSessionSerializer(session)

            return Response(
                {"status": "success", "session": serializer.data},
                status=status.HTTP_200_OK,
            )

        except ChatSession.DoesNotExist:
            return Response(
                {"status": "error", "message": "Chat session not found"},
                status=status.HTTP_404_NOT_FOUND,
            )
        except Exception as e:
            logger.error(f"Error fetching chat session: {str(e)}")
            return Response(
                {
                    "status": "error",
                    "message": "Error fetching chat session",
                    "error": str(e),
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    def delete(self, request, session_id):
        """Delete chat session"""
        try:
            session = ChatSession.objects.get(id=session_id, user=request.user)
            session.delete()

            return Response(
                {"status": "success", "message": "Chat session deleted successfully"},
                status=status.HTTP_200_OK,
            )

        except ChatSession.DoesNotExist:
            return Response(
                {"status": "error", "message": "Chat session not found"},
                status=status.HTTP_404_NOT_FOUND,
            )
        except Exception as e:
            logger.error(f"Error deleting chat session: {str(e)}")
            return Response(
                {
                    "status": "error",
                    "message": "Error deleting chat session",
                    "error": str(e),
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class ChatSessionMessagesView(APIView):
    """API view for fetching chat session messages"""

    permission_classes = [IsAuthenticated]
    serializer_class = ChatMessageSerializer

    def get(self, request, session_id):
        """Get chat session messages"""
        try:
            session = ChatSession.objects.get(id=session_id, user=request.user)
            messages = session.messages.all().order_by("created_at")
            serializer = ChatMessageSerializer(messages, many=True)

            return Response(
                {
                    "status": "success",
                    "session_id": session_id,
                    "messages": serializer.data,
                },
                status=status.HTTP_200_OK,
            )

        except ChatSession.DoesNotExist:
            return Response(
                {"status": "error", "message": "Chat session not found"},
                status=status.HTTP_404_NOT_FOUND,
            )
        except Exception as e:
            logger.error(f"Error fetching chat session messages: {str(e)}")
            return Response(
                {
                    "status": "error",
                    "message": "Error fetching chat session messages",
                    "error": str(e),
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class DocumentSearchView(APIView):
    """API view for searching documents"""

    permission_classes = [IsAuthenticated]
    serializer_class = DocumentSearchSerializer

    def post(self, request):
        """Search documents"""
        try:
            serializer = DocumentSearchSerializer(data=request.data)

            if serializer.is_valid():
                query = serializer.validated_data["query"]
                document_id = serializer.validated_data.get("document_id")
                max_results = serializer.validated_data.get("max_results", 5)

                # Get vector stores
                vector_stores = chat_service._get_vector_stores(
                    request.user.id, document_id
                )

                if not vector_stores:
                    return Response(
                        {
                            "status": "error",
                            "message": "No documents available for search",
                        },
                        status=status.HTTP_400_BAD_REQUEST,
                    )

                # Search documents
                results = chat_service._search_documents(
                    query, vector_stores, max_results
                )

                # Format results
                formatted_results = []
                for result in results:
                    formatted_results.append(
                        {
                            "content": result["content"],
                            "score": result["score"],
                            "document_title": result["metadata"].get(
                                "source", "Unknown"
                            ),
                            "chunk_index": result["metadata"].get("chunk_index"),
                            "page_number": result["metadata"].get("page_number"),
                        }
                    )

                return Response(
                    {"status": "success", "results": formatted_results},
                    status=status.HTTP_200_OK,
                )

            return Response(
                {
                    "status": "error",
                    "message": "Invalid data",
                    "errors": serializer.errors,
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return Response(
                {
                    "status": "error",
                    "message": "Error searching documents",
                    "error": str(e),
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class ProcessingQueueView(APIView):
    """API view for processing queue status"""

    permission_classes = [IsAuthenticated]
    serializer_class = ProcessingQueueSerializer

    def get(self, request):
        """Get processing queue status for user's documents"""
        try:
            queue_items = ProcessingQueue.objects.filter(
                document__user=request.user
            ).order_by("-created_at")

            serializer = ProcessingQueueSerializer(queue_items, many=True)

            return Response(
                {"status": "success", "queue_items": serializer.data},
                status=status.HTTP_200_OK,
            )

        except Exception as e:
            logger.error(f"Error fetching processing queue: {str(e)}")
            return Response(
                {
                    "status": "error",
                    "message": "Error fetching processing queue",
                    "error": str(e),
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class ReprocessDocumentView(APIView):
    """API view for reprocessing documents"""

    permission_classes = [IsAuthenticated]
    serializer_class = DocumentSerializer

    def post(self, request, document_id):
        """Reprocess a document"""
        try:
            document = Document.objects.get(id=document_id, user=request.user)

            # Reset processing status
            document.is_processed = False
            document.processing_error = None
            document.processed_at = None
            document.save()

            # Clear existing chunks
            DocumentChunk.objects.filter(document=document).delete()

            # Clear processing queue
            ProcessingQueue.objects.filter(document=document).delete()

            # Add to processing queue
            ProcessingQueue.objects.create(document=document, status="pending")

            # Process document
            success = document_service.process_document(document)

            if success:
                return Response(
                    {"status": "success", "message": "Document reprocessing started"},
                    status=status.HTTP_200_OK,
                )
            else:
                return Response(
                    {
                        "status": "error",
                        "message": "Failed to reprocess document",
                        "error": document.processing_error,
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )

        except Document.DoesNotExist:
            return Response(
                {"status": "error", "message": "Document not found"},
                status=status.HTTP_404_NOT_FOUND,
            )
        except Exception as e:
            logger.error(f"Error reprocessing document: {str(e)}")
            return Response(
                {
                    "status": "error",
                    "message": "Error reprocessing document",
                    "error": str(e),
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class GenericFileUploadView(APIView):
    """API view for uploading generic files (admin only)"""

    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]
    serializer_class = GenericFileUploadSerializer

    def post(self, request):
        """Upload a generic file (admin only)"""
        # Check if user is admin
        if not request.user.is_staff:
            return Response(
                {
                    "status": "error",
                    "message": "Only administrators can upload generic files",
                },
                status=status.HTTP_403_FORBIDDEN,
            )

        try:
            serializer = GenericFileUploadSerializer(
                data=request.data, context={"request": request}
            )

            if serializer.is_valid():
                # Create generic file (duplicate check handled in serializer)
                generic_file = serializer.save()

                return Response(
                    {
                        "status": "success",
                        "message": "Generic file uploaded and processed successfully",
                        "file": GenericFileSerializer(
                            generic_file, context={"request": request}
                        ).data,
                    },
                    status=status.HTTP_201_CREATED,
                )

            return Response(
                {
                    "status": "error",
                    "message": "Invalid data",
                    "errors": serializer.errors,
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        except Exception as e:
            logger.error(f"Error uploading generic file: {str(e)}")
            return Response(
                {
                    "status": "error",
                    "message": "Internal server error",
                    "error": str(e),
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    def get(self, request):
        """Get all generic files (admin only)"""
        # Check if user is admin
        if not request.user.is_staff:
            return Response(
                {
                    "status": "error",
                    "message": "Only administrators can view generic files",
                },
                status=status.HTTP_403_FORBIDDEN,
            )

        try:
            generic_files = GenericFile.objects.all().order_by("-created_at")
            serializer = GenericFileSerializer(
                generic_files, many=True, context={"request": request}
            )

            return Response(
                {"status": "success", "files": serializer.data},
                status=status.HTTP_200_OK,
            )

        except Exception as e:
            logger.error(f"Error fetching generic files: {str(e)}")
            return Response(
                {
                    "status": "error",
                    "message": "Error fetching generic files",
                    "error": str(e),
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class GenericFileDetailView(APIView):
    """API view for generic file details (admin only)"""

    permission_classes = [IsAuthenticated]
    serializer_class = GenericFileSerializer

    def get(self, request, file_id):
        """Get generic file details (admin only)"""
        # Check if user is admin
        if not request.user.is_staff:
            return Response(
                {
                    "status": "error",
                    "message": "Only administrators can view generic file details",
                },
                status=status.HTTP_403_FORBIDDEN,
            )

        try:
            generic_file = GenericFile.objects.get(id=file_id)
            serializer = GenericFileSerializer(
                generic_file, context={"request": request}
            )

            return Response(
                {"status": "success", "file": serializer.data},
                status=status.HTTP_200_OK,
            )

        except GenericFile.DoesNotExist:
            return Response(
                {"status": "error", "message": "Generic file not found"},
                status=status.HTTP_404_NOT_FOUND,
            )
        except Exception as e:
            logger.error(f"Error fetching generic file: {str(e)}")
            return Response(
                {
                    "status": "error",
                    "message": "Error fetching generic file",
                    "error": str(e),
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    def delete(self, request, file_id):
        """Delete generic file (admin only)"""
        # Check if user is admin
        if not request.user.is_staff:
            return Response(
                {
                    "status": "error",
                    "message": "Only administrators can delete generic files",
                },
                status=status.HTTP_403_FORBIDDEN,
            )

        try:
            generic_file = GenericFile.objects.get(id=file_id)
            generic_file.delete()

            return Response(
                {"status": "success", "message": "Generic file deleted successfully"},
                status=status.HTTP_200_OK,
            )

        except GenericFile.DoesNotExist:
            return Response(
                {"status": "error", "message": "Generic file not found"},
                status=status.HTTP_404_NOT_FOUND,
            )
        except Exception as e:
            logger.error(f"Error deleting generic file: {str(e)}")
            return Response(
                {
                    "status": "error",
                    "message": "Error deleting generic file",
                    "error": str(e),
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
