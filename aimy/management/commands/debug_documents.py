from django.core.management.base import BaseCommand
from aimy.models import Document
from aimy.services import DocumentProcessingService, ChatService
from accounts.models import CustomUser


class Command(BaseCommand):
    help = "Debug and repair document vector stores"

    def add_arguments(self, parser):
        parser.add_argument(
            "--document-id", type=int, help="Specific document ID to debug"
        )
        parser.add_argument(
            "--user-id", type=int, help="Specific user ID to debug documents for"
        )
        parser.add_argument(
            "--repair",
            action="store_true",
            help="Attempt to repair broken vector stores",
        )
        parser.add_argument(
            "--list", action="store_true", help="List all documents and their status"
        )

    def handle(self, *args, **options):
        doc_service = DocumentProcessingService()
        chat_service = ChatService()

        if options["list"]:
            self.list_documents()
            return

        if options["document_id"]:
            self.debug_specific_document(
                options["document_id"], options["repair"], doc_service, chat_service
            )
        elif options["user_id"]:
            self.debug_user_documents(
                options["user_id"], options["repair"], doc_service, chat_service
            )
        else:
            self.debug_all_documents(options["repair"], doc_service, chat_service)

    def list_documents(self):
        """List all documents and their processing status"""
        self.stdout.write(self.style.SUCCESS("Document Status Report"))
        self.stdout.write("=" * 50)

        documents = Document.objects.all().order_by("user_id", "id")
        for doc in documents:
            status = "✅ PROCESSED" if doc.is_processed else "❌ NOT PROCESSED"
            error = f" (Error: {doc.processing_error})" if doc.processing_error else ""

            self.stdout.write(
                f"ID: {doc.id} | User: {doc.user_id} | Title: {doc.title[:30]} | {status}{error}"
            )

    def debug_specific_document(self, document_id, repair, doc_service, chat_service):
        """Debug a specific document"""
        try:
            document = Document.objects.get(id=document_id)
            self.stdout.write(f"Debugging document {document_id}: {document.title}")

            # Check basic info
            self.stdout.write(f"  User ID: {document.user_id}")
            self.stdout.write(f"  Processed: {document.is_processed}")
            self.stdout.write(f"  Processed at: {document.processed_at}")
            self.stdout.write(f"  Error: {document.processing_error}")

            # Try to load vector store
            if document.is_processed:
                vector_store = chat_service._load_document_vector_store(document)
                if vector_store:
                    self.stdout.write(
                        self.style.SUCCESS("  ✅ Vector store loaded successfully")
                    )

                    # Test search
                    try:
                        results = vector_store.similarity_search("test", k=1)
                        self.stdout.write(
                            f"  ✅ Search test successful: {len(results)} results"
                        )
                    except Exception as e:
                        self.stdout.write(
                            self.style.ERROR(f"  ❌ Search test failed: {e}")
                        )
                else:
                    self.stdout.write(
                        self.style.ERROR("  ❌ Vector store failed to load")
                    )

                    if repair:
                        self.stdout.write("  🔧 Attempting repair...")
                        success = doc_service.verify_and_repair_document_vector_store(
                            document
                        )
                        if success:
                            self.stdout.write(
                                self.style.SUCCESS("  ✅ Repair successful!")
                            )
                        else:
                            self.stdout.write(self.style.ERROR("  ❌ Repair failed"))
            else:
                self.stdout.write("  ⏳ Document not processed yet")

                if repair:
                    self.stdout.write("  🔧 Attempting to process...")
                    success = doc_service.process_document(document)
                    if success:
                        self.stdout.write(
                            self.style.SUCCESS("  ✅ Processing successful!")
                        )
                    else:
                        self.stdout.write(self.style.ERROR("  ❌ Processing failed"))

        except Document.DoesNotExist:
            self.stdout.write(self.style.ERROR(f"Document {document_id} not found"))

    def debug_user_documents(self, user_id, repair, doc_service, chat_service):
        """Debug all documents for a specific user"""
        try:
            user = CustomUser.objects.get(id=user_id)
            self.stdout.write(f"Debugging documents for user {user_id}: {user.email}")

            documents = Document.objects.filter(user_id=user_id)
            self.stdout.write(f"Found {documents.count()} documents")

            for doc in documents:
                self.stdout.write(f"\n--- Document {doc.id}: {doc.title} ---")
                self.debug_specific_document(doc.id, repair, doc_service, chat_service)

        except CustomUser.DoesNotExist:
            self.stdout.write(self.style.ERROR(f"User {user_id} not found"))

    def debug_all_documents(self, repair, doc_service, chat_service):
        """Debug all documents in the system"""
        self.stdout.write("Debugging all documents...")

        documents = Document.objects.all()
        total = documents.count()
        processed = documents.filter(is_processed=True).count()

        self.stdout.write(f"Total documents: {total}")
        self.stdout.write(f"Processed documents: {processed}")
        self.stdout.write(f"Unprocessed documents: {total - processed}")

        if repair:
            self.stdout.write("\n🔧 Starting repair process...")

            # Check processed documents with potential vector store issues
            for doc in documents.filter(is_processed=True):
                vector_store = chat_service._load_document_vector_store(doc)
                if not vector_store:
                    self.stdout.write(f"  🔧 Repairing document {doc.id}: {doc.title}")
                    success = doc_service.verify_and_repair_document_vector_store(doc)
                    if success:
                        self.stdout.write(
                            self.style.SUCCESS(f"    ✅ Repaired document {doc.id}")
                        )
                    else:
                        self.stdout.write(
                            self.style.ERROR(
                                f"    ❌ Failed to repair document {doc.id}"
                            )
                        )

            # Process unprocessed documents
            for doc in documents.filter(is_processed=False):
                self.stdout.write(f"  🔧 Processing document {doc.id}: {doc.title}")
                success = doc_service.process_document(doc)
                if success:
                    self.stdout.write(
                        self.style.SUCCESS(f"    ✅ Processed document {doc.id}")
                    )
                else:
                    self.stdout.write(
                        self.style.ERROR(f"    ❌ Failed to process document {doc.id}")
                    )
