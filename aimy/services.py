from typing import List, Dict, Any, Optional
from django.conf import settings
from django.utils import timezone
from django.db import transaction

# LangChain imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_redis import RedisVectorStore, RedisConfig
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document as LangchainDocument
from langchain_community.vectorstores import Chroma  # Alternative vector store

# ChromaDB imports
import chromadb

# Local imports
from .models import (
    Document,
    DocumentChunk,
    ProcessingQueue,
    ChatSession,
    ChatMessage,
    GenericFile,
    GenericFileChunk,
    Reminder,
)
from .knust_tools import knust_tools
from loguru import logger

# Configuration
# Get Redis URL from Django settings
REDIS_URL = getattr(settings, "REDIS_URL", "redis://default:default@localhost:6380/0")
OPENAI_API_KEY = getattr(settings, "OPENAI_API_KEY", "")
OPENAI_MODEL = getattr(settings, "OPENAI_MODEL", "gpt-3.5-turbo")
EMBEDDING_MODEL = getattr(settings, "EMBEDDING_MODEL", "text-embedding-3-small")

# Log configuration for debugging
logger.info(f"Redis URL: {REDIS_URL}")
logger.info(f"OpenAI API Key configured: {'Yes' if OPENAI_API_KEY else 'No'}")
logger.info(f"OpenAI Model: {OPENAI_MODEL}")
logger.info(f"Embedding Model: {EMBEDDING_MODEL}")


class DocumentProcessingService:
    """Service for processing and indexing documents"""

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Smaller chunks for better search
            chunk_overlap=100,
            length_function=len,
        )

    def process_document(self, document: Document) -> bool:
        """Process a document and add it to vector store"""
        try:
            # Check if document is already processed
            if document.is_processed:
                logger.info(f"Document {document.id} already processed")
                return True

            # Check for duplicate documents (same file hash)
            if self.handle_duplicate_document(document):
                logger.info(f"Document {document.id} linked to existing vector store")
                return True

            # Update processing status
            queue_item, created = ProcessingQueue.objects.get_or_create(
                document=document, defaults={"status": "processing"}
            )
            if not created:
                queue_item.status = "processing"
                queue_item.started_at = timezone.now()
                queue_item.save()

            # Load document
            try:
                docs = self._load_document(document)
                if not docs:
                    raise Exception("No content extracted from document")
                logger.info(
                    f"Successfully loaded document with {len(docs)} pages/sections"
                )
            except Exception as load_error:
                logger.error(f"Error loading document: {str(load_error)}")
                raise Exception(f"Document loading failed: {str(load_error)}")

            # Split into chunks
            try:
                chunks = self.text_splitter.split_documents(docs)
                if not chunks:
                    raise Exception("No chunks created from document")
                logger.info(f"Split document into {len(chunks)} chunks")
            except Exception as split_error:
                logger.error(f"Error splitting document: {str(split_error)}")
                raise Exception(f"Document splitting failed: {str(split_error)}")

            # Test Redis connection first
            try:
                import redis

                redis_client = redis.from_url(REDIS_URL)
                redis_client.ping()
                logger.info(f"Redis connection successful: {REDIS_URL}")
            except Exception as redis_error:
                logger.error(f"Redis connection failed: {str(redis_error)}")
                # Don't raise here, just log and continue with Chroma

            # Prepare chunks with metadata first
            for i, chunk in enumerate(chunks):
                chunk.metadata.update(
                    {
                        "document_id": document.id,
                        "chunk_index": i,
                        "page_number": chunk.metadata.get("page", None),
                        "source": document.title,
                    }
                )

            # Try to create vector store with chunks - prioritize Redis over Chroma
            vector_store = None
            vector_store_type = None

            # Try Redis first (prioritized over Chroma)
            try:
                logger.info(
                    f"Attempting to create Redis vector store with index: {document.index_name}"
                )
                config = RedisConfig(
                    index_name=document.index_name,
                    redis_url=REDIS_URL,
                    metadata_schema=[
                        {"name": "document_id", "type": "tag"},
                        {"name": "chunk_index", "type": "tag"},
                        {"name": "page_number", "type": "tag"},
                    ],
                )
                vector_store = RedisVectorStore(self.embeddings, config=config)
                vector_store_type = "redis"
                logger.info(
                    f"Redis vector store created successfully for index: {document.index_name}"
                )
            except Exception as redis_error:
                logger.warning(
                    f"Redis vector store creation failed: {str(redis_error)}"
                )
                logger.warning(f"Redis error type: {type(redis_error).__name__}")
                logger.warning(f"Redis error details: {str(redis_error)}")
                import traceback

                logger.warning(f"Redis error traceback: {traceback.format_exc()}")

                # Fallback to Chroma vector store
                try:
                    logger.info(
                        f"Attempting to create Chroma vector store for document: {document.id}"
                    )

                    # Create Chroma client with persistent directory
                    import chromadb

                    client = chromadb.PersistentClient(path="./chroma_db")

                    vector_store = Chroma.from_documents(
                        documents=chunks,
                        embedding=self.embeddings,
                        collection_name=f"doc_{document.file_hash[:16]}",
                        client=client,
                    )
                    vector_store_type = "chroma"
                    logger.info(
                        f"Chroma vector store created successfully for document: {document.id}"
                    )
                except Exception as chroma_error:
                    logger.error(
                        f"Chroma vector store creation failed: {str(chroma_error)}"
                    )
                    logger.error(f"Chroma error type: {type(chroma_error).__name__}")
                    logger.error(f"Chroma error details: {str(chroma_error)}")
                    import traceback

                    logger.error(f"Chroma error traceback: {traceback.format_exc()}")
                    raise Exception(
                        f"Vector store creation error (both Redis and Chroma failed): {str(redis_error)} -> {str(chroma_error)}"
                    )

            if vector_store is None:
                raise Exception("Failed to create any vector store")

            # Add documents to vector store and save chunks
            with transaction.atomic():
                # Clear existing chunks
                DocumentChunk.objects.filter(document=document).delete()

                # Add to vector store and save chunks
                for i, chunk in enumerate(chunks):
                    # Add metadata
                    chunk.metadata.update(
                        {
                            "document_id": document.id,
                            "chunk_index": i,
                            "page_number": chunk.metadata.get("page", None),
                            "source": document.title,
                        }
                    )

                    # Save chunk to database
                    doc_chunk = DocumentChunk.objects.create(
                        document=document,
                        content=chunk.page_content,
                        chunk_index=i,
                        page_number=chunk.metadata.get("page", None),
                        vector_id=f"{document.index_name}_chunk_{i}",
                    )

                    # Update chunk with vector ID
                    chunk.metadata["vector_id"] = doc_chunk.vector_id

                # Add all chunks to vector store (only for Redis, Chroma already has them)
                if vector_store_type == "redis":
                    try:
                        vector_store.add_documents(chunks)
                        logger.info(
                            f"Successfully added {len(chunks)} chunks to Redis vector store"
                        )
                    except Exception as e:
                        logger.error(
                            f"Error adding documents to Redis vector store: {str(e)}"
                        )
                        raise Exception(f"Redis vector store operation error: {str(e)}")
                else:
                    logger.info(
                        f"Chroma vector store already contains {len(chunks)} chunks"
                    )

            # Update document status
            document.is_processed = True
            document.processed_at = timezone.now()
            document.save()

            # Update queue status
            queue_item.status = "completed"
            queue_item.completed_at = timezone.now()
            queue_item.save()

            logger.info(f"Successfully processed document {document.id}")
            return True

        except Exception as e:
            logger.error(f"Error processing document {document.id}: {str(e)}")

            # Update error status
            document.processing_error = str(e)
            document.save()

            if hasattr(queue_item, "id"):
                queue_item.status = "failed"
                queue_item.error_message = str(e)
                queue_item.completed_at = timezone.now()
                queue_item.save()

            return False

    def _load_document(self, document: Document) -> List[LangchainDocument]:
        """Load document based on file type"""
        logger.info(f"Attempting to load document {document.id}")

        # Check if file is accessible
        if not document.is_file_accessible():
            raise Exception(f"File is not accessible for document {document.id}")

        file_path = document.get_file_path()
        logger.info(f"Loading document {document.id} from path: {file_path}")

        file_extension = document.file_type.lower()
        logger.info(f"Loading document {document.id} with file type: {file_extension}")

        try:
            if file_extension == ".pdf":
                loader = PyPDFLoader(file_path)
                logger.info(f"Using PyPDFLoader for PDF file: {file_path}")
            elif file_extension == ".docx":
                loader = Docx2txtLoader(file_path)
                logger.info(f"Using Docx2txtLoader for DOCX file: {file_path}")
            elif file_extension == ".txt":
                loader = TextLoader(file_path)
                logger.info(f"Using TextLoader for TXT file: {file_path}")
            else:
                raise Exception(f"Unsupported file type: {file_extension}")

            logger.info(f"Loading documents from {file_path}...")
            docs = loader.load()
            logger.info(f"Successfully loaded {len(docs)} pages/chunks from document")

            # Clean the text content to remove null bytes and other problematic characters
            cleaned_docs = []
            for doc in docs:
                cleaned_content = self._clean_text_content(doc.page_content)
                cleaned_doc = LangchainDocument(
                    page_content=cleaned_content, metadata=doc.metadata
                )
                cleaned_docs.append(cleaned_doc)

            logger.info(f"Cleaned {len(cleaned_docs)} document pages")

            # Log some content for debugging
            if cleaned_docs:
                first_doc = cleaned_docs[0]
                logger.info(
                    f"First document content preview: {first_doc.page_content[:100]}..."
                )
                logger.info(f"First document metadata: {first_doc.metadata}")

            return cleaned_docs

        except ImportError as e:
            logger.error(f"Missing module for document loading: {str(e)}")
            raise Exception(f"Missing required module: {str(e)}")
        except Exception as e:
            logger.error(f"Error loading document {document.id}: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"File path: {file_path}")
            logger.error(f"File accessible: {document.is_file_accessible()}")
            import traceback

            logger.error(f"Document loading traceback: {traceback.format_exc()}")
            raise Exception(f"Document loading error: {str(e)}")

    def _clean_text_content(self, content: str) -> str:
        """
        Clean text content to remove null bytes and other problematic characters
        that could cause PostgreSQL errors.
        """
        if not content:
            return ""

        try:
            # Remove null bytes (0x00) that cause PostgreSQL errors
            cleaned_content = content.replace("\x00", "")

            # Remove other problematic control characters but keep common ones like \n, \t
            import re

            # Remove control characters except newline (\n), tab (\t), and carriage return (\r)
            cleaned_content = re.sub(
                r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", "", cleaned_content
            )

            # Normalize whitespace - replace multiple consecutive whitespace with single space
            # But preserve line breaks
            cleaned_content = re.sub(r"[ \t]+", " ", cleaned_content)
            cleaned_content = re.sub(r"\n\s*\n", "\n\n", cleaned_content)

            # Remove any leading/trailing whitespace
            cleaned_content = cleaned_content.strip()

            # Ensure the content is valid UTF-8
            cleaned_content = cleaned_content.encode("utf-8", errors="ignore").decode(
                "utf-8"
            )

            logger.debug(
                f"Cleaned text content: original length {len(content)}, cleaned length {len(cleaned_content)}"
            )

            return cleaned_content

        except Exception as e:
            logger.error(f"Error cleaning text content: {e}")
            # Return a safe fallback
            return content.replace("\x00", "") if content else ""

    def verify_and_repair_document_vector_store(self, document: Document) -> bool:
        """
        Verify if a document's vector store exists and is accessible.
        If not, attempt to repair by reprocessing the document.
        """
        try:
            # Check if document is marked as processed
            if not document.is_processed:
                logger.info(f"Document {document.id} is not marked as processed")
                return False

            # Try to load the vector store
            vector_store = self._load_document_vector_store(document)

            if vector_store:
                logger.info(f"Document {document.id} vector store is working correctly")
                return True
            else:
                logger.warning(
                    f"Document {document.id} vector store failed to load, attempting repair"
                )

                # Mark document as not processed to trigger reprocessing
                document.is_processed = False
                document.processed_at = None
                document.processing_error = None
                document.save()

                # Reprocess the document
                success = self.process_document(document)
                if success:
                    logger.info(
                        f"Successfully repaired document {document.id} vector store"
                    )
                    return True
                else:
                    logger.error(
                        f"Failed to repair document {document.id} vector store"
                    )
                    return False

        except Exception as e:
            logger.error(f"Error verifying/repairing document {document.id}: {e}")
            return False

    def check_document_exists(self, file_hash: str, user_id: int) -> Optional[Document]:
        """Check if document with same hash already exists for user"""
        return Document.objects.filter(
            file_hash=file_hash, user_id=user_id, is_processed=True
        ).first()

    def handle_duplicate_document(self, new_document: Document) -> bool:
        """
        Handle a document that might be a duplicate.
        If duplicate exists and is processed, mark new document as processed too.
        Returns True if duplicate was found and handled, False otherwise.
        """
        try:
            # Check if a processed document with same hash already exists for this user
            existing_doc = self.check_document_exists(
                new_document.file_hash, new_document.user_id
            )

            if existing_doc:
                logger.info(
                    f"Found existing processed document {existing_doc.id} with same hash as {new_document.id}"
                )

                # Mark the new document as processed since vector store already exists
                new_document.is_processed = True
                new_document.processed_at = timezone.now()
                new_document.processing_error = None
                new_document.save()

                # Copy chunks from existing document to new document for consistency
                existing_chunks = DocumentChunk.objects.filter(document=existing_doc)
                for chunk in existing_chunks:
                    DocumentChunk.objects.create(
                        document=new_document,
                        content=chunk.content,
                        chunk_index=chunk.chunk_index,
                        page_number=chunk.page_number,
                        vector_id=f"{new_document.index_name}_chunk_{chunk.chunk_index}",
                    )

                logger.info(
                    f"Successfully linked document {new_document.id} to existing vector store (hash: {new_document.file_hash[:16]})"
                )
                return True

            return False

        except Exception as e:
            logger.error(f"Error handling duplicate document {new_document.id}: {e}")
            return False

    def process_generic_file(self, generic_file: GenericFile) -> bool:
        """Process a generic file and add it to vector store"""
        try:
            # Check if file is already processed
            if generic_file.is_processed:
                logger.info(f"Generic file {generic_file.id} already processed")
                return True

            # Load document
            try:
                docs = self._load_generic_file(generic_file)
                if not docs:
                    raise Exception("No content extracted from generic file")
                logger.info(
                    f"Successfully loaded generic file with {len(docs)} pages/sections"
                )
            except Exception as load_error:
                logger.error(f"Error loading generic file: {str(load_error)}")
                raise Exception(f"Generic file loading failed: {str(load_error)}")

            # Split into chunks
            try:
                chunks = self.text_splitter.split_documents(docs)
                if not chunks:
                    raise Exception("No chunks created from generic file")
                logger.info(f"Split generic file into {len(chunks)} chunks")
            except Exception as split_error:
                logger.error(f"Error splitting generic file: {str(split_error)}")
                raise Exception(f"Generic file splitting failed: {str(split_error)}")

            # Test Redis connection first
            try:
                import redis

                redis_client = redis.from_url(REDIS_URL)
                redis_client.ping()
                logger.info(f"Redis connection successful: {REDIS_URL}")
            except Exception as redis_error:
                logger.error(f"Redis connection failed: {str(redis_error)}")

            # Create GenericFileChunk records and prepare chunks with metadata
            for i, chunk in enumerate(chunks):
                # Create chunk record in database
                chunk_record = GenericFileChunk.objects.create(
                    generic_file=generic_file,
                    content=chunk.page_content,
                    chunk_index=i,
                    page_number=chunk.metadata.get("page", None),
                    vector_id=f"generic_{generic_file.id}_{i}",
                )

                # Update chunk metadata
                chunk.metadata.update(
                    {
                        "generic_file_id": generic_file.id,
                        "chunk_index": i,
                        "source": generic_file.title,
                        "file_type": generic_file.file_type,
                        "chunk_id": chunk_record.id,
                    }
                )

            # Try Chroma first (prioritized over Redis), fallback to Redis
            vector_store = None
            try:
                # Try Chroma first
                import chromadb

                client = chromadb.PersistentClient(path="./chroma_db")

                vector_store = Chroma.from_documents(
                    documents=chunks,
                    embedding=self.embeddings,
                    collection_name=f"generic_{generic_file.file_hash[:16]}",
                    client=client,
                )
                logger.info(
                    f"Successfully created Chroma vector store for generic file {generic_file.id}"
                )
            except Exception as chroma_error:
                logger.error(
                    f"Chroma vector store creation failed: {str(chroma_error)}"
                )
                try:
                    # Fallback to Redis
                    vector_store = RedisVectorStore.from_documents(
                        documents=chunks,
                        embedding=self.embeddings,
                        redis_url=REDIS_URL,
                        index_name=generic_file.index_name,
                        key_prefix=f"generic_{generic_file.id}_",
                    )
                    logger.info(
                        f"Successfully created Redis vector store for generic file {generic_file.id}"
                    )
                except Exception as redis_error:
                    logger.error(
                        f"Redis vector store creation failed: {str(redis_error)}"
                    )
                    raise Exception(
                        f"Vector store creation error (both Chroma and Redis failed): {str(chroma_error)} -> {str(redis_error)}"
                    )

            # Update generic file status
            generic_file.is_processed = True
            generic_file.processed_at = timezone.now()
            generic_file.save()

            logger.info(f"Successfully processed generic file {generic_file.id}")
            return True

        except Exception as e:
            logger.error(f"Error processing generic file {generic_file.id}: {str(e)}")
            generic_file.processing_error = str(e)
            generic_file.save()
            return False

    def _load_generic_file(self, generic_file: GenericFile) -> List[LangchainDocument]:
        """Load generic file based on file type"""
        logger.info(f"Attempting to load generic file {generic_file.id}")

        # Check if file is accessible
        if not generic_file.is_file_accessible():
            raise Exception(
                f"File is not accessible for generic file {generic_file.id}"
            )

        file_path = generic_file.get_file_path()
        logger.info(f"Loading generic file {generic_file.id} from path: {file_path}")

        file_extension = generic_file.file_type.lower()

        try:
            if file_extension == "pdf":
                loader = PyPDFLoader(file_path)
            elif file_extension == "docx":
                loader = Docx2txtLoader(file_path)
            elif file_extension in ["txt", "md", "json", "csv"]:
                loader = TextLoader(file_path)
            else:
                raise Exception(f"Unsupported file type: {file_extension}")

            docs = loader.load()
            logger.info(
                f"Successfully loaded {len(docs)} documents from generic file {generic_file.id}"
            )

            # Clean the text content to remove null bytes and other problematic characters
            cleaned_docs = []
            for doc in docs:
                cleaned_content = self._clean_text_content(doc.page_content)
                cleaned_doc = LangchainDocument(
                    page_content=cleaned_content, metadata=doc.metadata
                )
                cleaned_docs.append(cleaned_doc)

            logger.info(f"Cleaned {len(cleaned_docs)} generic file pages")
            return cleaned_docs

        except Exception as e:
            logger.error(f"Error loading generic file {generic_file.id}: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"File path: {file_path}")
            logger.error(f"File accessible: {generic_file.is_file_accessible()}")
            import traceback

            logger.error(f"Generic file loading traceback: {traceback.format_exc()}")
            raise Exception(f"Generic file loading error: {str(e)}")


class ChatService:
    """Service for handling chat functionality"""

    def __init__(self):
        self.llm = ChatOpenAI(
            model=OPENAI_MODEL, openai_api_key=OPENAI_API_KEY, temperature=0.7
        )
        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY
        )
        self.reminder_service = ReminderService()

    def ask_question(
        self,
        question: str,
        user_id: int,
        session_id: Optional[int] = None,
        document_id: Optional[int] = None,
        max_results: int = 5,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """Ask a question and get response with priority system"""
        try:
            # Get or create chat session
            session = self._get_or_create_session(user_id, session_id, document_id)

            # Priority 0: Check if this is a greeting (highest priority, no file lookups)
            greeting_response = self._handle_greeting_request(
                question, session, user_id
            )
            if greeting_response:
                return greeting_response

            # Priority 1: Check if this is a reminder request
            reminder_response = self._handle_reminder_request(
                question, session, user_id
            )
            if reminder_response:
                return reminder_response

            # Priority 2: If document_id is provided, prioritize that document
            if document_id:
                return self._handle_document_specific_question(
                    question, session, user_id, document_id, max_results, temperature
                )

            # Priority 3: If session_id is provided, prioritize that session's document
            if session_id and session.document:
                return self._handle_session_specific_question(
                    question,
                    session,
                    user_id,
                    session.document.id,
                    max_results,
                    temperature,
                )

            # Priority 4: Use generic files (admin-uploaded files) - similar to document-specific approach
            return self._handle_generic_files_question(
                question, session, user_id, max_results, temperature
            )

        except Exception as e:
            logger.error(f"Error asking question: {str(e)}")
            return self._create_error_response(str(e))

    def _handle_greeting_request(
        self, question: str, session: ChatSession, user_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        Handle greeting messages without looking for documents.
        Returns a response if this is a greeting, None otherwise.
        """
        try:
            # Common greeting patterns
            greeting_patterns = [
                # Basic greetings
                r"\b(hi|hello|hey|hiya|howdy)\b",
                r"\b(good\s+(morning|afternoon|evening|day|night))\b",
                r"\b(greetings?)\b",
                r"\b(salutations?)\b",
                # Conversational starters
                r"\b(how\s+(are\s+you|is\s+it\s+going|you\s+doing))\b",
                r"\b(what\'?s\s+up)\b",
                r"\b(how\s+do\s+you\s+do)\b",
                r"\b(nice\s+to\s+(meet|see)\s+you)\b",
                r"\b(pleased\s+to\s+(meet|see)\s+you)\b",
                # Question about the assistant
                r"\b(who\s+are\s+you)\b",
                r"\b(what\s+are\s+you)\b",
                r"\b(introduce\s+yourself)\b",
                r"\b(tell\s+me\s+about\s+yourself)\b",
                # Closing greetings (could be mid or end conversation)
                r"\b(goodbye|bye|farewell|see\s+you|take\s+care)\b",
                r"\b(have\s+a\s+(good|great|nice)\s+(day|night|time))\b",
                r"\b(until\s+next\s+time)\b",
                r"\b(catch\s+you\s+later)\b",
                # Polite expressions
                r"\b(thank\s+you|thanks|much\s+appreciated)\b",
                r"\b(you\'?re\s+(welcome|awesome|great|helpful))\b",
            ]

            import re

            # Clean the question for pattern matching
            question_lower = question.lower().strip()

            # Check if the message matches any greeting pattern
            is_greeting = False
            for pattern in greeting_patterns:
                if re.search(pattern, question_lower, re.IGNORECASE):
                    is_greeting = True
                    break

            # Additional check for very short messages that might be greetings
            if not is_greeting and len(question_lower) <= 20:
                simple_greetings = [
                    "hi",
                    "hello",
                    "hey",
                    "yo",
                    "sup",
                    "morning",
                    "evening",
                    "afternoon",
                    "night",
                    "goodbye",
                    "bye",
                    "thanks",
                    "thank you",
                ]
                for greeting in simple_greetings:
                    if greeting in question_lower:
                        is_greeting = True
                        break

            if not is_greeting:
                return None  # Not a greeting

            # Save the user message first
            user_message = ChatMessage.objects.create(
                session=session, message_type="user", content=question
            )

            # Generate appropriate greeting response
            response_content = self._generate_greeting_response(question_lower, session)

            # Save the assistant response
            assistant_message = ChatMessage.objects.create(
                session=session,
                message_type="assistant",
                content=response_content,
                confidence_score=1.0,  # High confidence for greetings
            )

            # Update session
            session.last_message_at = timezone.now()
            if not session.title or session.title == "New Chat":
                session.title = "Greeting"
            session.save()

            return {
                "success": True,
                "answer": response_content,
                "session_id": session.id,
                "session_title": session.title,
                "message_id": assistant_message.id,
                "confidence_score": 1.0,
                "source_type": "greeting",
                "greeting_detected": True,
            }

        except Exception as e:
            logger.error(f"Error handling greeting: {e}")
            return None

    def _generate_greeting_response(
        self, question_lower: str, session: ChatSession
    ) -> str:
        """Generate appropriate greeting responses"""
        import random
        from datetime import datetime

        # Get user's first name if available
        user_name = session.user.first_name if session.user.first_name else "there"

        # Get current time for time-based greetings
        current_hour = datetime.now().hour

        if any(word in question_lower for word in ["morning", "good morning"]):
            responses = [
                f"Good morning, {user_name}! ☀️ How can I help you today?",
                f"Morning, {user_name}! Hope you're having a great start to your day. What can I assist you with?",
                f"Good morning! ☀️ Ready to tackle the day? How can I support you?",
            ]
        elif any(word in question_lower for word in ["afternoon", "good afternoon"]):
            responses = [
                f"Good afternoon, {user_name}! 🌤️ How's your day going? What can I help you with?",
                f"Afternoon, {user_name}! Hope you're having a productive day. How can I assist you?",
                f"Good afternoon! How can I make your day better?",
            ]
        elif any(
            word in question_lower
            for word in ["evening", "good evening", "night", "good night"]
        ):
            responses = [
                f"Good evening, {user_name}! 🌙 How can I help you tonight?",
                f"Evening, {user_name}! Hope you've had a great day. What can I assist you with?",
                f"Good evening! How can I help you wind down your day?",
            ]
        elif any(
            word in question_lower
            for word in ["how are you", "how you doing", "how is it going", "what's up"]
        ):
            responses = [
                f"I'm doing great, thank you for asking, {user_name}! 😊 I'm here and ready to help. How are you doing?",
                f"I'm fantastic, thanks! Always excited to help students like you. How can I assist you today?",
                f"I'm doing well, {user_name}! Thanks for asking. I'm here to help with your studies. What's on your mind?",
            ]
        elif any(
            word in question_lower
            for word in [
                "who are you",
                "what are you",
                "introduce yourself",
                "tell me about yourself",
            ]
        ):
            responses = [
                f"Hi {user_name}! I'm Aimy, your AI study assistant. I'm here to help you with your academic questions, documents, and reminders. I can analyze your study materials, answer questions, and even set reminders for important events. How can I help you today? 🎓",
                f"Hello! I'm Aimy, designed to be your personal AI academic companion. I can help you understand your course materials, answer questions about uploaded documents, and keep you organized with reminders. What would you like to work on? 📚",
                f"Nice to meet you, {user_name}! I'm Aimy, your intelligent study buddy. I specialize in helping students like you with document analysis, answering academic questions, and staying organized. Ready to dive into some learning? ✨",
            ]
        elif any(
            word in question_lower
            for word in [
                "goodbye",
                "bye",
                "farewell",
                "see you",
                "take care",
                "until next time",
                "catch you later",
            ]
        ):
            responses = [
                f"Goodbye, {user_name}! Take care and good luck with your studies! 👋",
                f"See you later, {user_name}! Don't hesitate to come back if you need any help. Have a great day! 🌟",
                f"Farewell! Remember, I'm always here when you need academic support. Take care! 📖",
                f"Bye for now, {user_name}! Hope our chat was helpful. See you next time! ✨",
            ]
        elif any(
            word in question_lower
            for word in ["thank you", "thanks", "much appreciated"]
        ):
            responses = [
                f"You're very welcome, {user_name}! 😊 Happy to help anytime!",
                f"My pleasure, {user_name}! That's what I'm here for. Feel free to ask if you need anything else! 🌟",
                f"Glad I could help! Don't hesitate to reach out whenever you need academic support, {user_name}! 📚",
            ]
        elif any(
            word in question_lower
            for word in [
                "you're welcome",
                "you're awesome",
                "you're great",
                "you're helpful",
            ]
        ):
            responses = [
                f"Thank you so much, {user_name}! That really means a lot! 😊 I love helping students succeed!",
                f"Aww, thanks {user_name}! You just made my day! I'm always here to support your learning journey! ✨",
                f"You're too kind, {user_name}! I really enjoy being part of your academic success! 🎓",
            ]
        else:
            # General greeting responses
            if current_hour < 12:
                time_greeting = "Good morning"
                emoji = "☀️"
            elif current_hour < 17:
                time_greeting = "Good afternoon"
                emoji = "🌤️"
            else:
                time_greeting = "Good evening"
                emoji = "🌙"

            responses = [
                f"Hello, {user_name}! {emoji} I'm Aimy, your AI study assistant. How can I help you today?",
                f"Hi there, {user_name}! Great to see you! I'm here to help with your studies. What can I assist you with? 😊",
                f"{time_greeting}, {user_name}! {emoji} I'm Aimy, ready to help you with your academic needs. What's on your mind?",
                f"Hey {user_name}! Welcome! I'm here to help you with documents, questions, and reminders. How can I support your studies today? 📚",
                f"Hello! I'm Aimy, your friendly AI academic assistant. Ready to help you learn and stay organized! What can I do for you, {user_name}? ✨",
            ]

        return random.choice(responses)

    def _handle_reminder_request(
        self, question: str, session: ChatSession, user_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        Handle reminder requests specifically.
        Returns a response if this is a reminder request, None otherwise.
        """
        try:
            # Check if this is a reminder request
            reminder_data = self.reminder_service.analyze_message_for_reminder(question)

            if not reminder_data:
                return None  # Not a reminder request

            # Save the user message first
            user_message = ChatMessage.objects.create(
                session=session, message_type="user", content=question
            )

            # Create the reminder
            reminder = self.reminder_service.create_reminder(
                user=session.user,
                chat_session=session,
                chat_message=user_message,
                reminder_data=reminder_data,
                user_timezone="UTC",  # TODO: Get from user preferences
            )

            if reminder:
                # Create a helpful response about the reminder
                response_content = (
                    f"✅ I've set a reminder for you: **{reminder.title}**\n\n"
                )
                response_content += f"📅 **When:** {reminder.reminder_datetime.strftime('%B %d, %Y at %I:%M %p')}\n"

                # Add delivery method with more detail
                delivery_method_map = {
                    "notification": "📱 In-app notification",
                    "email": "📧 Email notification",
                    "sms": "📱 SMS notification",
                    "email_sms": "📧📱 Email and SMS notifications",
                    "all": "📧📱🔔 All notification methods",
                }
                delivery_desc = delivery_method_map.get(
                    reminder.delivery_method, reminder.get_delivery_method_display()
                )
                response_content += f"🔔 **Delivery:** {delivery_desc}\n\n"

                # Add contact info status
                if reminder.delivery_method in ["email", "email_sms", "all"]:
                    response_content += (
                        f"📧 Email will be sent to: {session.user.email}\n"
                    )
                if reminder.delivery_method in ["sms", "email_sms", "all"]:
                    phone = session.user.phone_number or "Not provided"
                    response_content += f"📱 SMS will be sent to: {phone}\n"

                response_content += (
                    "\nI'll make sure to remind you at the scheduled time!"
                )

                # Save the assistant response
                assistant_message = ChatMessage.objects.create(
                    session=session,
                    message_type="assistant",
                    content=response_content,
                    confidence_score=1.0,  # High confidence for successful reminder creation
                )

                # Update session
                session.last_message_at = timezone.now()
                if not session.title or session.title == "New Chat":
                    session.title = f"Reminder: {reminder.title[:50]}"
                session.save()

                return {
                    "success": True,
                    "answer": response_content,
                    "confidence_score": 1.0,
                    "sources": [],
                    "session_id": session.id,
                    "session_title": session.title,
                    "reminder_created": True,
                    "reminder_id": reminder.id,
                    "reminder_datetime": reminder.reminder_datetime.isoformat(),
                }
            else:
                # Reminder creation failed
                error_message = "I understood that you want to set a reminder, but I had trouble creating it. Please try again or be more specific about the date and time."

                assistant_message = ChatMessage.objects.create(
                    session=session,
                    message_type="assistant",
                    content=error_message,
                    confidence_score=0.3,
                )

                return {
                    "success": False,
                    "answer": error_message,
                    "confidence_score": 0.3,
                    "sources": [],
                    "session_id": session.id,
                    "session_title": session.title,
                    "reminder_created": False,
                    "error": "Failed to create reminder",
                }

        except Exception as e:
            logger.error(f"Error handling reminder request: {e}")
            return None  # Fall back to normal question handling

    def _handle_document_specific_question(
        self,
        question: str,
        session: ChatSession,
        user_id: int,
        document_id: int,
        max_results: int,
        temperature: float,
    ) -> Dict[str, Any]:
        """Handle questions specifically about a user's document"""
        try:
            # Get vector stores for the specific document only
            vector_stores = self._get_vector_stores(user_id, document_id)
            if not vector_stores:
                logger.warning(
                    f"No vector stores available for user {user_id}, document {document_id}"
                )

                # Check if document exists and provide specific error message
                try:
                    doc = Document.objects.get(id=document_id, user_id=user_id)
                    if not doc.is_processed:
                        return self._create_error_response(
                            f"Document '{doc.title}' is still being processed. Please wait a moment and try again."
                        )
                    else:
                        return self._create_error_response(
                            f"Document '{doc.title}' is processed but vector store could not be loaded. Please try re-uploading the document."
                        )
                except Document.DoesNotExist:
                    return self._create_error_response(
                        f"Document with ID {document_id} does not exist or you don't have access to it."
                    )

            # Search for relevant content in the specific document
            relevant_docs = self._search_documents(question, vector_stores, max_results)

            if not relevant_docs:
                return self._create_error_response(
                    f"No relevant information found in the specified document"
                )

            # Generate response
            response = self._generate_response(
                question, relevant_docs, session, temperature
            )

            # Save messages
            user_message = ChatMessage.objects.create(
                session=session, message_type="user", content=question
            )

            assistant_message = ChatMessage.objects.create(
                session=session,
                message_type="assistant",
                content=response["answer"],
                confidence_score=response["confidence_score"],
                tokens_used=response.get("tokens_used", 0),
            )

            # Update session
            session.last_message_at = timezone.now()
            session.save()

            return {
                "success": True,
                "answer": response["answer"],
                "confidence_score": response["confidence_score"],
                "session_id": session.id,
                "message_id": assistant_message.id,
                "tokens_used": response.get("tokens_used", 0),
            }

        except Exception as e:
            logger.error(f"Error handling document-specific question: {str(e)}")
            return self._create_error_response(str(e))

    def _handle_session_specific_question(
        self,
        question: str,
        session: ChatSession,
        user_id: int,
        document_id: int,
        max_results: int,
        temperature: float,
    ) -> Dict[str, Any]:
        """Handle questions for a specific session's document"""
        try:
            # Get vector stores for the session's document only
            vector_stores = self._get_vector_stores(user_id, document_id)
            if not vector_stores:
                logger.warning(
                    f"No vector stores available for user {user_id}, session document {document_id}"
                )
                return self._create_error_response(
                    f"No document available for this session"
                )

            # Search for relevant content in the session's document
            relevant_docs = self._search_documents(question, vector_stores, max_results)

            if not relevant_docs:
                return self._create_error_response(
                    f"No relevant information found in the session's document"
                )

            # Generate response
            response = self._generate_response(
                question, relevant_docs, session, temperature
            )

            # Save messages
            user_message = ChatMessage.objects.create(
                session=session, message_type="user", content=question
            )

            assistant_message = ChatMessage.objects.create(
                session=session,
                message_type="assistant",
                content=response["answer"],
                confidence_score=response["confidence_score"],
                tokens_used=response.get("tokens_used", 0),
            )

            # Update session
            session.last_message_at = timezone.now()
            session.save()

            return {
                "success": True,
                "answer": response["answer"],
                "confidence_score": response["confidence_score"],
                "session_id": session.id,
                "message_id": assistant_message.id,
                "tokens_used": response.get("tokens_used", 0),
            }

        except Exception as e:
            logger.error(f"Error handling session-specific question: {str(e)}")
            return self._create_error_response(str(e))

    def _handle_general_question(
        self,
        question: str,
        session: ChatSession,
        user_id: int,
        max_results: int,
        temperature: float,
    ) -> Dict[str, Any]:
        """Handle general questions using all available sources"""
        try:
            # Check if it's a mathematical question first
            if self._is_mathematical_question(question):
                return self._handle_mathematical_question(question, session, user_id)

            # Check if it's a KNUST-specific question
            if self._is_knust_specific_question(question):
                return self._handle_knust_question(question, session, user_id)

            # Get all available vector stores (user documents + generic files)
            vector_stores = self._get_vector_stores(user_id, None)
            if not vector_stores:
                logger.warning(f"No vector stores available for user {user_id}")
                # Fallback to general knowledge when no vector stores available
                return self._handle_general_knowledge_question(
                    question, session, user_id, temperature
                )

            # Search for relevant content across all sources
            relevant_docs = self._search_documents(question, vector_stores, max_results)

            # Generate response
            response = self._generate_response(
                question, relevant_docs, session, temperature
            )

            # Save messages
            user_message = ChatMessage.objects.create(
                session=session, message_type="user", content=question
            )

            assistant_message = ChatMessage.objects.create(
                session=session,
                message_type="assistant",
                content=response["answer"],
                confidence_score=response["confidence_score"],
                tokens_used=response.get("tokens_used", 0),
            )

            # Update session
            session.last_message_at = timezone.now()
            session.save()

            return {
                "success": True,
                "answer": response["answer"],
                "confidence_score": response["confidence_score"],
                "session_id": session.id,
                "message_id": assistant_message.id,
                "tokens_used": response.get("tokens_used", 0),
            }

        except Exception as e:
            logger.error(f"Error handling general question: {str(e)}")
            return self._create_error_response(str(e))

    def _handle_generic_files_question(
        self,
        question: str,
        session: ChatSession,
        user_id: int,
        max_results: int,
        temperature: float,
    ) -> Dict[str, Any]:
        """Handle questions using generic files first, then fallback to KNUST tools or general knowledge"""
        try:
            # Get only generic files vector stores
            vector_stores = self._get_generic_files_vector_stores()

            if not vector_stores:
                logger.warning("No generic files vector stores available")
                # No generic files available, try KNUST tools first, then general knowledge
                if self._is_knust_specific_question(question):
                    return self._handle_knust_question(question, session, user_id)
                return self._handle_general_question(
                    question, session, user_id, max_results, temperature
                )

            # Search for relevant content in generic files only
            relevant_docs = self._search_documents(question, vector_stores, max_results)

            # Check if the found content is actually relevant (not just keyword matching)
            if not relevant_docs or self._is_low_relevance_response(
                relevant_docs, question
            ):
                # No relevant info in generic files, try KNUST tools first, then general knowledge
                if self._is_knust_specific_question(question):
                    return self._handle_knust_question(question, session, user_id)
                return self._handle_general_question(
                    question, session, user_id, max_results, temperature
                )

            # Generate response from generic files
            response = self._generate_response(
                question, relevant_docs, session, temperature
            )

            # Save messages
            user_message = ChatMessage.objects.create(
                session=session, message_type="user", content=question
            )

            assistant_message = ChatMessage.objects.create(
                session=session,
                message_type="assistant",
                content=response["answer"],
                confidence_score=response["confidence_score"],
                tokens_used=response.get("tokens_used", 0),
            )

            # Update session
            session.last_message_at = timezone.now()
            session.save()

            return {
                "success": True,
                "answer": response["answer"],
                "confidence_score": response["confidence_score"],
                "session_id": session.id,
                "message_id": assistant_message.id,
                "tokens_used": response.get("tokens_used", 0),
            }

        except Exception as e:
            logger.error(f"Error handling generic files question: {str(e)}")
            # Fallback to KNUST tools first, then general knowledge
            if self._is_knust_specific_question(question):
                return self._handle_knust_question(question, session, user_id)
            return self._handle_general_question(
                question, session, user_id, max_results, temperature
            )

    def _is_low_relevance_response(
        self, relevant_docs: List[Dict[str, Any]], question: str
    ) -> bool:
        """Check if the response from generic files is actually relevant to the question"""
        try:
            # If no docs found, it's low relevance
            if not relevant_docs:
                return True

            # Check if any of the found docs have a good relevance score
            for doc in relevant_docs:
                score = doc.get("score", 0)
                # If we have a good score (lower is better for similarity), consider it relevant
                if score < 0.6:  # More lenient threshold for better coverage
                    return False

            # If all scores are poor, it's low relevance
            return True

        except Exception as e:
            logger.error(f"Error checking relevance: {str(e)}")
            return True  # Default to low relevance if error

    def _is_knust_specific_question(self, question: str) -> bool:
        """Check if a question is specifically about KNUST (not just mentions KNUST)"""
        knust_specific_keywords = [
            "admission",
            "apply",
            "application",
            "requirements",
            "cut-off",
            "cutoff",
            "colleges",
            "faculty",
            "school",
            "program",
            "course",
            "degree",
            "contact",
            "phone",
            "email",
            "address",
            "location",
            "campus",
            "student life",
            "accommodation",
            "fees",
            "tuition",
            "scholarship",
            "research",
            "graduate",
            "postgraduate",
            "masters",
            "phd",
        ]

        question_lower = question.lower()
        return any(keyword in question_lower for keyword in knust_specific_keywords)

    def _is_mathematical_question(self, question: str) -> bool:
        """Check if a question is mathematical in nature"""
        import re

        # Mathematical operators and patterns
        math_patterns = [
            r"[\+\-\*\/\^\(\)]",  # Basic operators
            r"\d+\s*[\+\-\*\/\^]\s*\d+",  # Number operations
            r"calculate|compute|solve|evaluate|sum|add|subtract|multiply|divide",
            r"what is \d+[\+\-\*\/]\d+",
            r"\d+\s*plus\s*\d+|\d+\s*minus\s*\d+|\d+\s*times\s*\d+|\d+\s*divided by\s*\d+",
            r"percentage|percent|%",
            r"square root|sqrt",
            r"power|exponent",
            r"equation|formula",
            r"how much is|what\'s|what is \d+",
            r"\d+\s*[\+\-\*\/]\s*\d+",  # Simple arithmetic like "1+1"
        ]

        question_lower = question.lower()

        # Check for mathematical patterns
        for pattern in math_patterns:
            if re.search(pattern, question_lower):
                return True

        # Check for simple arithmetic expressions (just numbers and operators)
        if re.search(r"^\s*\d+\s*[\+\-\*\/]\s*\d+\s*$", question.strip()):
            return True

        # Check for questions that start with numbers and operators
        if re.search(r"^\s*\d+[\+\-\*\/]", question.strip()):
            return True

        return False

    def _handle_mathematical_question(
        self, question: str, session: ChatSession, user_id: int
    ) -> Dict[str, Any]:
        """Handle mathematical questions using safe evaluation"""
        try:
            import re

            # Clean the question and extract mathematical expression
            question_lower = question.lower()

            # Handle common mathematical phrases
            if "what is" in question_lower:
                # Extract the mathematical part after "what is"
                math_part = question_lower.split("what is")[-1].strip()
            elif "what's" in question_lower:
                # Extract the mathematical part after "what's"
                math_part = question_lower.split("what's")[-1].strip()
            elif "how much is" in question_lower:
                # Extract the mathematical part after "how much is"
                math_part = question_lower.split("how much is")[-1].strip()
            elif "calculate" in question_lower:
                # Extract the mathematical part after "calculate"
                math_part = question_lower.split("calculate")[-1].strip()
            else:
                # Try to extract mathematical expression directly
                math_part = question.strip()

            # Replace word operators with symbols
            math_part = re.sub(r"\s+plus\s+", "+", math_part)
            math_part = re.sub(r"\s+minus\s+", "-", math_part)
            math_part = re.sub(r"\s+times\s+", "*", math_part)
            math_part = re.sub(r"\s+multiplied by\s+", "*", math_part)
            math_part = re.sub(r"\s+divided by\s+", "/", math_part)
            math_part = re.sub(r"\s+over\s+", "/", math_part)

            # Remove non-mathematical characters except operators and numbers
            math_part = re.sub(r"[^\d\+\-\*\/\(\)\.\s]", "", math_part)
            math_part = math_part.strip()

            # Validate the expression is safe (only contains allowed characters)
            if not re.match(r"^[\d\+\-\*\/\(\)\.\s]+$", math_part):
                raise ValueError("Invalid mathematical expression")

            # Evaluate the expression safely
            try:
                # Use a safer approach for mathematical evaluation
                if (
                    "+" in math_part
                    or "-" in math_part
                    or "*" in math_part
                    or "/" in math_part
                ):
                    # Simple arithmetic evaluation with additional safety checks
                    # Only allow basic arithmetic operations
                    allowed_chars = set("0123456789+-*/(). ")
                    if not all(c in allowed_chars for c in math_part):
                        raise ValueError(
                            "Invalid characters in mathematical expression"
                        )

                    # Check for division by zero
                    if "/" in math_part and "0" in math_part.split("/")[1].split()[0]:
                        raise ZeroDivisionError("Division by zero")

                    result = eval(
                        math_part
                    )  # Note: eval is used here for mathematical expressions only
                else:
                    result = float(math_part)

                # Format the result appropriately
                if isinstance(result, int):
                    answer = f"The answer is {result}"
                elif isinstance(result, float):
                    # Round to 2 decimal places for cleaner output
                    answer = f"The answer is {result:.2f}"
                else:
                    answer = f"The answer is {result}"

            except (ValueError, ZeroDivisionError, SyntaxError) as e:
                answer = f"I cannot evaluate that mathematical expression: {str(e)}"

            # Save messages
            user_message = ChatMessage.objects.create(
                session=session, message_type="user", content=question
            )

            assistant_message = ChatMessage.objects.create(
                session=session,
                message_type="assistant",
                content=answer,
                confidence_score=0.95,  # High confidence for mathematical answers
                tokens_used=len(answer.split()),  # Approximate token count
            )

            # Update session
            session.last_message_at = timezone.now()
            session.save()

            return {
                "answer": answer,
                "confidence_score": 0.95,
                "session_id": session.id,
                "message_id": assistant_message.id,
                "tokens_used": len(answer.split()),
            }

        except Exception as e:
            logger.error(f"Error handling mathematical question: {str(e)}")
            return self._create_error_response(
                f"Error processing mathematical question: {str(e)}"
            )

    def _handle_general_knowledge_question(
        self, question: str, session: ChatSession, user_id: int, temperature: float
    ) -> Dict[str, Any]:
        """Handle general knowledge questions using LLM when no vector stores are available"""
        try:
            # Create a prompt for general knowledge questions
            prompt = f"""You are a helpful AI assistant. Please answer the following question based on your general knowledge.
            If you're not sure about something, please say so rather than making up information.

            Question: {question}

            Answer:"""

            # Generate response using LLM
            response = self.llm.invoke(prompt)

            answer = response.content

            # Save messages
            user_message = ChatMessage.objects.create(
                session=session, message_type="user", content=question
            )

            assistant_message = ChatMessage.objects.create(
                session=session,
                message_type="assistant",
                content=answer,
                confidence_score=0.7,  # Moderate confidence for general knowledge
                tokens_used=len(answer.split()),  # Approximate token count
            )

            # Update session
            session.last_message_at = timezone.now()
            session.save()

            return {
                "answer": answer,
                "confidence_score": 0.7,
                "session_id": session.id,
                "message_id": assistant_message.id,
                "tokens_used": len(answer.split()),
            }

        except Exception as e:
            logger.error(f"Error handling general knowledge question: {str(e)}")
            return self._create_error_response(
                f"Error processing general knowledge question: {str(e)}"
            )

    def _handle_knust_question(
        self, question: str, session: ChatSession, user_id: int
    ) -> Dict[str, Any]:
        """Handle KNUST-related questions using KNUST tools"""
        try:
            # Get KNUST information
            knust_answer = knust_tools.get_knust_info(question)

            # Save messages
            user_message = ChatMessage.objects.create(
                session=session, message_type="user", content=question
            )

            assistant_message = ChatMessage.objects.create(
                session=session,
                message_type="assistant",
                content=knust_answer,
                confidence_score=0.9,  # High confidence for KNUST info
                tokens_used=len(knust_answer.split()),  # Approximate token count
            )

            # Update session
            session.last_message_at = timezone.now()
            session.save()

            return {
                "answer": knust_answer,
                "confidence_score": 0.9,
                "session_id": session.id,
                "message_id": assistant_message.id,
                "tokens_used": len(knust_answer.split()),
            }

        except Exception as e:
            logger.error(f"Error handling KNUST question: {str(e)}")
            return self._create_error_response(str(e))

    def _get_or_create_session(
        self,
        user_id: int,
        session_id: Optional[int] = None,
        document_id: Optional[int] = None,
    ) -> ChatSession:
        """Get existing session or create new one"""
        if session_id:
            try:
                session = ChatSession.objects.get(id=session_id, user_id=user_id)
                return session
            except ChatSession.DoesNotExist:
                pass

        # Create new session
        document = None
        if document_id:
            try:
                document = Document.objects.get(id=document_id, user_id=user_id)
            except Document.DoesNotExist:
                pass

        return ChatSession.objects.create(
            user_id=user_id,
            document=document,
            title=f"Chat Session {timezone.now().strftime('%Y-%m-%d %H:%M')}",
        )

    def _get_vector_stores(
        self, user_id: int, document_id: Optional[int] = None
    ) -> List:
        """Get vector stores for user's documents and generic files"""
        vector_stores = []

        if document_id:
            # Get specific document
            documents = Document.objects.filter(
                id=document_id, user_id=user_id, is_processed=True
            )
            logger.info(
                f"Found {documents.count()} processed documents for user {user_id}, document_id {document_id}"
            )

            # Check if document exists at all (processed or not)
            all_documents = Document.objects.filter(id=document_id, user_id=user_id)
            if all_documents.exists():
                doc = all_documents.first()
                logger.info(
                    f"Document {document_id} exists: title='{doc.title}', is_processed={doc.is_processed}"
                )
                if not doc.is_processed:
                    logger.warning(
                        f"Document {document_id} exists but is not yet processed. Processing status may be pending."
                    )
            else:
                logger.error(
                    f"Document {document_id} does not exist for user {user_id}"
                )

            for doc in documents:
                vector_store = self._load_document_vector_store(doc)
                if vector_store:
                    vector_stores.append(vector_store)
                    logger.info(
                        f"Successfully loaded vector store for document {doc.id}"
                    )
                else:
                    logger.warning(f"Failed to load vector store for document {doc.id}")
        else:
            # Get all user's documents
            documents = Document.objects.filter(user_id=user_id, is_processed=True)
            logger.info(
                f"Found {documents.count()} processed documents for user {user_id}"
            )

            for doc in documents:
                vector_store = self._load_document_vector_store(doc)
                if vector_store:
                    vector_stores.append(vector_store)
                    logger.info(
                        f"Successfully loaded vector store for document {doc.id}"
                    )
                else:
                    logger.warning(f"Failed to load vector store for document {doc.id}")

            # Also get all generic files for general questions
            generic_files = GenericFile.objects.filter(is_processed=True)
            logger.info(f"Found {generic_files.count()} processed generic files")

            for generic_file in generic_files:
                vector_store = self._load_generic_file_vector_store(generic_file)
                if vector_store:
                    vector_stores.append(vector_store)
                    logger.info(
                        f"Successfully loaded vector store for generic file {generic_file.id}"
                    )
                else:
                    logger.warning(
                        f"Failed to load vector store for generic file {generic_file.id}"
                    )

        logger.info(f"Total vector stores loaded: {len(vector_stores)}")
        return vector_stores

    def _get_generic_files_vector_stores(self) -> List:
        """Get vector stores for generic files only"""
        vector_stores = []
        generic_files = GenericFile.objects.filter(is_processed=True)

        for generic_file in generic_files:
            vector_store = self._load_generic_file_vector_store(generic_file)
            if vector_store:
                vector_stores.append(vector_store)

        return vector_stores

    def _load_document_vector_store(self, doc: Document):
        """Load vector store for a specific document - prioritize Redis over Chroma"""
        try:
            # Try Redis first (prioritized over Chroma)
            config = RedisConfig(
                index_name=doc.index_name,
                redis_url=REDIS_URL,
                metadata_schema=[
                    {"name": "document_id", "type": "tag"},
                    {"name": "chunk_index", "type": "tag"},
                    {"name": "page_number", "type": "tag"},
                ],
            )
            vector_store = RedisVectorStore(self.embeddings, config=config)

            # Verify the Redis vector store is actually usable
            try:
                # Test a simple similarity search to ensure it's working
                test_results = vector_store.similarity_search("test", k=1)
                logger.info(
                    f"Successfully loaded and verified Redis vector store for document {doc.id} (found {len(test_results)} test results)"
                )
                return vector_store
            except Exception as test_error:
                logger.warning(
                    f"Redis vector store test failed for document {doc.id}: {test_error}"
                )
                raise test_error  # Fail and try Chroma

        except Exception as redis_error:
            logger.warning(
                f"Redis vector store failed for document {doc.id}: {str(redis_error)}"
            )
            # Try Chroma as fallback
            try:
                import chromadb

                # Create Chroma client with persistent directory
                client = chromadb.PersistentClient(path="./chroma_db")

                # Check if collection exists and has data
                collection_name = f"doc_{doc.file_hash[:16]}"
                try:
                    collection = client.get_collection(name=collection_name)
                    count = collection.count()
                    logger.info(
                        f"Chroma collection {collection_name} exists with {count} documents"
                    )

                    if count == 0:
                        logger.warning(
                            f"Chroma collection {collection_name} exists but is empty"
                        )
                        raise Exception(f"Collection {collection_name} is empty")

                except Exception as collection_error:
                    logger.warning(
                        f"Chroma collection {collection_name} not found or empty: {str(collection_error)}"
                    )
                    raise collection_error

                # Create Chroma vector store
                vector_store = Chroma(
                    embedding_function=self.embeddings,
                    collection_name=collection_name,
                    client=client,
                )

                # Verify the vector store is actually usable
                try:
                    # Test a simple similarity search to ensure it's working
                    test_results = vector_store.similarity_search("test", k=1)
                    logger.info(
                        f"Successfully loaded and verified Chroma vector store for document {doc.id} (found {len(test_results)} test results)"
                    )
                except Exception as test_error:
                    logger.warning(
                        f"Chroma vector store test failed for document {doc.id}: {test_error}"
                    )
                    # Don't fail here, just log the warning

                return vector_store

            except Exception as chroma_error:
                logger.error(
                    f"Both Redis and Chroma failed for document {doc.id}: {str(redis_error)} -> {str(chroma_error)}"
                )
                return None

    def _load_generic_file_vector_store(self, generic_file: GenericFile):
        """Load vector store for a specific generic file - prioritize Chroma over Redis"""
        try:
            # Try Chroma first (prioritized over Redis)
            import chromadb

            # Create Chroma client with persistent directory
            client = chromadb.PersistentClient(path="./chroma_db")

            # Check if collection exists and has data
            collection_name = f"generic_{generic_file.file_hash[:16]}"
            try:
                collection = client.get_collection(name=collection_name)
                count = collection.count()
                logger.info(
                    f"Chroma collection {collection_name} exists with {count} documents"
                )

                if count == 0:
                    logger.warning(
                        f"Chroma collection {collection_name} exists but is empty"
                    )
                    raise Exception("Collection is empty")

            except Exception as collection_error:
                logger.warning(
                    f"Chroma collection {collection_name} not found or empty: {str(collection_error)}"
                )
                raise collection_error

            # Create Chroma vector store
            vector_store = Chroma(
                embedding_function=self.embeddings,
                collection_name=collection_name,
                client=client,
            )
            logger.info(
                f"Successfully loaded Chroma vector store for generic file {generic_file.id}"
            )
            return vector_store

        except Exception as chroma_error:
            logger.warning(
                f"Chroma vector store failed for generic file {generic_file.id}: {str(chroma_error)}"
            )
            # Try Redis as fallback
            try:
                config = RedisConfig(
                    index_name=generic_file.index_name,
                    redis_url=REDIS_URL,
                    metadata_schema=[
                        {"name": "generic_file_id", "type": "tag"},
                        {"name": "chunk_index", "type": "tag"},
                        {"name": "source", "type": "tag"},
                    ],
                )
                vector_store = RedisVectorStore(self.embeddings, config=config)
                logger.info(
                    f"Successfully loaded Redis vector store for generic file {generic_file.id}"
                )
                return vector_store
            except Exception as redis_error:
                logger.error(
                    f"Both Chroma and Redis failed for generic file {generic_file.id}: {str(chroma_error)} -> {str(redis_error)}"
                )
                return None

    def _search_documents(
        self, question: str, vector_stores: List, max_results: int
    ) -> List[Dict[str, Any]]:
        """Search documents for relevant content with hybrid approach"""
        all_results = []

        for vector_store in vector_stores:
            try:
                # Handle both Redis and Chroma vector stores
                if hasattr(vector_store, "similarity_search_with_score"):
                    results = vector_store.similarity_search_with_score(
                        question,
                        k=max_results * 2,  # Get more results for hybrid search
                    )
                else:
                    # Fallback for Chroma
                    results = vector_store.similarity_search_with_relevance_scores(
                        question, k=max_results * 2
                    )

                logger.info(
                    f"Vector search returned {len(results)} results for question: {question}"
                )

                for doc, score in results:
                    logger.info(
                        f"Result score: {score}, content preview: {doc.page_content[:100]}..."
                    )
                    all_results.append(
                        {
                            "content": doc.page_content,
                            "metadata": doc.metadata,
                            "score": score,
                        }
                    )
            except Exception as e:
                logger.error(f"Error searching vector store: {str(e)}")

        # Sort by score first
        all_results.sort(key=lambda x: x["score"])
        logger.info(f"Total results after sorting: {len(all_results)}")

        # Apply hybrid filtering: boost exact matches
        enhanced_results = []
        for result in all_results:
            content = result["content"].lower()
            question_lower = question.lower()

            # Check for exact course code matches (e.g., CSM 254, CHEM 292, etc.)
            import re

            course_pattern = r"\b[A-Z]{2,4}\s+\d{3}\b"
            course_matches = re.findall(course_pattern, question_lower)

            # If we find course codes in the question, boost matching content
            if course_matches:
                for course_code in course_matches:
                    if course_code in content:
                        # Boost this result by reducing its score
                        result["score"] = result["score"] * 0.5  # Better score
                        break

            enhanced_results.append(result)

        logger.info(f"Enhanced results: {len(enhanced_results)}")
        if enhanced_results:
            logger.info(f"Best score: {enhanced_results[0]['score']}")
            logger.info(f"Worst score: {enhanced_results[-1]['score']}")

        # Re-sort by enhanced scores
        enhanced_results.sort(key=lambda x: x["score"])

        # If no good results from vector search, try direct database search
        # Be more lenient for general questions like "summary", "overview", etc.
        general_question_keywords = [
            "summary",
            "overview",
            "what is",
            "tell me about",
            "explain",
            "describe",
        ]
        is_general_question = any(
            keyword in question.lower() for keyword in general_question_keywords
        )

        if not enhanced_results or (
            not is_general_question and all(r["score"] > 0.5 for r in enhanced_results)
        ):
            enhanced_results = self._search_database_directly(question, max_results)

        return enhanced_results[:max_results]

    def _search_database_directly(
        self, question: str, max_results: int
    ) -> List[Dict[str, Any]]:
        """Search database directly when vector search fails"""
        try:
            from .models import GenericFileChunk, DocumentChunk

            # Extract course codes from question
            import re

            course_pattern = r"\b[A-Z]{2,4}\s+\d{3}\b"
            course_matches = re.findall(course_pattern, question.upper())

            # For general questions, get chunks from both document and generic file chunks
            general_question_keywords = [
                "summary",
                "overview",
                "what is",
                "tell me about",
                "explain",
                "describe",
            ]
            is_general_question = any(
                keyword in question.lower() for keyword in general_question_keywords
            )

            if is_general_question:
                # For general questions, return chunks from both document and generic file chunks
                results = []

                # Get document chunks
                doc_chunks = DocumentChunk.objects.all()[:max_results]
                for chunk in doc_chunks:
                    results.append(
                        {
                            "content": chunk.content,
                            "metadata": {
                                "document_id": chunk.document.id,
                                "chunk_index": chunk.chunk_index,
                                "source": chunk.document.title,
                            },
                            "score": 0.3,  # Good score for general questions
                        }
                    )

                # Get generic file chunks
                gen_chunks = GenericFileChunk.objects.all()[:max_results]
                for chunk in gen_chunks:
                    results.append(
                        {
                            "content": chunk.content,
                            "metadata": {
                                "generic_file_id": chunk.generic_file.id,
                                "chunk_index": chunk.chunk_index,
                                "source": chunk.generic_file.title,
                            },
                            "score": 0.3,  # Good score for general questions
                        }
                    )

                return results[:max_results]

            if not course_matches:
                return []

            # Search for chunks containing the course codes
            results = []
            for course_code in course_matches:
                chunks = GenericFileChunk.objects.filter(
                    content__icontains=course_code.upper()
                )[:max_results]

                for chunk in chunks:
                    # Get additional context from nearby chunks
                    context_chunks = []

                    # Get the previous chunk if it contains session info
                    if chunk.chunk_index > 0:
                        prev_chunk = GenericFileChunk.objects.filter(
                            generic_file_id=chunk.generic_file.id,
                            chunk_index=chunk.chunk_index - 1,
                        ).first()
                        if prev_chunk and (
                            "MORNING SESSION" in prev_chunk.content
                            or "EVENING SESSION" in prev_chunk.content
                        ):
                            context_chunks.append(prev_chunk.content)

                    # Get the next chunk if it contains date info
                    next_chunk = GenericFileChunk.objects.filter(
                        generic_file_id=chunk.generic_file.id,
                        chunk_index=chunk.chunk_index + 1,
                    ).first()
                    if next_chunk and (
                        "AUGUST" in next_chunk.content or "MONDAY" in next_chunk.content
                    ):
                        context_chunks.append(next_chunk.content)

                    # Combine content with context
                    full_content = chunk.content
                    if context_chunks:
                        full_content = (
                            "\n\n".join(context_chunks) + "\n\n" + chunk.content
                        )

                    results.append(
                        {
                            "content": full_content,
                            "metadata": {
                                "generic_file_id": chunk.generic_file.id,
                                "chunk_index": chunk.chunk_index,
                                "source": chunk.generic_file.title,
                            },
                            "score": 0.1,  # Very good score for direct match
                        }
                    )

            return results

        except Exception as e:
            logger.error(f"Error in direct database search: {str(e)}")
            return []

    def _generate_response(
        self,
        question: str,
        relevant_docs: List[Dict[str, Any]],
        session: ChatSession,
        temperature: float,
    ) -> Dict[str, Any]:
        """Generate response using LLM"""
        if not relevant_docs:
            return {
                "answer": "I don't have enough information to answer your question. Please upload some documents first.",
                "confidence_score": 0.0,
            }

        # Prepare context
        context = "\n\n".join([doc["content"] for doc in relevant_docs])

        # Create prompt
        prompt = f"""Based on the following context, answer the user's question.
        If you cannot answer the question based on the context, say so.

        Context:
        {context}

        Question: {question}

        Answer:"""

        # Generate response
        response = self.llm.invoke(prompt)

        return {
            "answer": response.content,
            "confidence_score": 0.8,  # Placeholder - could be calculated based on scores
        }

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create error response"""
        return {
            "answer": f"Sorry, I encountered an error: {error_message}",
            "confidence_score": 0.0,
            "session_id": None,
            "message_id": None,
            "tokens_used": 0,
        }

    def get_chat_history(self, session_id: int, user_id: int) -> List[Dict[str, Any]]:
        """Get chat history for a session"""
        try:
            session = ChatSession.objects.get(id=session_id, user_id=user_id)
            messages = session.messages.all()

            return [
                {
                    "id": msg.id,
                    "type": msg.message_type,
                    "content": msg.content,
                    "created_at": msg.created_at.isoformat(),
                    "confidence_score": (
                        msg.confidence_score
                        if msg.message_type == "assistant"
                        else None
                    ),
                }
                for msg in messages
            ]
        except ChatSession.DoesNotExist:
            return []

    def get_user_sessions(self, user_id: int) -> List[Dict[str, Any]]:
        """Get all chat sessions for a user"""
        sessions = ChatSession.objects.filter(user_id=user_id).order_by("-updated_at")

        return [
            {
                "id": session.id,
                "title": session.title,
                "document_title": session.document.title if session.document else None,
                "message_count": session.get_message_count(),
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "is_active": session.is_active,
            }
            for session in sessions
        ]


class ReminderService:
    """Service for handling reminder functionality"""

    def __init__(self):
        self.llm = ChatOpenAI(
            model=OPENAI_MODEL,
            openai_api_key=OPENAI_API_KEY,
            temperature=0.1,  # Lower temperature for more consistent parsing
        )

    def analyze_message_for_reminder(
        self, message: str, user_timezone: str = "UTC"
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze a user message to detect if it contains a reminder request.
        Returns reminder details if found, None otherwise.
        """
        import json
        import re
        from datetime import datetime, timedelta

        # Keywords that indicate reminder intent
        reminder_keywords = [
            "remind",
            "reminder",
            "remember",
            "don't forget",
            "alert me",
            "notify me",
            "schedule",
            "set a reminder",
            "alarm",
        ]

        # Check if message contains reminder keywords
        message_lower = message.lower()
        has_reminder_keyword = any(
            keyword in message_lower for keyword in reminder_keywords
        )

        if not has_reminder_keyword:
            return None

        # Create a prompt to extract reminder information using LLM
        system_prompt = f"""
        You are a reminder extraction assistant. Analyze the user message and extract reminder information if present.
        Current timezone: {user_timezone}
        Current datetime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

        If the message contains a reminder request, extract:
        1. title: Brief description (max 100 chars)
        2. description: Detailed description (optional)
        3. datetime: When to remind (YYYY-MM-DD HH:MM:SS format)
        4. relative_time: If time is relative (e.g., "tomorrow", "in 2 hours")

        Return ONLY a JSON object with these fields if a reminder is detected, or null if no reminder is found.
        Examples:
        - "remind me about my exam at 2:30 tomorrow" -> {{"title": "Exam reminder", "datetime": "2024-01-15 14:30:00", "relative_time": "tomorrow"}}
        - "don't forget to call mom at 7pm" -> {{"title": "Call mom", "datetime": "2024-01-14 19:00:00", "relative_time": "today"}}
        """

        try:
            # Use LLM to extract reminder information
            response = self.llm.invoke(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message},
                ]
            )

            response_text = response.content.strip()
            logger.info(f"LLM reminder extraction response: {response_text}")

            # Try to parse JSON response
            if response_text.lower() == "null" or not response_text:
                return None

            # Clean response - sometimes LLM adds extra text
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                reminder_data = json.loads(json_str)

                # Validate required fields
                if "title" in reminder_data and "datetime" in reminder_data:
                    # Parse datetime and ensure it's in the future
                    try:
                        reminder_datetime = datetime.strptime(
                            reminder_data["datetime"], "%Y-%m-%d %H:%M:%S"
                        )

                        # If datetime is in the past, try to adjust to next occurrence
                        if reminder_datetime <= datetime.now():
                            # If it's the same day but past time, move to tomorrow
                            if reminder_datetime.date() == datetime.now().date():
                                reminder_datetime += timedelta(days=1)
                            # If it's a past date, move to next year
                            elif reminder_datetime.date() < datetime.now().date():
                                reminder_datetime = reminder_datetime.replace(
                                    year=datetime.now().year + 1
                                )

                        reminder_data["datetime"] = reminder_datetime.strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                        return reminder_data

                    except ValueError as e:
                        logger.error(f"Invalid datetime format in reminder: {e}")
                        return None

            return None

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            return None
        except Exception as e:
            logger.error(f"Error in reminder analysis: {e}")
            return None

    def create_reminder(
        self,
        user,
        chat_session: ChatSession,
        chat_message: ChatMessage,
        reminder_data: Dict[str, Any],
        user_timezone: str = "UTC",
    ) -> Optional[Reminder]:
        """
        Create a reminder object from extracted reminder data.
        """
        try:
            from datetime import datetime

            # Parse the datetime
            reminder_datetime = datetime.strptime(
                reminder_data["datetime"], "%Y-%m-%d %H:%M:%S"
            )

            # Determine the best delivery method based on user's available contact info
            delivery_method = self._determine_delivery_method(user)

            # Create reminder
            reminder = Reminder.objects.create(
                user=user,
                chat_session=chat_session,
                chat_message=chat_message,
                title=reminder_data["title"][:255],  # Ensure within limit
                description=reminder_data.get("description", ""),
                original_message=chat_message.content,
                reminder_datetime=reminder_datetime,
                timezone=user_timezone,
                delivery_method=delivery_method,
            )

            logger.info(
                f"Created reminder {reminder.id} for user {user.id}: {reminder.title}"
            )
            return reminder

        except Exception as e:
            logger.error(f"Error creating reminder: {e}")
            return None

    def _determine_delivery_method(self, user) -> str:
        """
        Determine the best delivery method based on user's available contact information.
        Priority: SMS > Email + SMS > Email > Notification only
        """
        has_phone = bool(user.phone_number and user.phone_number.strip())
        has_email = bool(user.email and user.email.strip())

        if has_phone and has_email:
            return "email_sms"  # Both email and SMS for best coverage
        elif has_phone:
            return "sms"  # SMS only if they have phone but no email (unlikely)
        elif has_email:
            return "email"  # Email only if they have email but no phone
        else:
            return "notification"  # Fallback to in-app notification only

    def process_message_for_reminders(
        self,
        message_content: str,
        user,
        chat_session: ChatSession,
        chat_message: ChatMessage,
        user_timezone: str = "UTC",
    ) -> Optional[Reminder]:
        """
        Process a chat message to detect and create reminders.
        This should be called after a user message is saved.
        """
        # Analyze message for reminder intent
        reminder_data = self.analyze_message_for_reminder(
            message_content, user_timezone
        )

        if reminder_data:
            # Create the reminder
            reminder = self.create_reminder(
                user, chat_session, chat_message, reminder_data, user_timezone
            )
            return reminder

        return None

    def get_user_reminders(
        self, user, status: Optional[str] = None, limit: int = 50
    ) -> List[Reminder]:
        """Get user's reminders, optionally filtered by status"""
        queryset = Reminder.objects.filter(user=user)

        if status:
            queryset = queryset.filter(status=status)

        return queryset.order_by("reminder_datetime")[:limit]

    def get_due_reminders(self) -> List[Reminder]:
        """Get all reminders that are due to be sent"""
        from django.utils import timezone

        return Reminder.objects.filter(
            status="pending", reminder_datetime__lte=timezone.now()
        ).order_by("reminder_datetime")

    def mark_reminder_sent(self, reminder: Reminder):
        """Mark a reminder as sent"""
        reminder.mark_as_sent()
        logger.info(f"Marked reminder {reminder.id} as sent")

    def mark_reminder_failed(self, reminder: Reminder, error_message: str):
        """Mark a reminder as failed"""
        reminder.mark_as_failed(error_message)
        logger.error(f"Marked reminder {reminder.id} as failed: {error_message}")

    def cancel_reminder(self, reminder: Reminder):
        """Cancel a reminder"""
        reminder.status = "cancelled"
        reminder.save()
        logger.info(f"Cancelled reminder {reminder.id}")
