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

# Local imports
from .models import (
    Document,
    DocumentChunk,
    ProcessingQueue,
    ChatSession,
    ChatMessage,
    GenericFile,
    GenericFileChunk,
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

            # Try to create vector store with chunks
            vector_store = None
            vector_store_type = None

            # Try Redis first
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
            except Exception as vector_error:
                logger.warning(
                    f"Redis vector store creation failed: {str(vector_error)}"
                )
                logger.warning(f"Redis error type: {type(vector_error).__name__}")
                logger.warning(f"Redis error details: {str(vector_error)}")
                import traceback

                logger.warning(f"Redis error traceback: {traceback.format_exc()}")

                # Fallback to Chroma vector store
                try:
                    logger.info(
                        f"Attempting to create Chroma vector store for document: {document.id}"
                    )
                    vector_store = Chroma.from_documents(
                        documents=chunks,
                        embedding=self.embeddings,
                        collection_name=f"doc_{document.id}",
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
                        f"Vector store creation error (both Redis and Chroma failed): {str(vector_error)} -> {str(chroma_error)}"
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

            # Log some content for debugging
            if docs:
                first_doc = docs[0]
                logger.info(
                    f"First document content preview: {first_doc.page_content[:100]}..."
                )
                logger.info(f"First document metadata: {first_doc.metadata}")

            return docs

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

    def check_document_exists(self, file_hash: str, user_id: int) -> Optional[Document]:
        """Check if document with same hash already exists for user"""
        return Document.objects.filter(
            file_hash=file_hash, user_id=user_id, is_processed=True
        ).first()

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

            # Try Redis first, fallback to Chroma
            vector_store = None
            try:
                # Try Redis
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
                logger.error(f"Redis vector store creation failed: {str(redis_error)}")
                try:
                    # Fallback to Chroma
                    _ = Chroma.from_documents(
                        documents=chunks,
                        embedding=self.embeddings,
                        collection_name=generic_file.index_name,
                    )
                    logger.info(
                        f"Successfully created Chroma vector store for generic file {generic_file.id}"
                    )
                except Exception as chroma_error:
                    logger.error(
                        f"Chroma vector store creation failed: {str(chroma_error)}"
                    )
                    raise Exception(
                        f"Vector store creation error (both Redis and Chroma failed): {str(redis_error)} -> {str(chroma_error)}"
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
            return docs

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

            # Priority 1: If document_id is provided, prioritize that document
            if document_id:
                return self._handle_document_specific_question(
                    question, session, user_id, document_id, max_results, temperature
                )

            # Priority 2: If session_id is provided, prioritize that session's document
            if session_id and session.document:
                return self._handle_session_specific_question(
                    question,
                    session,
                    user_id,
                    session.document.id,
                    max_results,
                    temperature,
                )

            # Priority 3: Use generic files (admin-uploaded files) - similar to document-specific approach
            return self._handle_generic_files_question(
                question, session, user_id, max_results, temperature
            )

        except Exception as e:
            logger.error(f"Error asking question: {str(e)}")
            return self._create_error_response(str(e))

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
                return self._create_error_response(
                    f"No document with ID {document_id} available for search"
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
            _ = ChatMessage.objects.create(
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
            _ = ChatMessage.objects.create(
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
            # Get all available vector stores (user documents + generic files)
            vector_stores = self._get_vector_stores(user_id, None)
            if not vector_stores:
                return self._create_error_response("No documents available for search")

            # Search for relevant content across all sources
            relevant_docs = self._search_documents(question, vector_stores, max_results)

            # Generate response
            response = self._generate_response(
                question, relevant_docs, session, temperature
            )

            # Save messages
            _ = ChatMessage.objects.create(
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
            _ = ChatMessage.objects.create(
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

    def _handle_knust_question(
        self, question: str, session: ChatSession, user_id: int
    ) -> Dict[str, Any]:
        """Handle KNUST-related questions using KNUST tools"""
        try:
            # Get KNUST information
            knust_answer = knust_tools.get_knust_info(question)

            # Save messages
            _ = ChatMessage.objects.create(
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

            for doc in documents:
                vector_store = self._load_document_vector_store(doc)
                if vector_store:
                    vector_stores.append(vector_store)
        else:
            # Get all user's documents
            documents = Document.objects.filter(user_id=user_id, is_processed=True)

            for doc in documents:
                vector_store = self._load_document_vector_store(doc)
                if vector_store:
                    vector_stores.append(vector_store)

            # Also get all generic files for general questions
            generic_files = GenericFile.objects.filter(is_processed=True)
            for generic_file in generic_files:
                vector_store = self._load_generic_file_vector_store(generic_file)
                if vector_store:
                    vector_stores.append(vector_store)

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
        """Load vector store for a specific document"""
        try:
            # Try Redis first
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
            logger.info(f"Loaded Redis vector store for document {doc.id}")
            return vector_store
        except Exception as e:
            logger.warning(f"Redis vector store failed for document {doc.id}: {str(e)}")
            # Try Chroma as fallback
            try:
                vector_store = Chroma(
                    embedding_function=self.embeddings,
                    collection_name=f"doc_{doc.id}",
                )
                logger.info(f"Loaded Chroma vector store for document {doc.id}")
                return vector_store
            except Exception as chroma_error:
                logger.error(
                    f"Both Redis and Chroma failed for document {doc.id}: {str(e)} -> {str(chroma_error)}"
                )
                return None

    def _load_generic_file_vector_store(self, generic_file: GenericFile):
        """Load vector store for a specific generic file"""
        try:
            # Try Redis first
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
            logger.info(f"Loaded Redis vector store for generic file {generic_file.id}")
            return vector_store
        except Exception as e:
            logger.warning(
                f"Redis vector store failed for generic file {generic_file.id}: {str(e)}"
            )
            # Try Chroma as fallback
            try:
                vector_store = Chroma(
                    embedding_function=self.embeddings,
                    collection_name=f"generic_{generic_file.id}",
                )
                logger.info(
                    f"Loaded Chroma vector store for generic file {generic_file.id}"
                )
                return vector_store
            except Exception as chroma_error:
                logger.error(
                    f"Both Redis and Chroma failed for generic file {generic_file.id}: {str(e)} -> {str(chroma_error)}"
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

                for doc, score in results:
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

        # Re-sort by enhanced scores
        enhanced_results.sort(key=lambda x: x["score"])

        # If no good results from vector search, try direct database search
        if not enhanced_results or all(r["score"] > 0.5 for r in enhanced_results):
            enhanced_results = self._search_database_directly(question, max_results)

        return enhanced_results[:max_results]

    def _search_database_directly(
        self, question: str, max_results: int
    ) -> List[Dict[str, Any]]:
        """Search database directly when vector search fails"""
        try:
            from .models import GenericFileChunk

            # Extract course codes from question
            import re

            course_pattern = r"\b[A-Z]{2,4}\s+\d{3}\b"
            course_matches = re.findall(course_pattern, question.upper())

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
