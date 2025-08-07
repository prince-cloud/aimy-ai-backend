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
from .models import Document, DocumentChunk, ProcessingQueue, ChatSession, ChatMessage
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
            chunk_size=1000,
            chunk_overlap=200,
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
        """Ask a question and get response"""
        try:
            # Get or create chat session
            session = self._get_or_create_session(user_id, session_id, document_id)

            # Get relevant documents
            vector_stores = self._get_vector_stores(user_id, document_id)
            if not vector_stores:
                return self._create_error_response("No documents available for search")

            # Search for relevant content
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
            logger.error(f"Error asking question: {str(e)}")
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
        """Get vector stores for user's documents"""
        documents = Document.objects.filter(user_id=user_id, is_processed=True)

        if document_id:
            documents = documents.filter(id=document_id)

        vector_stores = []
        for doc in documents:
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
                vector_stores.append(vector_store)
                logger.info(f"Loaded Redis vector store for document {doc.id}")
            except Exception as e:
                logger.warning(
                    f"Redis vector store failed for document {doc.id}: {str(e)}"
                )
                # Try Chroma as fallback
                try:
                    vector_store = Chroma(
                        embedding_function=self.embeddings,
                        collection_name=f"doc_{doc.id}",
                    )
                    vector_stores.append(vector_store)
                    logger.info(f"Loaded Chroma vector store for document {doc.id}")
                except Exception as chroma_error:
                    logger.error(
                        f"Both Redis and Chroma failed for document {doc.id}: {str(e)} -> {str(chroma_error)}"
                    )

        return vector_stores

    def _search_documents(
        self, question: str, vector_stores: List, max_results: int
    ) -> List[Dict[str, Any]]:
        """Search documents for relevant content"""
        all_results = []

        for vector_store in vector_stores:
            try:
                # Handle both Redis and Chroma vector stores
                if hasattr(vector_store, "similarity_search_with_score"):
                    results = vector_store.similarity_search_with_score(
                        question, k=max_results
                    )
                else:
                    # Fallback for Chroma
                    results = vector_store.similarity_search_with_relevance_scores(
                        question, k=max_results
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

        # Sort by score and return top results
        all_results.sort(key=lambda x: x["score"])
        return all_results[:max_results]

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
