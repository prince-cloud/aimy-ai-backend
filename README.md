# Aimy AI - Document Chat System

A comprehensive AI-powered document chat system built with Django, Redis, and OpenAI. Users can upload documents (PDF, DOCX, TXT), have them processed and indexed in a Redis vector store, and then ask questions about the content using natural language.

## 🚀 Features

### 📄 Document Management
- **Multi-format Support**: Upload PDF, DOCX, and TXT files
- **Duplicate Detection**: Automatic detection of duplicate files using SHA256 hashing
- **Processing Queue**: Background processing with status tracking
- **File Optimization**: Prevents re-processing of identical files to save OpenAI credits

### 🤖 AI-Powered Chat
- **Vector Search**: Semantic search through document content using OpenAI embeddings
- **Contextual Responses**: AI generates answers based on relevant document chunks
- **Chat History**: Persistent chat sessions with message history
- **Source Attribution**: Responses include references to source documents and pages

### 🔍 Advanced Search
- **Semantic Search**: Find relevant content using natural language queries
- **Multi-document Search**: Search across all user documents or specific documents
- **Relevance Scoring**: Results ranked by relevance to query

### 📊 Admin Interface
- **Document Management**: View, manage, and reprocess documents
- **Chat Monitoring**: Monitor chat sessions and messages
- **Processing Queue**: Track document processing status
- **User Analytics**: View user activity and document usage

## 🛠️ Technology Stack

- **Backend**: Django 5.2.4
- **Database**: PostgreSQL
- **Vector Store**: Redis with LangChain Redis integration
- **AI/ML**: OpenAI GPT-3.5-turbo, text-embedding-3-small
- **Document Processing**: PyPDF, python-docx, LangChain
- **Admin**: Django Unfold for modern admin interface
- **API**: Django REST Framework with comprehensive documentation

## 📋 API Endpoints

### Document Management
- `POST /aimy/documents/` - Upload document
- `GET /aimy/documents/` - List user documents
- `GET /aimy/documents/{id}/` - Get document details
- `DELETE /aimy/documents/{id}/` - Delete document
- `POST /aimy/documents/{id}/reprocess/` - Reprocess document

### Chat Functionality
- `POST /aimy/chat/ask/` - Ask a question
- `GET /aimy/chat/sessions/` - List chat sessions
- `POST /aimy/chat/sessions/` - Create new chat session
- `GET /aimy/chat/sessions/{id}/` - Get chat session details
- `DELETE /aimy/chat/sessions/{id}/` - Delete chat session

### Search & Processing
- `POST /aimy/documents/search/` - Search documents
- `GET /aimy/processing-queue/` - Get processing queue status

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Clone the repository
git clone <repository-url>
cd aimy-ai-backend

# Create virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration
Create a `.env` file with your configuration:
```env
# Django Settings
SECRET_KEY=your-secret-key
DEBUG=True
DATABASE_URL=postgresql://user:password@localhost:5432/aimy_db

# Redis Configuration
REDIS_URL=redis://localhost:6379/0

# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-3.5-turbo
EMBEDDING_MODEL=text-embedding-3-small

# Media Files
MEDIA_URL=/media/
MEDIA_ROOT=media/
```

### 3. Database Setup
```bash
# Run migrations
python manage.py migrate

# Create superuser
python manage.py createsuperuser
```

### 4. Start Services
```bash
# Start Redis (if not running)
redis-server

# Start Django development server
python manage.py runserver
```

## 📖 Usage Examples

### Upload a Document
```bash
curl -X POST http://localhost:8000/aimy/documents/ \
  -H "Authorization: Bearer <your-token>" \
  -F "title=My Document" \
  -F "file=@document.pdf"
```

### Ask a Question
```bash
curl -X POST http://localhost:8000/aimy/chat/ask/ \
  -H "Authorization: Bearer <your-token>" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the main topics discussed in the document?",
    "document_id": 1,
    "max_results": 5
  }'
```

### Search Documents
```bash
curl -X POST http://localhost:8000/aimy/documents/search/ \
  -H "Authorization: Bearer <your-token>" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning algorithms",
    "max_results": 10
  }'
```

## 🔧 Configuration Options

### Document Processing
- **Chunk Size**: 1000 characters (configurable)
- **Chunk Overlap**: 200 characters (configurable)
- **Max File Size**: 10MB
- **Supported Formats**: PDF, DOCX, TXT

### AI Configuration
- **LLM Model**: GPT-3.5-turbo (configurable)
- **Embedding Model**: text-embedding-3-small (configurable)
- **Temperature**: 0.7 (configurable)
- **Max Results**: 5 (configurable)

### Redis Configuration
- **Index Schema**: Document ID, chunk index, page number
- **Metadata**: Source document, chunk content, relevance scores

## 📊 Database Schema

### Core Models
- **Document**: File metadata, processing status, vector store info
- **DocumentChunk**: Text chunks with vector IDs and page numbers
- **ChatSession**: User chat sessions with document associations
- **ChatMessage**: Individual messages with sources and confidence scores
- **ProcessingQueue**: Background processing status tracking

### Key Features
- **File Hash Detection**: SHA256-based duplicate detection
- **Vector Store Integration**: Redis-based semantic search
- **Chat Persistence**: Complete message history with metadata
- **Processing Status**: Real-time processing queue monitoring

## 🔒 Security Features

- **Authentication**: JWT-based authentication
- **Authorization**: User-specific document access
- **File Validation**: Type and size validation
- **Input Sanitization**: XSS protection
- **Rate Limiting**: API rate limiting (configurable)

## 🚀 Deployment

### Production Setup
1. **Environment Variables**: Configure all required environment variables
2. **Database**: Set up PostgreSQL with proper credentials
3. **Redis**: Configure Redis for vector storage
4. **Media Storage**: Configure media file storage (local or cloud)
5. **Static Files**: Collect and serve static files
6. **WSGI**: Configure WSGI server (Gunicorn recommended)

### Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up -d
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue on GitHub
- Contact the development team
- Check the documentation

## 🔮 Future Enhancements

- **Real-time Chat**: WebSocket-based real-time chat
- **Document Collaboration**: Multi-user document sharing
- **Advanced Analytics**: Usage analytics and insights
- **Mobile API**: Mobile-optimized endpoints
- **Batch Processing**: Bulk document processing
- **Custom Models**: Support for custom AI models
