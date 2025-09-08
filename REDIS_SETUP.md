# Redis Vector Store Setup Guide

Your system is now configured to prioritize Redis over ChromaDB for vector storage, but Redis needs the RediSearch module for vector operations.

## Current Status

✅ **Code Updated**: System now tries Redis first, falls back to ChromaDB
❌ **Redis Module Missing**: RediSearch module not installed
🔄 **Fallback Working**: ChromaDB fallback is functioning correctly

## Setup Options

### Option 1: Install Redis Stack (Recommended)

Redis Stack includes RediSearch module needed for vector operations.

#### Using Docker (Easiest):
```bash
# Stop current Redis container
docker stop redis-container-name

# Run Redis Stack
docker run -d \
  --name redis-stack \
  -p 6379:6379 \
  -p 8001:8001 \
  redis/redis-stack:latest
```

#### Using Homebrew (macOS):
```bash
# Install Redis Stack
brew tap redis-stack/redis-stack
brew install redis-stack

# Start Redis Stack
redis-stack-server
```

#### Update your settings:
```bash
# In your environment or docker-compose.yml
REDIS_URL=redis://localhost:6379/0
```

### Option 2: Use Redis Cloud (Production Ready)

1. **Sign up**: Go to [Redis Cloud](https://redis.com/redis-enterprise-cloud/)
2. **Create database**: Choose a plan with RediSearch
3. **Get connection string**: Copy the Redis URL
4. **Update environment**:
```bash
REDIS_URL=redis://username:password@host:port/db
```

### Option 3: Docker Compose Setup

Add to your `docker-compose.yml`:

```yaml
services:
  redis:
    image: redis/redis-stack:latest
    ports:
      - "6379:6379"
      - "8001:8001"  # Redis Insight UI
    volumes:
      - redis_data:/data
    environment:
      - REDIS_ARGS=--appendonly yes

volumes:
  redis_data:
```

### Option 4: Keep Using ChromaDB

If you prefer to stick with ChromaDB:

```bash
# Simply upload a new document and it will use ChromaDB
# No additional setup needed
```

## Verification

After setting up Redis Stack, test with:

```bash
# Check Redis modules
redis-cli MODULE LIST

# Should show:
# 1) 1) "name"
#    2) "search"
#    3) "ver"
#    4) "20804"

# Test document processing
python manage.py debug_documents --document-id 20
```

## Current Fallback Behavior

Your system now works as follows:

1. **Try Redis first** (currently fails due to missing module)
2. **Fallback to ChromaDB** (works perfectly)
3. **Provide helpful error messages**

## Log Output

You'll see logs like:
```
INFO - Attempting to create Redis vector store...
WARNING - Redis vector store failed: RediSearch module not found
INFO - Chroma vector store created successfully
```

## Benefits of Redis vs ChromaDB

### Redis Advantages:
- ✅ **Better performance** for large datasets
- ✅ **Production-ready** clustering and scaling
- ✅ **Memory-based** for faster access
- ✅ **Better integration** with existing Redis infrastructure

### ChromaDB Advantages:
- ✅ **Easier setup** (no additional modules needed)
- ✅ **File-based storage** (no server required)
- ✅ **Good for development** and smaller datasets
- ✅ **Currently working** without additional setup

## Recommendation

For production use, set up **Redis Stack** (Option 1) for better performance and scalability.

For development/testing, the current **ChromaDB fallback** works perfectly fine.
