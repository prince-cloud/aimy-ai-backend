#!/usr/bin/env python
"""
Test script to check vector store loading and searching
"""
import os
import sys
import django

# Add the project directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up Django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from aimy.models import Document, GenericFile
from aimy.services import ChatService


def test_vector_stores():
    """Test loading and searching vector stores"""
    print("=== TESTING VECTOR STORES ===")

    # Initialize the service
    service = ChatService()

    # Test document vector stores
    print("\n--- Testing Document Vector Stores ---")
    documents = Document.objects.filter(is_processed=True)

    for doc in documents:
        print(f"\nTesting document {doc.id}: {doc.title}")

        # Try to load vector store
        vector_store = service._load_document_vector_store(doc)
        if vector_store:
            print(f"  ✓ Vector store loaded successfully")

            # Try a simple search
            try:
                results = vector_store.similarity_search_with_score("test", k=1)
                print(f"  ✓ Search test successful - found {len(results)} results")

                # Try a more specific search
                results = vector_store.similarity_search_with_score("time table", k=3)
                print(
                    f"  ✓ 'time table' search successful - found {len(results)} results"
                )

                if results:
                    print(f"  ✓ First result score: {results[0][1]}")
                    print(
                        f"  ✓ First result content preview: {results[0][0].page_content[:100]}..."
                    )

            except Exception as e:
                print(f"  ✗ Search test failed: {str(e)}")
        else:
            print(f"  ✗ Failed to load vector store")

    # Test generic file vector stores
    print("\n--- Testing Generic File Vector Stores ---")
    generic_files = GenericFile.objects.filter(is_processed=True)

    for file in generic_files:
        print(f"\nTesting generic file {file.id}: {file.title}")

        # Try to load vector store
        vector_store = service._load_generic_file_vector_store(file)
        if vector_store:
            print(f"  ✓ Vector store loaded successfully")

            # Try a simple search
            try:
                results = vector_store.similarity_search_with_score("test", k=1)
                print(f"  ✓ Search test successful - found {len(results)} results")

                # Try a more specific search
                results = vector_store.similarity_search_with_score("college", k=3)
                print(f"  ✓ 'college' search successful - found {len(results)} results")

                if results:
                    print(f"  ✓ First result score: {results[0][1]}")
                    print(
                        f"  ✓ First result content preview: {results[0][0].page_content[:100]}..."
                    )

            except Exception as e:
                print(f"  ✗ Search test failed: {str(e)}")
        else:
            print(f"  ✗ Failed to load vector store")


def test_ask_question():
    """Test the ask_question method"""
    print("\n=== TESTING ASK QUESTION ===")

    from aimy.services import chat_service

    # Test with a simple question
    test_question = "What is the time table for semester 2?"

    print(f"Testing question: '{test_question}'")

    try:
        response = chat_service.ask_question(
            question=test_question,
            user_id=1,  # Assuming user 1 exists
            max_results=5,
            temperature=0.7,
        )

        print(f"Response: {response}")

        if "answer" in response:
            print(f"Answer: {response['answer']}")
        if "error" in response:
            print(f"Error: {response['error']}")

    except Exception as e:
        print(f"Error testing ask_question: {str(e)}")


if __name__ == "__main__":
    test_vector_stores()
    test_ask_question()
