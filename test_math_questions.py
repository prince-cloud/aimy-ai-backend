#!/usr/bin/env python3
"""
Test script to verify mathematical question handling
"""

import os
import sys
import django

# Add the project directory to the Python path
sys.path.append("/Users/acheampongprince/dev/projects/suetrex/aimy-ai-backend")

# Set up Django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from aimy.services import ChatService
from aimy.models import ChatSession, CustomUser


def test_mathematical_questions():
    """Test various mathematical questions"""

    # Create a test user (or use existing one)
    try:
        user = CustomUser.objects.get(email="test@example.com")
    except CustomUser.DoesNotExist:
        user = CustomUser.objects.create_user(
            email="test@example.com",
            password="testpass123",
            first_name="Test",
            last_name="User",
        )

    # Create a chat service instance
    chat_service = ChatService()

    # Test questions
    test_questions = [
        "1+1",
        "What is 2+3?",
        "Calculate 10*5",
        "What's 15/3?",
        "How much is 7-4?",
        "What is the capital of Ghana?",  # Non-mathematical question
        "Tell me about KNUST admission requirements",  # KNUST question
    ]

    print("Testing mathematical question handling...")
    print("=" * 50)

    for question in test_questions:
        print(f"\nQuestion: {question}")
        try:
            # Create a session for testing
            session = ChatSession.objects.create(
                user=user, title=f"Test session for: {question[:30]}"
            )

            # Ask the question
            response = chat_service.ask_question(
                question=question, user_id=user.id, session_id=session.id
            )

            print(f"Answer: {response.get('answer', 'No answer')}")
            print(f"Confidence: {response.get('confidence_score', 'N/A')}")

        except Exception as e:
            print(f"Error: {str(e)}")

        print("-" * 30)


if __name__ == "__main__":
    test_mathematical_questions()
