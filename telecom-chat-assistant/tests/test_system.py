import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from app.services.assistant import TelecomEgyptAssistant


def test_system():
    """Test the chatbot system with workflow and chat history"""

    print("=" * 60)
    print("TESTING TELECOM EGYPT ASSISTANT")
    print("=" * 60)

    assistant = TelecomEgyptAssistant()

    try:
        assistant.load_existing_index()
    except:
        print("No existing index found. Initializing from web...")
        assistant.initialize_from_web(max_pages=20)

    conv_id = assistant.create_new_conversation(user_id="test_user")
    print(f"\nCreated conversation: {conv_id}")

    test_queries = [
        "What are your internet packages?",
        "Tell me more about the fastest one",
        "ما هي خدمات تليكوم مصر؟",
        "عايز أعرف عن الأسعار",
        "How can I contact customer service?"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'=' * 60}")
        print(f"TEST {i}: {query}")
        print(f"Conversation ID: {conv_id}")
        print(f"{'=' * 60}")

        result = assistant.get_detailed_response(query, conv_id)

        print(f"Language: {result['language']}")
        print(f"Workflow Step: {result['step']}")
        print(f"\nAnswer:\n{result['answer']}")

        if result.get('sources'):
            print(f"\nSources ({len(result['sources'])}):")
            for j, src in enumerate(result['sources'][:3], 1):
                print(f"  {j}. {src['title']} (Score: {src['relevance_score']:.3f})")

        time.sleep(1)

    print(f"\n{'=' * 60}")
    print("CONVERSATION HISTORY")
    print(f"{'=' * 60}")

    history = assistant.get_conversation_history(conv_id)
    print(f"Total messages: {len(history)}")
    for msg in history[:5]:
        print(f"\n{msg['role'].upper()} [{msg['timestamp']}]:")
        print(f"{msg['content'][:100]}...")

    print(f"\n{'=' * 60}")
    print("ALL CONVERSATIONS")
    print(f"{'=' * 60}")

    conversations = assistant.get_all_conversations()
    for conv in conversations[:5]:
        print(f"ID: {conv['conversation_id']}")
        print(f"  Created: {conv['created_at']}")
        print(f"  Updated: {conv['last_updated']}\n")

    print(f"\n{'=' * 60}")
    print("ALL TESTS COMPLETED")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    test_system()