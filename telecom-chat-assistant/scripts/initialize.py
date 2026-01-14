import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from app.services.assistant import TelecomEgyptAssistant
from app.core.config import settings


def main():
    print("=" * 60)
    print("TELECOM EGYPT ASSISTANT - INITIALIZATION")
    print("=" * 60)
    print()
    
    print(f"Model Mode: {'LOCAL (Ollama)' if settings.USE_LOCAL_MODEL else 'CLOUD (OpenAI)'}")
    if settings.USE_LOCAL_MODEL:
        print(f"Ollama URL: {settings.OLLAMA_BASE_URL}")
        print(f"Text Model: {settings.LOCAL_MODEL_NAME}")
        print(f"Vision Model: {settings.LOCAL_VISION_MODEL_NAME}")
    print()
    
    try:
        assistant = TelecomEgyptAssistant()
        
        if settings.FAISS_INDEX_PATH.exists():
            print("Existing FAISS index found.")
            choice = input("Do you want to (l)oad existing or (r)ebuild? [l/r]: ").lower()
            
            if choice == 'l':
                print("\nLoading existing index...")
                assistant.load_existing_index()
                print("System loaded successfully!")
                return
        
        print(f"\nScraping website: {settings.BASE_URL}")
        print(f"Max pages: {settings.MAX_PAGES}")
        print()
        
        assistant.initialize_from_web(max_pages=settings.MAX_PAGES)
        
        print()
        print("=" * 60)
        print("INITIALIZATION COMPLETE!")
        print("=" * 60)
        print()
        print("You can now:")
        print("1. Start the API: uvicorn app.main:app --reload")
        print("2. Start Streamlit: streamlit run app/streamlit_app.py")
        print("3. Use Docker: docker-compose up -d")
        
    except KeyboardInterrupt:
        print("\n\nInitialization cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during initialization: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()