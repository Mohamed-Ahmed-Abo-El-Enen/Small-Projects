import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from app.core.config import settings
from app.services.assistant import TelecomEgyptAssistant


st.set_page_config(
    page_title="Telecom Egypt Assistant",
    page_icon="📱",
    layout="wide"
)

st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .main-header h1 {
        color: white;
        margin: 0;
    }
    .chat-message {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .user-message {
        background-color: #e0f2fe;
        border-left: 4px solid #3b82f6;
    }
    .assistant-message {
        background-color: #f0fdf4;
        border-left: 4px solid #10b981;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="main-header">
        <h1>📱 Telecom Egypt Intelligent Assistant</h1>
        <p style="color: white; margin: 0;">Your AI-powered customer support</p>
    </div>
""", unsafe_allow_html=True)

if 'assistant' not in st.session_state:
    st.session_state.assistant = None

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'conversation_id' not in st.session_state:
    st.session_state.conversation_id = None

if 'all_conversations' not in st.session_state:
    st.session_state.all_conversations = []

if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None

with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("🤖 Model Configuration")
    if settings.USE_LOCAL_MODEL:
        st.success(f"✓ Local Model: {settings.LOCAL_MODEL_NAME}")
        st.info(f"Ollama: {settings.OLLAMA_BASE_URL}")
    else:
        st.success(f"✓ Cloud Model: {settings.LLM_MODEL}")

    st.divider()

    st.subheader("1. Initialize System")

    init_option = st.radio(
        "Choose initialization method:",
        ["Load Existing Index", "Scrape Website (First Time)"]
    )

    if init_option == "Scrape Website (First Time)":
        max_pages = st.slider("Max pages to crawl", 10, 100, 50)
        if st.button("🚀 Initialize from Website"):
            with st.spinner("Initializing... This may take a few minutes..."):
                assistant = TelecomEgyptAssistant()
                assistant.initialize_from_web(max_pages)
                st.session_state.assistant = assistant

                st.session_state.conversation_id = assistant.create_new_conversation()
                st.success("✅ System initialized!")
    else:
        if st.button("📂 Load Existing Index"):
            with st.spinner("Loading..."):
                assistant = TelecomEgyptAssistant()
                assistant.load_existing_index()
                st.session_state.assistant = assistant

                st.session_state.conversation_id = assistant.create_new_conversation()
                st.success("✅ System loaded!")

    st.divider()

    if st.session_state.assistant:
        st.subheader("2. Conversations")

        if st.button("➕ New Conversation"):
            st.session_state.conversation_id = st.session_state.assistant.create_new_conversation()
            st.session_state.chat_history = []
            st.success("✓ New conversation started!")
            st.rerun()

        if st.button("🔄 Refresh Conversations"):
            st.session_state.all_conversations = st.session_state.assistant.get_all_conversations()

        if st.session_state.all_conversations:
            conv_options = {
                f"{conv['conversation_id'][:20]}... ({conv['last_updated'][:10]})": conv['conversation_id']
                for conv in st.session_state.all_conversations
            }

            selected = st.selectbox("Load conversation:", list(conv_options.keys()))

            if st.button("📖 Load Selected"):
                st.session_state.conversation_id = conv_options[selected]
                history = st.session_state.assistant.get_conversation_history(st.session_state.conversation_id)
                st.session_state.chat_history = history
                st.success(f"✓ Loaded conversation with {len(history)} messages")
                st.rerun()

        if st.session_state.conversation_id:
            st.info(f"📌 Current: {st.session_state.conversation_id[:30]}...")

    st.divider()

    if st.session_state.assistant:
        st.subheader("3. Upload Documents")
        uploaded_file = st.file_uploader(
            "Add documents to knowledge base",
            type=['pdf', 'docx', 'txt', 'html', 'png', 'jpg', 'jpeg', 'webp']
        )

        if uploaded_file:
            if st.button("📄 Add Document"):
                save_path = f"./temp_{uploaded_file.name}"
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                with st.spinner("Processing document..."):
                    st.session_state.assistant.add_document(save_path)
                    st.success(f"✅ Added: {uploaded_file.name}")

                os.remove(save_path)

    st.divider()

    if st.session_state.assistant:
        st.subheader("4. Upload Image for Query")
        query_image = st.file_uploader(
            "Upload an image to analyze",
            type=['png', 'jpg', 'jpeg', 'webp'],
            key="query_image"
        )

        if query_image:
            st.image(query_image, caption="Uploaded Image", use_container_width=True)

            save_path = f"./temp_query_{query_image.name}"
            with open(save_path, "wb") as f:
                f.write(query_image.getbuffer())
            st.session_state.uploaded_image = save_path
            st.success("✓ Image ready for analysis")
        elif st.session_state.uploaded_image:
            if st.button("🗑️ Clear Image"):
                if os.path.exists(st.session_state.uploaded_image):
                    os.remove(st.session_state.uploaded_image)
                st.session_state.uploaded_image = None
                st.rerun()

    st.divider()

    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

if st.session_state.assistant is None:
    st.info("👈 Please initialize the system using the sidebar")
else:
    chat_container = st.container()

    with chat_container:
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>👤 You:</strong><br>
                        {message['content']}
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <strong>🤖 Assistant:</strong><br>
                        {message['content']}
                    </div>
                """, unsafe_allow_html=True)

    query = st.chat_input("Ask me anything about Telecom Egypt... / اسألني عن تليكوم مصر...")

    if query:
        st.session_state.chat_history.append({
            'role': 'user',
            'content': query
        })

        with st.spinner("🤔 Thinking..."):
            image_path = st.session_state.uploaded_image
            response = st.session_state.assistant.chat(
                query,
                st.session_state.conversation_id,
                image_path=image_path
            )

            if image_path and os.path.exists(image_path):
                os.remove(image_path)
                st.session_state.uploaded_image = None

        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': response
        })

        st.rerun()