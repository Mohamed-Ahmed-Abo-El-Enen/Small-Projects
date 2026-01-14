import re
from typing import Dict, List, Optional
from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.core.config import settings
from app.services.vector_store import VectorStoreManager
from app.services.chat_history import ChatHistoryManager
from app.services.image_processor import ImageProcessor


class AgentState(TypedDict):
    """State for the LangGraph agent"""
    query: str
    conversation_id: str
    language: str
    chat_history: List[HumanMessage | AIMessage]
    retrieved_docs: List[Dict]
    context: str
    answer: str
    sources: List[Dict]
    error: Optional[str]
    step: str
    image_path: Optional[str]
    has_image: bool


class RAGWorkflowGraph:
    """LangGraph-based RAG workflow orchestration (Singleton)"""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.vector_store_manager = VectorStoreManager()
        self.chat_history_manager = ChatHistoryManager()
        self.image_processor = ImageProcessor()

        if settings.USE_LOCAL_MODEL:
            print(f"Initializing Ollama model: {settings.LOCAL_MODEL_NAME}")
            self.llm = ChatOllama(
                model=settings.LOCAL_MODEL_NAME,
                base_url=settings.OLLAMA_BASE_URL,
                temperature=0.3,
            )
        else:
            print(f"Initializing OpenAI model: {settings.LLM_MODEL}")
            self.llm = ChatOpenAI(
                model_name=settings.LLM_MODEL,
                temperature=0.3,
                max_tokens=1000
            )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """أنت مساعد ذكي لشركة تليكوم مصر.
You are an intelligent assistant for Telecom Egypt.

استخدم المعلومات التالية والمحادثة السابقة للإجابة على سؤال العميل.
Use the following information and previous conversation to answer the customer's question.

Instructions / التعليمات:
1. Answer ONLY based on the provided context / أجب فقط بناءً على السياق المقدم
2. Use conversation history for context and continuity / استخدم تاريخ المحادثة للسياق والاستمرارية
3. If the answer is not in the context, say "I don't have this information" / إذا لم تكن الإجابة في السياق، قل "ليس لدي هذه المعلومة"
4. Respond in the same language as the question ({language}) / أجب بنفس لغة السؤال
5. Be concise, professional, and conversational / كن موجزاً، مهنياً، ومحاوراً
6. Cite sources when possible / اذكر المصادر عند الإمكان
7. If analyzing an image, describe what you see and extract any relevant information / إذا كنت تحلل صورة، صف ما تراه واستخرج أي معلومات ذات صلة"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", """Context / السياق:
{context}

Question / السؤال:
{question}""")
        ])

        self.output_parser = StrOutputParser()

        self.workflow = self._build_workflow()
        self.app = self.workflow.compile(checkpointer=MemorySaver())

        self._initialized = True
        print("RAGWorkflowGraph initialized (Singleton)")

    def _detect_language(self, state: AgentState) -> AgentState:
        """Node: Detect query language and load chat history"""
        text = state["query"]
        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
        total_chars = len(re.findall(r'[a-zA-Z\u0600-\u06FF]', text))

        if total_chars == 0:
            language = "en"
        else:
            language = "ar" if (arabic_chars / total_chars) > 0.3 else "en"

        state["language"] = language

        conversation_id = state.get("conversation_id", "default")
        history = self.chat_history_manager.get_formatted_history(conversation_id)
        state["chat_history"] = history

        has_image = state.get("has_image", False)

        state["step"] = "language_detected"
        print(f"Language detected: {language}")
        print(f"Loaded {len(history)} previous messages")
        if has_image:
            print(f"Image detected in query")
        return state

    def _process_image(self, state: AgentState) -> AgentState:
        """Node: Process image if present"""
        try:
            if not state.get("has_image", False):
                state["step"] = "no_image"
                return state

            image_path = state.get("image_path")
            query = state.get("query", "")

            if not image_path:
                state["error"] = "Image path not provided"
                state["step"] = "error"
                return state

            print(f"Analyzing image: {image_path}")

            image_analysis = self.image_processor.analyze_image(image_path, query)

            state["context"] = f"Image Analysis:\n{image_analysis}\n\n"
            state["step"] = "image_processed"
            print(f"Image analyzed successfully")

        except Exception as e:
            state["error"] = f"Image processing error: {str(e)}"
            state["step"] = "error"
            print(f"Error in image processing: {str(e)}")

        return state

    def _retrieve_documents(self, state: AgentState) -> AgentState:
        """Node: Retrieve relevant documents"""
        try:
            if state.get("has_image") and not state.get("query", "").strip():
                state["retrieved_docs"] = []
                state["step"] = "documents_skipped"
                print(f"✓ Document retrieval skipped (image-only query)")
                return state

            query = state["query"]
            results = self.vector_store_manager.search(query, k=settings.TOP_K_RETRIEVAL)

            state["retrieved_docs"] = results
            state["step"] = "documents_retrieved"
            print(f"Retrieved {len(results)} relevant documents")

        except Exception as e:
            state["error"] = f"Retrieval error: {str(e)}"
            state["step"] = "error"
            print(f"Error in retrieval: {str(e)}")

        return state

    def _prepare_context(self, state: AgentState) -> AgentState:
        """Node: Prepare context from retrieved documents and image analysis"""
        try:
            results = state["retrieved_docs"]
            existing_context = state.get("context", "")

            context_parts = [existing_context] if existing_context else []
            sources = []

            for i, result in enumerate(results, 1):
                context_parts.append(
                    f"[Source {i}: {result['metadata'].get('title', 'Unknown')}]\n"
                    f"{result['content']}\n"
                )

                sources.append({
                    'title': result['metadata'].get('title', 'Unknown'),
                    'source': result['metadata'].get('source', 'Unknown'),
                    'relevance_score': result['score']
                })

            context = "\n".join(context_parts)
            state["context"] = context
            state["sources"] = sources
            state["step"] = "context_prepared"
            print(f"Context prepared from {len(sources)} sources")

        except Exception as e:
            state["error"] = f"Context preparation error: {str(e)}"
            state["step"] = "error"
            print(f"Error in context preparation: {str(e)}")

        return state

    def _generate_answer(self, state: AgentState) -> AgentState:
        """Node: Generate answer using LLM"""
        try:
            query = state["query"]
            context = state["context"]
            language = state["language"]
            chat_history = state.get("chat_history", [])

            chain = self.prompt | self.llm | self.output_parser

            answer = chain.invoke({
                "context": context,
                "question": query,
                "language": language,
                "chat_history": chat_history
            })

            state["answer"] = answer
            state["step"] = "answer_generated"
            print(f"✓ Answer generated successfully")

        except Exception as e:
            state["error"] = f"Generation error: {str(e)}"
            state["step"] = "error"
            print(f"Error in answer generation: {str(e)}")

        return state

    def _check_quality(self, state: AgentState) -> AgentState:
        """Node: Check answer quality"""
        try:
            answer = state["answer"]

            if len(answer) < 10:
                state["error"] = "Answer too short"
                state["step"] = "quality_failed"
            elif "I don't have this information" in answer or "ليس لدي هذه المعلومة" in answer:
                state["step"] = "no_answer_found"
            else:
                state["step"] = "quality_passed"

            print(f"Quality check: {state['step']}")

        except Exception as e:
            state["error"] = f"Quality check error: {str(e)}"
            state["step"] = "error"

        return state

    def _format_response(self, state: AgentState) -> AgentState:
        """Node: Format final response and save to history"""
        try:
            answer = state["answer"]
            sources = state.get("sources", [])
            conversation_id = state.get("conversation_id", "default")
            query = state["query"]
            language = state.get("language", "en")

            if sources and state["step"] == "quality_passed":
                answer += "\n\nSources / المصادر:\n"
                for i, src in enumerate(sources[:3], 1):
                    answer += f"{i}. {src['title']}\n"

            state["answer"] = answer
            state["step"] = "completed"

            self.chat_history_manager.add_message(
                conversation_id=conversation_id,
                role="user",
                content=query,
                language=language
            )

            self.chat_history_manager.add_message(
                conversation_id=conversation_id,
                role="assistant",
                content=answer,
                language=language,
                sources=sources
            )

            print(f"Response formatted with {len(sources)} sources")
            print(f"Conversation saved to database")

        except Exception as e:
            state["error"] = f"Formatting error: {str(e)}"
            state["step"] = "error"

        return state

    def _handle_error(self, state: AgentState) -> AgentState:
        """Node: Handle errors gracefully"""
        error_msg = state.get("error", "Unknown error")
        language = state.get("language", "en")

        if language == "ar":
            state["answer"] = f"عذراً، حدث خطأ: {error_msg}\nيرجى المحاولة مرة أخرى."
        else:
            state["answer"] = f"Sorry, an error occurred: {error_msg}\nPlease try again."

        state["step"] = "error_handled"
        return state

    def _should_continue(self, state: AgentState) -> str:
        """Conditional edge: Decide next step"""
        step = state.get("step", "")

        if step == "error":
            return "handle_error"
        elif step == "language_detected":
            if state.get("has_image", False):
                return "process_image"
            else:
                return "retrieve_documents"
        elif step == "image_processed" or step == "no_image":
            return "retrieve_documents"
        elif step == "quality_passed":
            return "format_response"
        elif step == "quality_failed" or step == "no_answer_found":
            return "format_response"
        else:
            return "continue"

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)

        workflow.add_node("detect_language", self._detect_language)
        workflow.add_node("process_image", self._process_image)
        workflow.add_node("retrieve_documents", self._retrieve_documents)
        workflow.add_node("prepare_context", self._prepare_context)
        workflow.add_node("generate_answer", self._generate_answer)
        workflow.add_node("check_quality", self._check_quality)
        workflow.add_node("format_response", self._format_response)
        workflow.add_node("handle_error", self._handle_error)

        workflow.set_entry_point("detect_language")

        workflow.add_conditional_edges(
            "detect_language",
            self._should_continue,
            {
                "process_image": "process_image",
                "retrieve_documents": "retrieve_documents",
                "handle_error": "handle_error"
            }
        )

        workflow.add_conditional_edges(
            "process_image",
            self._should_continue,
            {
                "retrieve_documents": "retrieve_documents",
                "handle_error": "handle_error"
            }
        )

        workflow.add_edge("retrieve_documents", "prepare_context")
        workflow.add_edge("prepare_context", "generate_answer")
        workflow.add_edge("generate_answer", "check_quality")

        workflow.add_conditional_edges(
            "check_quality",
            self._should_continue,
            {
                "format_response": "format_response",
                "handle_error": "handle_error",
                "continue": "format_response"
            }
        )

        workflow.add_edge("format_response", END)
        workflow.add_edge("handle_error", END)

        return workflow

    def process_query(self, query: str, conversation_id: str = "default",
                     image_path: str = None) -> Dict:
        """Process a query through the workflow"""
        print(f"\n{'='*60}")
        print(f"Processing Query: {query}")
        print(f"Conversation ID: {conversation_id}")
        if image_path:
            print(f"Image: {image_path}")
        print(f"{'='*60}\n")

        initial_state = {
            "query": query,
            "conversation_id": conversation_id,
            "language": "",
            "chat_history": [],
            "retrieved_docs": [],
            "context": "",
            "answer": "",
            "sources": [],
            "error": None,
            "step": "started",
            "image_path": image_path,
            "has_image": image_path is not None
        }

        config_dict = {"configurable": {"thread_id": conversation_id}}
        final_state = self.app.invoke(initial_state, config_dict)

        print(f"\n{'='*60}")
        print(f"Workflow completed: {final_state['step']}")
        print(f"{'='*60}\n")

        return {
            "query": final_state["query"],
            "answer": final_state["answer"],
            "language": final_state["language"],
            "sources": final_state.get("sources", []),
            "step": final_state["step"],
            "conversation_id": conversation_id,
            "had_image": final_state.get("has_image", False)
        }