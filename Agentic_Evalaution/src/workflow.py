from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from src.agent_state import AgentState
from src.pydantic_models import QuestionResponse
from src.vector_store import VectorStoreService
from src.config import config
from src.agents import QuestionGeneratorAgent, EvaluatorAgent


class QuestionGenerationWorkflow:
    def __init__(self, vector_service: VectorStoreService):
        self.vector_service = vector_service
        self.llm = ChatOpenAI(
            model=config.MODEL_NAME,
            temperature=config.TEMPERATURE
        )
        self.question_agent = QuestionGeneratorAgent(self.llm)
        self.evaluator_agent = EvaluatorAgent(self.llm)
        self.graph = self._build_graph()

    def _retrieve_node(self, state: AgentState) -> AgentState:
        try:
            docs = self.vector_service.retrieve_context(state["query"], k=config.SIMILARITY_SEARCH_K)
            state["retrieved_context"] = [doc.page_content for doc in docs]
        except Exception as e:
            state["error"] = f"Retrieval error: {str(e)}"
            state["retrieved_context"] = []
        return state

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("generate", self.question_agent.generate)
        workflow.add_node("evaluate", self.evaluator_agent.evaluate)
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "evaluate")
        workflow.add_edge("evaluate", END)
        return workflow.compile()

    def run(self, query: str, num_questions: int = 5,
            difficulty: str = "medium") -> QuestionResponse:
        initial_state = AgentState(
            query=query,
            num_questions=num_questions,
            difficulty=difficulty,
            retrieved_context=[],
            generated_questions=[],
            evaluated_questions=[],
            error=""
        )

        final_state = self.graph.invoke(initial_state)

        if final_state["error"]:
            raise Exception(final_state["error"])

        questions = final_state["evaluated_questions"]
        avg_score = sum(q.evaluation_score for q in questions) / len(questions) if questions else 0

        return QuestionResponse(
            questions=questions,
            total_generated=len(questions),
            average_quality_score=round(avg_score, 2)
        )

