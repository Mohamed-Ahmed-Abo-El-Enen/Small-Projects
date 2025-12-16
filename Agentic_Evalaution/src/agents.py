import json
import re
from langchain.prompts import ChatPromptTemplate
from src.agent_state import AgentState
from src.pydantic_models import MCQQuestion


class QuestionGeneratorAgent:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert educational content creator specializing in mathematics.
Your task is to generate high-quality Multiple Choice Questions (MCQs) based on the provided context.

Guidelines:
- Create clear, unambiguous questions
- Ensure all options are plausible
- Only one option should be correct
- Include detailed explanations
- Match the specified difficulty level
- Base questions strictly on the provided context"""),
            ("user", """Context:
{context}

Generate {num_questions} MCQ questions about: {topic}
Difficulty level: {difficulty}

Return a JSON array with this exact structure:
[
  {{
    "question": "What is...",
    "options": {{
      "A": "Option 1",
      "B": "Option 2",
      "C": "Option 3",
      "D": "Option 4"
    }},
    "correct_answer": "B",
    "explanation": "Detailed explanation...",
    "topic": "Topic name"
  }}
]

IMPORTANT: Return ONLY the JSON array, no additional text.""")
        ])

    def generate(self, state: AgentState) -> AgentState:
        try:
            context = "\n\n".join(state["retrieved_context"])
            chain = self.prompt | self.llm
            response = chain.invoke({
                "context": context,
                "num_questions": state["num_questions"],
                "topic": state["query"],
                "difficulty": state["difficulty"]
            })

            content = response.content.strip()
            if content.startswith("```"):
                content = re.sub(r'^```json?\n|```$', '', content, flags=re.MULTILINE)

            questions = json.loads(content)

            for i, q in enumerate(questions):
                q["id"] = i + 1
                q["difficulty"] = state["difficulty"]

            state["generated_questions"] = questions
        except Exception as e:
            state["error"] = f"Question generation error: {str(e)}"
            state["generated_questions"] = []

        return state


class EvaluatorAgent:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert educational content evaluator.
Evaluate the quality of MCQ questions based on these criteria:
1. Clarity (0-1): Is the question clear and unambiguous?
2. Correctness (0-1): Is the correct answer actually correct?
3. Distractor Quality (0-1): Are wrong options plausible?
4. Explanation Quality (0-1): Is the explanation clear and helpful?
5. Alignment (0-1): Does it match the difficulty level and topic?

Provide an overall score (0-1) and specific feedback."""),
            ("user", """Evaluate this question:

Question: {question}
Options: {options}
Correct Answer: {correct_answer}
Explanation: {explanation}
Difficulty: {difficulty}

Return JSON:
{{
  "overall_score": 0.0-1.0,
  "feedback": "Brief evaluation",
  "pass": true/false
}}

IMPORTANT: Return ONLY the JSON object.""")
        ])

    def evaluate(self, state: AgentState) -> AgentState:
        try:
            evaluated = []

            for question in state["generated_questions"]:
                chain = self.prompt | self.llm
                response = chain.invoke({
                    "question": question["question"],
                    "options": json.dumps(question["options"]),
                    "correct_answer": question["correct_answer"],
                    "explanation": question["explanation"],
                    "difficulty": question["difficulty"]
                })

                content = response.content.strip()
                if content.startswith("```"):
                    content = re.sub(r'^```json?\n|```$', '', content, flags=re.MULTILINE)

                evaluation = json.loads(content)

                if evaluation.get("pass", True) and evaluation["overall_score"] >= 0.6:
                    question["evaluation_score"] = evaluation["overall_score"]
                    evaluated.append(MCQQuestion(**question))

            state["evaluated_questions"] = evaluated
        except Exception as e:
            state["error"] = f"Evaluation error: {str(e)}"
            state["evaluated_questions"] = [
                MCQQuestion(**{**q, "evaluation_score": 0.7})
                for q in state["generated_questions"]
            ]

        return state
