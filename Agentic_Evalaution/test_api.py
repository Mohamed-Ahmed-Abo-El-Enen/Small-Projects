import pytest
from fastapi.testclient import TestClient
from pathlib import Path
from main import app

client = TestClient(app)


class TestHealthCheck:
    """Test health check endpoint"""
    def test_health_endpoint(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data


class TestIngestEndpoint:
    """Test PDF ingestion endpoint"""
    def test_ingest_pdf_success(self):
        """Test successful PDF ingestion"""
        pdf_path = Path("./data/uploads/A Quick Algebra Review (1).pdf")

        if not pdf_path.exists():
            pytest.skip("Test PDF not found")

        with open(pdf_path, "rb") as f:
            response = client.post(
                "/ingest",
                files={"file": ("test.pdf", f, "application/pdf")}
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "document_id" in data
        assert "table_of_contents" in data
        assert "total_chunks" in data
        assert len(data["table_of_contents"]) > 0

    def test_ingest_without_file(self):
        """Test ingestion without file"""
        response = client.post("/ingest")
        assert response.status_code == 422


class TestQuestionGeneration:
    """Test question generation endpoint"""

    def test_generate_questions_valid_request(self):
        """Test question generation with valid request"""
        request_data = {
            "query": "quadratic equations",
            "num_questions": 3,
            "difficulty": "medium"
        }

        response = client.post("/generate/questions", json=request_data)

        if response.status_code == 200:
            data = response.json()
            assert "questions" in data
            assert "total_generated" in data
            assert "average_quality_score" in data
            assert len(data["questions"]) > 0

            question = data["questions"][0]
            assert "id" in question
            assert "question" in question
            assert "options" in question
            assert "correct_answer" in question
            assert "explanation" in question
            assert "difficulty" in question
            assert "evaluation_score" in question

    def test_generate_questions_invalid_difficulty(self):
        """Test with invalid difficulty level"""
        request_data = {
            "query": "linear equations",
            "num_questions": 3,
            "difficulty": "invalid"
        }

        response = client.post("/generate/questions", json=request_data)
        assert response.status_code == 422

    def test_generate_questions_too_many(self):
        """Test requesting too many questions"""
        request_data = {
            "query": "algebra",
            "num_questions": 15,
            "difficulty": "easy"
        }

        response = client.post("/generate/questions", json=request_data)
        assert response.status_code == 422

    def test_generate_questions_missing_query(self):
        """Test without query parameter"""
        request_data = {
            "num_questions": 5,
            "difficulty": "medium"
        }

        response = client.post("/generate/questions", json=request_data)
        assert response.status_code == 422


class TestEndToEnd:
    """End-to-end integration tests"""

    def test_full_workflow(self):
        """Test complete workflow: ingest then generate"""
        pdf_path = Path("./data/uploads/A Quick Algebra Review (1).pdf")

        if not pdf_path.exists():
            pytest.skip("Test PDF not found")

        with open(pdf_path, "rb") as f:
            ingest_response = client.post(
                "/ingest",
                files={"file": ("test.pdf", f, "application/pdf")}
            )

        assert ingest_response.status_code == 200

        request_data = {
            "query": "solving equations",
            "num_questions": 2,
            "difficulty": "easy"
        }

        gen_response = client.post("/generate/questions", json=request_data)

        if gen_response.status_code == 200:
            data = gen_response.json()
            assert data["total_generated"] > 0


class TestQuestionQuality:
    """Test quality of generated questions"""

    def test_question_format(self):
        """Test that questions follow correct format"""
        request_data = {
            "query": "exponents",
            "num_questions": 1,
            "difficulty": "medium"
        }

        response = client.post("/generate/questions", json=request_data)

        if response.status_code == 200:
            data = response.json()
            if len(data["questions"]) > 0:
                question = data["questions"][0]

                assert set(question["options"].keys()) == {"A", "B", "C", "D"}

                assert question["correct_answer"] in ["A", "B", "C", "D"]

                assert 0 <= question["evaluation_score"] <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])