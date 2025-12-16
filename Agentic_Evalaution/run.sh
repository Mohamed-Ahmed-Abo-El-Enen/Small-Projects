docker build -t rag-question-api .

docker-compose up -d

docker run -p 8000:8000 --env-file .env rag-question-api

docker-compose logs -f