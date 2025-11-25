# LF Jobs RAG: Hybrid Job Search Engine

This project implements a Retrieval-Augmented Generation (RAG) pipeline for semantic job search. It utilizes a microservice architecture to deliver high-precision results by combining Dense Vector Search (Cohere) with Sparse Keyword Search (BM25) via LlamaIndex.

The system is fully containerized using Docker and is optimized for high-latency network environments by hosting the Vector Database and Cache layer locally.

## Documentation

For a detailed engineering report covering architectural decisions, system design diagrams, and trade-off analysis, please refer to the full documentation:

[**View Full Architecture Report (PDF)**](./RAG%20Pipeline%20for%20Job%20Data%20Retrieval.pdf)

## Key Features

*   **Hybrid Search:** Implements an ensemble retriever using LlamaIndex's `QueryFusionRetriever`, combining Qdrant (Dense) and BM25 (Sparse) with Reciprocal Rank Fusion.
*   **Semantic Reranking:** Re-scores retrieved candidates using the Cohere Rerank v3 model to ensure maximum relevance before generation.
*   **Performance Caching:** Integrated Redis layer caches identical queries, reducing response times to sub-millisecond latency.
*   **Contextual Ingestion:** Features a robust ETL pipeline that cleans HTML, injects metadata headers into text chunks, and generates deterministic UUIDs to prevent data duplication.
*   **Microservice Architecture:** Orchestrated entirely via Docker Compose, isolating the API, Database, and Cache.

## Tech Stack

*   **Framework:** FastAPI (Python 3.11)
*   **Orchestration:** LlamaIndex 0.12+
*   **Vector Database:** Qdrant (Local Docker Container)
*   **Keyword Search:** BM25 (Local Disk Persistence)
*   **LLM:** Google Gemini 1.5 Flash
*   **Embeddings:** Cohere embed-english-v3.0
*   **Cache:** Redis

## Setup and Installation

### 1. Prerequisites
*   Docker Desktop installed and running.
*   Git.
*   API Keys for Google Gemini and Cohere.

### 2. Clone Repository

```bash
git clone https://github.com/parajulitilak/RAG-Pipeline-for-Job-Data-Retrieval.git
cd RAG-Pipeline-for-Job-Data-Retrieval
```

### 3. Configuration
Create a file named `.env` in the root directory with the following configuration:

```ini
# API Keys
GOOGLE_API_KEY=your_google_api_key
COHERE_API_KEY=your_cohere_api_key

# Infrastructure (Docker Internal Networking)
QDRANT_URL=http://qdrant:6333
QDRANT_COLLECTION=lf_jobs_hybrid
REDIS_URL=redis://redis:6379

# Models
LLM_MODEL=models/gemini-1.5-flash
EMBEDDING_MODEL=embed-english-v3.0
RERANK_MODEL=rerank-english-v3.0
```

### 4. Build and Run
Start the services using Docker Compose:

```bash
docker-compose up --build -d
docker-compose up
```

*   **API Endpoint:** http://localhost:8000
*   **Swagger Documentation:** http://localhost:8000/docs
*   **Qdrant Dashboard:** http://localhost:6333/dashboard

## Data Ingestion

The system requires an initial ingestion of the `LF Jobs.xlsx` dataset to build the Vector and Keyword indices. This process creates the Qdrant collection and generates the local `docstore.json` required for BM25.

Run the following command to trigger the background ingestion process:

```bash
curl -X POST "http://localhost:8000/api/ingest"
```

**Note:** The ingestion pipeline includes a rate limiter (12-second delay per batch) to adhere to Cohere Trial API limits. The process takes approximately 80 minutes to index 1,000 jobs.

## Example Usage

### 1. Hybrid Search Query
Perform a search query via the API. The system will retrieve, rerank, and generate a structured response.

```bash
curl -X POST "http://localhost:8000/api/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "Show me Data Scientist jobs in the USA with at least 3 years of experience."}'
```

**Response:**
```json
{
  "answer": "I found 4 matches:\n* **Data Scientist** at **Grainger** (Green Bay, WI)\n    - *Match:* The job requires \"3+ years of experience in a data science role.\"\n    - *ID:* LF0033\n    - *Posted:* 2025-06-20T23:32:13Z\n* **Data Scientist, Product Analytics** at **TikTok** (Los Angeles, CA)\n .........",
  "citations": [
    {
      "id": "LF0033",
      "title": "Data Scientist",
      "company": "Grainger",
      "location": "Green Bay, WI",
      "score": 0.99437994
    },
    {citation2},
    {citation3},
    {citation4},
    {citation5}
  ],
  "processing_time": 3.7199151515960693
}

```

### 2. Cache Verification
Running the same query immediately after will trigger the Redis cache.

```json
{
  "answer": "I found 4 matches...",
  "citations": [...],
  "processing_time": 0.002749681472778
}
```
