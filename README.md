Ethiopian Travel QA System
Overview
This project is a Question-Answering (QA) system for travel-related queries about Ethiopia, developed by Saron Zeleke. It combines web scraping, database queries, hybrid retrieval (vector + BM25), and LLM-based answer generation to deliver accurate, fast responses. The system features a FastAPI backend with CORS middleware for seamless integration with a Streamlit frontend, providing an interactive chat interface for users.
The backend uses the original RetrievalSystem (retrieval.py) for its accuracy in capturing specific Ethiopian terms (e.g., "Lalibela") and async compatibility with FastAPI. The Streamlit frontend offers a user-friendly chat UI with theme toggling and feedback collection.
Features

Hybrid Retrieval: Combines ChromaDB (vector search with all-MiniLM-L6-v2), BM25 (keyword search), and CrossEncoder reranking (ms-marco-MiniLM-L-6-v2).
Web Scraping: Crawls dynamic/static pages using Selenium/aiohttp, respecting robots.txt.
Database Integration: Queries MySQL tables (e.g., hotels, tours).
Answer Generation: Uses Mistral (via OpenAI API) for list or paragraph answers.
Frontend: Streamlit-based chat UI with dark/light themes and feedback buttons.
Caching: Redis for fast response caching.
CORS: Enables secure frontend-backend communication.
Endpoints: /ask for queries, /health for status, /feedback for user feedback.

Why This Implementation?
The original RetrievalSystem is chosen for:

Accuracy: Independent BM25 search captures specific terms (e.g., "Axum"), critical for travel queries.
Performance: Async hybrid_retrieval suits FastAPI’s concurrency.
Simplicity: Minimal dependencies, fits moderate corpus (hundreds to thousands of documents).

Project Structure

main.py: FastAPI backend with CORS, web scraping, database queries, retrieval, and answer generation.
frontend.py: Streamlit frontend for interactive chat UI.
retrieval.py: Hybrid retrieval logic (vector + BM25 + reranking).
db_manager.py: MySQL connection and query handling.
settings.py: Configuration (e.g., chunk_size, database credentials).
qa.log: Logs system activity.

Installation
Prerequisites

Python 3.8+
MySQL database with travel-related tables (e.g., bravo_hotels, bravo_tours)
Redis server
ChromeDriver (auto-installed via webdriver_manager or manual placement)
Dependencies: Install via requirements.txt

Setup

Clone the repository:
git clone https://github.com/Saronzeleke/RAG_Website_Scrap_QA.git
cd ethiopian-travel-qa


Install dependencies:
pip install -r requirements.txt

Required packages: fastapi, uvicorn, aiohttp, aiomysql, redis, sentence-transformers, chromadb, rank-bm25, beautifulsoup4, selenium, webdriver_manager, openai, streamlit.

Configure settings:

Create settings.py based on settings.example.py.
Set:
target_url: Base URL for crawling (e.g., https://visitethiopia.et).
Database credentials (mysql_host, mysql_user, etc.).
Redis settings (redis_host, redis_port, redis_db).
OpenAI API key (openrouter_api_key).
Retrieval params (chunk_size, retrieval_k, use_reranking).




Start MySQL and Redis:

Ensure MySQL and Redis servers are running.
Populate MySQL with travel data (e.g., hotels, tours).


Run the backend:
python main.py


FastAPI runs on http://localhost:8000.
API docs at http://localhost:8000/docs.


Run the frontend:
streamlit run frontend.py


Streamlit runs on http://127.0.0.1:8501.



Usage

Frontend: Open http://127.0.0.1:8501, click the chat button, and ask questions (e.g., "What are the top 3 cultural sites in Ethiopia?").
Backend API:
POST /ask:{
  "question": "What are the top 3 cultural sites in Ethiopia?"
}

Response:{
  "answer": "1. Lalibela Churches - 12th century rock-hewn churches [https://visitethiopia.et/lalibela]\n2. Axum Obelisks - Ancient stelae from the Aksumite Empire [https://visitethiopia.et/axum]\n3. Gondar Castles - Royal enclosures from the 17th century [https://visitethiopia.et/gondar]",
  "sources": ["https://visitethiopia.et/lalibela", "..."],
  "format": "list"
}


GET /health: Returns {"status": "healthy", "message": "Backend is running"}.
POST /feedback:{
  "question": "What are the top hotels in Addis Ababa?",
  "answer": "1. Sheraton Addis - Luxury hotel with great amenities...",
  "was_helpful": true
}





How It Works

Startup:

Backend crawls target_url (up to 10,000 pages, 1,000 depth).
Queries MySQL tables for structured data.
Indexes documents in RetrievalSystem (ChromaDB + BM25).


Retrieval (retrieval.py):

Chunking: Splits documents into fixed-size chunks at natural boundaries.
Vector Search: Uses all-MiniLM-L6-v2 and ChromaDB (cosine similarity).
BM25 Search: Independent keyword search with BM25Okapi.
Fusion: Combines vector and BM25 scores (alpha=0.5) for balanced relevance.
Reranking: Refines top k results with CrossEncoder.


Query Processing:

Frontend sends query to /ask.
Backend checks Redis cache, queries DB, retrieves chunks, builds context, and generates answer.
Response is cached in Redis and displayed in the Streamlit chat UI.
Users can provide feedback via buttons, sent to /feedback.


CORS: Allows Streamlit (http://127.0.0.1:8501) to communicate with FastAPI (http://localhost:8000).


Why Fusion?

Definition: Combines vector (semantic) and BM25 (keyword) scores to rank documents.
Why Used:
Handles diverse queries: Broad (e.g., "cultural sites") and specific (e.g., "Sheraton Addis Ababa").
Improves recall: Captures both meaning and exact terms (e.g., "Lalibela").
Enhances efficiency: Narrows candidates before reranking.


Implementation: hybrid_scores = 0.5 * vector_score + 0.5 * bm25_score.

Performance

Speed: Async backend and Redis caching achieve sub-second responses (≤200 ms for cached queries).
Accuracy: Independent BM25 excels for specific terms; CrossEncoder boosts precision.
Scalability: Suitable for moderate corpora. In-memory storage limits large-scale use.

Future Improvements

Tuning Fusion: Adjust alpha based on query type.
Language Support: Add Amharic tokenization for BM25.
Persistence: Use persistent ChromaDB for larger corpora.
Evaluation: Add precision/recall metrics (e.g., NDCG).

Contributing

Submit issues/pull requests: https://github.com/Saronzeleke/RAG_Website_Scrap_QA.
Contact the developer for major changes.

Contact

Developer: Saron Zeleke
Email: sharonkuye369@gmail.com

License
MIT License. See LICENSE for details.