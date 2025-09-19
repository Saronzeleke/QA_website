
Ethiopian Travel QA System
Overview
This project is a Question-Answering (QA) system designed to deliver fast, accurate answers to travel-related queries about Ethiopia. It employs a hybrid retrieval approach, combining vector search (semantic understanding), BM25 (keyword matching), and CrossEncoder reranking to ensure high relevance. The system integrates web-scraped data and MySQL database records, served via a FastAPI web application.
Developed by Saron Zeleke, this system is optimized for a moderate-scale corpus, handling diverse queries like "Top hotels in Addis Ababa" or "Best cultural sites in Ethiopia" with precision and speed. It is built for production-ready performance, accuracy, and maintainability.
Features

Hybrid Retrieval: Combines ChromaDB (vector search with all-MiniLM-L6-v2), BM25 (keyword search), and CrossEncoder reranking (ms-marco-MiniLM-L-6-v2) for robust document retrieval.
Web Scraping: Crawls dynamic/static pages using Selenium and aiohttp, respecting robots.txt.
Database Integration: Queries MySQL tables (e.g., hotels, tours) for structured data.
Answer Generation: Uses an LLM (Mistral via OpenAI API) to generate list-based or paragraph answers.
Caching: Redis for fast response caching.
Async API: FastAPI for high-concurrency web serving.

Why This Implementation?
This system uses the original RetrievalSystem from retrieval.py for:

Accuracy: Independent BM25 search captures specific Ethiopian terms (e.g., "Lalibela", "Axum"), critical for travel queries.
Performance: Asynchronous hybrid_retrieval integrates seamlessly with FastAPI, handling concurrent requests efficiently.
Simplicity: Minimal dependencies (no NLTK) and straightforward chunking suit the moderate corpus size (hundreds to thousands of documents).

Project Structure

main.py: FastAPI app, orchestrates web scraping, database queries, retrieval, and answer generation.
retrieval.py: Core retrieval logic with hybrid search (vector + BM25) and reranking.
db_manager.py: Handles MySQL connections and queries for structured data.
settings.py: Configuration (e.g., chunk_size, retrieval_k, database credentials).
qa.log: Logs system activity for debugging and monitoring.

Installation
Prerequisites

Python 3.8+
MySQL database with travel-related tables (e.g., bravo_hotels, bravo_tours)
Redis server
ChromeDriver (for Selenium, auto-installed via webdriver_manager or manual placement)
Dependencies: Install via requirements.txt

Setup

Clone the repository:
git clone https://github.com/Saronzeleke/RAG_Website_Scrap_QA.git
cd ethiopian-travel-qa


Install dependencies:
pip install -r requirements.txt

Required packages: fastapi, uvicorn, aiohttp, aiomysql, redis, sentence-transformers, chromadb, rank-bm25, beautifulsoup4, selenium, webdriver_manager, openai.

Configure settings:

Create settings.py based on settings.example.py.
Set:
target_url: Base URL for web crawling (e.g., https://visitethiopia.et).
Database credentials (mysql_host, mysql_user, etc.).
Redis settings (redis_host, redis_port, redis_db).
OpenAI API key (openrouter_api_key) for LLM.
Retrieval params (chunk_size, retrieval_k, use_reranking).




Start MySQL and Redis:

Ensure MySQL and Redis servers are running.
Populate MySQL with relevant travel data (e.g., hotels, tours).


Run the application:
python main.py


Access the API at http://localhost:8000.
API docs at http://localhost:8000/docs.



Usage

Endpoint: POST /ask
Request Body:{
  "question": "What are the top 3 cultural sites in Ethiopia?"
}


Response:{
  "answer": "1. Lalibela Churches - 12th century rock-hewn churches [https://visitethiopia.et/lalibela]\n2. Axum Obelisks - Ancient stelae from the Aksumite Empire [https://visitethiopia.et/axum]\n3. Gondar Castles - Royal enclosures from the 17th century [https://visitethiopia.et/gondar]",
  "sources": ["https://visitethiopia.et/lalibela", "..."],
  "format": "list"
}



How It Works

Startup:

Crawls target_url (up to 10,000 pages, 1,000 depth) using Selenium/aiohttp.
Queries MySQL tables for structured data.
Indexes documents in RetrievalSystem (ChromaDB + BM25).


Retrieval (retrieval.py):

Chunking: Splits documents into fixed-size chunks (from settings.chunk_size) at natural boundaries.
Vector Search: Encodes query/documents with all-MiniLM-L6-v2, queries ChromaDB (cosine similarity).
BM25 Search: Independent keyword search using BM25Okapi.
Fusion: Combines normalized vector and BM25 scores (alpha=0.5) for balanced relevance.
Reranking: Uses CrossEncoder to refine top k results (default k=5).


Query Processing:

Checks Redis cache for cached answers.
Queries DB for full-text matches.
Retrieves top chunks via hybrid_retrieval.
Builds context (up to 16,000 chars).
Generates answer with LLM (list or paragraph based on query type).
Caches result in Redis.



Why Fusion?

Definition: Fusion combines vector (semantic) and BM25 (keyword) scores to rank documents.
Why Used:
Handles diverse queries: Broad (e.g., "cultural sites") and specific (e.g., "Sheraton Addis Ababa").
Improves recall: Captures both meaning and exact terms, critical for Ethiopian place names (e.g., "Lalibela").
Enhances efficiency: Narrows candidates before costly reranking.


Implementation: hybrid_scores = 0.5 * vector_score + 0.5 * bm25_score. Equal weights ensure balance for travel queries.

Performance

Speed: Async retrieval and Redis caching achieve sub-second responses for cached queries (≤200 ms).
Accuracy: Independent BM25 excels for specific terms; CrossEncoder boosts precision.
Scalability: Suitable for moderate corpora (hundreds to thousands of documents). In-memory storage limits large-scale use.

Future Improvements

Tuning Fusion: Adjust alpha (e.g., 0.7 for semantic-heavy queries) based on query analysis.
Language Support: Add Amharic tokenization for BM25 to handle local terms.
Persistence: Adopt persistent ChromaDB for larger corpora.
Evaluation: Implement precision/recall metrics (e.g., NDCG) using test queries.

Contributing

Report issues or submit pull requests on the repository: https://github.com/Saronzeleke/RAG_Website_Scrap_QA.
For major changes, contact the developer.

Contact

Developer: Saron Zeleke
Email: sharonkuye369@gmail.com

License
MIT License. See LICENSE for details.