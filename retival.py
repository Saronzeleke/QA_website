# import logging
# from sentence_transformers import SentenceTransformer, CrossEncoder
# import chromadb
# from typing import List, Dict
# import numpy as np
# from setting import settings
# import json

# logger = logging.getLogger(__name__)

# class RetrievalSystem:
#     def __init__(self):
#         self.client = chromadb.Client()
#         self.collection = self.client.create_collection("ethiopia_travel")
#         self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
#         self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2') if settings.use_reranking else None
#         logger.info("Use pytorch device_name: cpu")
#         logger.info("Load pretrained SentenceTransformer: all-MiniLM-L6-v2")

#     def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
#         chunks = []
#         for doc in documents:
#             content = doc["content"] if isinstance(doc["content"], str) else json.dumps(doc["content"])
#             start = 0
#             doc_id = doc["id"]
#             while start < len(content):
#                 end = start + settings.chunk_size
#                 if end > len(content):
#                     end = len(content)
#                 else:
#                     while end > start and content[end] not in [' ', '\n', '.', '!', '?']:
#                         end -= 1
#                     if end == start:
#                         end = start + settings.chunk_size
#                 chunk_text = content[start:end].strip()
#                 if len(chunk_text) < 10:
#                     start = end
#                     continue
#                 metadata = {
#                     "source": doc["source"],
#                     "url": doc.get("url", ""),
#                     "title": doc["title"],
#                     "chunk_id": f"{doc_id}_{start}",
#                     "updated_at": doc["updated_at"]
#                 }
#                 if isinstance(doc["content"], dict):
#                     # Serialize lists and tables to JSON strings to comply with ChromaDB metadata requirements
#                     metadata["lists"] = json.dumps(doc["content"].get("metadata", {}).get("lists", []))
#                     metadata["tables"] = json.dumps(doc["content"].get("metadata", {}).get("tables", []))
#                 chunks.append({
#                     "id": f"{doc_id}_{start}",
#                     "text": chunk_text,
#                     "metadata": metadata
#                 })
#                 start = end
#         return chunks

#     def add_documents(self, documents: List[Dict]):
#         chunks = self.chunk_documents(documents)
#         embeddings = self.encoder.encode([chunk["text"] for chunk in chunks], show_progress_bar=True)
#         self.collection.add(
#             ids=[chunk["id"] for chunk in chunks],
#             embeddings=embeddings.tolist(),
#             metadatas=[chunk["metadata"] for chunk in chunks],
#             documents=[chunk["text"] for chunk in chunks]
#         )
#         logger.info(f"Added {len(chunks)} docs to Chroma (total docs={self.collection.count()})")

#     def _rerank_with_cross_encoder(self, query: str, results: List[Dict]) -> List[Dict]:
#         if not self.cross_encoder or len(results) <= 1:
#             return results
#         try:
#             # Truncate query and documents to avoid token limit (512 for ms-marco-MiniLM-L-6-v2)
#             max_query_len = 1000
#             max_doc_len = 400
#             query = query[:max_query_len]
#             pairs = [[query, doc["text"][:max_doc_len]] for doc in results]
#             scores = self.cross_encoder.predict(pairs)
#             scored_results = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)
#             return [result for _, result in scored_results]
#         except Exception as e:
#             logger.error(f"Reranking failed: {e}")
#             return results

#     async def hybrid_retrieval(self, query: str, k: int = settings.retrieval_k) -> List[Dict]:
#         embedding = self.encoder.encode([query])[0]
#         keyword_results = self.collection.query(
#             query_texts=[query],
#             n_results=k,
#             include=["documents", "metadatas"]
#         )
#         vector_results = self.collection.query(
#             query_embeddings=[embedding.tolist()],
#             n_results=k,
#             include=["documents", "metadatas"]
#         )
#         combined_results = []
#         seen_ids = set()
#         for results in [keyword_results, vector_results]:
#             for i in range(len(results["ids"][0])):
#                 doc_id = results["ids"][0][i]
#                 if doc_id not in seen_ids:
#                     seen_ids.add(doc_id)
#                     combined_results.append({
#                         "id": doc_id,
#                         "text": results["documents"][0][i],
#                         "metadata": results["metadatas"][0][i]
#                     })
#         return self._rerank_with_cross_encoder(query, combined_results[:k])
import logging
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from typing import List, Dict
import numpy as np
from setting import settings
import json
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

class RetrievalSystem:
    def __init__(self):
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(
            name="ethiopia_travel",
            metadata={"hnsw:space": "cosine"}
        )
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2') if settings.use_reranking else None
        self.documents = []
        self.metadatas = []
        self.ids = []
        self.bm25 = None
        self.alpha = 0.5  # Hybrid fusion weight: 0.5 for vector, 0.5 for BM25
        logger.info("Use pytorch device_name: cpu")
        logger.info("Load pretrained SentenceTransformer: all-MiniLM-L6-v2")

    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        chunks = []
        for doc in documents:
            content = doc["content"] if isinstance(doc["content"], str) else json.dumps(doc["content"])
            start = 0
            doc_id = doc["id"]
            while start < len(content):
                end = start + settings.chunk_size
                if end > len(content):
                    end = len(content)
                else:
                    while end > start and content[end] not in [' ', '\n', '.', '!', '?']:
                        end -= 1
                    if end == start:
                        end = start + settings.chunk_size
                chunk_text = content[start:end].strip()
                if len(chunk_text) < 10:
                    start = end
                    continue
                metadata = {
                    "source": doc["source"],
                    "url": doc.get("url", ""),
                    "title": doc["title"],
                    "chunk_id": f"{doc_id}_{start}",
                    "updated_at": doc["updated_at"]
                }
                if isinstance(doc["content"], dict):
                    # Serialize lists and tables to JSON strings to comply with ChromaDB metadata requirements
                    metadata["lists"] = json.dumps(doc["content"].get("metadata", {}).get("lists", []))
                    metadata["tables"] = json.dumps(doc["content"].get("metadata", {}).get("tables", []))
                chunks.append({
                    "id": f"{doc_id}_{start}",
                    "text": chunk_text,
                    "metadata": metadata
                })
                start = end
        return chunks

    def add_documents(self, documents: List[Dict]):
        chunks = self.chunk_documents(documents)
        embeddings = self.encoder.encode([chunk["text"] for chunk in chunks], show_progress_bar=True)
        self.collection.add(
            ids=[chunk["id"] for chunk in chunks],
            embeddings=embeddings.tolist(),
            metadatas=[chunk["metadata"] for chunk in chunks],
            documents=[chunk["text"] for chunk in chunks]
        )
        self.documents = [chunk["text"] for chunk in chunks]
        self.metadatas = [chunk["metadata"] for chunk in chunks]
        self.ids = [chunk["id"] for chunk in chunks]
        tokenized_corpus = [doc.lower().split(" ") for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"Added {len(chunks)} docs to Chroma and BM25 (total docs={self.collection.count()})")

    def _rerank_with_cross_encoder(self, query: str, results: List[Dict]) -> List[Dict]:
        if not self.cross_encoder or len(results) <= 1:
            return results
        try:
            # Truncate query and documents to avoid token limit (512 for ms-marco-MiniLM-L-6-v2)
            max_query_len = 1000
            max_doc_len = 400
            query = query[:max_query_len]
            pairs = [[query, doc["text"][:max_doc_len]] for doc in results]
            scores = self.cross_encoder.predict(pairs)
            scored_results = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)
            return [result for _, result in scored_results]
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return results

    async def hybrid_retrieval(self, query: str, k: int = settings.retrieval_k) -> List[Dict]:
        fetch_k = k * 2  # Fetch more candidates for better hybrid
        embedding = self.encoder.encode([query])[0]
        
        # Vector search with Chroma (cosine similarity)
        vector_results = self.collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=fetch_k,
            include=["documents", "metadatas", "distances"]
        )
        vector_ids = vector_results['ids'][0]
        vector_docs = vector_results['documents'][0]
        vector_metas = vector_results['metadatas'][0]
        vector_distances = vector_results['distances'][0]
        # Convert cosine distance to similarity (assuming normalized embeddings)
        vector_sims = [1 - dist for dist in vector_distances]
        
        # BM25 keyword search
        tokenized_query = query.lower().split(" ")
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_indices = np.argsort(bm25_scores)[::-1][:fetch_k]
        bm25_ids = [self.ids[i] for i in bm25_indices]
        bm25_docs = [self.documents[i] for i in bm25_indices]
        bm25_metas = [self.metadatas[i] for i in bm25_indices]
        bm25_scores_top = [bm25_scores[i] for i in bm25_indices]
        
        # Combine using reciprocal rank fusion or weighted scores
        # Here, we use normalized weighted sum
        all_ids = set(vector_ids + bm25_ids)
        hybrid_scores = {}
        
        # Normalize vector sims
        max_vec_sim = max(vector_sims) if vector_sims else 1.0
        vec_dict = {vector_ids[i]: vector_sims[i] / max_vec_sim if max_vec_sim > 0 else 0 for i in range(len(vector_ids))}
        
        # Normalize BM25 scores
        max_bm25 = max(bm25_scores_top) if bm25_scores_top else 1.0
        bm25_dict = {bm25_ids[i]: bm25_scores_top[i] / max_bm25 if max_bm25 > 0 else 0 for i in range(len(bm25_ids))}
        
        for id_ in all_ids:
            vec_score = vec_dict.get(id_, 0)
            bm25_score = bm25_dict.get(id_, 0)
            hybrid_scores[id_] = self.alpha * vec_score + (1 - self.alpha) * bm25_score
        
        # Sort by hybrid score descending
        sorted_ids = sorted(hybrid_scores, key=hybrid_scores.get, reverse=True)[:fetch_k]
        
        # Collect results for top hybrid
        combined_results = []
        id_to_doc = {vector_ids[i]: {"text": vector_docs[i], "metadata": vector_metas[i]} for i in range(len(vector_ids))}
        id_to_doc.update({bm25_ids[i]: {"text": bm25_docs[i], "metadata": bm25_metas[i]} for i in range(len(bm25_ids))})
        
        for id_ in sorted_ids:
            if id_ in id_to_doc:
                combined_results.append({
                    "id": id_,
                    "text": id_to_doc[id_]["text"],
                    "metadata": id_to_doc[id_]["metadata"]
                })
        
        # Rerank the combined top results
        reranked = self._rerank_with_cross_encoder(query, combined_results)
        
        # Return top k after rerank
        return reranked[:k]