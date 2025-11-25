import time
import logging
import hashlib
from app.core.dependencies import get_redis
from rag.engine import RAGEngineSingleton
from app.models.schemas import QueryResponse, JobCitation

logger = logging.getLogger(__name__)

class RAGService:
    """
    Service layer handling search logic, caching, and RAG orchestration.
    """

    def __init__(self):
        self.engine = RAGEngineSingleton()

    async def process_query(self, query_text: str) -> QueryResponse:
        """
        Process a user query with Redis caching and Hybrid RAG.
        
        1. Check Redis cache for normalized query hash.
        2. If miss, execute RAG pipeline (Retrieval -> Rerank -> Generate).
        3. Cache and return results.
        """
        total_start = time.time()
        redis = await get_redis()
        
        # Normalize query for cache consistency
        normalized_query = query_text.strip().lower()
        query_hash = hashlib.sha256(normalized_query.encode()).hexdigest()
        cache_key = f"rag_cache:{query_hash}"

        # 1. Check Cache
        t_cache = time.time()
        cached_json = await redis.get(cache_key)
        
        if cached_json:
            fetch_time = time.time() - t_cache
            logger.info(f"Redis Cache Hit: {normalized_query} (Time: {fetch_time:.4f}s)")
            
            cached_response = QueryResponse.model_validate_json(cached_json)
            # Update processing time to reflect actual fetch speed for this request
            cached_response.processing_time = fetch_time
            return cached_response

        logger.info(f"Redis Cache Miss. Starting Pipeline for: {normalized_query}")

        # 2. Run RAG Pipeline
        # Using aquery_detailed to log specific timing for Retrieval/Rerank/Gen steps
        response = await self.engine.aquery_detailed(query_text)
        
        # 3. Format Response
        citations = []
        seen_ids = set()
        
        # Extract and deduplicate citations from source nodes
        for node in response.source_nodes:
            meta = node.metadata
            job_id = meta.get("job_id", "Unknown")
            
            if job_id not in seen_ids:
                citations.append(JobCitation(
                    id=job_id,
                    title=meta.get("title", "Unknown"),
                    company=meta.get("company", "Unknown"),
                    location=meta.get("location", "Unknown"),
                    score=node.score if node.score else 0.0
                ))
                seen_ids.add(job_id)

        total_time = time.time() - total_start
        logger.info(f"Total Request Time: {total_time:.4f}s")

        result = QueryResponse(
            answer=str(response),
            citations=citations,
            processing_time=total_time
        )

        # 4. Save to Cache (TTL: 1 Hour)
        await redis.set(cache_key, result.model_dump_json(), ex=3600)
        
        return result