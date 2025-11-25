from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from app.models.schemas import QueryResponse, QueryRequest
from app.services.rag_service import RAGService
from app.core.dependencies import get_rag_service
from rag.ingest import ingest_data_wrapper

router = APIRouter()

@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Search for Job Listings",
    description="Executes a Hybrid RAG search (Vector + Keyword) with Reranking and Redis Caching."
)
async def query_jobs(
    request: QueryRequest,
    service: RAGService = Depends(get_rag_service)
):
    """
    Endpoint to process natural language job queries.
    
    Args:
        request (QueryRequest): Contains the user's query string.
        service (RAGService): Injected instance of the business logic layer.

    Returns:
        QueryResponse: The AI-generated answer and list of job citations.
    """
    try:
        return await service.process_query(request.query)
    except Exception as e:
        # In production, log the stack trace here before raising 500
        raise HTTPException(status_code=500, detail=str(e))

@router.post(
    "/ingest",
    summary="Trigger Data Ingestion",
    description="Starts the ETL pipeline in the background to process 'LF Jobs.xlsx' into Qdrant and local storage."
)
async def trigger_ingestion(background_tasks: BackgroundTasks):
    """
    Asynchronous endpoint to trigger the ingestion pipeline.
    
    This allows the API to respond immediately while the heavy lifting (chunking,
    embedding, indexing) happens in a background thread.
    """
    # The file path is fixed for this assignment
    target_file = "data/LF Jobs.xlsx"
    
    background_tasks.add_task(ingest_data_wrapper, target_file)
    
    return {
        "status": "accepted", 
        "message": "Ingestion started in background. Please check server logs for progress."
    }