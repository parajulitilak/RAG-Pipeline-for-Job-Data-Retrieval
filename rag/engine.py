import os
import time
import logging
import re
from typing import List, Any

from llama_index.core import (
    VectorStoreIndex, 
    StorageContext, 
    Settings, 
    QueryBundle
)
from llama_index.core.base.response.schema import Response
from llama_index.core.retrievers import VectorIndexRetriever, QueryFusionRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.postprocessor.cohere_rerank import CohereRerank
from rag.components import RAGFactory
from app.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

GEMINI_PROMPT_TEMPLATE = """
<system_instructions>
You are an Elite AI Technical Recruiter for the "LF Jobs" platform.
Your task is to match candidates to job openings with extreme precision using *only* the provided database.

### CRITICAL ANALYSIS PROCESS:
1. Filter by Constraints: Check Location, Level, Category, and Date if specified.
2. Location Logic:
   - If user says "New York", match "NYC", "Manhattan", "New York".
   - If user says "Remote", strictly match "Remote".
3. Evidence Check: Quote specific skills from <details> or <tags>.

### NEGATIVE CONSTRAINTS:
- Do not invent information.
- Do not output XML tags in the final response.

### OUTPUT FORMATTING:
- Headline: "I found [X] matches:"
- Job List (Max 5):
  * **[Title]** at **[Company]** ([Location])
    - *Match:* [Reason citing specific skills/tags]
    - *ID:* [LF####]
    - *Posted:* [Date]
</system_instructions>

<available_jobs>
{context_str}
</available_jobs>

<user_query>
{query_str}
</user_query>

<assistant_response>
"""

class RAGEngineSingleton:
    """
    Singleton class managing the RAG (Retrieval-Augmented Generation) pipeline.
    
    Implements a Hybrid Search architecture:
    1. Dense Retrieval: Qdrant (Semantic Search)
    2. Sparse Retrieval: BM25 (Keyword Search from local DocStore)
    3. Fusion: Reciprocal Rank Fusion
    4. Reranking: Cohere Rerank v3
    5. Synthesis: Google Gemini via XML Prompting
    """
    
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RAGEngineSingleton, cls).__new__(cls)
        return cls._instance

    def initialize(self) -> None:
        """
        Initializes the LlamaIndex components and loads indices.
        Idempotent: Will simply return if already initialized.
        """
        if self._initialized:
            return

        logger.info("Initializing Hybrid Ensemble Engine...")
        
        # 1. Configure Global LlamaIndex Settings
        Settings.llm = RAGFactory.get_llm()
        Settings.embed_model = RAGFactory.get_embedding_model(input_type="search_query")

        # 2. Setup Dense Vector Retriever (Qdrant)
        vector_store = RAGFactory.get_vector_store()
        vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        vector_retriever = VectorIndexRetriever(
            index=vector_index,
            similarity_top_k=20
        )

        # 3. Setup Sparse Keyword Retriever (BM25)
        # Requires local docstore.json generated during ingestion
        bm25_retriever = None
        try:
            storage_path = "./data/storage"
            if os.path.exists(storage_path) and os.path.exists(os.path.join(storage_path, "docstore.json")):
                logger.info("Loading DocStore for BM25...")
                storage_context = StorageContext.from_defaults(persist_dir=storage_path)
                
                bm25_retriever = BM25Retriever.from_defaults(
                    docstore=storage_context.docstore,
                    similarity_top_k=20
                )
                logger.info("BM25 Keyword Index loaded successfully.")
            else:
                logger.warning("No local DocStore found. Running in Vector-Only mode.")
        except Exception as e:
            logger.error(f"BM25 Load Failed: {e}")

        # 4. Setup Hybrid Fusion
        if bm25_retriever:
            logger.info("Fusing Vector + BM25 Retrievers...")
            self.retriever = QueryFusionRetriever(
                [vector_retriever, bm25_retriever],
                similarity_top_k=20,
                num_queries=1,
                mode="reciprocal_rerank",
                use_async=True
            )
        else:
            logger.info("Using Vector Retriever Only.")
            self.retriever = vector_retriever

        # 5. Setup Reranker
        self.reranker = CohereRerank(
            api_key=settings.COHERE_API_KEY,
            model=settings.RERANK_MODEL,
            top_n=5
        )
        
        self._initialized = True
        logger.info("RAG Engine initialized successfully.")

    def _format_context_xml(self, nodes: List[NodeWithScore]) -> str:
        """
        Formats retrieved nodes into an XML structure for the LLM prompt.
        
        Args:
            nodes (List[NodeWithScore]): List of ranked nodes.

        Returns:
            str: A single string containing XML-formatted job data.
        """
        context_list = []
        for node in nodes:
            meta = node.metadata
            
            # Get Raw Text
            raw_text = node.get_content()
            
            # Remove the "Header" injected during ingestion to avoid token duplication.
            # The header typically looks like "Job Title: ... | ... Level: ... .\n"
            clean_text = re.sub(r"^Job Title:.*?Level:.*?\.\n", "", raw_text, flags=re.DOTALL)
            clean_text = clean_text.strip() 
            
            # Construct XML Block
            job_block = f"""
            <job id="{meta.get('job_id', 'N/A')}">
                <title>{meta.get('title', 'N/A')}</title>
                <company>{meta.get('company', 'N/A')}</company>
                <location>{meta.get('location', 'N/A')}</location>
                <level>{meta.get('level', 'N/A')}</level>
                <category>{meta.get('category', 'N/A')}</category>
                <tags>{meta.get('tags', '')}</tags>
                <date>{meta.get('date', 'N/A')}</date>
                <details>
                {clean_text[:5000]}
                </details>
            </job>
            """
            context_list.append(job_block)
        
        return "\n".join(context_list)

    async def aquery(self, query_text: str) -> Response:
        """
        Execute a standard async query.
        
        Args:
            query_text (str): The user's search query.

        Returns:
            Response: LlamaIndex Response object containing text and source nodes.
        """
        return await self._run_pipeline(query_text, debug=False)

    async def aquery_detailed(self, query_text: str) -> Response:
        """
        Execute a query with detailed timing logs printed to stdout.
        Useful for debugging latency.
        """
        return await self._run_pipeline(query_text, debug=True)

    async def _run_pipeline(self, query_text: str, debug: bool = False) -> Response:
        """
        Internal pipeline execution logic:
        Retrieve -> Rerank -> Format XML -> Generate.
        """
        if not self._initialized:
            self.initialize()
        
        if debug:
            print("\n[START HYBRID PIPELINE]")
        
        # 1. Retrieval
        t0 = time.time()
        nodes = await self.retriever.aretrieve(query_text)
        if debug:
            print(f"[TIMING] 1. Retrieval:       {time.time()-t0:.4f}s | Found {len(nodes)} nodes")

        # 2. Reranking
        t1 = time.time()
        query_bundle = QueryBundle(query_text)
        ranked_nodes = self.reranker.postprocess_nodes(nodes, query_bundle)
        if debug:
            print(f"[TIMING] 2. Reranking:       {time.time()-t1:.4f}s | Selected Top {len(ranked_nodes)}")

        # 3. XML Formatting
        xml_context = self._format_context_xml(ranked_nodes)
        
        # 4. Prompt Construction
        final_prompt = GEMINI_PROMPT_TEMPLATE.format(
            context_str=xml_context,
            query_str=query_text
        )

        if debug:
            print("\n[DEBUG] XML CONTEXT SENT TO GEMINI")
            print("-" * 40)
            print(xml_context[:600] + "... (truncated)")
            print("-" * 40 + "\n")

        # 5. Generation
        t2 = time.time()
        response_text = await Settings.llm.acomplete(final_prompt)
        
        if debug: 
            print(f"[TIMING] 3. Generation:      {time.time()-t2:.4f}s")
            print("[END PIPELINE]\n")
        
        # 6. Return Response
        return Response(
            response=str(response_text),
            source_nodes=ranked_nodes
        )