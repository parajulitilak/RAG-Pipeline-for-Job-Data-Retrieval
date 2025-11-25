from llama_index.llms.gemini import Gemini
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, AsyncQdrantClient
from app.core.config import get_settings

settings = get_settings()

class RAGFactory:
    """
    Factory class to instantiate RAG components (LLM, Embedding, Vector Store).
    Ensures consistent configuration and singleton-like behavior via configuration settings.
    """

    @staticmethod
    def get_llm() -> Gemini:
        """
        Creates the Google Gemini LLM instance.
        
        Returns:
            Gemini: Configured LLM instance.
        """
        return Gemini(
            api_key=settings.GOOGLE_API_KEY,
            model_name=settings.LLM_MODEL,
            temperature=0.1
        )

    @staticmethod
    def get_embedding_model(input_type: str = "search_query") -> CohereEmbedding:
        """
        Creates the Cohere Embedding model.
        
        Args:
            input_type (str): Usage context. 
                              Use 'search_query' for user queries.
                              Use 'search_document' for indexing data.
        
        Returns:
            CohereEmbedding: Configured embedding model.
        """
        return CohereEmbedding(
            cohere_api_key=settings.COHERE_API_KEY,
            model_name=settings.EMBEDDING_MODEL,
            input_type=input_type,
            embed_batch_size=48
        )

    @staticmethod
    def get_vector_store() -> QdrantVectorStore:
        """
        Creates the Qdrant Vector Store client.
        
        Note: 'enable_hybrid' is set to False because we are implementing 
        hybrid search manually using an Ensemble (Vector + BM25) approach 
        in the engine layer.
        
        Returns:
            QdrantVectorStore: Configured vector store instance.
        """
        # Initialize both sync and async clients for optimal performance
        client = QdrantClient(url=settings.QDRANT_URL)
        aclient = AsyncQdrantClient(url=settings.QDRANT_URL)

        return QdrantVectorStore(
            collection_name=settings.QDRANT_COLLECTION,
            client=client,
            aclient=aclient,
            enable_hybrid=False, 
            batch_size=48,
        )