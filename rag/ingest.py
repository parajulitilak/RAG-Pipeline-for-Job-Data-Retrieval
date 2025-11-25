import os
import time
import logging
import re
import uuid
import pandas as pd
from bs4 import BeautifulSoup
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import LangchainNodeParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag.components import RAGFactory

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', force=True)

class JobIngestor:
    """
    JobIngestor handles the ETL (Extract, Transform, Load) pipeline for job data.
    
    It performs the following operations:
    1. Loads job data from Excel.
    2. Cleans HTML and normalizes location data.
    3. Chunks text using recursive splitting with context injection.
    4. Generates deterministic IDs to ensure idempotency.
    5. Persists text to a local DocStore (for BM25 retrieval).
    6. Uploads vectors to Qdrant in rate-limited batches (for Dense retrieval).
    """

    # Mapping for canonical location normalization
    LOCATION_MAP = {
        "nyc": "new york", "sf": "san francisco", "bay area": "san francisco",
        "remote": "remote", "uk": "london",
    }
    
    # Configuration for batch processing
    BATCH_SIZE = 48
    DELAY_SECONDS = 12
    PERSIST_DIR = "./data/storage"
    LOCK_FILE = "/tmp/ingest.lock"

    def __init__(self):
        """
        Initialize the Ingestion Pipeline components.
        Sets up LlamaIndex settings and connects to the Qdrant Vector Store.
        """
        # 1. Configure Global Settings
        Settings.embed_model = RAGFactory.get_embedding_model(input_type="search_document")
        Settings.llm = RAGFactory.get_llm()
        
        # 2. Connect to Qdrant
        self.vector_store = RAGFactory.get_vector_store()
        
        # 3. Initialize Storage Context (Manages DocStore & VectorStore connection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

    def _clean_html(self, html_content: str) -> str:
        """
        Clean HTML content from job descriptions while preserving structural meaning.
        
        Args:
            html_content (str): Raw HTML string from the dataset.

        Returns:
            str: Cleaned text with <li> converted to bullets and excessive whitespace removed.
        """
        if not isinstance(html_content, str) or "<" not in html_content:
            return str(html_content).strip()
            
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Preserve list structure
        for li in soup.find_all("li"):
            li.insert_before("• ")
            li.insert_after("\n")
            
        # Preserve line breaks
        for br in soup.find_all("br"):
            br.replace_with("\n")
            
        text = soup.get_text()
        # Normalize paragraphs (max 2 newlines)
        return re.sub(r'\n{3,}', '\n\n', text).strip()

    def _normalize_location(self, loc: str) -> str:
        """
        Normalize raw location strings to canonical formats (e.g., "NYC" -> "new york").
        
        Args:
            loc (str): Raw location string.

        Returns:
            str: Normalized location string.
        """
        if not isinstance(loc, str): return "unknown"
        loc_lower = loc.lower().strip()
        for key, val in self.LOCATION_MAP.items():
            if key in loc_lower: return val
        return loc_lower

    def _create_nodes_from_row(self, row: pd.Series, parser: LangchainNodeParser) -> list[Document]:
        """
        Process a single dataframe row into chunked LlamaIndex Documents.
        
        Features:
        - Extracts metadata.
        - Cleans text.
        - Injects context header (Title/Company) into every chunk.
        - Generates deterministic UUIDs based on Job ID.

        Args:
            row (pd.Series): A row from the jobs DataFrame.
            parser (LangchainNodeParser): The text splitter instance.

        Returns:
            list[Document]: A list of processed nodes ready for ingestion.
        """
        meta = {
            "job_id": str(row.get("ID", "LF0000")),
            "title": str(row.get("Job Title", "Unknown")),
            "company": str(row.get("Company Name", "Unknown")),
            "location": str(row.get("Job Location", "Unknown")),
            "location_canonical": self._normalize_location(str(row.get("Job Location", ""))),
            "level": str(row.get("Job Level", "Unknown")),
            "category": str(row.get("Job Category", "Unknown")),
            "tags": str(row.get("Tags", "")),
            "date": str(row.get("Publication Date", ""))
        }

        raw_desc = row.get("Job Description", "")
        clean_desc = self._clean_html(raw_desc)

        # Context Injection: Prepended to every chunk for semantic clarity
        context_header = (
            f"Job Title: {meta['title']} | "
            f"Company: {meta['company']} | "
            f"Location: {meta['location']} | "
            f"Level: {meta['level']}.\n"
        )

        doc = Document(text=clean_desc, metadata=meta)
        job_nodes = parser.get_nodes_from_documents([doc])

        processed_nodes = []
        for i, node in enumerate(job_nodes):
            # Deterministic ID Generation (Idempotency)
            # Ensures re-running ingestion updates existing vectors instead of duplicating
            unique_string = f"{meta['job_id']}_{i}"
            node.id_ = str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_string))
            
            # Inject header
            node.text = context_header + node.text
            
            # Exclude internal metadata from LLM context to save tokens
            node.excluded_llm_metadata_keys = ["location_canonical", "job_id", "date"]
            node.excluded_embed_metadata_keys = ["job_id", "location_canonical", "date"]
            processed_nodes.append(node)
            
        return processed_nodes

    def ingest(self, file_path: str = "data/LF Jobs.xlsx") -> dict:
        """
        Main execution method for the ingestion pipeline.
        
        Flow:
        1. Acquire Lock (prevent concurrent runs).
        2. Load and Process Data into Nodes.
        3. Persist Nodes to Disk (DocStore) for BM25.
        4. Upload Vectors to Qdrant in Batches.
        """
        # Simple File Lock Mechanism
        if os.path.exists(self.LOCK_FILE):
            logger.warning("Ingestion already running (Lock file exists).")
            return {"status": "error", "message": "Ingestion already running"}

        with open(self.LOCK_FILE, 'w') as f: f.write("locked")

        try:
            logger.info("Starting Production Ingestion...")

            if not os.path.exists(file_path):
                return {"status": "error", "message": "File not found"}

            # 1. Load Data
            df = pd.read_excel(file_path).fillna("")
            logger.info(f"Loaded {len(df)} rows.")

            # 2. Configure Splitter
            lc_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, 
                chunk_overlap=50, 
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            node_parser = LangchainNodeParser(lc_splitter)

            # 3. Create Nodes
            all_nodes = []
            for _, row in df.iterrows():
                nodes = self._create_nodes_from_row(row, node_parser)
                all_nodes.extend(nodes)

            logger.info(f"Generated {len(all_nodes)} nodes.")

            # 4. Save DocStore to Disk (Critical for BM25 Keyword Search)
            # We save BEFORE uploading vectors to ensure hybrid search works even if upload fails later.
            logger.info("Registering nodes in DocStore...")
            self.storage_context.docstore.add_documents(all_nodes)
            
            if not os.path.exists(self.PERSIST_DIR):
                os.makedirs(self.PERSIST_DIR)
            
            self.storage_context.persist(persist_dir=self.PERSIST_DIR)
            logger.info("DocStore persisted to disk. Text data is safe.")

            # 5. Batch Vector Upload (Qdrant)
            # We re-initialize index from storage to link DocStore + VectorStore
            index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store, 
                storage_context=self.storage_context
            )

            logger.info(f"Upserting vectors to Qdrant... Batch: {self.BATCH_SIZE}")

            for i in range(0, len(all_nodes), self.BATCH_SIZE):
                batch = all_nodes[i : i + self.BATCH_SIZE]
                
                # This generates embeddings via Cohere and uploads to Qdrant
                index.insert_nodes(batch)
                
                logger.info(f"Batch {(i//self.BATCH_SIZE)+1} upserted.")
                
                # Rate limit protection for Cohere Trial Key
                if i + self.BATCH_SIZE < len(all_nodes):
                    time.sleep(self.DELAY_SECONDS)

            # Final persist to ensure all index metadata is consistent
            self.storage_context.persist(persist_dir=self.PERSIST_DIR)
            logger.info("Ingestion Fully Complete.")
            return {"status": "success", "message": f"Ingested {len(all_nodes)} nodes"}

        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            return {"status": "error", "message": str(e)}
        
        finally:
            # Release Lock
            if os.path.exists(self.LOCK_FILE):
                os.remove(self.LOCK_FILE)

def ingest_data_wrapper(file_path: str):
    """Wrapper function for API calls."""
    ingestor = JobIngestor()
    ingestor.ingest(file_path)