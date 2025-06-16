# config.py

# PDF source directory
PDF_DIR = "rag/data/pdf"

# Output paths
CHUNKS_PATH = "data/chunks.pkl"
EMBEDDINGS_PATH = "data/chunks_embeddings.npy"

# Chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 20

# Embedding model
EMBEDDING_MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# LLM config
OPENAI_API_KEY = ""
OPENAI_API_BASE = ""
OPENAI_MODEL_NAME = ""


# metadata
INDEX_PATH = "../data/index/faiss_index.index"
METADATA_PATH = "../data/index/faiss_metadata.pkl"
EMBEDDING_DIM = 384  
