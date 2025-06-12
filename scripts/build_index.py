# build_index.py

import logging
from pathlib import Path
from tqdm import tqdm
from ..core.config import INDEX_PATH, METADATA_PATH, EMBEDDING_DIM
from ..core.loader import load_pdfs
from ..core.chunker import chunk_texts
from ..core.embedder import embed_chunks
from ..core.vector_store import FaissIndex


logging.basicConfig(level=logging.INFO, format="✅ [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("Lecture des documents PDF...")
    docs = load_pdfs()
    

    logger.info("Chunking...")
    chunks = chunk_texts(docs)

    logger.info("Embedding des chunks...")
    embeddings = embed_chunks(chunks)


    logger.info("Construction de l'index FAISS...")
    metadatas = [{"doc_id": chunk["doc_id"], "text": chunk["text"]} for chunk in chunks]

    index = FaissIndex(dim=embeddings.shape[1])
    index.add(embeddings, metadatas)

    logger.info("Sauvegarde...")
    Path(INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
    index.save(INDEX_PATH, METADATA_PATH)
    logger.info("✅ Index sauvegardé avec succès.")


if __name__ == "__main__":
    main()
