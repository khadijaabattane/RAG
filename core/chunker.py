from collections.abc import Iterable
from typing import Any, Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from .config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_texts(docs: Iterable[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Split a collection of raw documents into fixed‑size, overlapping text chunks.

    The function is a light wrapper around
    :class:`langchain_text_splitters.RecursiveCharacterTextSplitter`.  
    It converts each input document into one or more shorter fragments
    (“chunks”) that are easier to embed, index, or process with large‑language‑
    model pipelines.

    Parameters
    ----------
    docs
        An iterable of dictionaries where **each** dictionary must contain  

        • ``"doc_id"`` – a unique identifier for the original document  
        • ``"text"``   – the full textual content of the document

    Returns
    -------
    list[dict[str, str]]
        A flat list of chunk dictionaries.  Each dictionary preserves the
        original ``"doc_id"`` and adds the chunk under the key ``"text"``.  The
        order of chunks follows the order of the source documents and the order
        produced by the text splitter.

    Notes
    -----
    * Chunk boundaries are controlled by the global configuration constants
      ``CHUNK_SIZE`` (maximum characters per chunk) and ``CHUNK_OVERLAP``  
      (number of characters overlapped between consecutive chunks).

    * The ``separators`` argument is tuned to respect natural language
      structure—paragraphs, sentences, and words—before falling back to
      character‑level splitting.

    Examples
    --------
    >>> docs = [
    ...     {"doc_id": 42, "text": "Hello world. How are you today?"},
    ... ]
    >>> output = chunk_texts(docs)
    >>> output
    [{'doc_id': 42, 'text': 'Hello world.'},
     {'doc_id': 42, 'text': ' How are you today?'}]
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
    )

    all_chunks: List[Dict[str, str]] = []
    for doc in docs:
        chunks = splitter.split_text(doc["text"])
        for chunk in chunks:
            all_chunks.append({"doc_id": doc["doc_id"], "text": chunk})
    return all_chunks
