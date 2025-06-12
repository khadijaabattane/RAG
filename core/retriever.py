from __future__ import annotations

from typing import List, Tuple, Dict

import numpy as np
from .vector_store import FaissIndex
from .embedder import embed
from .config import INDEX_PATH, METADATA_PATH


class RAGRetriever:
    """Lightweight wrapper around a FAISS vector index for RAG pipelines.

    The retriever converts the user query into an embedding with
    :func:`core.embedder.embed`, performs an ANN (approximate nearest-neighbour)
    search against a pre-built FAISS index, and returns the *top-k* most similar
    document chunks together with their similarity scores.

    Parameters
    ----------
    dim
        Dimensionality of the embedding space.  **Must** match the encoder used
        when the index was created.  Default: ``384`` (common for MiniLM-based
        models).
    index_path
        File path to the serialized FAISS index (e.g. ``"faiss.index"``).
    metadata_path
        File path to the NumPy ``.npy`` file storing per-vector metadata
        (typically a list of chunk dictionaries).

    Attributes
    ----------
    index : FaissIndex
        In-memory FAISS index ready for similarity search.

    Notes
    -----
    * The constructor eagerly loads both the index and its metadata to minimise
      latency at inference time.  If start-up time is critical, consider lazy
      loading or a separate warm-up step.
    * The current implementation always places the index on CPU.  Move it to
      GPU with ``faiss.index_cpu_to_gpu`` if your deployment stack supports it.
    """

    def __init__(
        self,
        dim: int = 384,
        index_path: str | None = INDEX_PATH,
        metadata_path: str | None = METADATA_PATH,
    ) -> None:
        self.index: FaissIndex = FaissIndex(dim=dim)
        self.index.load(index_path, metadata_path)

    # --------------------------------------------------------------------- #
    # Public API                                                            #
    # --------------------------------------------------------------------- #
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Tuple[Dict[str, str], float]]:
        """Return the *top-k* chunks most relevant to *query*.

        Parameters
        ----------
        query
            A natural-language question or statement.
        top_k
            The maximum number of passages to return.  Default is ``5``.

        Returns
        -------
        list[tuple[dict[str, str], float]]
            A list of ``(chunk, score)`` tuples ordered from most to least
            similar, where *chunk* is the metadata dictionary originally stored
            alongside the vector and *score* is the cosine similarity
            (ℓ2-normalised dot product).

        Examples
        --------
        >>> retriever = RAGRetriever()
        >>> passages = retriever.retrieve("Quelle est la capitale de la France ?", top_k=3)
        >>> passages[0][0]["text"]
        'Paris est la capitale de la France …'
        """
        # 1. Encode the query into the same latent space as the index.
        query_emb: np.ndarray = embed(query).reshape(1, -1)

        # 2. Perform ANN search and return the results.
        return self.index.search(query_emb, top_k=top_k)
