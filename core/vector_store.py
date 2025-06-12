from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pickle

import faiss
import numpy as np


class FaissIndex:
    """A minimal wrapper around *FAISS* for similarity search in RAG pipelines.

    The class stores an in-memory **flat** (exact) index with the squared
    Euclidean (ℓ2) distance metric and keeps per-vector metadata in a plain
    Python list.  It is intentionally lightweight—no shards, no GPU off-load—
    so that it can be embedded in simple inference services or Jupyter demos.

    Parameters
    ----------
    dim
        The dimensionality of the embedding vectors.  **Must** match the
        encoder used to create the embeddings.

    Attributes
    ----------
    index
        ``faiss.IndexFlatL2`` instance storing the raw float32 matrix.
    metadata
        ``list[dict]`` holding arbitrary JSON-serialisable information for each
        vector (e.g. chunk text, source file).  A **one-to-one** alignment
        between vectors and metadata rows is enforced.

    Notes
    -----
    * The distance ``dist`` returned by :meth:`search` is the *squared* ℓ2
      distance because ``IndexFlatL2`` follows the FAISS convention.  For
      cosine similarity you should **ℓ2-normalise** your embeddings before
      calling :meth:`add`.
    * For large corpora or production workloads, switch to an
      :class:`faiss.IndexIVFFlat` or HNSW index and consider GPU acceleration.
    """

    # --------------------------------------------------------------------- #
    # Construction & I/O                                                    #
    # --------------------------------------------------------------------- #
    def __init__(self, dim: int) -> None:
        self.index: faiss.Index = faiss.IndexFlatL2(dim)
        self.metadata: List[Dict] = []

    def save(self, index_path: str | Path, metadata_path: str | Path) -> None:
        """Persist the FAISS index and its metadata to disk.

        Parameters
        ----------
        index_path
            File path for the binary FAISS index (e.g. ``"faiss.index"``).
        metadata_path
            File path for the pickled metadata list (e.g. ``"meta.pkl"``).
        """
        faiss.write_index(self.index, str(index_path))
        with open(metadata_path, "wb") as handle:
            pickle.dump(self.metadata, handle)

    def load(self, index_path: str | Path, metadata_path: str | Path) -> None:
        """Load an index previously saved with :meth:`save`.

        Parameters
        ----------
        index_path
            Path to a ``faiss.write_index`` output file.
        metadata_path
            Pickled list created by :meth:`save`.
        """
        self.index = faiss.read_index(str(index_path))
        with open(metadata_path, "rb") as handle:
            self.metadata = pickle.load(handle)

    # --------------------------------------------------------------------- #
    # Data management                                                       #
    # --------------------------------------------------------------------- #
    def add(
        self,
        embeddings: np.ndarray,
        metadatas: Sequence[Dict],
    ) -> None:
        """Insert vectors and their metadata into the index.

        Parameters
        ----------
        embeddings
            A 2-D ``float32`` NumPy array of shape ``(n, dim)``.  The array must
            be contiguous **C-order**; call ``np.ascontiguousarray`` otherwise.
        metadatas
            A sequence of dictionaries with length **exactly** equal to
            ``len(embeddings)``.  Each dictionary can store any JSON-serialisable
            fields (e.g. ``{"doc_id": "...", "text": "..."}``).

        Raises
        ------
        ValueError
            If the number of vectors and metadata rows does not match.
        """
        if len(embeddings) != len(metadatas):
            raise ValueError(
                "Mismatch between number of embeddings "
                f"({len(embeddings)}) and metadata entries ({len(metadatas)})"
            )

        self.index.add(embeddings.astype(np.float32, copy=False))
        self.metadata.extend(list(metadatas))

    # --------------------------------------------------------------------- #
    # Query interface                                                       #
    # --------------------------------------------------------------------- #
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[Dict, float]]:
        """Return the *top-k* nearest neighbours of *query_embedding*.

        Parameters
        ----------
        query_embedding
            A single ℓ2-normalised query vector with shape ``(1, dim)``.
        top_k
            Number of neighbours to retrieve.  Default ``5``.

        Returns
        -------
        list[tuple[dict, float]]
            A list of ``(metadata, distance)`` tuples ordered from closest to
            farthest.  *Distance* is the **squared** ℓ2 distance.

        Examples
        --------
        >>> idx = FaissIndex(dim=384)
        >>> idx.add(embeddings, metadatas)
        >>> results = idx.search(query_emb, top_k=3)
        >>> results[0][0]["text"]  # metadata of best match
        'Paris est la capitale de la France …'
        """
        D, I = self.index.search(query_embedding.astype(np.float32, copy=False), top_k)

        results: List[Tuple[Dict, float]] = []
        for vec_id, dist in zip(I[0], D[0]):
            if vec_id < len(self.metadata):  # guard against empty slots
                results.append((self.metadata[vec_id], float(dist)))
        return results
