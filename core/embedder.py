from collections.abc import Iterable
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from .config import EMBEDDING_MODEL_ID

# --------------------------------------------------------------------------- #
# Model initialisation                                                        #
# --------------------------------------------------------------------------- #
# These objects are loaded **once** at import-time so that subsequent calls
# to `embed()` or `embed_chunks()` are fast.  Move them to GPU manually if
# your production environment allows it.
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_ID)
model = (
    AutoModel.from_pretrained(EMBEDDING_MODEL_ID)
    .eval()               # disable dropout, etc.
    .to("cpu")            # keep CPU by default; switch to "cuda" if available
)


def embed(text: str) -> np.ndarray:
    """Encode a single piece of text into a unit-length embedding vector.

    The function performs the minimal preprocessing required by the model
    tokenizer, runs a forward pass through the *sentence-transformer-style*
    encoder, and returns an **ℓ2-normalised** representation suitable for
    cosine-similarity search.

    Parameters
    ----------
    text
        The raw text string to embed.  It may contain arbitrary Unicode
        characters.  Very long texts are truncated to the model’s maximum
        sequence length.

    Returns
    -------
    numpy.ndarray
        A 1-D float32 vector of size ``model.config.hidden_size`` lying on the
        unit hypersphere.

    Notes
    -----
    * The mean pooling strategy ``output.last_hidden_state.mean(dim=1)`` is the
      simplest way to obtain a sentence embedding.  Depending on the model, a
      different pooling method (e.g. CLS token) might yield better results.
    * The operation is wrapped in ``torch.no_grad()`` to disable gradient
      tracking and reduce memory usage.
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
    ).to("cpu")

    with torch.no_grad():
        output = model(**inputs)

    embedding: torch.Tensor = F.normalize(
        output.last_hidden_state.mean(dim=1), p=2, dim=1
    )[0]

    return embedding.cpu().numpy()


def embed_chunks(chunks: Iterable[Dict[str, str]]) -> np.ndarray:
    """Vectorise a sequence of pre-split text chunks.

    Parameters
    ----------
    chunks
        An iterable of dictionaries—typically the output of
        :func:`chunker.chunk_texts`—each containing a ``"text"`` field.

    Returns
    -------
    numpy.ndarray
        A 2-D array of shape ``(n_chunks, embedding_dim)`` where
        ``embedding_dim`` equals ``model.config.hidden_size``.
    """
    return np.vstack([embed(chunk["text"]) for chunk in chunks])
