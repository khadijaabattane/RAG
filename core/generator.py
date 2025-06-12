from __future__ import annotations

from typing import Dict, Iterable, Tuple

from langchain_openai import ChatOpenAI
from .config import OPENAI_API_KEY, OPENAI_API_BASE, OPENAI_MODEL_NAME


#: Singleton LLM instance reused across requests to avoid costly re-instantiation.
llm: ChatOpenAI = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    openai_api_base=OPENAI_API_BASE,
    model_name=OPENAI_MODEL_NAME,
)


def format_prompt(
    chunks: Iterable[Tuple[Dict[str, str], float]],
    query: str,
) -> str:
    """Create the full prompt sent to the language model.

    Each *chunk*—a short text passage retrieved from the document corpus—is
    numbered and labelled with its ``doc_id`` so that the model can reference
    them in its answer.  The resulting template instructs the model to answer
    **in French** and to cite passage numbers when relevant.

    Parameters
    ----------
    chunks
        An iterable of ``(chunk_dict, score)`` tuples, where *chunk_dict* is a
        mapping that **must** contain the keys

        * ``"doc_id"`` – the source document identifier  
        * ``"text"``   – the extracted passage text

        The *score* (e.g. cosine similarity) is not used directly but is kept
        in the signature so that the function can accept the raw output of most
        retrieval pipelines.

    query
        The user’s question in natural language.

    Returns
    -------
    str
        A formatted prompt string ready for :py:meth:`langchain_openai.ChatOpenAI.invoke`.

    Notes
    -----
    * Adapt the hard-coded French response instruction if you need answers in a
      different language.
    * You can tweak the citation style or add system-level instructions without
      changing the rest of the pipeline.
    """
    chunk_block = "\n".join(
        f"[{i + 1} — {c['doc_id']}]: {c['text']}"
        for i, (c, _) in enumerate(chunks)
    )

    return (
        "Utilise les passages suivants pour répondre à la question. "
        "Cite le numéro de passage si pertinent.\n\n"
        "<chunks>\n"
        f"{chunk_block}\n"
        "</chunks>\n\n"
        "<question>\n"
        f"{query}\n"
        "</question>\n\n"
        "Réponds en français de façon claire et concise."
    )


def generate_answer(
    chunks: Iterable[Tuple[Dict[str, str], float]],
    query: str,
) -> str:
    """Produce a grounded answer to *query* using the supplied *chunks*.

    The helper builds a prompt with :func:`format_prompt`, submits it to the
    global *llm* instance, and returns the model’s textual response.

    Parameters
    ----------
    chunks
        Same iterable accepted by :func:`format_prompt`.

    query
        The user’s information need.

    Returns
    -------
    str
        The assistant’s answer, stripped of leading and trailing whitespace.

    Examples
    --------
    >>> ranked_chunks = [({"doc_id": "regulation.pdf", "text": "..."}, 0.92)]
    >>> generate_answer(ranked_chunks, "Qu'est-ce que le RGPD ?")
    'Le RGPD (Règlement général sur la protection des données) est …'
    """
    prompt = format_prompt(chunks, query)
    response_obj = llm.invoke(prompt)

    # The ChatCompletion-like object exposes the reply text through `.content`.
    return getattr(response_obj, "content", str(response_obj)).strip()
