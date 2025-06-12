from collections.abc import Iterable
from typing import Dict, List
from pathlib import Path

from PyPDF2 import PdfReader
from .config import PDF_DIR


def load_pdfs(pdf_dir: str | Path = PDF_DIR) -> List[Dict[str, str]]:
    """Read every “*.pdf” file in a directory and return their textual contents.

    Each PDF is parsed page-by-page with :class:`PyPDF2.PdfReader`; the extracted
    page strings are concatenated with newline separators to form the full
    document.  The function returns a **flat list** of records that keeps track
    of the original file name, making downstream mapping between source PDF and
    derived chunks or embeddings trivial.

    Parameters
    ----------
    pdf_dir
        Directory path (either ``str`` or :class:`~pathlib.Path`) that will be
        searched *non-recursively* for files matching the glob pattern
        ``"*.pdf"``.  Defaults to :data:`core.config.PDF_DIR`.

    Returns
    -------
    list[dict[str, str]]
        A list where each element is a dictionary with keys

        * ``"doc_id"`` – the PDF file name (including extension)  
        * ``"text"``   – a single string containing all extracted page texts

    Raises
    ------
    FileNotFoundError
        If *pdf_dir* does not exist.
    ValueError
        If no PDF files are found inside *pdf_dir*.

    Notes
    -----
    * Some PDFs contain pages with non-extractable content (e.g. scanned
      images).  Such pages are silently skipped because
      :pymeth:`PyPDF2.PageObject.extract_text` returns ``None`` in those cases.
    * For large corpora you might prefer a generator interface
      (``yield``) to reduce peak memory usage.

    Examples
    --------
    >>> from loader import load_pdfs
    >>> documents = load_pdfs("/path/to/invoices")
    >>> len(documents)
    25
    >>> documents[0]["doc_id"]
    'invoice-001.pdf'
    """
    pdf_path = Path(pdf_dir)

    if not pdf_path.exists():
        raise FileNotFoundError(f"Directory {pdf_path!s} does not exist")

    docs: List[Dict[str, str]] = []
    for pdf_file in pdf_path.glob("*.pdf"):
        reader = PdfReader(str(pdf_file))
        text = "\n".join(
            page_text
            for page in reader.pages
            if (page_text := page.extract_text())  # skip non-text pages
        )
        docs.append({"doc_id": pdf_file.name, "text": text})

    if not docs:
        raise ValueError(f"No PDF files found in directory {pdf_path!s}")

    return docs
