"""PDF text extraction module using PyMuPDF (fitz).

Extracts text from lecture PDFs page-by-page while preserving structured
metadata (course, lecture, source file, 1-based page number).
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import pymupdf

try:
    from .pdf_utils import normalize_symbols
except ImportError:
    from pdf_utils import normalize_symbols


def extract_pdf(
    pdf_path: str,
    course: str,
    lecture: str,
    warn_threshold: int = 40,
    lecture_resolver: Optional[Callable[[int], str]] = None,
    chapter_resolver: Optional[Callable[[int], Optional[int]]] = None,
) -> List[Dict[str, Any]]:
    """Extract page text and metadata from a PDF file.

    Args:
        pdf_path: Path to target PDF file.
        course: Course code (e.g. 'MTH441' or 'EE602').
        lecture: Default lecture label (e.g. 'Lecture 01' or 'Chapter 01').
        warn_threshold: Character count below which a page triggers a warning.
        lecture_resolver: Optional callable (page_num: int) -> str to resolve chapter/lecture dynamically per page.
        chapter_resolver: Optional callable (page_num: int) -> Optional[int] to resolve chapter number per page.

    Returns:
        List of dictionaries containing 'text' and 'metadata'.
    """
    path_obj = Path(pdf_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    doc = pymupdf.open(str(path_obj))
    source_name = path_obj.name
    pages: List[Dict[str, Any]] = []

    for page_idx, page in enumerate(doc):
        page_num = page_idx + 1
        raw_text = page.get_text("text", sort=True)
        text = normalize_symbols(raw_text).strip()

        if len(text) == 0:
            print(f"[Warning] Page {page_num} of {source_name} is completely blank.")
        elif len(text) < warn_threshold:
            print(
                f"[Warning] Page {page_num} of {source_name} has only {len(text)} characters (potential scan/figure)."
            )

        page_lecture = lecture_resolver(page_num) if lecture_resolver else lecture
        page_chapter = chapter_resolver(page_num) if chapter_resolver else None

        page_meta = {
            "course": course,
            "lecture": page_lecture,
            "source": source_name,
            "page": page_num,
        }
        if page_chapter is not None:
            page_meta["chapter"] = page_chapter

        pages.append(
            {
                "text": text,
                "metadata": page_meta,
            }
        )

    doc.close()
    return pages
