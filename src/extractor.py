import pymupdf
from pathlib import Path
from pdf_utils import normalize_symbols


def extract_pdf(pdf_path, course, lecture):

    doc = pymupdf.open(pdf_path)

    source = Path(pdf_path).name

    pages = []

    for page_number, page in enumerate(doc):

        text = page.get_text("text", sort=True)

        text = normalize_symbols(text)

        pages.append({
            "text": text,
            "metadata": {
                "course": course,
                "lecture": lecture,
                "source": source,
                "page": page_number + 1
            }
        })

    doc.close()

    return pages