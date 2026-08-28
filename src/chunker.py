def clean_page_text(text):
    footer = "Regression Analysis and ANOVA | Chapter 1 | Introduction | Shalabh, IIT Kanpur"

    text = text.replace(footer, "")

    return text.strip()


def chunk_pages(pages, chunk_size=800, chunk_overlap=120):

    chunks = []

    for page in pages:

        text = page["text"].strip()

        if not text:
            continue

        text=clean_page_text(text)
        start = 0
        chunk_number = 1
        text_length = len(text)

        while start < text_length:

            end = start + chunk_size

            chunk_text = text[start:end]

            chunks.append({
                "text": chunk_text,

                "metadata": {
                    **page["metadata"],
                    "chunk": chunk_number,
                    "character_count": len(chunk_text)
                }
            })

            chunk_number += 1

            if end >= text_length:
                break

            start = end - chunk_overlap

    return chunks