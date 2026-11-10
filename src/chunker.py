"""Page-based text chunking module.

Splits extracted lecture text into page-scoped chunks of target character size
with in-page overlap. Uses natural boundary snapping (headers, paragraphs, sentences,
bullet points) so chunks begin and end on coherent thoughts without orphaned words.
"""

import re
from typing import Any, Dict, List, Optional

try:
    from .cleaner import clean_page_text
except ImportError:
    from cleaner import clean_page_text

# Common lecture topic header patterns to prioritize clean topic splits
TOPIC_HEADER_REGEX = re.compile(
    r"\n\n([A-Z][A-Za-z0-9\- ]*(?:distribution|functions|theorem|models?|analysis|forms|variables?)[^\n]*)\n",
    re.IGNORECASE,
)


def chunk_pages(
    pages: List[Dict[str, Any]],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    ensure_cleaned: bool = True,
) -> List[Dict[str, Any]]:
    """Split pages into overlapping naturally-bounded chunks per page.

    Args:
        pages: List of page dicts with 'text' and 'metadata'.
        chunk_size: Target maximum characters per chunk (default: 800).
        chunk_overlap: Target in-page overlap in characters (default: 120).
        ensure_cleaned: If True, passes text through clean_page_text.

    Returns:
        List of chunk dictionaries with 'text' and enriched 'metadata'.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
        )

    chunks: List[Dict[str, Any]] = []

    for page in pages:
        text = page["text"]
        if ensure_cleaned:
            text = clean_page_text(text)
        else:
            text = text.strip()

        if not text:
            continue

        text_length = len(text)
        start = 0
        chunk_number = 1

        while start < text_length:
            candidate_end = start + chunk_size
            is_header_split = False

            if candidate_end >= text_length:
                end = text_length
            else:
                # Search for a natural boundary within a window before candidate_end
                buffer_start = max(start + 200, candidate_end - 220)
                window = text[buffer_start:candidate_end]

                # 1. Check for topic headers in window (best split point)
                header_match = list(TOPIC_HEADER_REGEX.finditer(window))
                if header_match:
                    end = buffer_start + header_match[-1].start() + 2
                    is_header_split = True
                else:
                    # 2. Check for paragraph break
                    p_break = window.rfind("\n\n")
                    if p_break != -1:
                        end = buffer_start + p_break + 2
                    else:
                        # 3. Check for bullet item start
                        bullet_match = list(re.finditer(r"\n[ \t]*[•➢\-\*]\s*", window))
                        if bullet_match:
                            end = buffer_start + bullet_match[-1].start() + 1
                        else:
                            # 4. Check for sentence boundary (. followed by space/newline)
                            s_matches = list(re.finditer(r"\.\s+", window))
                            if s_matches:
                                end = buffer_start + s_matches[-1].end()
                            else:
                                l_break = window.rfind("\n")
                                if l_break != -1:
                                    end = buffer_start + l_break + 1
                                else:
                                    sp = window.rfind(" ")
                                    if sp != -1:
                                        end = buffer_start + sp + 1
                                    else:
                                        end = candidate_end

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "text": chunk_text,
                        "metadata": {
                            **page["metadata"],
                            "chunk": chunk_number,
                            "character_count": len(chunk_text),
                        },
                    }
                )
                chunk_number += 1

            if end >= text_length:
                break

            # Calculate next chunk start
            if is_header_split:
                # When cleanly splitting before a major topic header, start fresh without dragging in previous equations
                next_start = end
            else:
                raw_next_start = max(start + 1, end - chunk_overlap)
                snap_min = max(start + 1, raw_next_start - 60)
                snap_max = min(text_length, raw_next_start + 60)
                snap_window = text[snap_min:snap_max]

                header_snap = list(TOPIC_HEADER_REGEX.finditer(snap_window))
                if header_snap:
                    next_start = snap_min + header_snap[0].start() + 2
                else:
                    p_in_snap = snap_window.find("\n\n")
                    if p_in_snap != -1:
                        next_start = snap_min + p_in_snap + 2
                    else:
                        bullet_in_snap = list(re.finditer(r"\n[ \t]*[•➢\-\*]\s*", snap_window))
                        if bullet_in_snap:
                            next_start = snap_min + bullet_in_snap[0].start() + 1
                        else:
                            sent_in_snap = list(re.finditer(r"\.\s+", snap_window))
                            if sent_in_snap:
                                next_start = snap_min + sent_in_snap[0].end()
                            else:
                                sp = text.find(" ", raw_next_start, min(text_length, raw_next_start + 40))
                                if sp != -1:
                                    next_start = sp + 1
                                else:
                                    next_start = raw_next_start

            start = next_start

    return chunks
