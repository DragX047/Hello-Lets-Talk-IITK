"""PDF text and symbol utilities for mathematical lecture notes."""

import re

SYMBOL_MAP = {
    # Greek letters
    "\uf062": "β",
    "\uf065": "ε",
    "\uf06d": "μ",
    "\uf06c": "λ",
    "\uf063": "χ",
    "\uf070": "ρ",
    "\uf073": "σ",
    "\uf053": "Σ",
    "\uf047": "Γ",
    "\uf04d": "M",
    # Mathematical operators & relations
    "\uf0b4": "×",
    "\uf0b6": "∂",
    "\uf0b9": "≠",
    "\uf0a3": "≤",
    "\uf0b3": "≥",
    "\uf0a5": "∞",
    "\uf0de": "⇒",
    "\uf0e5": "∑",
    "\uf03c": "<",
    "\uf03e": ">",
    "\uf03d": "=",
    "\uf02b": "+",
    "\uf02d": "-",
}

STRETCHY_DELIMITERS = [
    # Large parentheses pieces
    "\uf0e6",
    "\uf0e7",
    "\uf0e8",
    "\uf0f6",
    "\uf0f7",
    "\uf0f8",
    # Large square bracket pieces
    "\uf0e9",
    "\uf0ea",
    "\uf0eb",
    "\uf0f9",
    "\uf0fa",
    "\uf0fb",
]


def normalize_symbols(text: str) -> str:
    """Replace private-use Unicode math symbols and strip stretchy delimiter fragments."""
    if not text:
        return ""

    for old, new in SYMBOL_MAP.items():
        text = text.replace(old, new)

    for symbol in STRETCHY_DELIMITERS:
        text = text.replace(symbol, "")

    return text


def normalize_whitespace(text: str) -> str:
    """Clean up indentation, multiple spaces, and awkward line breaks."""
    if not text:
        return ""

    # Normalize Windows line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Add space after punctuation if missing (e.g. "λ;Y" -> "λ; Y", "0,and" -> "0, and")
    text = re.sub(r";(?=[^\s\d])", "; ", text)
    text = re.sub(r",(?=[^\s\d])", ", ", text)

    # Clean lines individually: collapse multiple spaces/tabs into a single space
    lines = []
    for raw_line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        lines.append(line)
    text = "\n".join(lines)

    # Join orphaned bullets on their own line: "•\nText" -> "• Text"
    text = re.sub(r"(^[•➢\-\*])\s*\n+([^\n])", r"\1 \2", text, flags=re.MULTILINE)

    # Indent sub-bullets neatly and ensure space after bullet symbol
    text = re.sub(r"^[ \t]*➢\s*", "  ➢ ", text, flags=re.MULTILINE)
    text = re.sub(r"^[ \t]*•\s*", "• ", text, flags=re.MULTILINE)

    # Strip duplicate slide page number stamps (e.g. "9 9\n", "11 11\n")
    text = re.sub(r"^\s*\d+\s+\d+\s*\n", "", text)
    text = re.sub(r"\n\s*\d+\s+\d+\s*$", "", text)

    # Merge wrapped lines within paragraphs/bullets:
    # If a line does not end with sentence-ending punctuation (. : ? !) or equals/operator,
    # and the next line is NOT a bullet, merge them with a single space.
    paragraphs = text.split("\n\n")
    cleaned_paras = []
    for para in paragraphs:
        p_lines = para.split("\n")
        merged = []
        for line in p_lines:
            line_str = line.strip()
            if not line_str:
                continue
            if not merged:
                merged.append(line_str)
            else:
                prev = merged[-1]
                is_bullet = bool(re.match(r"^[ \t]*[•➢\-\*]", line_str))
                prev_ends_incomplete = not bool(re.search(r"[\.:\?!]$", prev))
                prev_is_bullet_header = bool(re.match(r"^[•➢\-\*]\s*If$", prev)) or (
                    len(prev) < 25 and not prev_ends_incomplete
                )

                if prev_ends_incomplete and not is_bullet and not prev_is_bullet_header:
                    merged[-1] = prev + " " + line_str
                else:
                    merged.append(line_str)
        cleaned_paras.append("\n".join(merged))

    text = "\n\n".join(cleaned_paras)

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
