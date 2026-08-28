import pymupdf
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

    # Mathematical operators
    "\uf0b4": "×",
    "\uf0b6": "∂",
    "\uf0b9": "≠",
    "\uf0a3": "≤",
    "\uf0b3": "≥",
    "\uf0a5": "∞",
    "\uf0de": "⇒",
    "\uf0e5": "∑",
}


STRETCHY_DELIMITERS = [

    # Large parentheses
    "\uf0e6",
    "\uf0e7",
    "\uf0e8",
    "\uf0f6",
    "\uf0f7",
    "\uf0f8",

    # Large square brackets
    "\uf0e9",
    "\uf0ea",
    "\uf0eb",
    "\uf0f9",
    "\uf0fa",
    "\uf0fb",
]


def normalize_symbols(text):

    # Convert mathematical symbols
    for old, new in SYMBOL_MAP.items():
        text = text.replace(old, new)

    # Remove decorative pieces of large delimiters
    for symbol in STRETCHY_DELIMITERS:
        text = text.replace(symbol, "")

    return text



def normalize_whitespace(text):

    # Normalize Windows-style line endings
    text = text.replace("\r\n", "\n")

    # Replace multiple spaces/tabs with one space
    text = re.sub(r"[ \t]+", " ", text)

    # Fix line wrapping inside sentences.
    # A newline followed by a lowercase letter is almost
    # certainly a continuation of the same sentence.
    text = re.sub(r"(?<=[a-z,.;:)])\n(?=[a-z])", " ", text)

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()