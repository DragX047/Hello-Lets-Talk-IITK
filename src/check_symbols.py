import pymupdf
from collections import Counter

pdf_path = "data/lectures/MTH441/lecture01.pdf"

doc = pymupdf.open(pdf_path)

symbol_chars = Counter()

for page in doc:
    data = page.get_text("dict")

    for block in data["blocks"]:
        if "lines" not in block:
            continue

        for line in block["lines"]:
            for span in line["spans"]:
                if "SymbolMT" in span["font"]:
                    for char in span["text"]:
                        symbol_chars[repr(char)] += 1

print("Symbol font characters found:\n")

for char, count in symbol_chars.items():
    print(char, "->", count)