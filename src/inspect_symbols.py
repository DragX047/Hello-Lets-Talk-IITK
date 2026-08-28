import pymupdf

pdf_path = "data/lectures/MTH441/lecture01.pdf"

doc = pymupdf.open(pdf_path)

for page_number, page in enumerate(doc):

    data = page.get_text("dict")

    for block in data["blocks"]:

        if "lines" not in block:
            continue

        for line in block["lines"]:

            for span in line["spans"]:

                if "SymbolMT" not in span["font"]:
                    continue

                for char in span["text"]:

                    if ord(char) >= 0xE000:

                        print(
                            f"Page {page_number + 1}: "
                            f"{repr(char)} "
                            f"Unicode={hex(ord(char))} "
                            f"font={span['font']}"
                        )