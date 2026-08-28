from extractor import extract_pdf


pdf_path = "data/lectures/MTH441/lecture01.pdf"

pages = extract_pdf(
    pdf_path,
    course="MTH441",
    lecture="Lecture 01"
)

page_number = 15

page = pages[page_number - 1]

print("=" * 70)
print(f"PAGE {page['metadata']['page']}")
print("=" * 70)

print("\nMetadata:")
print(page["metadata"])

print("\nText:")
print(page["text"])