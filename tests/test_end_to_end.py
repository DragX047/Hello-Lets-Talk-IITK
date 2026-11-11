"""End-to-end acceptance test for Hello Lets Talk IITK pipeline."""

import sys
from pathlib import Path

# Ensure UTF-8 output on Windows terminals
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extractor import extract_pdf
from src.cleaner import clean_pages
from src.chunker import chunk_pages
from src.embedder import Embedder
from src.vector_store import VectorStore
from src.retriever import Retriever


def run_test():
    print("=== 1. Extracting PDF ===")
    pdf_path = "data/lectures/MTH441/lecture01.pdf"
    pages = extract_pdf(pdf_path, course="MTH441", lecture="Lecture 01")
    print(f"Extracted {len(pages)} pages.")
    assert len(pages) == 20, f"Expected 20 pages, got {len(pages)}"

    print("\n=== 2. Cleaning Pages ===")
    cleaned = clean_pages(pages)
    print(f"Cleaned {len(cleaned)} pages.")
    assert "Shalabh, IIT Kanpur" not in cleaned[0]["text"], "Footer was not removed!"

    print("\n=== 3. Chunking Pages ===")
    chunks = chunk_pages(cleaned, chunk_size=800, chunk_overlap=120)
    print(f"Generated {len(chunks)} chunks.")
    assert len(chunks) > 0, "No chunks generated!"
    assert all("character_count" in c["metadata"] for c in chunks), "Metadata missing character_count"

    print("\n=== 4. Embedding Chunks ===")
    embedder = Embedder()
    embeddings = embedder.embed_chunks(chunks)
    print(f"Computed embeddings: shape {embeddings.shape}")
    assert embeddings.shape == (len(chunks), 384), f"Shape mismatch: {embeddings.shape}"

    print("\n=== 5. Persistent ChromaDB Storage ===")
    vs = VectorStore(persist_dir="storage/chroma", collection_name="hello_lets_talk_iitk_e2e_test_collection")
    vs.reset_collection()
    added = vs.add_chunks(chunks, embeddings)
    print(f"Upserted {added} chunks. Collection count = {vs.count()}")
    assert vs.count() == len(chunks), f"Expected {len(chunks)}, got {vs.count()}"

    print("\n=== 6. Retrieval & Search ===")
    retriever = Retriever(vector_store=vs, embedder=embedder)
    query = "difference between linear and nonlinear models"
    matches = retriever.retrieve(query, top_k=3)
    print(f"Query: '{query}' -> Found {len(matches)} matches.\n")
    assert len(matches) > 0, "Retrieval returned 0 matches!"

    for i, match in enumerate(matches, 1):
        meta = match["metadata"]
        print(f"Match {i}: Page {meta['page']} | Chunk {meta['chunk']} | Distance: {match['distance']} | Sim: {match['similarity']}")
        print(f"   Excerpt: {match['text'][:120].strip()}...\n")

    print("=== 7. LLM Context Formatting ===")
    context = retriever.format_context_for_llm(matches)
    print("Formatted LLM Context Preview:")
    print(context[:400])
    print("\nALL ACCEPTANCE TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_test()

