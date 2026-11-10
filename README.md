# ZombAI V1 — Local PDF-to-RAG Search and Retrieval Pipeline

ZombAI V1 is a modular, local search and retrieval baseline built for academic lecture notes (specifically tested on IITK MTH441 - Regression Analysis). It extracts lecture PDFs, normalizes mathematical symbols, removes recurring slide footers, chunks text with metadata, computes 384-dimensional dense embeddings (`all-MiniLM-L6-v2`), stores them persistently in ChromaDB, and retrieves relevant evidence with citations and formatted LLM context.

---

## Architecture & Project Structure

```
Hello-Lets-Talk-IITK/
├── data/
│   └── lectures/MTH441/       # Source lecture PDFs (lecture01.pdf, etc.)
├── storage/
│   └── chroma/                 # Persistent local ChromaDB database
├── src/
│   ├── __init__.py             # Exposes pipeline classes and functions
│   ├── extractor.py            # Page-by-page PDF extraction with PyMuPDF
│   ├── cleaner.py              # Targeted slide footer & artefact removal
│   ├── chunker.py              # Page-scoped in-page chunking (800 chars, 120 overlap)
│   ├── embedder.py             # all-MiniLM-L6-v2 dense embeddings (384-d)
│   ├── vector_store.py         # Persistent ChromaDB collection manager
│   ├── retriever.py            # Nearest-neighbor search & LLM context assembler
│   └── pdf_utils.py            # Greek & math symbol normalization
├── tests/
│   └── test_end_to_end.py      # End-to-end acceptance test suite
├── main.ipynb                  # Interactive walkthrough notebook with CLI interface
├── main.py                     # CLI entrypoint for ingestion and querying
├── requirements.txt            # Project dependencies
└── ZombAI_V1_Roadmap.md        # Comprehensive V1 specification and roadmap
```

---

## Environment Setup

The project uses the Python environment located in the parent folder's `.venv`:

```powershell
# From project directory (Hello-Lets-Talk-IITK):
..\.venv\Scripts\Activate.ps1
```

If needed, install dependencies:
```powershell
pip install -r requirements.txt
```

---

## Usage

### 1. Interactive Jupyter Notebook (Recommended)
Launch Jupyter Notebook or JupyterLab:
```powershell
jupyter lab main.ipynb
```
The notebook:
- Explains why each modular file exists (`extractor.py`, `cleaner.py`, `chunker.py`, `embedder.py`, `vector_store.py`, `retriever.py`).
- Walks through each stage step-by-step with real outputs.
- Provides an **interactive CLI-style testing interface** and an **ipywidgets search GUI** to query the indexed lecture notes.

### 2. Command Line Interface (CLI)

#### Ingest a lecture:
```powershell
python main.py ingest --pdf data/lectures/MTH441/lecture01.pdf --course MTH441 --lecture "Lecture 01" --reset
```

#### Query directly:
```powershell
python main.py query "difference between linear and nonlinear models" --top-k 3
```

#### Launch the Interactive Shell:
```powershell
python main.py interactive
```

### 3. Run Acceptance Tests
```powershell
python tests/test_end_to_end.py
```
All unit tests and end-to-end integration checks will verify extraction, cleaning, chunking, embeddings, ChromaDB persistence, and semantic relevance.

