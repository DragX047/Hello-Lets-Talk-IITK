# Hello Lets Talk IITK (V1) --- Complete Roadmap from Scratch

## PDF-to-RAG Search and Retrieval Pipeline

**Goal:** Build a local pipeline that extracts lecture PDFs, cleans
text, chunks it with metadata, embeds the chunks, stores them
persistently, and retrieves relevant evidence for a query.

This roadmap reflects the decisions made so far. V1 focuses on a
reliable search-and-retrieve baseline. Section-aware chunking,
reranking, hybrid search, and advanced math-layout handling are later
versions.

## 1. V1 scope

V1 must: 1. Extract PDF text page by page. 2. Preserve course, lecture,
source filename, and page number. 3. Remove recurring headers/footers
before chunking. 4. Split text into page-based chunks. 5. Preserve
metadata and add chunk number and character count. 6. Generate
embeddings with one consistent model. 7. Store IDs, text, vectors, and
metadata in persistent ChromaDB. 8. Embed user queries with the same
model. 9. Retrieve top-k semantically similar chunks. 10. Return text,
metadata, and distance, and format results as context for a future LLM.

Not in V1: section-aware chunking, cross-page overlap, hybrid retrieval,
reranking, query rewriting, agent workflows, UI, production deployment,
or complex equation reconstruction.

## 2. Architecture

``` text
PDF
 ↓
Extract page text + metadata
 ↓
Clean recurring headers/footers
 ↓
Page-based chunking
 ↓
Chunk text + metadata
 ↓
Embedding model
 ↓
Persistent ChromaDB
                  User query
                      ↓
              Same embedding model
                      ↓
               Vector similarity search
                      ↓
              Top-k chunks + metadata
                      ↓
             Context for a future LLM
```

Suggested structure:

``` text
Hello-Lets-Talk-IITK/
├── data/
│   ├── raw/
│   └── processed/
├── storage/
│   └── chroma/
├── src/
│   ├── extractor.py
│   ├── cleaner.py
│   ├── chunker.py
│   ├── embedder.py
│   ├── vector_store.py
│   ├── retriever.py
│   └── __init__.py
├── tests/
├── .gitignore
├── requirements.txt
└── README.md
```

Keep each module focused: extraction does not chunk; cleaning does not
embed; chunking does not access Chroma; embedding does not parse PDFs;
the retriever orchestrates query embedding and search.

## 3. Environment setup (Windows)

### Prerequisites

Install Python 3.11, Git (recommended), and an editor such as VS Code.
Open a fresh PowerShell terminal and verify:

``` powershell
python --version
python -m pip --version
git --version
```

If `python` is not recognized, try `py --version`.

### Create or open the project

For a new project:

``` powershell
cd D:\Placement
mkdir Hello-Lets-Talk-IITK
cd Hello-Lets-Talk-IITK
mkdir data
mkdir data\raw
mkdir data\processed
mkdir storage
mkdir storage\chroma
mkdir src
mkdir tests
```

If the project already exists, do not recreate it; open the existing
directory and preserve current code.

### Create and activate a virtual environment

From the project root:

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks activation, use this only for the current terminal,
if appropriate:

``` powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Or use Command Prompt:

``` bat
.venv\Scripts\activate.bat
```

Confirm `(.venv)` appears in the terminal. Activate the environment
before installing packages or running the project.

### Install dependencies

``` powershell
python -m pip install pymupdf llama-index-core sentence-transformers chromadb numpy
python -m pip freeze > requirements.txt
```

Packages: - `pymupdf`: page-by-page PDF text extraction. -
`llama-index-core`: provides `SentenceSplitter` if retaining the current
chunker. - `sentence-transformers`: local embedding generation. -
`chromadb`: persistent vector store. - `numpy`: embedding array handling
and inspection.

On another environment, install the pinned package list with:

``` powershell
python -m pip install -r requirements.txt
```

Create `.gitignore`:

``` gitignore
.venv/
__pycache__/
*.py[cod]
.env
storage/chroma/
data/processed/
.pytest_cache/
```

Decide deliberately whether to commit source PDFs; large or copyrighted
lecture PDFs usually should not be committed.

## 4. Step A --- PDF extraction

Use PyMuPDF (`fitz`) to extract text one page at a time. Return
dictionaries, because the current chunker expects dictionaries rather
than PyMuPDF page objects.

Expected shape:

``` python
[
    {
        "text": "Extracted text from this page...",
        "metadata": {
            "course": "MTH441",
            "lecture": "Lecture 01",
            "source": "lecture01.pdf",
            "page": 1
        }
    }
]
```

Rules: - Use 1-based page numbers so metadata matches the PDF viewer. -
Preserve source, course, and lecture metadata. - Report blank or
suspiciously short pages instead of silently ignoring them. - Scanned
pages may require OCR; that is not included in the basic V1 pipeline. -
PDF extraction can scramble equation symbols, matrix brackets,
subscripts, superscripts, and fraction layouts. Keep extracted text
available for debugging.

Test page count, several page texts, metadata, and equation-heavy pages
before proceeding.

## 5. Step B --- Clean headers and footers

The recurring footer identified in the lecture PDF is:

``` text
Regression Analysis and ANOVA | Chapter 1 | Introduction | Shalabh, IIT Kanpur
```

Extracted text may append page-number artefacts, for example
`... IIT Kanpur 1515`. Remove the recurring footer and its trailing
page-number junk from the **body text before chunking and embedding**.
Keep the page number in metadata.

Put cleaning in `cleaner.py`. Use a narrowly scoped rule or regex based
on the actual extracted text; do not use a broad rule that could delete
legitimate body text. Strip excess whitespace after removal.

Test several pages: - Footer is absent. - Final body paragraph remains
intact. - Metadata page number remains correct. - A legitimate mention
in the middle of the page is not accidentally removed.

## 6. Step C --- Chunking

Current V1 decisions: - Process each page independently. - Target: 800
characters. - Overlap: 120 characters within a page. - No overlap across
page boundaries. - Chunk numbering starts at 1 and resets on each
page. - Preserve page metadata and add `chunk` and `character_count`.

Example metadata:

``` python
{
    "course": "MTH441",
    "lecture": "Lecture 01",
    "source": "lecture01.pdf",
    "page": 15,
    "chunk": 1,
    "character_count": 800
}
```

The current implementation uses `SentenceSplitter`:

``` python
from llama_index.core.node_parser import SentenceSplitter

splitter = SentenceSplitter(
    chunk_size=800,
    chunk_overlap=120
)
```

**Important size caveat:** LlamaIndex's `SentenceSplitter` normally
measures `chunk_size` and `chunk_overlap` in tokens, not raw characters.
Do not assume `chunk_size=800` guarantees a maximum of 800 characters.
The reported output `1935 → 800 + 800 + 335` is evidence of the tested
behavior, but verify the exact installed version and code. If V1
promises a strict 800-character limit, explicitly enforce and test that
requirement or use a character-based splitter. If using
`SentenceSplitter` as intended, document 800 as a token-based target
instead. Make this decision consistent throughout the code and
documentation.

Each output chunk should have this shape:

``` python
{
    "text": "A section of cleaned page text...",
    "metadata": {
        "course": "MTH441",
        "lecture": "Lecture 01",
        "source": "lecture01.pdf",
        "page": 15,
        "chunk": 1,
        "character_count": 800
    }
}
```

Always compute `character_count` from the final text with
`len(chunk_text)`.

Tests: - Empty text creates no chunk. - Correct page metadata is
preserved. - Chunk numbering resets per page. -
`character_count == len(text)`. - The documented size rule is
enforced. - In-page overlap exists. - No cross-page overlap is
expected. - Footer text is absent. - Every chunk later receives a unique
storage ID.

The first three checks and in-page overlap were reported as passing in
the current project.

## 7. Step D --- Embedding generation

An embedding is a vector of numbers representing aspects of text
meaning. You do not calculate the numbers manually; an embedding model
generates them.

V1 baseline model:

``` text
sentence-transformers/all-MiniLM-L6-v2
```

It produces 384-dimensional embeddings. It is lightweight and suitable
for getting a baseline working, but has not been proven best for
mathematical lecture content.

Create `src/embedder.py`. The model should be loaded once per process
and expose two operations: - `embed_chunks(chunks)`: embed each
`chunk["text"]`. - `embed_query(query)`: embed the user query.

Starting-point example:

``` python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_chunks(chunks):
    texts = [chunk["text"] for chunk in chunks]
    return model.encode(texts, show_progress_bar=True)

def embed_query(query):
    return model.encode(query)
```

Validate input and add type annotations as the module is integrated.

Embedding checks: - Number of vectors equals number of chunks. - Each
vector has 384 dimensions for this model. - Values are finite. - Each
vector remains paired with its original chunk and metadata. - Document
and query embeddings use the exact same model. - The model is not
reloaded for every chunk or query.

The first run may download model files. Once the model is cached, it can
normally run locally. Do not generate embeddings again for every search.
If the model changes, re-embed the corpus and rebuild the collection;
vectors from different models are not interchangeable.

## 8. Step E --- Persistent ChromaDB storage

Create `src/vector_store.py`. Use a persistent local directory such as
`storage/chroma/` and a named collection such as
`hello_lets_talk_iitk_chunks_v1`.

Each record needs: - **ID:** unique chunk ID, ideally derived from
source/lecture/page/chunk. - **Document:** chunk text. - **Embedding:**
vector generated by the selected model. - **Metadata:** course, lecture,
source, page, chunk, character count.

Example:

``` text
id: MTH441_Lecture01_page15_chunk1
document: "The theory and fundamentals of linear models..."
metadata:
    course = "MTH441"
    lecture = "Lecture 01"
    source = "lecture01.pdf"
    page = 15
    chunk = 1
    character_count = 800
embedding: [384 floating-point values]
```

Use one consistent strategy: 1. **Recommended for this design:**
generate vectors in `embedder.py`, then pass them into ChromaDB. Embed
queries in the same module and pass query vectors to Chroma. 2.
Alternative: let Chroma's configured embedding function handle both
document and query embeddings.

Do not mix the two approaches casually. This roadmap uses explicit
embeddings from Sentence Transformers.

Chroma metadata should use supported scalar values such as strings,
integers, floats, or booleans; the metadata above is compatible.

Avoid duplicate ingestion. For a small development corpus, clearing and
rebuilding the project's own collection on a full ingestion run is
simple. Alternatively, use stable IDs and upsert records. Do not blindly
append duplicate records on every run. Verify persistence by closing the
process and reopening the database.

## 9. Step F --- Semantic retrieval

Create `src/retriever.py`. For a query such as "What is the difference
between a linear and nonlinear model?":

1.  Validate the query.
2.  Embed it with the same model.
3.  Query ChromaDB for nearest vectors.
4.  Start with `top_k=5`.
5.  Return each match's text, metadata, and distance.
6.  Preserve source/page information for traceability.

Conceptual Chroma query (confirm the API against the installed version):

``` python
results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=5,
    include=["documents", "metadatas", "distances"]
)
```

Return a clean list rather than exposing Chroma's nested response format
everywhere:

``` python
[
    {
        "text": "A relevant chunk...",
        "metadata": {
            "course": "MTH441",
            "lecture": "Lecture 01",
            "source": "lecture01.pdf",
            "page": 14,
            "chunk": 1,
            "character_count": 790
        },
        "distance": 0.23
    }
]
```

The distance is illustrative, not a predicted result. Its interpretation
depends on the configured metric. For common distance measures, smaller
values mean closer vectors; distance is not a probability or a
guaranteed relevance score.

V1 retrieval rules: - Start with `top_k=5`. - Handle empty queries and
empty collections. - Do not invent a relevance threshold before
evaluating results. - Include source/page metadata in every result. -
Keep the retriever responsible for evidence retrieval, not answer
generation.

## 10. Step G --- Format context for a future LLM

Prepare a formatter that converts results into a context string, without
throwing away the structured results.

Example:

``` text
[Source: lecture01.pdf | Page: 14 | Chunk: 1]
A linear model is linear in its parameters...

[Source: lecture01.pdf | Page: 15 | Chunk: 1]
Regression analysis uses observed data to determine model parameters...
```

Keep both structured records (for programmatic use and citations) and
formatted context (for a future LLM call). V1 search-and-retrieve can be
complete without generating a final natural-language answer.

## 11. Testing plan

### Extraction

-   [ ] PDF opens and page count is correct.
-   [ ] Page numbering is 1-based and accurate.
-   [ ] Course, lecture, and source metadata are attached.
-   [ ] Blank/scanned pages are detected and reported.
-   [ ] Mathematical pages are manually inspected.

### Cleaning

-   [ ] Recurring footer is removed, including trailing page-number
    artefacts.
-   [ ] Body text remains intact.
-   [ ] Metadata page number remains.
-   [ ] Legitimate body mentions are not deleted.

### Chunking

-   [ ] Empty pages do not create empty chunks.
-   [ ] Every chunk has correct metadata.
-   [ ] Chunk numbering resets per page.
-   [ ] `character_count == len(text)`.
-   [ ] The character-vs-token size rule is explicitly decided and
    tested.
-   [ ] In-page overlap works.
-   [ ] No cross-page overlap is introduced.
-   [ ] Footer text is absent.

### Embeddings

-   [ ] Number of vectors equals number of chunks.
-   [ ] Dimensions are consistent.
-   [ ] Values are finite.
-   [ ] Query and document use the same model.
-   [ ] Model is loaded once per process where practical.

### Vector store

-   [ ] All records are inserted.
-   [ ] IDs are unique.
-   [ ] Metadata is preserved.
-   [ ] Database survives process exit.
-   [ ] Collection can be reopened.
-   [ ] Re-ingestion does not create accidental duplicates.

### Retrieval

Test queries with known answers in the lecture: - "What makes a model
linear?" - "How does regression analysis estimate model parameters?" -
"What is the difference between a mathematical model and a statistical
model?" - "Which conditions make a linear form independent of a
quadratic form?"

For each query, inspect whether relevant pages appear in the top five,
whether the chunks actually answer the question, and whether source/page
metadata is correct. Code execution alone does not prove retrieval
quality.

## 12. Recommended implementation order

1.  Create/activate `.venv`; install dependencies and save
    `requirements.txt`.
2.  Implement and test `extractor.py`.
3.  Implement and test `cleaner.py`.
4.  Keep the working V1 chunker; validate metadata, overlap, and size
    behavior.
5.  Implement `embedder.py`; validate count and dimensions.
6.  Implement `vector_store.py`; verify persistent storage and reload.
7.  Implement `retriever.py`; inspect top-k results.
8.  Add context formatting.
9.  Run an end-to-end test: PDF → extraction → cleaning → chunks →
    embeddings → Chroma → query → results.
10. Freeze V1 and record known limitations before starting V2.

Build and validate each layer independently instead of hiding everything
inside one large script.

## 13. End-to-end acceptance test

V1 passes when you can: 1. Ingest `lecture01.pdf` with correct
course/lecture metadata. 2. Confirm footer text is absent. 3. Confirm
all chunks have course, lecture, source, page, chunk, and character
count. 4. Confirm the documented chunk-size rule. 5. Generate one
embedding per chunk. 6. Persist the records to ChromaDB. 7. Exit Python
and start a new process. 8. Reopen the collection without re-ingesting
the PDF. 9. Search "difference between linear and nonlinear models." 10.
Print the top five matches with source, page, chunk, and distance. 11.
Manually verify that useful evidence from the relevant lecture pages is
retrieved.

Illustrative terminal output:

``` text
Query: difference between linear and nonlinear models

1. Source: lecture01.pdf | Page: 14 | Chunk: 1 | Distance: ...
   Text: A model is linear if it is linear in its parameters...

2. Source: lecture01.pdf | Page: 13 | Chunk: 1 | Distance: ...
   Text: Linear models play a central part in modern statistical methods...
```

Actual order and distance depend on the data and model.

## 14. Known limitations

Record these rather than trying to solve everything in V1: - PDF
extraction may scramble matrix brackets, subscripts, superscripts,
summation limits, and complex fractions. - Scanned pages and figure-only
information may require OCR or visual processing. - The chosen embedding
model may be weak on specialist mathematics. - Page-wise chunking does
not understand logical sections. - A chunk can split a sentence or
derivation. - No reranker or hybrid keyword search is included. -
Semantic similarity does not guarantee a chunk answers the question. -
Distance is not a calibrated confidence score.

## 15. V1 completion checklist

-   [x] Fresh environment can be created from these instructions.
-   [x] Dependencies install from `requirements.txt`.
-   [x] PDF pages extract with correct metadata.
-   [x] Recurring footer is removed without damaging body text.
-   [x] Page-based chunks follow the documented size/overlap behavior.
-   [x] Chunk metadata and character counts are correct.
-   [x] One consistent embedding model generates vectors.
-   [x] Chunks, vectors, IDs, and metadata persist in ChromaDB.
-   [x] Database can be reopened in a new process.
-   [x] Queries use the same embedding model.
-   [x] Top-k results include text, metadata, and distance.
-   [x] Results can be formatted as LLM context.
-   [x] Representative queries return useful lecture chunks.
-   [x] Configuration and limitations are documented.

**V1 Status: COMPLETED & FROZEN.**
For a detailed changelog of practical enhancements made during implementation (such as math equation reconstruction, natural boundary chunking, topic isolation, and dual-backend embedding), see **[V1.md](file:///d:/Placement/ZombAI/Hello-Lets-Talk-IITK/V1.md)**.
