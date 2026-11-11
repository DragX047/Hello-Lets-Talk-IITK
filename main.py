"""Hello Lets Talk IITK (V2) - Multi-Lecture Graph-RAG System CLI & Entrypoint.

Provides CLI interfaces for:
- Batch course ingestion (multi-lecture PDF extraction, chunking, embedding, graph construction)
- Semantic vector search
- Hybrid Graph retrieval (text chunks + discrete LaTeX equations)
- Interactive LLM Chat with local Llama 3.1 8B via Ollama
- Knowledge graph inspection and summary
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Ensure UTF-8 output on Windows terminals
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.ingest_manager import IngestManager
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.graph_store import CourseGraph
from src.graph_retriever import GraphRetriever
from src.generator import LLMGenerator
from src.equation_parser import EquationStore


def run_ingest(
    lectures_dir: Optional[str] = None,
    course: str = "MTH441",
    reset: bool = False,
):
    """Run batch multi-lecture ingestion and build knowledge graph."""
    target_dir = lectures_dir or f"data/lectures/{course}"
    print("=" * 65)
    print(f" Hello Lets Talk IITK - Ingesting Course: {course}")
    print(f" Source Directory: {target_dir}")
    print("=" * 65)

    manager = IngestManager(course_id=course)
    report = manager.ingest_all(lectures_dir=target_dir, reset_chroma=reset)
    return report


def run_vector_query(query: str, top_k: int = 5, course: Optional[str] = None):
    """Query ChromaDB semantic vector store directly."""
    retriever = Retriever()
    results = retriever.retrieve(query, top_k=top_k, course_filter=course)

    print(f"\nFound {len(results)} matches for: '{query}'" + (f" [Course: {course}]" if course else ""))
    print("=" * 65)
    for i, res in enumerate(results, 1):
        meta = res["metadata"]
        dist = res["distance"]
        sim = res["similarity"]
        src = meta.get("source", "unknown")
        crs = meta.get("course", "")
        lec = meta.get("lecture", "Lecture")
        page = meta.get("page", "?")
        c_num = meta.get("chunk", "?")

        lec_tag = f"{crs} {lec}" if crs else lec
        print(f"#{i} [{lec_tag} | Page {page} | Chunk {c_num}] (Dist: {dist}, Sim: {sim})")
        print(f"{res['text'].strip()}\n" + "-" * 65)


def run_graph_query(query: str, top_k: int = 5, course: Optional[str] = None):
    """Perform hybrid retrieval: vector search + knowledge graph traversal."""
    graph_retriever = GraphRetriever()
    result = graph_retriever.retrieve(query=query, top_k=top_k, course_filter=course)

    print("=" * 65)
    print(f" Hello Lets Talk IITK - Hybrid Graph-RAG Retrieval: '{query}'" + (f" [Course: {course}]" if course else ""))
    print("=" * 65)

    print(f"\n[Retrieved Text Chunks ({len(result['chunks'])})]:")
    for i, c in enumerate(result["chunks"], 1):
        m = c["metadata"]
        crs = m.get("course", "")
        lec_tag = f"{crs} {m.get('lecture')}" if crs else m.get("lecture")
        print(f"  #{i} [{lec_tag} Page {m.get('page')}] (Cosine Sim: {c['similarity']})")

    print(f"\n[Linked Discrete Equations ({len(result['equations'])})]:")
    for eq in result["equations"]:
        crs = eq.get("course") or (eq.get("id", "").split("_")[0] if "_" in eq.get("id", "") else "")
        lec = eq.get("lecture")
        if not lec:
            lid = eq.get("lecture_id", "")
            if "_CH" in lid:
                lec = f"Chapter {lid.split('_CH')[-1]}"
            elif "_L" in lid:
                lec = f"Lecture {lid.split('_L')[-1]}"
            elif "_APP" in lid:
                lec = f"Appendix {lid.split('_APP_')[-1]}"
            else:
                lec = "Lecture"
        lec_tag = f"{crs} {lec}" if crs and not lec.startswith(crs) else (lec or crs)
        print(f"  • {eq['id']} - {eq['name']} ({lec_tag}, Page {eq.get('page')})")
        print(f"    LaTeX: $${eq['latex']}$$")
        print(f"    Variables: {', '.join(eq.get('variables', []))}")
        if eq.get("conditions"):
            print(f"    Conditions: {eq['conditions']}")

    print(f"\n[Covered Topics]: {', '.join(result['topics']) or 'None explicit'}")
    print(f"[Lectures Cited]: {', '.join(result['lectures'])}")
    print("=" * 65)


def run_graph_info(course: str = "MTH441", graph_path: Optional[str] = None):
    """Display statistics and summary of the course knowledge graph."""
    target_path = graph_path or f"storage/graph/{course}_graph.json"
    if not Path(target_path).exists():
        print(f"Knowledge graph not found at '{target_path}'. Please run ingest first.")
        return

    cg = CourseGraph.load_json(target_path)
    summary = cg.summary()
    print("=" * 65)
    print(f" Hello Lets Talk IITK - Knowledge Graph Structure ({cg.course_id})")
    print(f" Path: {target_path}")
    print("=" * 65)
    print(f" Total Nodes: {summary['total_nodes']}")
    print(f" Total Edges: {summary['total_edges']}")
    print("\n Node Breakdown:")
    for ntype, count in summary["node_types"].items():
        print(f"   • {ntype.capitalize():<12}: {count}")

    print("\n Edge Relations:")
    for rel, count in summary["edge_relations"].items():
        print(f"   • {rel:<22}: {count}")
    print("=" * 65)


def interactive_chat(
    top_k: int = 5,
    single_query: Optional[str] = None,
    course: Optional[str] = None,
):
    """Launch interactive grounded chat session with local Llama 3.1 8B, or run a single query."""
    graph_retriever = GraphRetriever()

    if single_query:
        print(f"\n[Student Question]: {single_query}" + (f" [Course: {course}]" if course else ""))
        print("\n[Hello Lets Talk IITK Thinking & Traversal...]")
        response = graph_retriever.ask(query=single_query, top_k=top_k, course_filter=course)

        print("\n" + "=" * 68)
        print("[Hello Lets Talk IITK TA Answer]:")
        print(response["answer"])
        print("-" * 68)

        citations = response.get("citations", [])
        equations = response.get("referenced_equations", [])

        if citations:
            print(f"[Citations]: {', '.join(citations)}")
        if equations:
            print(f"[Equations Used]: {', '.join(equations)}")
        print("=" * 68)
        return

    course_label = course or "IIT Kanpur Courses (MTH441 / CS771 / EE602)"
    print("=" * 68)
    print(" Hello Lets Talk IITK - Teaching Assistant Interactive Shell (Llama 3.1 8B)")
    print(f" Grounded on {course_label}")
    print(" Powered by ChromaDB + Multi-Course Knowledge Graph + Local Ollama (llama3.1:8b)")
    print(" Type your academic question or 'exit' / 'q' to quit.")
    print("=" * 68)

    while True:
        try:
            query = input("\n[Student] > ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("Session closed. Happy studying!")
                break

            print("\n[Hello Lets Talk IITK Thinking & Traversal...]")
            response = graph_retriever.ask(query=query, top_k=top_k, course_filter=course)

            print("\n" + "=" * 68)
            print("[Hello Lets Talk IITK TA Answer]:")
            print(response["answer"])
            print("-" * 68)

            citations = response.get("citations", [])
            equations = response.get("referenced_equations", [])

            if citations:
                print(f"[Citations]: {', '.join(citations)}")
            if equations:
                print(f"[Equations Used]: {', '.join(equations)}")
            print("=" * 68)

        except KeyboardInterrupt:
            print("\nInterrupted. Exiting.")
            break
        except Exception as e:
            print(f"\n[Error]: {e}")


def main():
    parser = argparse.ArgumentParser(description="Hello Lets Talk IITK (V2) Multi-Lecture Graph-RAG System")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Ingest command
    ingest_p = subparsers.add_parser("ingest", help="Batch ingest lectures and build Knowledge Graph")
    ingest_p.add_argument("--dir", type=str, default=None, help="Path to lecture PDFs")
    ingest_p.add_argument("--course", "-c", type=str, default="MTH441", help="Course code (e.g. MTH441, CS771, or EE602)")
    ingest_p.add_argument("--reset", action="store_true", help="Reset ChromaDB collection before indexing")

    # Vector query
    query_p = subparsers.add_parser("query", help="Query ChromaDB vector store directly")
    query_p.add_argument("query_text", type=str, help="Search query string")
    query_p.add_argument("--top-k", type=int, default=5, help="Number of results")
    query_p.add_argument("--course", "-c", type=str, default=None, help="Course code filter (e.g. MTH441, CS771, or EE602)")

    # Graph query
    gquery_p = subparsers.add_parser("graph-query", help="Hybrid vector search + Knowledge Graph retrieval")
    gquery_p.add_argument("query_text", type=str, help="Search query string")
    gquery_p.add_argument("--top-k", type=int, default=5, help="Number of results")
    gquery_p.add_argument("--course", "-c", type=str, default=None, help="Course code filter (e.g. MTH441, CS771, or EE602)")

    # Graph info
    info_p = subparsers.add_parser("graph-info", help="Display Knowledge Graph statistics")
    info_p.add_argument("--course", "-c", type=str, default="MTH441", help="Course code (e.g. MTH441, CS771, or EE602)")
    info_p.add_argument("--path", type=str, default=None, help="Direct path to graph JSON file")

    # Interactive Chat
    chat_p = subparsers.add_parser("chat", help="Launch interactive grounded chat with Llama 3.1 8B")
    chat_p.add_argument("--query", "-q", type=str, default=None, help="Optional single query to execute non-interactively")
    chat_p.add_argument("--top-k", type=int, default=5, help="Number of grounding chunks")
    chat_p.add_argument("--course", "-c", type=str, default=None, help="Course code filter (e.g. MTH441, CS771, or EE602)")

    args = parser.parse_args()

    if args.command == "ingest":
        run_ingest(lectures_dir=args.dir, course=args.course, reset=args.reset)
    elif args.command == "query":
        run_vector_query(args.query_text, top_k=args.top_k, course=args.course)
    elif args.command == "graph-query":
        run_graph_query(args.query_text, top_k=args.top_k, course=args.course)
    elif args.command == "graph-info":
        run_graph_info(course=args.course, graph_path=args.path)
    elif args.command == "chat":
        interactive_chat(top_k=args.top_k, single_query=args.query, course=args.course)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
