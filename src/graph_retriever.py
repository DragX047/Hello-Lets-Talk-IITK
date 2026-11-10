"""Hybrid Graph-RAG Retrieval and Context Assembly Module.

Unifies semantic dense vector search (ChromaDB) with knowledge graph traversal
(NetworkX CourseGraph) and discrete mathematical equation lookup (EquationStore).
Provides enriched dual context (explanatory text + discrete LaTeX equations)
and end-to-end question answering via local Llama 3.1 8B.
"""

from typing import Any, Dict, List, Optional, Set
from pathlib import Path

try:
    from .vector_store import VectorStore
    from .embedder import Embedder, get_default_embedder
    from .graph_store import CourseGraph
    from .equation_parser import Equation, EquationStore
    from .generator import LLMGenerator
except ImportError:
    from vector_store import VectorStore
    from embedder import Embedder, get_default_embedder
    from graph_store import CourseGraph
    from equation_parser import Equation, EquationStore
    from generator import LLMGenerator


class GraphRetriever:
    """Orchestrates hybrid vector search, knowledge graph traversal, and RAG generation."""

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedder: Optional[Embedder] = None,
        graph_store: Optional[CourseGraph] = None,
        equation_store: Optional[EquationStore] = None,
        generator: Optional[LLMGenerator] = None,
        graph_path: str = "storage/graph/MTH441_graph.json",
    ):
        """Initialize hybrid graph retriever components.

        Args:
            vector_store: ChromaDB vector store instance.
            embedder: SentenceTransformer / ONNX embedding generator.
            graph_store: CourseGraph knowledge graph instance.
            equation_store: EquationStore catalog instance.
            generator: LLMGenerator instance for Llama 3.1 8B inference.
            graph_path: Path to serialized knowledge graph JSON if graph_store is None.
        """
        self.vector_store = vector_store or VectorStore()
        self.embedder = embedder or get_default_embedder()
        self.equation_store = equation_store or EquationStore()
        self._generator = generator

        # Multi-course graph management: load all available graphs in storage/graph/
        self.graphs: Dict[str, CourseGraph] = {}
        if graph_store is not None:
            self.graphs[graph_store.course_id.upper()] = graph_store
            self.default_course = graph_store.course_id.upper()
        else:
            self.default_course = "MTH441"
            # Auto-discover course graphs from storage/graph/
            graph_dir = Path("storage/graph")
            if graph_dir.exists():
                for gf in graph_dir.glob("*_graph.json"):
                    cid = gf.stem.replace("_graph", "").upper()
                    try:
                        self.graphs[cid] = CourseGraph.load_json(str(gf))
                    except Exception as e:
                        print(f"[Warning] Failed loading graph from {gf}: {e}")

            # Fallback to loading specified graph_path if needed
            if not self.graphs and Path(graph_path).exists():
                try:
                    loaded = CourseGraph.load_json(graph_path)
                    self.graphs[loaded.course_id.upper()] = loaded
                except Exception:
                    pass

    @property
    def graph_store(self) -> CourseGraph:
        """Backward-compatible accessor for primary course graph."""
        if self.default_course in self.graphs:
            return self.graphs[self.default_course]
        if self.graphs:
            return next(iter(self.graphs.values()))
        return CourseGraph(self.default_course)

    def get_graph(self, course_id: Optional[str] = None) -> Optional[CourseGraph]:
        """Fetch course graph for course_id, loading on-demand if available on disk."""
        if not course_id:
            course_id = self.default_course
        course_key = course_id.upper()

        if course_key in self.graphs:
            return self.graphs[course_key]

        # Attempt to load on demand
        expected_path = Path(f"storage/graph/{course_key}_graph.json")
        if expected_path.exists():
            try:
                g = CourseGraph.load_json(str(expected_path))
                self.graphs[course_key] = g
                return g
            except Exception as e:
                print(f"[Warning] Could not load graph for {course_key}: {e}")

        return self.graphs.get(self.default_course)

    @property
    def generator(self) -> LLMGenerator:
        """Lazy-initialize LLM generator to avoid startup overhead if not querying."""
        if self._generator is None:
            self._generator = LLMGenerator()
        return self._generator

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        course_filter: Optional[str] = None,
        lecture_filter: Optional[str] = None,
        expand_equations: bool = True,
        expand_sequential: bool = False,
    ) -> Dict[str, Any]:
        """Execute hybrid search: dense vector retrieval followed by graph traversal.

        Args:
            query: The user's natural language question or topic.
            top_k: Number of nearest text chunks to retrieve.
            course_filter: Optional filter by course code (e.g., 'MTH441').
            lecture_filter: Optional filter by lecture name (e.g., 'Lecture 01').
            expand_equations: Whether to discover and link discrete equations.
            expand_sequential: Whether to pull in neighboring sequential chunks.

        Returns:
            Dict containing retrieved chunks, linked equations, topics, and formatted blocks.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            return {
                "query": query,
                "chunks": [],
                "equations": [],
                "topics": [],
                "lectures": [],
                "formatted_context": "",
                "formatted_equations": "",
            }

        # 1. Semantic Vector Search via ChromaDB
        where_clause = {}
        if course_filter:
            where_clause["course"] = course_filter
        if lecture_filter:
            where_clause["lecture"] = lecture_filter
        where = where_clause if where_clause else None

        query_vec = self.embedder.embed_query(cleaned_query)
        raw_results = self.vector_store.query(
            query_embedding=query_vec,
            top_k=top_k,
            where=where,
        )

        ids = raw_results.get("ids", [[]])[0]
        documents = raw_results.get("documents", [[]])[0]
        metadatas = raw_results.get("metadatas", [[]])[0]
        distances = raw_results.get("distances", [[]])[0]

        chunks: List[Dict[str, Any]] = []
        found_chunk_ids: Set[str] = set()

        for chunk_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
            similarity = round(1.0 - dist, 4) if dist is not None else None
            chunk_dict = {
                "id": chunk_id,
                "text": doc,
                "metadata": meta,
                "distance": round(dist, 4) if dist is not None else None,
                "similarity": similarity,
            }
            chunks.append(chunk_dict)
            found_chunk_ids.add(chunk_id)

        # 2. Graph Traversal: Pull in Connected Equations and Topic Metadata
        linked_equations_map: Dict[str, Dict[str, Any]] = {}
        covered_topics: Set[str] = set()
        covered_lectures: Set[str] = set()

        for chunk in chunks:
            c_id = chunk["id"]
            meta = chunk["metadata"]
            lec = meta.get("lecture", "")
            page = int(meta.get("page", 0)) if str(meta.get("page", "")).isdigit() else 0
            crs = meta.get("course", course_filter or self.default_course)
            cg = self.get_graph(crs)

            if lec:
                lec_display = f"{crs} {lec}" if crs else lec
                covered_lectures.add(lec_display)

            if cg:
                # Check direct graph edges (REFERENCES_EQUATION)
                graph_eqs = cg.get_equations_for_chunk(c_id)
                for eq_data in graph_eqs:
                    eq_id = eq_data.get("id") or eq_data.get("equation_id")
                    if eq_id:
                        linked_equations_map[eq_id] = eq_data

                # Check chunk node's parent topic
                node_data = cg.get_node(c_id)
                if node_data:
                    topic_id = node_data.get("topic_id", "")
                    if topic_id:
                        covered_topics.add(topic_id)

                # Pull in neighboring sequential chunk if requested
                if expand_sequential:
                    adj = cg.get_adjacent_chunks(c_id, include_previous=False, include_next=True)
                    next_c = adj.get("next")
                    if next_c and next_c["id"] not in found_chunk_ids:
                        # Append sequence extension
                        found_chunk_ids.add(next_c["id"])
                        # Do not alter primary ranking, but flag as expansion

        # 3. Supplemental Equation Discovery from EquationStore & Dynamic Extraction
        if expand_equations:
            chunk_metas = [c["metadata"] for c in chunks]
            chunk_texts = [c.get("text", "") for c in chunks]
            catalog_matches = self.equation_store.find_relevant_equations(
                query=cleaned_query,
                chunk_metadatas=chunk_metas,
                course_filter=course_filter,
                chunk_texts=chunk_texts,
            )
            for eq in catalog_matches:
                if eq.id not in linked_equations_map:
                    linked_equations_map[eq.id] = eq.to_dict()

        # Deduplicate equations: prefer curated catalog over auto-extracted, eliminate duplicates by (page, norm_latex)
        deduped_equations: Dict[str, Dict[str, Any]] = {}
        for eq_data in linked_equations_map.values():
            p = eq_data.get("page", 0)
            raw_lat = eq_data.get("latex", "")
            norm_key = (p, "".join(c for c in raw_lat.lower() if c.isalnum()))
            
            if norm_key not in deduped_equations:
                deduped_equations[norm_key] = eq_data
            else:
                existing = deduped_equations[norm_key]
                # If existing is auto and new one is curated catalog, replace it
                if "_auto_" in existing.get("id", "") and "_auto_" not in eq_data.get("id", ""):
                    deduped_equations[norm_key] = eq_data

        linked_equations = list(deduped_equations.values())

        # 4. Format Context Blocks for LLM
        formatted_context = self.format_chunks_context(chunks)
        formatted_equations = self.format_equations_context(linked_equations)

        return {
            "query": cleaned_query,
            "chunks": chunks,
            "equations": linked_equations,
            "topics": sorted(list(covered_topics)),
            "lectures": sorted(list(covered_lectures)),
            "formatted_context": formatted_context,
            "formatted_equations": formatted_equations,
        }

    def ask(
        self,
        query: str,
        top_k: int = 5,
        course_filter: Optional[str] = None,
        lecture_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """End-to-end grounded Q&A with Llama 3.1 8B using retrieved evidence and equations.

        Args:
            query: The student's question.
            top_k: Number of chunks to retrieve for grounding.
            course_filter: Optional course code filter.
            lecture_filter: Optional lecture filter.

        Returns:
            Dict containing 'answer', 'citations', 'referenced_equations', and 'retrieval'.
        """
        retrieval = self.retrieve(
            query=query,
            top_k=top_k,
            course_filter=course_filter,
            lecture_filter=lecture_filter,
            expand_equations=True,
        )

        rag_result = self.generator.generate_rag_answer(
            query=query,
            context_chunks=retrieval["chunks"],
            discrete_equations=retrieval["equations"],
        )

        return {
            "query": query,
            "answer": rag_result["answer"],
            "citations": rag_result["citations"],
            "referenced_equations": rag_result["referenced_equations"],
            "retrieval": retrieval,
            "model": rag_result["model"],
        }

    @staticmethod
    def format_chunks_context(chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved text chunks with clear academic citations."""
        if not chunks:
            return "No matching lecture notes found."

        blocks = []
        for i, c in enumerate(chunks, 1):
            meta = c["metadata"]
            course = meta.get("course", "")
            lec = meta.get("lecture", "Lecture")
            page = meta.get("page", "?")
            c_num = meta.get("chunk", "?")
            dist = c.get("distance", "")

            lec_tag = f"{course} {lec}" if course else lec
            dist_label = f" | Dist: {dist}" if dist is not None else ""
            block = (
                f"Evidence #{i} [{lec_tag}, Page {page}] (Chunk {c_num}{dist_label}):\n"
                f"{c['text'].strip()}"
            )
            blocks.append(block)

        return "\n\n".join(blocks)

    @staticmethod
    def format_equations_context(equations: List[Dict[str, Any]]) -> str:
        """Format discrete equations into clean LaTeX blocks with parameter descriptions."""
        if not equations:
            return "No verified discrete equations linked."

        blocks = []
        for eq in equations:
            eq_id = eq.get("id", "EQ")
            name = eq.get("name", "Equation")
            latex = eq.get("latex", "")
            variables = ", ".join(eq.get("variables", []))
            conditions = eq.get("conditions", "")
            page = eq.get("page", "?")
            course = eq.get("course") or (eq_id.split("_")[0] if "_" in eq_id else "")
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

            lec_tag = f"{course} {lec}" if course and not lec.startswith(course) else (lec or course)
            block = (
                f"[{eq_id}] - {name} ({lec_tag}, Page {page})\n"
                f"LaTeX: $${latex}$$\n"
                f"Variables: {variables}"
            )
            if conditions:
                block += f"\nConditions: {conditions}"
            blocks.append(block)

        return "\n\n".join(blocks)
