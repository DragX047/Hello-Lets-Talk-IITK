"""Unit and Integration Test Suite for Hello Lets Talk IITK (V2).

Tests:
1. CourseGraph CRUD, edge traversal, and JSON serialization.
2. EquationStore catalog lookup and keyword matching.
3. GraphRetriever hybrid retrieval (text chunks + discrete LaTeX equations).
4. LLMGenerator prompt assembly and local Ollama execution.
"""

import os
import sys
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output on Windows terminals
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.graph_store import CourseGraph
from src.equation_parser import EquationStore, Equation
from src.graph_retriever import GraphRetriever
from src.vector_store import VectorStore
from src.generator import LLMGenerator
from src.embedder import Embedder


class TestHelloLetsTalkIITKV2(unittest.TestCase):
    """Test suite covering V2 components."""

    def test_course_graph_crud_and_serialization(self):
        """Test node/edge additions, traversal, and JSON persistence in CourseGraph."""
        cg = CourseGraph(course_id="MTH441")

        # Add Course, Lecture, Topic, Chunk, and Equation
        cg.add_course(course_id="MTH441", name="Regression Analysis", instructor="Prof. Shalabh")
        cg.add_lecture(
            lecture_id="MTH441_L01",
            course_id="MTH441",
            lecture_num=1,
            title="Introduction",
            source_file="lecture01.pdf",
        )
        cg.add_topic(
            topic_id="MTH441_L01_TOPIC_chi2",
            lecture_id="MTH441_L01",
            title="Chi-Square Distribution",
        )
        cg.add_chunk(
            chunk_id="chunk_01",
            lecture_id="MTH441_L01",
            page=9,
            chunk_num=1,
            text="Chi-square distribution definition and properties.",
            topic_id="MTH441_L01_TOPIC_chi2",
        )
        cg.add_chunk(
            chunk_id="chunk_02",
            lecture_id="MTH441_L01",
            page=9,
            chunk_num=2,
            text="Probability density function of chi-square.",
            topic_id="MTH441_L01_TOPIC_chi2",
        )
        cg.add_equation(
            equation_id="eq_chi2_pdf",
            name="Chi-Square PDF",
            latex=r"f(x) = \frac{1}{\Gamma(k/2) 2^{k/2}} x^{k/2 - 1} e^{-x/2}",
            lecture_id="MTH441_L01",
            page=9,
            variables=["x", "k"],
            topic_id="MTH441_L01_TOPIC_chi2",
        )

        # Link edges
        cg.link_equation_to_chunk("chunk_02", "eq_chi2_pdf")
        cg.link_sequential_chunks(["chunk_01", "chunk_02"])

        summary = cg.summary()
        self.assertEqual(summary["total_nodes"], 6)
        self.assertEqual(summary["node_types"]["course"], 1)
        self.assertEqual(summary["node_types"]["lecture"], 1)
        self.assertEqual(summary["node_types"]["topic"], 1)
        self.assertEqual(summary["node_types"]["chunk"], 2)
        self.assertEqual(summary["node_types"]["equation"], 1)

        # Check edge traversal
        eqs = cg.get_equations_for_chunk("chunk_02")
        self.assertEqual(len(eqs), 1)
        self.assertEqual(eqs[0]["id"], "eq_chi2_pdf")

        adj = cg.get_adjacent_chunks("chunk_01")
        self.assertIsNotNone(adj["next"])
        self.assertEqual(adj["next"]["id"], "chunk_02")

        # Save and reload JSON
        test_path = "storage/graph/test_unittest_graph.json"
        cg.save_json(test_path)
        self.assertTrue(Path(test_path).exists())

        reloaded = CourseGraph.load_json(test_path)
        self.assertEqual(reloaded.summary()["total_nodes"], 6)
        self.assertEqual(len(reloaded.get_equations_for_chunk("chunk_02")), 1)

        if Path(test_path).exists():
            os.remove(test_path)

    def test_equation_store_catalog(self):
        """Verify EquationStore contains verified equations across all 6 lectures."""
        store = EquationStore()
        all_eqs = store.all_equations()
        self.assertGreaterEqual(len(all_eqs), 40)

        # Lecture queries for all 6 lectures
        l1_eqs = store.get_by_lecture("Lecture 01")
        l2_eqs = store.get_by_lecture("Lecture 02")
        l3_eqs = store.get_by_lecture("Lecture 03")
        l4_eqs = store.get_by_lecture("Lecture 04")
        l5_eqs = store.get_by_lecture("Lecture 05")
        l6_eqs = store.get_by_lecture("Lecture 06")

        self.assertGreaterEqual(len(l1_eqs), 10)
        self.assertGreaterEqual(len(l2_eqs), 5)
        self.assertGreaterEqual(len(l3_eqs), 8)
        self.assertGreaterEqual(len(l4_eqs), 5)
        self.assertGreaterEqual(len(l5_eqs), 5)
        self.assertGreaterEqual(len(l6_eqs), 5)

        # Keyword search across chapters
        chi_matches = store.find_relevant_equations("Tell me about chi-square distribution")
        self.assertTrue(any("chi2" in eq.id for eq in chi_matches))

        ols_matches = store.find_relevant_equations("What are normal equations in multiple regression?")
        self.assertTrue(any("normal_equations" in eq.id for eq in ols_matches))

        gauss_matches = store.find_relevant_equations("Gauss-Markov BLUE theorem")
        self.assertTrue(any("gauss_markov" in eq.id for eq in gauss_matches))

        cook_matches = store.find_relevant_equations("Cook's distance and influential points")
        self.assertTrue(any("cooks_distance" in eq.id for eq in cook_matches))

        vif_matches = store.find_relevant_equations("Variance inflation factor and multicollinearity")
        self.assertTrue(any("vif" in eq.id for eq in vif_matches))

    def test_graph_retriever_hybrid(self):
        """Verify GraphRetriever returns both text chunks and discrete equations."""
        vs = VectorStore()
        self.assertGreater(vs.count(), 0, "VectorStore must have indexed chunks from ingest.")

        retriever = GraphRetriever()
        result = retriever.retrieve(query="Gauss-Markov theorem and BLUE", top_k=3)

        self.assertIn("chunks", result)
        self.assertIn("equations", result)
        self.assertGreater(len(result["chunks"]), 0)
        self.assertGreater(len(result["equations"]), 0)

        # Verify discrete equation content
        eq_names = [e["name"] for e in result["equations"]]
        self.assertTrue(any("Gauss-Markov" in name for name in eq_names))

        # Check formatted blocks
        self.assertIn("Evidence #", result["formatted_context"])
        self.assertIn("LaTeX:", result["formatted_equations"])

    def test_llm_generator_prompt_assembly(self):
        """Verify LLM prompt formatting with citations and discrete LaTeX."""
        gen = LLMGenerator()
        chunks = [
            {
                "id": "c1",
                "text": "The OLS estimator of beta is given by solving normal equations.",
                "metadata": {"course": "MTH441", "lecture": "Lecture 03", "page": 7, "chunk": 1},
            }
        ]
        equations = [
            {
                "id": "MTH441_L03_eq_normal_equations",
                "name": "Normal Equations",
                "latex": r"X'X \hat{\beta} = X'y",
                "variables": ["X", r"\hat{\beta}", "y"],
                "conditions": "rank(X) = p",
            }
        ]

        # Test generation with local Ollama
        res = gen.generate_rag_answer(
            query="What are the normal equations for OLS?",
            context_chunks=chunks,
            discrete_equations=equations,
        )

        self.assertIn("answer", res)
        self.assertGreater(len(res["answer"]), 10)
        self.assertTrue(
            any("Lecture 03, Page 7" in c for c in res["citations"]),
            f"Expected citation containing 'Lecture 03, Page 7', got {res['citations']}",
        )
        self.assertIn("MTH441_L03_eq_normal_equations", res["referenced_equations"])

    def test_ee602_retrieval_and_equations(self):
        """Verify EE602 hybrid retrieval returns book chapters and discrete equations."""
        store = VectorStore()
        embedder = Embedder()
        eq_store = EquationStore()
        retriever = GraphRetriever(vector_store=store, embedder=embedder, equation_store=eq_store)

        res = retriever.retrieve("Explain the Cramer-Rao Lower Bound and Fisher Information", top_k=3, course_filter="EE602")
        self.assertGreaterEqual(len(res["chunks"]), 1)
        self.assertEqual(res["chunks"][0]["metadata"]["course"], "EE602")
        self.assertTrue(any("Chapter 03" in c["metadata"].get("lecture", "") for c in res["chunks"]))

        # Verify discrete equations for CRLB
        eq_ids = [eq["id"] for eq in res["equations"]]
        self.assertIn("EE602_CH03_eq_crlb_scalar", eq_ids)


if __name__ == "__main__":
    unittest.main()
