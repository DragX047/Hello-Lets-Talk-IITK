"""Unit and Integration Tests for EquationExtractor and Multi-Course Equation Retrieval."""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.equation_extractor import EquationExtractor
from src.equation_parser import EquationStore, Equation
from src.graph_retriever import GraphRetriever
from src.vector_store import VectorStore


class TestEquationExtractor(unittest.TestCase):
    """Test suite covering EquationExtractor and dynamic retrieval."""

    def test_unicode_and_latex_cleaning(self):
        """Verify Unicode mathematical typography is cleanly converted to LaTeX."""
        raw_cs = "min_{w, b} 𝒘⊤𝒙𝒏 + 𝑏 ≥ 1 ∀𝑛"
        cleaned = EquationExtractor.clean_latex(raw_cs)
        self.assertIn(r"\mathbf{w}^\top", cleaned)
        self.assertIn(r"\mathbf{x}_n", cleaned)
        self.assertIn(r"\ge", cleaned)

    def test_variable_extraction(self):
        """Verify variables are correctly parsed from LaTeX formulas."""
        latex = r"\mathcal{L}(w) = -\sum_{n=1}^N \left[ y_n \log \sigma(w^\top x_n) + (1 - y_n) \log(1 - \sigma(w^\top x_n)) \right]"
        vars_found = EquationExtractor.extract_variables(latex)
        self.assertIn(r"\sigma", vars_found)
        self.assertIn("w", vars_found)
        self.assertIn("x", vars_found)
        self.assertIn("y", vars_found)
        self.assertIn("N", vars_found)

    def test_extract_from_text_cs771(self):
        """Test extraction from a CS771 SVM text block."""
        snippet = """
        Support Vector Machines: Maximum Margin
        min_{w, b} \\frac{1}{2} ||w||^2 + C \\sum_{n=1}^N \\max(0, 1 - y_n (w^T x_n + b))
        where C > 0 controls regularization.
        """
        eqs = EquationExtractor.extract_from_text(snippet, page=11, course="CS771", lecture="Lecture 10")
        self.assertGreaterEqual(len(eqs), 1)
        eq = eqs[0]
        self.assertEqual(eq.course, "CS771")
        self.assertEqual(eq.lecture, "Lecture 10")
        self.assertEqual(eq.page, 11)
        self.assertIn("Support Vector", eq.name)
        self.assertIn(r"\sum", eq.latex)

    def test_extract_from_text_ee602(self):
        """Test extraction from an EE602 Neyman-Fisher text block."""
        snippet = """
        Theorem 5.1 (Neyman-Fisher Factorization) If we can factor the PDF p(x; \\theta) as
        p(x; \\theta) = g(T(x), \\theta) h(x) (5.3)
        then T(x) is a sufficient statistic for \\theta.
        """
        eqs = EquationExtractor.extract_from_text(snippet, page=113, course="EE602", lecture="Chapter 05")
        self.assertGreaterEqual(len(eqs), 1)
        eq = eqs[0]
        self.assertEqual(eq.course, "EE602")
        self.assertEqual(eq.lecture, "Chapter 05")
        self.assertIn("Neyman-Fisher", eq.name)
        self.assertIn(r"\theta", eq.latex)
        self.assertIn(r"\theta", eq.variables)

    def test_dynamic_retrieval_across_courses(self):
        """Test that GraphRetriever retrieves discrete equations for all 3 courses."""
        retriever = GraphRetriever()

        # 1. Query CS771 (SVM & Kernel Trick)
        res_cs = retriever.retrieve("Explain SVM soft margin and Gaussian RBF kernel", top_k=3, course_filter="CS771")
        self.assertGreater(len(res_cs["chunks"]), 0)
        self.assertGreater(len(res_cs["equations"]), 0)
        eq_names_cs = [e["name"] for e in res_cs["equations"]]
        self.assertTrue(any("Kernel" in n or "SVM" in n or "Support Vector" in n for n in eq_names_cs))

        # 2. Query EE602 (CRLB & Fisher Information)
        res_ee = retriever.retrieve("What is Cramér-Rao Lower Bound and Fisher Information?", top_k=3, course_filter="EE602")
        self.assertGreater(len(res_ee["chunks"]), 0)
        self.assertGreater(len(res_ee["equations"]), 0)
        eq_names_ee = [e["name"] for e in res_ee["equations"]]
        self.assertTrue(any("Cramér-Rao" in n or "Fisher" in n or "CRLB" in n for n in eq_names_ee))

        # 3. Query MTH441 (Gauss-Markov & BLUE)
        res_mth = retriever.retrieve("Gauss-Markov theorem and BLUE", top_k=3, course_filter="MTH441")
        self.assertGreater(len(res_mth["chunks"]), 0)
        self.assertGreater(len(res_mth["equations"]), 0)


if __name__ == "__main__":
    unittest.main()

