"""Automated Equation Extraction and OCR Normalization Engine.

Parses text chunks or raw document pages, detects mathematical formulas,
normalizes OCR/Unicode mathematical typography into clean LaTeX,
extracts constituent variables, and constructs discrete Equation entities.
"""

import re
import sys
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from .equation_parser import Equation
except ImportError:
    from equation_parser import Equation


class EquationExtractor:
    """Robust extractor for mathematical expressions and equations from lecture text."""

    # Map of Unicode characters, math symbols, and OCR artifacts to standard LaTeX equivalents
    UNICODE_TO_LATEX = {
        # Bold / Italic variable forms common in PDF extraction
        "𝒘": r"\mathbf{w}",
        "𝑤": "w",
        "𝐰": r"\mathbf{w}",
        "𝒘⊤": r"\mathbf{w}^\top",
        "w⊤": r"\mathbf{w}^\top",
        "w^T": r"\mathbf{w}^\top",
        "𝒙": r"\mathbf{x}",
        "𝑥": "x",
        "𝐱": r"\mathbf{x}",
        "𝒙𝒏": r"\mathbf{x}_n",
        "x_n": r"\mathbf{x}_n",
        "𝒚": r"\mathbf{y}",
        "𝑦": "y",
        "𝐲": r"\mathbf{y}",
        "𝒚𝒏": r"y_n",
        "y_n": r"y_n",
        "𝒛": r"\mathbf{z}",
        "𝛾": r"\gamma",
        "𝛼": r"\alpha",
        "𝛼𝑖": r"\alpha_i",
        "𝛽": r"\beta",
        "𝜃": r"\theta",
        "𝜽": r"\boldsymbol{\theta}",
        "𝜎": r"\sigma",
        "𝜇": r"\mu",
        "𝜆": r"\lambda",
        "𝜖": r"\epsilon",
        "𝜀": r"\varepsilon",
        "𝜙": r"\phi",
        "𝜓": r"\psi",
        "𝜔": r"\omega",
        "𝛴": r"\Sigma",
        "ℒ": r"\mathcal{L}",
        "𝒟": r"\mathcal{D}",
        "ℋ": r"\mathcal{H}",
        "top": r"\top",
        "⊤": r"^\top",
        "ᵀ": r"^\top",
        "෍": r"\sum",
        "෠": r"\hat",
        "∑": r"\sum",
        "∏": r"\prod",
        "∫": r"\int",
        "∂": r"\partial",
        "∇": r"\nabla",
        "√": r"\sqrt",
        "≤": r"\le",
        "≥": r"\ge",
        "≈": r"\approx",
        "≠": r"\ne",
        "≡": r"\equiv",
        "∈": r"\in",
        "∉": r"\notin",
        "∝": r"\propto",
        "∼": r"\sim",
        "~": r"\sim",
        "×": r"\times",
        "·": r"\cdot",
        "±": r"\pm",
        "→": r"\to",
        "⇒": r"\implies",
        "⇔": r"\iff",
        "∀": r"\forall ",
        "∃": r"\exists ",
        "∥": r"\|",
        "‖": r"\|",
        # OCR artifact cleanup common in scanned texts (Kay, etc.)
        "==": "=",
        ":=": "=",
        ":o": "=",
        ":о": "=",
        "д": r"\partial",
        "Ьу": "by",
        "а": "a",
        "е": "e",
        "о": "o",
        "р": "p",
        "с": "c",
        "х": "x",
        "у": "y",
        "А": "A",
        "В": "B",
        "С": "C",
        "Е": "E",
        "Н": "H",
        "Т": "T",
    }

    # Relation indicators required for an expression to qualify as an equation
    RELATION_REGEX = re.compile(
        r"(=|\\le|\\ge|\\approx|\\propto|\\in|\\sim|\\to|\\equiv|\\ne|<|>|\\sim|"
        r"\bargmin\b|\bargmax\b|\bmin\b|\bmax\b)",
        re.IGNORECASE,
    )

    # Core mathematical symbol & operator patterns
    MATH_FEATURE_REGEX = re.compile(
        r"(\\sum|\\prod|\\int|\\exp|\\ln|\\log|\\mathcal|\\mathbf|\\boldsymbol|"
        r"\^|\|\||_|\{|\}|\\mu|\\sigma|\\gamma|\\theta|\\alpha|\\beta|\\lambda|"
        r"\\epsilon|\\phi|\\psi|\\omega|\\Sigma|\\Gamma|\\partial|\\nabla|\\sqrt|"
        r"[pPfgh]\([a-zA-Z0-9_,\s\(\)\|\;]+\)|E\[|E\(|Var\(|Cov\(|Bias\(|MSE\(|"
        r"\([0-9]+[A-Za-z]?\.[0-9]+[A-Za-z]?\))"
    )

    # Explicit equation numbering patterns like (3.14), (A.5), (10.2)
    EQ_NUM_REGEX = re.compile(
        r"(?:^|\s)\(([0-9]+[A-Za-z]?\.[0-9]+[A-Za-z]?|[A-Z][0-9]*\.[0-9]+)\)\s*$"
    )

    # Blacklist of non-equation noise patterns
    EXCLUDED_PATTERNS = [
        re.compile(r"^(https?://|www\.)", re.IGNORECASE),
        re.compile(r"^(figure|table|page|slide|chapter|lecture)\s+\d+", re.IGNORECASE),
        re.compile(r"^copyright\b", re.IGNORECASE),
        re.compile(r"^all rights reserved", re.IGNORECASE),
        re.compile(r"\b(click here|download|author|department of)\b", re.IGNORECASE),
    ]

    @classmethod
    def clean_latex(cls, text: str) -> str:
        """Normalize raw text with Unicode characters and OCR quirks into valid LaTeX."""
        s = text.strip()
        # Replace compound symbols first (sorted by length descending)
        for char in sorted(cls.UNICODE_TO_LATEX.keys(), key=len, reverse=True):
            if char in s:
                s = s.replace(char, cls.UNICODE_TO_LATEX[char])

        # Convert remaining mathematical alphanumeric unicode characters (U+1D400 to U+1D7FF)
        chars = []
        for ch in s:
            code = ord(ch)
            if 0x1D400 <= code <= 0x1D7FF:
                norm = unicodedata.normalize("NFKD", ch)
                chars.append(norm)
            else:
                chars.append(ch)
        s = "".join(chars)

        # Standardize common functions into LaTeX backslash commands
        s = re.sub(r"\bexp\(", r"\\exp(", s)
        s = re.sub(r"\bln\(", r"\\ln(", s)
        s = re.sub(r"\blog\(", r"\\log(", s)
        s = re.sub(r"\bmax_", r"\\max_", s)
        s = re.sub(r"\bmin_", r"\\min_", s)
        s = re.sub(r"\bargmax_", r"\\arg\\max_", s)
        s = re.sub(r"\bargmin_", r"\\arg\\min_", s)
        s = re.sub(r"\barg\s*max", r"\\arg\\max", s)
        s = re.sub(r"\barg\s*min", r"\\arg\\min", s)
        s = re.sub(r"\bVar\(", r"\\text{Var}(", s)
        s = re.sub(r"\bCov\(", r"\\text{Cov}(", s)
        s = re.sub(r"\bBias\(", r"\\text{Bias}(", s)
        s = re.sub(r"\bMSE\(", r"\\text{MSE}(", s)

        # Convert patterns like \mathbf{x}n to \mathbf{x}_n
        s = re.sub(r"(\\mathbf\{[a-zA-Z]\})([a-zA-Z0-9])", r"\1_\2", s)

        # Fix multiple spaces
        s = re.sub(r"[ \t]+", " ", s)
        return s.strip()

    @classmethod
    def is_likely_equation(cls, line: str) -> bool:
        """Evaluate if a given line of text represents a mathematical formula."""
        s = line.strip()
        if len(s) < 5 or len(s) > 280:
            return False

        for excl in cls.EXCLUDED_PATTERNS:
            if excl.search(s):
                return False

        # Must have a relation indicator
        if not cls.RELATION_REGEX.search(s):
            # Check if it has a numbered equation tag at end
            if not cls.EQ_NUM_REGEX.search(s):
                return False

        # Must have mathematical symbols or functional notation
        if not cls.MATH_FEATURE_REGEX.search(s):
            return False

        # Mathematical density check: count symbols vs letters
        letters = len(re.findall(r"[a-zA-Z]", s))
        symbols = len(re.findall(r"[=+\-*/^_{}()\[\]\\~<>≤≥≈∑∫∂∇0-9]", s))

        # A valid equation line should have significant symbol density
        if symbols < 2:
            return False

        if letters > 0 and (symbols / letters) < 0.10 and not cls.EQ_NUM_REGEX.search(s):
            # Too much plain English prose relative to math symbols
            return False

        # Exclude typical natural language sentences that happen to have '='
        lower_s = s.lower()
        if any(lower_s.startswith(w) for w in ["note that", "recall that", "we have", "assume that", "where", "because", "since", "here"]):
            # If the formula is embedded after an introductory phrase, allow if symbol ratio is high
            if symbols < 4 or (symbols / max(letters, 1)) < 0.15:
                return False

        return True

    @classmethod
    def extract_variables(cls, latex: str) -> List[str]:
        """Extract constituent mathematical variables and parameters from LaTeX formula."""
        vars_found: Set[str] = set()

        # Greek symbols
        greek = re.findall(
            r"\\(alpha|beta|gamma|delta|epsilon|varepsilon|zeta|eta|theta|iota|kappa|"
            r"lambda|mu|nu|xi|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega|"
            r"Gamma|Delta|Theta|Lambda|Xi|Pi|Sigma|Upsilon|Phi|Psi|Omega)",
            latex,
        )
        for g in greek:
            vars_found.add(f"\\{g}")

        # Bold vectors/matrices like \mathbf{w}, \boldsymbol{\theta}
        bolds = re.findall(r"\\(?:mathbf|boldsymbol)\{([a-zA-Z0-9]+)\}", latex)
        for b in bolds:
            vars_found.add(f"\\mathbf{{{b}}}")

        # Subscripted variables like x_n, y_i, \theta_j
        subscripts = re.findall(r"([a-zA-Z])_[0-9a-zA-Z]+", latex)
        for sub in subscripts:
            vars_found.add(sub)

        # Single variable identifiers like w, x, y, z, n, k, C, J, H, S
        singles = re.findall(r"(?<![a-zA-Z\\])[a-zA-Z](?![a-zA-Z])", latex)
        stopwords = {"a", "i", "e", "d", "c", "s", "t", "T", "n"}  # Common grammatical or differential indices
        for s in singles:
            if s not in stopwords or s in {"w", "x", "y", "z", "C", "J", "H", "S", "K", "N", "P"}:
                vars_found.add(s)

        # Special functions / operators
        if r"\hat{" in latex:
            hats = re.findall(r"\\hat\{([^}]+)\}", latex)
            for h in hats:
                vars_found.add(f"\\hat{{{h}}}")

        return sorted(list(vars_found))

    @classmethod
    def infer_equation_name(cls, line: str, context: str = "") -> str:
        """Infer a human-readable title for the extracted equation from context."""
        combined = f"{context} {line}".lower()

        # Check for numbered equation first
        num_match = cls.EQ_NUM_REGEX.search(line)
        num_str = f" ({num_match.group(1)})" if num_match else ""

        # Domain concept heuristics
        concept_patterns = [
            (r"\b(cramer[-\s]?rao|crlb)\b", "Cramér-Rao Lower Bound"),
            (r"\b(fisher\s+information)\b", "Fisher Information"),
            (r"\b(mvue|unbiased\s+estimator)\b", "MVUE Definition / Unbiasedness Condition"),
            (r"\b(neyman[-\s]?fisher|factorization)\b", "Neyman-Fisher Factorization Theorem"),
            (r"\b(rao[-\s]?blackwell)\b", "Rao-Blackwell Theorem"),
            (r"\b(kalman\s+(?:filter|gain|state))\b", "Kalman Filter Update"),
            (r"\b(lmmse|linear\s+mmse)\b", "Linear MMSE (LMMSE) Estimator"),
            (r"\b(bayesian\s+posterior|bayes|posterior)\b", "Bayesian Posterior / Inference"),
            (r"\b(support\s+vector|svm|max-margin|maximum\s+margin|margin)\b", "Support Vector Machine Formulation"),
            (r"\b(hinge\s+loss)\b", "Hinge Loss Formulation"),
            (r"\b(kernel\s+trick|gaussian\s+rbf|rbf\s+kernel|polynomial\s+kernel|kernel)\b", "Kernel Function / Representation"),
            (r"\b(least\s+squares|\bols\b)\b", "Least Squares Estimator"),
            (r"\b(maximum\s+likelihood|\bmle\b)\b", "Maximum Likelihood Estimator (MLE)"),
            (r"\b(cross[-\s]?entropy|logistic)\b", "Cross-Entropy / Logistic Objective"),
            (r"\b(gradient\s+descent|\bsgd\b)\b", "Gradient Descent Update Rule"),
            (r"\b(decision\s+tree|entropy|gini)\b", "Decision Tree Impurity / Split Criterion"),
            (r"\b(k[-\s]?means|distortion)\b", "K-Means Distortion Objective"),
            (r"\b(principal\s+component|\bpca\b)\b", "Principal Component Analysis (PCA)"),
            (r"\b(normality|chi[-\s]?square)\b", "Statistical Distribution"),
            (r"\b(gauss[-\s]?markov|\bblue\b)\b", "Gauss-Markov BLUE Estimator"),
            (r"\b(residual|studentized)\b", "Regression Residual Metric"),
        ]

        for pat, name in concept_patterns:
            if re.search(pat, combined):
                return f"{name}{num_str}"

        if num_match:
            return f"Equation ({num_match.group(1)})"

        # Fallback to key variable relation
        clean_snip = re.sub(r"[\\{}]", "", line)[:45].strip()
        return f"Mathematical Relation: {clean_snip}"

    @classmethod
    def infer_conditions(cls, line: str, context: str = "") -> str:
        """Infer mathematical validity conditions or constraints from formula context."""
        conds = []
        if r"\forall" in line or "for all" in context.lower():
            conds.append("Applies for all sample points / indices")
        if "s.t." in line or "subject to" in context.lower():
            conds.append("Subject to optimization constraints")
        if "i.i.d." in context.lower() or "iid" in context.lower():
            conds.append("Assumes i.i.d. random variables")
        if "gaussian" in context.lower() or "normal" in context.lower():
            conds.append("Assumes Gaussian observation noise")
        if "unbiased" in context.lower():
            conds.append("Unbiasedness condition holds")
        return "; ".join(conds) if conds else "Standard mathematical formulation"

    @classmethod
    def extract_from_text(
        cls,
        text: str,
        page: int = 1,
        course: str = "CS771",
        lecture: str = "Lecture 01",
    ) -> List[Equation]:
        """Parse text block and extract all valid Equation objects."""
        if not text:
            return []

        lines = text.split("\n")
        equations: List[Equation] = []
        seen_latex: Set[str] = set()

        for idx, line in enumerate(lines):
            line_str = line.strip()
            if not cls.is_likely_equation(line_str):
                continue

            clean_eq = cls.clean_latex(line_str)
            # Remove any trailing equation number tags from the core LaTeX body
            clean_eq_body = cls.EQ_NUM_REGEX.sub("", clean_eq).strip()

            # Deduplicate similar formulas
            norm_key = re.sub(r"[\s_{}^()\\]", "", clean_eq_body).lower()
            if norm_key in seen_latex or len(norm_key) < 4:
                continue
            seen_latex.add(norm_key)

            # Context window around equation
            ctx_start = max(0, idx - 2)
            ctx_end = min(len(lines), idx + 3)
            surrounding_ctx = " ".join(l.strip() for l in lines[ctx_start:ctx_end] if l.strip())

            eq_name = cls.infer_equation_name(line_str, surrounding_ctx)
            variables = cls.extract_variables(clean_eq_body)
            conditions = cls.infer_conditions(line_str, surrounding_ctx)

            # Construct safe deterministic equation ID
            lec_clean = re.sub(r"[^0-9a-zA-Z]", "", lecture)
            eq_id = f"{course.upper()}_{lec_clean}_eq_auto_p{page}_{len(equations) + 1}"

            topic_name = eq_name.split("(")[0].strip()

            eq_obj = Equation(
                id=eq_id,
                name=eq_name,
                latex=clean_eq_body,
                variables=variables,
                conditions=conditions,
                course=course.upper(),
                lecture=lecture,
                page=page,
                topic=topic_name,
                description=f"Automated extraction from {course} {lecture}, page {page}: {clean_eq_body}",
            )
            equations.append(eq_obj)

        return equations

