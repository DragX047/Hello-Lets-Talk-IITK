"""Text cleaning module for lecture notes.

Removes recurring slide footers, headers, and trailing page-number artefacts.
Reconstructs fragmented mathematical equations (fractions, exponents, densities,
and summations) that get scrambled by PDF text extractors.
Normalizes spacing and line formatting so text reads naturally.
"""

import re
from typing import Any, Dict, List

try:
    from .pdf_utils import normalize_symbols, normalize_whitespace
except ImportError:
    from pdf_utils import normalize_symbols, normalize_whitespace

# Regex to match recurring slide footers across chapters:
# e.g., "Regression Analysis and ANOVA | Chapter 1 | Introduction | Shalabh, IIT Kanpur"
FOOTER_PATTERN = re.compile(
    r"Regression Analysis and ANOVA\s*\|\s*Chapter\s*\d+\s*\|\s*[^|\n]+\|\s*Shalabh,\s*IIT Kanpur(?:\s*\d+)*",
    re.IGNORECASE,
)

# Regex to match recurring CS771 header and slide footer patterns:
CS771_HEADER_PATTERN = re.compile(
    r"(?:CS771:\s*Introduction to Machine Learning\s*\([^\)]*\)|Intro to Machine Learning \(CS771A\)\s*Maths Refresher(?:\s*\d+)*)",
    re.IGNORECASE,
)


def reconstruct_math_equations(text: str) -> str:
    """Repair fragmented slide math equations, fractions, and density functions."""
    if not text:
        return ""

    # 1. Collapse duplicate Greek characters from PPT bold/shadow rendering: "μμ" -> "μ", "ββ" -> "β"
    text = re.sub(r"([μβλσεχρΓΣ])\1+", r"\1", text)

    # 2. Repair vector indices: "(μ1 , 2 ,..., μn )" -> "(μ1, μ2, ..., μn)"
    text = re.sub(r"μ\s*1\s*,\s*2\s*,\s*\.\.\.\s*,\s*μ\s*([nk])", r"μ1, μ2, ..., μ\1", text)
    text = re.sub(r"β\s*1\s*,\s*2\s*,\s*\.\.\.\s*,\s*β\s*([nk])", r"β1, β2, ..., β\1", text)
    text = re.sub(r"Y\s*1\s*,\s*2\s*,\s*\.\.\.\s*,\s*Y\s*([nk])", r"Y1, Y2, ..., Y\1", text)
    text = re.sub(r"X\s*1\s*,\s*2\s*,\s*\.\.\.\s*,\s*X\s*([nk])", r"X1, X2, ..., X\1", text)

    # 3. Repair t-distribution statistic fraction:
    text = re.sub(
        r"X\s+then the distribution of the statistic T\s*=\s*is called the t-distribution with n degrees of freedom\.\s*Y\s*/\s*n",
        "then the distribution of the statistic T = X / √(Y / n) is called the t-distribution with n degrees of freedom.",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(?:X\s*\n+)?•\s*If the mean of X is nonzero then the distribution of\s+(?:is called\s+)?(?:the\s+)?noncentral t-?\s*distribution\s*Y\s*/\s*n\s+with",
        "• If the mean of X is nonzero then the distribution of T = X / √(Y / n) is called the noncentral t-distribution with",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"the distribution of\s+X\s*\n*\s*Y\s*/\s*n\s+is called the noncentral t-?\s*distribution",
        "the distribution of T = X / √(Y / n) is called the noncentral t-distribution",
        text,
        flags=re.IGNORECASE,
    )

    # 4. Repair F-distribution statistic fractions:
    text = re.sub(
        r"X\s*/\s*m\s+respectively,\s*then the distribution of the statistic F\s*=\s*is called the F-distribution with m and\s*Y\s*/\s*n",
        "then the distribution of the statistic F = (X / m) / (Y / n) is called the F-distribution with m and",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"X\s*/\s*m\s+variables,\s*then the distribution of F\s*=\s*is the noncentral F distribution with m and n degrees\s*Y\s*/\s*n\s+of freedom",
        "variables, then the distribution of F = (X / m) / (Y / n) is the noncentral F distribution with m and n degrees of freedom",
        text,
        flags=re.IGNORECASE,
    )

    # 5. Reconstruct scrambled probability density functions:
    # F-distribution density
    text = re.sub(
        r"The probability density function of F is\s+m\s*\+\s*n.*?0\s*<\s*f\s*<\s*∞.*?(?=\n\n|•|$)",
        "The probability density function of F is:\n  f_F(f) = [Γ((m+n)/2) (m/n)^(m/2) / (Γ(m/2) Γ(n/2))] * f^(m/2 - 1) * (1 + (m/n)f)^(-(m+n)/2);  0 < f < ∞\n\n",
        text,
        flags=re.DOTALL,
    )
    # t-distribution density
    text = re.sub(
        r"The probability density function of t is.*?-\s*∞\s*<\s*<\s*∞\s*t.*?(?=\n\n|•|$)",
        "The probability density function of t is:\n  f_T(t) = [Γ((n+1)/2) / (√(nπ) Γ(n/2))] * (1 + t^2/n)^(-(n+1)/2);  -∞ < t < ∞\n\n",
        text,
        flags=re.DOTALL,
    )
    # Chi-square distribution density
    text = re.sub(
        r"The probability density function of χ[^•\n]+with k degrees of freedom is given as.*?(?=•|$)",
        "The probability density function of χ^2-distribution with k degrees of freedom is given as:\n  f(x) = [1 / (Γ(k/2) 2^(k/2))] * x^(k/2 - 1) * exp(-x/2);  0 < x < ∞\n\n",
        text,
        flags=re.DOTALL,
    )
    # Multivariate Normal density
    text = re.sub(
        r"(?:1\s+)?1\s+[−-]\s*1\s+f\s*\(\s*Y\s*\|\s*μ,\s*Σ\s*=\s*\)\s*exp.*?(?=\n\n|assuming|•|$)",
        "f(Y | μ, Σ) = [1 / ((2π)^(n/2) |Σ|^(1/2))] * exp[-(1/2) (Y - μ)' Σ^(-1) (Y - μ)]\n\n",
        text,
        flags=re.DOTALL,
    )

    # 6. Reconstruct summation and distribution relations:
    # Chi-square from standard normal sum of squares
    text = re.sub(
        r"(?:k\s+)?common mean 0 and common variance 1.*?distribution with k degrees of freedom\.",
        "common mean 0 and common variance 1, then the distribution of ∑_{i=1}^k Yi^2 is called the χ^2-distribution with k degrees of freedom.",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"2\s+1\s+k\s+2\s+2\s+and common variance σ,\s*then\s+2\s+Yi\s+has χ\s*distribution with k degrees of freedom\.\s*σ\s*=\s*∑i\s*1",
        "and common variance σ^2, then (1/σ^2) ∑_{i=1}^k Yi^2 has χ^2-distribution with k degrees of freedom.",
        text,
    )
    # Noncentral chi-square from normal with non-null means
    text = re.sub(
        r"(?:k\s+)?common variance 1.*?freedom and noncentrality parameter\s*λ\s*μi2\s*\.\s*=\s*∑\s*i\s*=\s*1",
        "common variance 1, then the distribution of ∑_{i=1}^k Yi^2 has noncentral χ^2-distribution with k degrees of freedom and noncentrality parameter λ = ∑_{i=1}^k μi^2.",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"(?:2\s+1\s+k\s+2\s+2\s+)?means\s+μ1,\s*μ2,\s*\.\.\.,\s*μk\s+but common variance σ.*?degrees of freedom and noncentrality parameter\s*λ\s*=\s*2\s*∑\s*σ\s*i\s*=\s*1",
        "means μ1, μ2, ..., μk but common variance σ^2, then (1/σ^2) ∑_{i=1}^k Yi^2 has noncentral χ^2-distribution with k degrees of freedom and noncentrality parameter λ = (1/σ^2) ∑_{i=1}^k μi^2.",
        text,
        flags=re.DOTALL,
    )
    # Additive property of independent noncentral chi-squares
    text = re.sub(
        r"(?:k\s+k\s+)?then\s+U\s*i\s+has noncentral Chi-square distribution with\s+ni\s+degrees of freedom and noncentrality.*?(?=•|$)",
        "then ∑_{i=1}^k Ui has noncentral Chi-square distribution with ∑_{i=1}^k ni degrees of freedom and noncentrality parameter ∑_{i=1}^k λi.\n\n",
        text,
        flags=re.DOTALL,
    )

    # 7. Linear function of random variables (Page 9):
    text = re.sub(
        r"n\s+If Y1\s*,\s*Y2\s*,\s*\.\.\.\s*,\s*Yn are n random variables.*?then k i Yi is called a linear function of =∑i 1",
        "If Y1, Y2, ..., Yn are n random variables, and k1, k2, ..., kn are scalars, then ∑_{i=1}^n ki Yi is called a linear function of",
        text,
    )
    text = re.sub(r"n\s*kYi\s*i\s*\.If\s*Y\s*=", "If Y =", text)
    text = re.sub(r"K\s*'\s*Y\s*=\s*∑\s*i\s*=\s*1\s*n\s*ki\s*E\s*\(\s*Yi\s*\)", "K' Y = ∑_{i=1}^n ki Yi", text)

    # 8. Matrix independence theorems:
    text = re.sub(r"B\s*Σ\s*=\s*A\s*0", "B Σ A = 0", text)
    text = re.sub(r"A\s+A\s+Σ\s*=\s*0", "A1 Σ A2 = 0", text)

    # 9. Normalize chi-square symbol:
    text = re.sub(r"χ-?2\b", "χ^2", text)
    text = re.sub(r"χ-?distribution2?\b", "χ^2-distribution", text)

    return text


def clean_page_text(text: str) -> str:
    """Clean extracted page text by stripping footers, normalising symbols, and polishing layout.

    Args:
        text: Raw text string from a PDF page.

    Returns:
        Cleaned body text with polished spacing, readable formulas, and clean lines.
    """
    if not text:
        return ""

    # 1. Normalize math and Greek symbols
    cleaned = normalize_symbols(text)

    # 2. Strip recurring slide footers and trailing page numbers
    cleaned = FOOTER_PATTERN.sub("", cleaned)
    cleaned = CS771_HEADER_PATTERN.sub("", cleaned)

    # 3. Reconstruct fragmented slide math formulas
    cleaned = reconstruct_math_equations(cleaned)

    # 4. Polish whitespace, bullet indentation, and line wrapping
    cleaned = normalize_whitespace(cleaned)

    return cleaned.strip()


def clean_pages(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply cleaning across a list of page dictionaries while preserving metadata.

    Args:
        pages: List of dictionaries with 'text' and 'metadata'.

    Returns:
        List of dictionaries with cleaned 'text' and identical 'metadata'.
    """
    cleaned_pages: List[Dict[str, Any]] = []

    for page in pages:
        cleaned_text = clean_page_text(page["text"])
        cleaned_pages.append(
            {
                "text": cleaned_text,
                "metadata": dict(page["metadata"]),
            }
        )

    return cleaned_pages
