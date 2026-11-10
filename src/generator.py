"""Local LLM Generation module using Ollama and Llama 3.1 8B.

Connects to the local Ollama instance (http://localhost:11434) to provide
academically grounded, citation-backed answers using retrieved lecture evidence
and discrete LaTeX mathematical equations.
"""

import json
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Union

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.1:8b"

SYSTEM_PROMPT = r"""You are ZombAI, an authoritative, rigorous Teaching Assistant for academic lecture courses at IIT Kanpur (including MTH441: Regression Analysis and ANOVA taught by Prof. Shalabh, CS771: Introduction to Machine Learning taught by Prof. Piyush Rai, and EE602: Statistical Signal Processing and Estimation Theory based on Steven M. Kay's textbook).

Your task is to provide clear, comprehensive, and mathematically accurate answers to the student's question based strictly on the provided lecture/textbook context and discrete equations.

Grounding & Formatting Rules:
1. Grounding: Answer strictly using the provided lecture/textbook context and equations. If the question cannot be answered from the provided notes, explicitly state: "This concept is not covered in the provided lecture notes."
2. Citations: Every key claim or definition must include an exact citation pointing to the course, lecture/chapter, and page number, formatted like [MTH441 Lecture 01, Page 14], [CS771 Lecture 10, Page 11], or [EE602 Chapter 03, Page 39].
3. Mathematics: Render all mathematical expressions using clean LaTeX:
   - Inline formulas: $y = X\beta + \varepsilon$
   - Display equations: $$f(x) = \frac{1}{\Gamma(k/2) 2^{k/2}} x^{k/2 - 1} e^{-x/2}$$
4. Equation Reference: When discrete equations are provided in the context, explicitly reference their Equation ID and explain their parameters and validity conditions.
5. Tone: Professional, structured, and pedagogical.
6. Non-Repetition: Provide your response exactly once without repeating sections, paragraphs, or disclaimers.
"""


class LLMGenerator:
    """Manages inference requests to local Ollama Llama 3.1 8B instance."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_OLLAMA_URL,
        timeout: int = 60,
        temperature: float = 0.2,
    ):
        """Initialize the Ollama client.

        Args:
            model_name: Model identifier in Ollama (default: 'llama3.1:8b').
            base_url: Ollama API URL (default: 'http://localhost:11434').
            timeout: Request timeout in seconds.
            temperature: Sampling temperature (0.0 - 1.0).
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self._verify_connection()

    def _verify_connection(self) -> bool:
        """Check if local Ollama daemon is reachable and model is installed."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as res:
                if res.status == 200:
                    data = json.loads(res.read().decode("utf-8"))
                    models = [m.get("name", "") for m in data.get("models", [])]
                    # Check matching model
                    has_model = any(self.model_name in m for m in models)
                    if not has_model:
                        print(
                            f"[Warning] Model '{self.model_name}' not listed in Ollama. Available: {models}"
                        )
                    return True
        except Exception as e:
            print(f"[Warning] Ollama server check failed at {self.base_url}: {e}")
            return False
        return False

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        stream: bool = False,
    ) -> str:
        """Send a prompt to the Ollama model and return the generated text.

        Args:
            prompt: User prompt string.
            system: Optional system instructions override.
            stream: Whether to stream the response (default False for batch return).

        Returns:
            Generated text string.
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "system": system or SYSTEM_PROMPT,
            "stream": stream,
            "options": {
                "temperature": self.temperature,
                "repeat_penalty": 1.15,
                "repeat_last_n": 128,
                "top_p": 0.9,
                "top_k": 40,
                "num_predict": 1024,
                "stop": [
                    "<|eot_id|>",
                    "<|start_header_id|>",
                    "<|end_header_id|>",
                    "<|end_of_text|>",
                    "=== RETRIEVED LECTURE CONTEXT ===",
                    "=== STUDENT QUESTION ===",
                ],
            },
        }

        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=data_bytes,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                raw_text = result.get("response", "").strip()
                return self.deduplicate_text(raw_text)
        except urllib.error.URLError as err:
            raise ConnectionError(
                f"Failed to connect to Ollama server at {self.base_url}: {err}"
            )
        except Exception as ex:
            raise RuntimeError(f"Ollama generation failed: {ex}")

    @staticmethod
    def deduplicate_text(text: str) -> str:
        """Detect and remove repetitive loops or duplicate paragraphs from LLM output.

        Args:
            text: Raw generated string.

        Returns:
            Clean string with degenerate repetition loops removed.
        """
        if not text:
            return text

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if len(paragraphs) <= 1:
            return text

        unique_paragraphs = []
        seen = set()

        for p in paragraphs:
            norm = " ".join(p.lower().split())
            if norm in seen:
                continue
            if unique_paragraphs and norm == " ".join(unique_paragraphs[-1].lower().split()):
                continue
            seen.add(norm)
            unique_paragraphs.append(p)

        deduped = "\n\n".join(unique_paragraphs)

        # Detect cyclical tail repetitions across lines
        lines = [line.strip() for line in deduped.split("\n") if line.strip()]
        n = len(lines)
        for cycle_len in range(1, n // 2 + 1):
            if lines[-cycle_len:] == lines[-2 * cycle_len : -cycle_len]:
                deduped = "\n".join(lines[:-cycle_len])
                break

        return deduped

    def generate_rag_answer(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        discrete_equations: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Generate a fully grounded RAG response with citations and equations.

        Args:
            query: The student's question.
            context_chunks: Top-k retrieved text chunk dictionaries.
            discrete_equations: Optional list of discrete equation dicts from the graph.

        Returns:
            Dict containing 'answer', 'citations', 'referenced_equations', and 'prompt'.
        """
        # Format textual evidence blocks
        evidence_blocks = []
        citations_found = set()
        for i, chunk in enumerate(context_chunks, 1):
            meta = chunk.get("metadata", {})
            src = meta.get("source", "lecture.pdf")
            crs = meta.get("course", "")
            lec = meta.get("lecture", "Lecture")
            page = meta.get("page", "?")
            c_num = meta.get("chunk", "?")

            lec_tag = f"{crs} {lec}" if crs else lec
            cite_label = f"[{lec_tag}, Page {page}]"
            citations_found.add(cite_label)

            block = (
                f"Evidence #{i} {cite_label} (Chunk {c_num}):\n"
                f"{chunk.get('text', '').strip()}"
            )
            evidence_blocks.append(block)

        formatted_context = (
            "\n\n".join(evidence_blocks)
            if evidence_blocks
            else "No matching lecture text found."
        )

        # Format discrete equations if present
        eq_blocks = []
        eq_ids = []
        if discrete_equations:
            for eq in discrete_equations:
                eq_id = eq.get("id", "EQ_UNKNOWN")
                eq_ids.append(eq_id)
                eq_name = eq.get("name", "Equation")
                latex = eq.get("latex", "")
                conds = eq.get("conditions", "")
                vars_list = ", ".join(eq.get("variables", []))

                eq_text = (
                    f"Equation ID: {eq_id} ({eq_name})\n"
                    f"LaTeX: {latex}\n"
                    f"Variables: {vars_list}"
                )
                if conds:
                    eq_text += f"\nConditions: {conds}"
                eq_blocks.append(eq_text)

        formatted_equations = (
            "\n\n".join(eq_blocks)
            if eq_blocks
            else "No discrete equations linked to this context."
        )

        # Assemble prompt
        user_prompt = (
            f"=== RETRIEVED LECTURE CONTEXT ===\n"
            f"{formatted_context}\n\n"
            f"=== VERIFIED DISCRETE EQUATIONS ===\n"
            f"{formatted_equations}\n\n"
            f"=== STUDENT QUESTION ===\n"
            f"{query}\n\n"
            f"Provide a structured, step-by-step academic response with exact citations:"
        )

        answer = self.generate(prompt=user_prompt)

        return {
            "query": query,
            "answer": answer,
            "citations": sorted(list(citations_found)),
            "referenced_equations": eq_ids,
            "raw_context": formatted_context,
            "model": self.model_name,
        }

