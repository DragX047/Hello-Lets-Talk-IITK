"""Hierarchical Knowledge Graph storage module for academic courses.

Implements CourseGraph using NetworkX to organize course entities into a rich
heterogeneous graph: Course -> Lecture -> Topic -> Chunk <-> Equation.
Provides graph serialization (JSON), edge traversal, and localized subgraph extraction.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import networkx as nx


class CourseGraph:
    """Manages the heterogeneous hierarchical knowledge graph for a course."""

    def __init__(self, course_id: str = "MTH441"):
        """Initialize an empty directed knowledge graph.

        Args:
            course_id: Identifier for the primary course (default: 'MTH441').
        """
        self.course_id = course_id
        self.graph = nx.DiGraph()

    # -------------------------------------------------------------------------
    # Node Creation Methods
    # -------------------------------------------------------------------------

    def add_course(
        self,
        course_id: str,
        name: str,
        instructor: str = "Prof. Shalabh",
        institution: str = "IIT Kanpur",
        description: str = "",
    ) -> str:
        """Add or update a Course root node."""
        node_id = f"COURSE_{course_id}"
        self.graph.add_node(
            node_id,
            node_type="course",
            id=node_id,
            course_id=course_id,
            name=name,
            instructor=instructor,
            institution=institution,
            description=description,
        )
        return node_id

    def add_lecture(
        self,
        lecture_id: str,
        course_id: str,
        lecture_num: int,
        title: str,
        source_file: str,
        chapter: Optional[int] = None,
        description: str = "",
    ) -> str:
        """Add a Lecture node and link it from its parent Course."""
        course_node = f"COURSE_{course_id}"
        if course_node not in self.graph:
            self.add_course(course_id, name=f"Course {course_id}")

        self.graph.add_node(
            lecture_id,
            node_type="lecture",
            id=lecture_id,
            course_id=course_id,
            lecture_num=lecture_num,
            title=title,
            source_file=source_file,
            chapter=chapter if chapter is not None else lecture_num,
            description=description,
        )

        self.graph.add_edge(course_node, lecture_id, relation="CONTAINS_LECTURE")
        return lecture_id

    def add_topic(
        self,
        topic_id: str,
        lecture_id: str,
        title: str,
        description: str = "",
    ) -> str:
        """Add a Topic node and link it from its parent Lecture."""
        self.graph.add_node(
            topic_id,
            node_type="topic",
            id=topic_id,
            lecture_id=lecture_id,
            title=title,
            description=description,
        )

        if lecture_id in self.graph:
            self.graph.add_edge(lecture_id, topic_id, relation="COVERS_TOPIC")
        return topic_id

    def add_chunk(
        self,
        chunk_id: str,
        lecture_id: str,
        page: int,
        chunk_num: int,
        text: str,
        source: str = "",
        topic_id: Optional[str] = None,
    ) -> str:
        """Add a Text Chunk node and connect it to its Topic and Lecture."""
        self.graph.add_node(
            chunk_id,
            node_type="chunk",
            id=chunk_id,
            lecture_id=lecture_id,
            page=page,
            chunk_num=chunk_num,
            text=text,
            source=source,
            topic_id=topic_id or "",
            char_count=len(text),
        )

        # Link to Topic if provided, otherwise link directly to Lecture
        if topic_id and topic_id in self.graph:
            self.graph.add_edge(topic_id, chunk_id, relation="HAS_CHUNK")
        elif lecture_id in self.graph:
            self.graph.add_edge(lecture_id, chunk_id, relation="HAS_CHUNK")

        return chunk_id

    def add_equation(
        self,
        equation_id: str,
        name: str,
        latex: str,
        lecture_id: str,
        page: int,
        variables: Optional[List[str]] = None,
        conditions: str = "",
        topic: str = "",
        description: str = "",
        topic_id: Optional[str] = None,
        course: Optional[str] = None,
        lecture: Optional[str] = None,
    ) -> str:
        """Add a discrete Equation node and optionally link to a Topic."""
        crs = course or self.course_id
        if not lecture:
            if "_CH" in lecture_id:
                lecture = f"Chapter {lecture_id.split('_CH')[-1]}"
            elif "_L" in lecture_id:
                lecture = f"Lecture {lecture_id.split('_L')[-1]}"
            elif "_APP" in lecture_id:
                lecture = f"Appendix {lecture_id.split('_APP_')[-1]}"
            else:
                lecture = "Lecture"

        self.graph.add_node(
            equation_id,
            node_type="equation",
            id=equation_id,
            name=name,
            latex=latex,
            variables=variables or [],
            conditions=conditions,
            lecture_id=lecture_id,
            page=page,
            topic=topic,
            description=description,
            course=crs,
            lecture=lecture,
        )

        if topic_id and topic_id in self.graph:
            self.graph.add_edge(topic_id, equation_id, relation="HAS_EQUATION")

        return equation_id

    # -------------------------------------------------------------------------
    # Edge Creation Methods
    # -------------------------------------------------------------------------

    def link_equation_to_chunk(
        self,
        chunk_id: str,
        equation_id: str,
        relation: str = "REFERENCES_EQUATION",
    ) -> bool:
        """Create a directed edge from a Chunk to an Equation."""
        if chunk_id in self.graph and equation_id in self.graph:
            self.graph.add_edge(chunk_id, equation_id, relation=relation)
            return True
        return False

    def link_sequential_chunks(self, chunk_ids: List[str]) -> int:
        """Link consecutive chunks in reading order via NEXT_CHUNK edges."""
        links = 0
        for i in range(len(chunk_ids) - 1):
            c_curr = chunk_ids[i]
            c_next = chunk_ids[i + 1]
            if c_curr in self.graph and c_next in self.graph:
                self.graph.add_edge(c_curr, c_next, relation="NEXT_CHUNK")
                links += 1
        return links

    # -------------------------------------------------------------------------
    # Traversal & Lookup Methods
    # -------------------------------------------------------------------------

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Return node attributes dictionary or None if not found."""
        if node_id in self.graph:
            d = dict(self.graph.nodes[node_id])
            d["id"] = node_id
            return d
        return None

    def get_equations_for_chunk(self, chunk_id: str) -> List[Dict[str, Any]]:
        """Find all discrete equations linked from a given chunk."""
        if chunk_id not in self.graph:
            return []

        equations = []
        for successor in self.graph.successors(chunk_id):
            edge_data = self.graph.get_edge_data(chunk_id, successor)
            if edge_data.get("relation") == "REFERENCES_EQUATION":
                eq_dict = dict(self.graph.nodes[successor])
                eq_dict["id"] = successor
                if "course" not in eq_dict or not eq_dict["course"]:
                    eq_dict["course"] = self.course_id
                if "lecture" not in eq_dict or not eq_dict["lecture"]:
                    lec_id = str(eq_dict.get("lecture_id", ""))
                    if "_L" in lec_id:
                        eq_dict["lecture"] = f"Lecture {lec_id.split('_L')[-1]}"
                equations.append(eq_dict)
        return equations

    def get_adjacent_chunks(
        self,
        chunk_id: str,
        include_previous: bool = True,
        include_next: bool = True,
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """Find preceding and succeeding chunks via NEXT_CHUNK edges."""
        result: Dict[str, Optional[Dict[str, Any]]] = {"previous": None, "next": None}

        if chunk_id not in self.graph:
            return result

        # Check succeeding chunks
        if include_next:
            for succ in self.graph.successors(chunk_id):
                if self.graph.get_edge_data(chunk_id, succ).get("relation") == "NEXT_CHUNK":
                    d = dict(self.graph.nodes[succ])
                    d["id"] = succ
                    result["next"] = d
                    break

        # Check preceding chunks
        if include_previous:
            for pred in self.graph.predecessors(chunk_id):
                if self.graph.get_edge_data(pred, chunk_id).get("relation") == "NEXT_CHUNK":
                    d = dict(self.graph.nodes[pred])
                    d["id"] = pred
                    result["previous"] = d
                    break

        return result

    def get_topics_for_lecture(self, lecture_id: str) -> List[Dict[str, Any]]:
        """Return all topics covered in a given lecture."""
        if lecture_id not in self.graph:
            return []

        topics = []
        for succ in self.graph.successors(lecture_id):
            if self.graph.get_edge_data(lecture_id, succ).get("relation") == "COVERS_TOPIC":
                d = dict(self.graph.nodes[succ])
                d["id"] = succ
                topics.append(d)
        return topics

    def get_equations_for_page(self, lecture_id: str, page: int) -> List[Dict[str, Any]]:
        """Find all equations occurring on a specific page of a lecture."""
        matched = []
        for node_id, data in self.graph.nodes(data=True):
            if (
                data.get("node_type") == "equation"
                and data.get("lecture_id") == lecture_id
                and data.get("page") == page
            ):
                d = dict(data)
                d["id"] = node_id
                matched.append(d)
        return matched

    def get_subgraph(self, node_id: str, radius: int = 1) -> Dict[str, Any]:
        """Extract an ego subgraph centered on node_id with the given radius."""
        if node_id not in self.graph:
            return {"nodes": [], "edges": []}

        # Ego graph in undirected view for neighborhood extraction
        undirected_view = self.graph.to_undirected()
        sub_nodes = set(nx.ego_graph(undirected_view, node_id, radius=radius).nodes())

        nodes_data = [dict(self.graph.nodes[n]) for n in sub_nodes]
        edges_data = []

        for u, v, data in self.graph.edges(data=True):
            if u in sub_nodes and v in sub_nodes:
                edges_data.append({"source": u, "target": v, **data})

        return {"nodes": nodes_data, "edges": edges_data}

    # -------------------------------------------------------------------------
    # Serialization & Statistics
    # -------------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return summary counts of nodes and edge relations."""
        node_counts: Dict[str, int] = {}
        for _, data in self.graph.nodes(data=True):
            ntype = data.get("node_type", "unknown")
            node_counts[ntype] = node_counts.get(ntype, 0) + 1

        edge_counts: Dict[str, int] = {}
        for _, _, data in self.graph.edges(data=True):
            rel = data.get("relation", "unknown")
            edge_counts[rel] = edge_counts.get(rel, 0) + 1

        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_types": node_counts,
            "edge_relations": edge_counts,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Export graph to a portable JSON-serializable dictionary."""
        nodes = []
        for n_id, data in self.graph.nodes(data=True):
            node_item = dict(data)
            node_item["id"] = n_id
            nodes.append(node_item)

        edges = []
        for u, v, data in self.graph.edges(data=True):
            edge_item = dict(data)
            edge_item["source"] = u
            edge_item["target"] = v
            edges.append(edge_item)

        return {
            "course_id": self.course_id,
            "summary": self.summary(),
            "nodes": nodes,
            "edges": edges,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CourseGraph":
        """Reconstruct CourseGraph from an exported dictionary."""
        cg = cls(course_id=data.get("course_id", "MTH441"))
        for n in data.get("nodes", []):
            n_id = n["id"]
            attrs = {k: v for k, v in n.items() if k != "id"}
            attrs["id"] = n_id
            cg.graph.add_node(n_id, **attrs)

        for e in data.get("edges", []):
            src = e["source"]
            tgt = e["target"]
            attrs = {k: v for k, v in e.items() if k not in ("source", "target")}
            cg.graph.add_edge(src, tgt, **attrs)

        return cg

    def save_json(self, filepath: str = "storage/graph/MTH441_graph.json") -> str:
        """Persist graph structure to a JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return str(path.resolve())

    @classmethod
    def load_json(cls, filepath: str = "storage/graph/MTH441_graph.json") -> "CourseGraph":
        """Load and reconstruct CourseGraph from a JSON file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Knowledge graph file not found at: {filepath}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_cytoscape_elements(self) -> List[Dict[str, Any]]:
        """Export nodes and edges in standard Cytoscape.js format for visual rendering."""
        elements = []
        for n_id, data in self.graph.nodes(data=True):
            label = (
                data.get("name")
                or data.get("title")
                or (f"Chunk p{data.get('page')}" if data.get("node_type") == "chunk" else n_id)
            )
            elements.append({
                "data": {
                    "id": n_id,
                    "label": label,
                    "type": data.get("node_type", "default"),
                    **{k: v for k, v in data.items() if k not in ("text", "latex")},
                }
            })

        for u, v, data in self.graph.edges(data=True):
            elements.append({
                "data": {
                    "source": u,
                    "target": v,
                    "relation": data.get("relation", "CONNECTED_TO"),
                }
            })
        return elements
