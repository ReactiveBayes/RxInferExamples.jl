from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import networkx as nx

try:
    import pydot
except Exception:  # pragma: no cover
    pydot = None

@dataclass
class GraphAssets:
    dot_path: Path
    mmd_path: Path


def read_dot_graph(dot_path: Path) -> nx.DiGraph:
    if pydot is None:
        raise RuntimeError("pydot is required to parse DOT files")
    graphs = pydot.graph_from_dot_file(str(dot_path))
    if not graphs:
        raise ValueError(f"No graph parsed from {dot_path}")
    return nx.nx_pydot.from_pydot(graphs[0])


def graph_layout_positions(g: nx.DiGraph) -> dict[str, tuple[float, float]]:
    # Use graphviz-like hierarchy if available; fallback to spring layout
    try:
        return nx.nx_agraph.graphviz_layout(g, prog="dot")  # type: ignore
    except Exception:
        return nx.spring_layout(g, seed=42)


def export_graph_metadata(g: nx.DiGraph) -> dict:
    return {
        "num_nodes": g.number_of_nodes(),
        "num_edges": g.number_of_edges(),
        "nodes": list(map(str, g.nodes())),
    }


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
