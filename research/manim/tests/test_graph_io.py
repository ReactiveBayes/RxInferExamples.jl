from pathlib import Path
from manim_project.configs.paths import MOUNTAIN_CAR_DOT
from manim_project.utils.graph_io import read_dot_graph, export_graph_metadata


def test_read_dot_graph_parses_nodes():
    g = read_dot_graph(Path(MOUNTAIN_CAR_DOT))
    meta = export_graph_metadata(g)
    assert meta["num_nodes"] > 0
    assert meta["num_edges"] > 0
