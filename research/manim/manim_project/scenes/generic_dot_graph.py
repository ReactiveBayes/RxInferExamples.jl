from __future__ import annotations
import os
from pathlib import Path
from manim import *
from manim_project.utils.graph_io import read_dot_graph, graph_layout_positions
import re


def _normalize_positions(pos: dict[str, tuple[float, float]]):
    xs = [p[0] for p in pos.values()] or [0.0]
    ys = [p[1] for p in pos.values()] or [0.0]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    def norm(v, a, b):
        return (2 * (v - a) / (b - a + 1e-6)) - 0.0
    return {n: (norm(x, minx, maxx), norm(y, miny, maxy)) for n, (x, y) in pos.items()}


class DotGraphScene(MovingCameraScene):
    """Render a graph from DOT file specified via env var DOT_PATH.

    Optional env vars:
      - TITLE: override scene title
      - PNG_PATH: show an image thumbnail on the left
    """
    def construct(self) -> None:
        dot_path = os.environ.get("DOT_PATH")
        if not dot_path:
            raise ValueError("DOT_PATH env var is required")
        dot_path = str(Path(dot_path))
        title_text = os.environ.get("TITLE") or Path(dot_path).parent.name
        # Use Text to avoid LaTeX dependency and escaping issues
        title = Text(title_text).scale(0.6).to_edge(UP)
        self.play(Write(title), run_time=1.5)

        png_path = os.environ.get("PNG_PATH")
        if png_path and Path(png_path).exists():
            img = ImageMobject(str(png_path)).scale(0.8)
            img.to_corner(UL).shift(DOWN * 0.5 + RIGHT * 0.5)
            self.play(FadeIn(img))

        g = read_dot_graph(Path(dot_path))
        pos = _normalize_positions(graph_layout_positions(g))

        def classify_node(name: str, label: str) -> str:
            lname = label.lower()
            if re.search(r"mvnormal|gamma|normal|bernoulli|poisson", lname):
                return "distribution"
            if re.fullmatch(r"u|u_.*|.*_u", label):
                return "control"
            if re.fullmatch(r"x|x_.*|.*_x", label):
                return "state"
            if re.fullmatch(r"s|s_.*|.*_s|y|y_.*|.*_y", label):
                return "observation"
            if re.match(r"theta|sigma|v_|v\w+|cov|precision", lname):
                return "parameter"
            if label.startswith("RxInfer"):
                return "internal"
            return "other"

        styles = {
            "distribution": dict(color=PURPLE_B, fill_opacity=0.2, shape="rectangle"),
            "control": dict(color=GREEN_B, fill_opacity=0.2, shape="circle"),
            "state": dict(color=ORANGE, fill_opacity=0.2, shape="circle"),
            "observation": dict(color=BLUE_B, fill_opacity=0.2, shape="circle"),
            "parameter": dict(color=GRAY, fill_opacity=0.15, shape="rectangle"),
            "internal": dict(color=DARK_GRAY, fill_opacity=0.05, shape="rectangle"),
            "other": dict(color=WHITE, fill_opacity=0.05, shape="circle"),
        }

        node_mobs: dict[str, VGroup] = {}
        for n, (px, py) in pos.items():
            label = str(g.nodes[n].get("label", n))
            kind = classify_node(n, label)
            style = styles[kind]
            center = np.array([px + 2.5, py - 0.5, 0.0])
            if style["shape"] == "rectangle":
                base = RoundedRectangle(corner_radius=0.05, width=0.5, height=0.25, color=style["color"])
            else:
                base = Circle(radius=0.16, color=style["color"])
            base.set_fill(style["color"], opacity=style["fill_opacity"]).move_to(center)
            lbl = Text(label, font_size=16)
            lbl.scale(0.6)
            lbl.next_to(base, UP, buff=0.06)
            node_mobs[n] = VGroup(base, lbl)

        def classify_edge(u: str, v: str) -> tuple[float, any]:
            uk = classify_node(u, str(g.nodes[u].get("label", u)))
            vk = classify_node(v, str(g.nodes[v].get("label", v)))
            # Likelihood edge to observation
            if vk == "observation":
                return 2.4, YELLOW_B
            # State transition
            if uk == "state" and vk == "state":
                return 2.0, ORANGE
            # Parameter influence
            if uk == "parameter":
                return 1.2, GRAY
            # Control to state/observation
            if uk == "control":
                return 1.8, GREEN_B
            # Distribution generating something
            if uk == "distribution":
                return 1.6, PURPLE_B
            return 1.4, WHITE

        edges = VGroup()
        for u, v in g.edges():
            if u in node_mobs and v in node_mobs:
                a = node_mobs[u][0].get_center()
                b = node_mobs[v][0].get_center()
                sw, col = classify_edge(u, v)
                edges.add(Arrow(a, b, stroke_width=sw, color=col, max_tip_length_to_length_ratio=0.05))

        graph_group = VGroup(edges, *node_mobs.values())
        self.play(LaggedStart(*[FadeIn(m) for m in graph_group], lag_ratio=0.02), run_time=3)
        self.wait(1)
        self.play(Circumscribe(graph_group, color=YELLOW), run_time=2)
        # Legend
        legend_items = []
        for key, text in [
            ("state", "State"),
            ("control", "Control"),
            ("observation", "Observation"),
            ("distribution", "Distribution"),
            ("parameter", "Parameter"),
        ]:
            style = styles[key]
            marker = Square(side_length=0.18, color=style["color"]).set_fill(style["color"], opacity=style["fill_opacity"]) if style["shape"]=="rectangle" else Dot(radius=0.09, color=style["color"]).set_fill(style["color"]) \
            
            label = Text(text, font_size=20)
            legend_items.append(VGroup(marker, label).arrange(RIGHT, buff=0.15))
        legend = VGroup(*legend_items).arrange(DOWN, buff=0.1)
        legend.to_corner(UR).shift(LEFT * 0.5 + DOWN * 0.5)
        self.play(FadeIn(legend), run_time=1)
        # Stats panel and caption
        kind_counts: dict[str, int] = {k: 0 for k in styles.keys()}
        for n in g.nodes:
            label = str(g.nodes[n].get("label", n))
            kind_counts[classify_node(n, label)] += 1
        stats_lines = [
            f"Nodes: {g.number_of_nodes()}  Edges: {g.number_of_edges()}",
            f"States: {kind_counts['state']}  Obs: {kind_counts['observation']}  Ctrls: {kind_counts['control']}",
            f"Dists: {kind_counts['distribution']}  Params: {kind_counts['parameter']}",
        ]
        stats = VGroup(*[Text(line, font_size=18) for line in stats_lines]).arrange(DOWN, buff=0.05)
        stats.to_corner(UL).shift(RIGHT * 0.3 + DOWN * 0.3)
        self.play(FadeIn(stats), run_time=1)
        caption = Text("Model graph", font_size=20).to_edge(DOWN)
        self.play(Write(caption), run_time=1)
        # Slow camera pan to showcase the graph
        frame = self.camera.frame
        self.play(frame.animate.shift(RIGHT * 0.5), run_time=2)
        self.play(frame.animate.shift(LEFT * 1.0), run_time=2)
        self.play(frame.animate.shift(RIGHT * 0.5), run_time=2)
        self.wait(1)
