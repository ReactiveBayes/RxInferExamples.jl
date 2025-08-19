from __future__ import annotations
from manim import *
from pathlib import Path

from manim_project.configs.paths import (
    MOUNTAIN_CAR_PNG,
    MOUNTAIN_CAR_DOT,
)
from manim_project.utils.graph_io import read_dot_graph, graph_layout_positions


class MountainCarGraph(Scene):
    def construct(self) -> None:
        title = Tex("Mountain Car Factor Graph").scale(0.8).to_edge(UP)
        self.play(Write(title))

        # Left: image snapshot
        if Path(MOUNTAIN_CAR_PNG).exists():
            img = ImageMobject(str(MOUNTAIN_CAR_PNG)).scale(0.8)
            img.to_corner(UL).shift(DOWN * 0.5 + RIGHT * 0.5)
            self.play(FadeIn(img))

        # Right: parsed DOT graph drawn using small circles and labels
        g = read_dot_graph(Path(MOUNTAIN_CAR_DOT))
        pos = graph_layout_positions(g)

        # Normalize positions
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        def norm(v, a, b):
            return (2 * (v - a) / (b - a + 1e-6)) - 0.0

        node_mobs = {}
        for n, (x, y) in pos.items():
            px = norm(x, minx, maxx)
            py = norm(y, miny, maxy)
            dot = Dot(point=np.array([px + 2.5, py - 0.5, 0]), radius=0.02, color=WHITE)
            label = Text(str(n), font_size=16).next_to(dot, UP, buff=0.05)
            node_mobs[n] = VGroup(dot, label)

        edges = VGroup()
        for u, v in g.edges():
            if u in node_mobs and v in node_mobs:
                a = node_mobs[u][0].get_center()
                b = node_mobs[v][0].get_center()
                edges.add(Arrow(a, b, stroke_width=1.5, max_tip_length_to_length_ratio=0.05))

        graph_group = VGroup(edges, *node_mobs.values())
        self.play(LaggedStart(*[FadeIn(m) for m in graph_group], lag_ratio=0.01))
        self.wait(1)

        self.play(Circumscribe(graph_group, color=YELLOW, run_time=2.0))
        self.wait(1)
