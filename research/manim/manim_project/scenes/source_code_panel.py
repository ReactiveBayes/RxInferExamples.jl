from __future__ import annotations
from pathlib import Path
import os
from manim import *

class SourceCodePanel(Scene):
    def construct(self):
        self.random_seed = 42
        src_path = self.get_src_path()
        # Use Text instead of Tex to avoid LaTeX escaping issues on filenames
        title = Text(Path(src_path).parent.name).to_edge(UP)
        self.next_section("title")
        self.play(Write(title), run_time=1.5)
        # Use Manim's Code API for v0.19: pass the file path positionally
        code = Code(
            src_path,
            tab_width=4,
            background="window",
            language="julia",
        ).scale(0.6)
        code.to_edge(DOWN)
        self.next_section("code")
        self.play(FadeIn(code), run_time=1.5)
        # Scroll the code block slowly to show more content
        self.next_section("scroll")
        for _ in range(3):
            self.play(code.animate.shift(UP * 0.5), run_time=2)
        for _ in range(3):
            self.play(code.animate.shift(DOWN * 0.5), run_time=2)
        self.wait(1)

    def get_src_path(self) -> str:
        # Prefer environment variable if provided (set by render script)
        env_src = os.environ.get("SRC_PATH")
        if env_src and Path(env_src).exists():
            return str(Path(env_src))
        # Fallback to Mountain Car default for backward compatibility
        from manim_project.configs.paths import MOUNTAIN_CAR_SOURCE
        return str(MOUNTAIN_CAR_SOURCE)
