from pathlib import Path
from manim_project.configs import paths

def test_assets_exist():
    assert Path(paths.MOUNTAIN_CAR_DOT).exists()
    assert Path(paths.MOUNTAIN_CAR_MMD).exists()
    assert Path(paths.MOUNTAIN_CAR_SOURCE).exists()
    assert Path(paths.MOUNTAIN_CAR_PNG).exists()
