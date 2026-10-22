"""
Automated Smoke Tests for Streamlit App and Navigation Modules
"""
import py_compile
from pathlib import Path


def test_app_syntax_and_compilation():
    """Verifies that app.py compiles cleanly without syntax errors."""
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    assert app_path.exists()
    # py_compile raises error on syntax errors
    py_compile.compile(str(app_path), doraise=True)


def test_all_page_modules_importable():
    """Verifies that all 9 application page renderers import cleanly."""
    from src.ui.page_overview import render_page_overview
    from src.ui.page_data_explorer import render_page_data_explorer
    from src.ui.page_eda import render_page_eda
    from src.ui.page_training import render_page_training
    from src.ui.page_evaluation import render_page_evaluation
    from src.ui.page_simulation import render_page_simulation
    from src.ui.page_predict import render_page_predict
    from src.ui.page_educational import render_page_educational
    from src.ui.page_history import render_page_history

    pages = [
        render_page_overview,
        render_page_data_explorer,
        render_page_eda,
        render_page_training,
        render_page_evaluation,
        render_page_simulation,
        render_page_predict,
        render_page_educational,
        render_page_history,
    ]
    for p in pages:
        assert callable(p)
