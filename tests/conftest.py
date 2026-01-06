import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Custom CLI options used by our video regression tests.

    NOTE: pytest discovers this hook only from conftest.py (or plugins), not from test modules.
    """

    parser.addoption(
        "--show",
        action="store_true",
        default=False,
        help="Show debug window (OpenCV imshow).",
    )
    parser.addoption(
        "--save-debug",
        action="store_true",
        default=False,
        help="Save debug-annotated video into tests/_out/",
    )
    parser.addoption(
        "--sample-fps",
        action="store",
        default="10",
        help="Sampling FPS for tests (default: 10).",
    )
