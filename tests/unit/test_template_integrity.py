import tomllib
from importlib import import_module
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def load_pyproject() -> dict[str, Any]:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_package_can_be_imported() -> None:
    package = import_module("rag_service")

    assert package.__all__ == ()


def test_project_metadata_describes_rag_service() -> None:
    pyproject = load_pyproject()
    project = pyproject["project"]

    assert project["name"] == "rag-service"
    expected_description = "Source-backed document ingestion and retrieval API."
    assert project["description"] == expected_description
    assert project["requires-python"] == ">=3.14"
    assert {"fastapi", "uvicorn[standard]"}.issubset(project["dependencies"])


def test_project_declares_typed_src_package() -> None:
    pyproject = load_pyproject()
    project = pyproject["project"]
    tool = pyproject["tool"]
    hatch_targets = tool["hatch"]["build"]["targets"]

    assert (ROOT / "src" / "rag_service" / "py.typed").is_file()
    assert "Typing :: Typed" in project["classifiers"]
    assert hatch_targets["wheel"]["packages"] == [
        "src/rag_service",
    ]
    assert tool["coverage"]["run"]["source"] == ["rag_service"]
