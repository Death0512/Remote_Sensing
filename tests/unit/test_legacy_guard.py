import ast
from pathlib import Path

LEGACY_DIRS = ("core", "detectors", "engine")
FORBIDDEN_MODULE_PREFIXES = (
    "src.core",
    "src.detectors",
    "src.engine",
    "core",
    "detectors",
    "engine",
)


def _get_active_src_files() -> list[Path]:
    repo_root = Path(__file__).resolve().parent.parent.parent
    src_dir = repo_root / "src"
    return sorted(src_dir.rglob("*.py"))


def test_legacy_directories_completely_removed() -> None:
    repo_root = Path(__file__).resolve().parent.parent.parent
    src_dir = repo_root / "src"
    for d in ("core", "detectors", "engine"):
        assert not (src_dir / d).exists()


def test_legacy_directories_do_not_exist() -> None:
    test_legacy_directories_completely_removed()


def test_no_legacy_imports_in_active_tree() -> None:
    repo_root = Path(__file__).resolve().parent.parent.parent
    active_files = _get_active_src_files()
    assert len(active_files) > 0

    violations: list[tuple[str, str]] = []
    for py_file in active_files:
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if any(
                        alias.name == prefix or alias.name.startswith(f"{prefix}.")
                        for prefix in FORBIDDEN_MODULE_PREFIXES
                    ):
                        violations.append((str(py_file.relative_to(repo_root)), alias.name))
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if any(
                    mod == prefix or mod.startswith(f"{prefix}.")
                    for prefix in FORBIDDEN_MODULE_PREFIXES
                ):
                    violations.append((str(py_file.relative_to(repo_root)), mod))

    assert violations == [], f"Found forbidden legacy imports in active tree: {violations}"


def test_no_detection_from_legacy_contracts() -> None:
    repo_root = Path(__file__).resolve().parent.parent.parent
    active_files = _get_active_src_files()
    assert len(active_files) > 0

    violations: list[tuple[str, str, str]] = []
    for py_file in active_files:
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if "core.contracts" in mod:
                    for alias in node.names:
                        if alias.name == "Detection":
                            violations.append(
                                (str(py_file.relative_to(repo_root)), mod, alias.name)
                            )

    assert violations == [], f"Found forbidden legacy Detection import in active tree: {violations}"
