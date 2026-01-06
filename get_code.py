from pathlib import Path

root = Path(".").resolve()
output = root / "code.txt"

EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    ".env",
    ".idea",
    ".vscode",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
    "dist",
    "build",
}

EXCLUDE_FILES = {"code.txt"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".pyd", ".so", ".dll", ".dylib"}


def should_skip(path: Path) -> bool:
    rel = path.relative_to(root)
    if set(rel.parts) & EXCLUDE_DIRS:
        return True

    if path.is_file():
        if path.name in EXCLUDE_FILES:
            return True
        if path.suffix.lower() in EXCLUDE_SUFFIXES:
            return True

    return False


def print_tree(out, root_path: Path) -> None:
    out.write("STRUCTURE:\n")
    out.write(f"{root_path.name}/\n")

    for path in sorted(root_path.rglob("*")):
        if should_skip(path):
            continue

        rel = path.relative_to(root_path)
        depth = len(rel.parts) - 1
        indent = "  " * depth

        if path.is_dir():
            out.write(f"{indent}├── {path.name}/\n")
        else:
            out.write(f"{indent}├── {path.name}\n")

    out.write("\n\n")


with output.open("w", encoding="utf-8") as out:
    print_tree(out, root)

    py_files = sorted(p for p in root.rglob("*.py") if p.is_file() and not should_skip(p))
    for p in py_files:
        out.write(f"FILE: {p.relative_to(root)}\n\n")
        try:
            out.write(p.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            out.write(p.read_text(encoding="latin-1", errors="replace"))
        out.write("\n\n")

print("Filtered directory tree + Python code saved to code.txt")
