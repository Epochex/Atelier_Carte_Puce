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

INCLUDE_SUFFIXES = {".py", ".c", ".cpp", ".h", ".hpp", ".sh"}


def should_skip(path: Path) -> bool:
    rel = path.relative_to(root)

    # Skip any path that contains an excluded directory in its components
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

        suffix = "/" if path.is_dir() else ""
        out.write(f"{indent}├── {path.name}{suffix}\n")

    out.write("\n\n")


def iter_source_files(root_path: Path):
    for p in sorted(root_path.rglob("*")):
        if not p.is_file():
            continue
        if should_skip(p):
            continue
        # include selected suffixes; also include files with no suffix but executable scripts if needed
        if p.suffix.lower() in INCLUDE_SUFFIXES:
            yield p


with output.open("w", encoding="utf-8") as out:
    print_tree(out, root)

    for p in iter_source_files(root):
        out.write(f"FILE: {p.relative_to(root)}\n")
        out.write("=" * 80 + "\n")
        try:
            out.write(p.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            out.write(p.read_text(encoding="latin-1", errors="replace"))
        out.write("\n\n")

print(f"Filtered directory tree + source files saved to {output}")
