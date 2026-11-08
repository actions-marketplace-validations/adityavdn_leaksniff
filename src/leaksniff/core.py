"""Loading .py and .ipynb files, mapping lines back to notebook cells, and noqa handling."""
from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

LEVELS = {"info": 0, "warning": 1, "error": 2}
NOQA = re.compile(r"#\s*(?:noqa|leaksniff:\s*ignore)(?::\s*(?P<codes>[A-Z0-9, ]+))?", re.I)


@dataclass(frozen=True)
class Finding:
    code: str
    level: str
    line: int
    message: str
    hint: str
    path: str = ""
    cell: int | None = None
    cell_line: int | None = None

    def where(self) -> str:
        if self.cell is not None:
            return f"{self.path}:cell {self.cell}:{self.cell_line}"
        return f"{self.path}:{self.line}"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Source:
    text: str
    path: str = "<string>"
    cell_of_line: dict[int, tuple[int, int]] = field(default_factory=dict)
    broken_cells: list[int] = field(default_factory=list)


def _clean_cell_line(line: str) -> str:
    """IPython magics and shell escapes are not Python; keep the line so numbering holds."""
    stripped = line.lstrip()
    if stripped.startswith(("%", "!", "?")) or stripped.endswith("?"):
        return "\n"
    return line if line.endswith("\n") else line + "\n"


def load_notebook(path: Path) -> Source:
    nb = json.loads(path.read_text(encoding="utf-8"))
    lines, mapping, broken = [], {}, []
    for number, cell in enumerate(nb.get("cells", []), 1):
        if cell.get("cell_type") != "code":
            continue
        raw = cell.get("source", [])
        body = [_clean_cell_line(l) for l in (raw.splitlines(True) if isinstance(raw, str) else raw)]
        try:
            ast.parse("".join(body))
        except SyntaxError:
            broken.append(number)
            body = ["\n"] * len(body)
        for i, text in enumerate(body, 1):
            lines.append(text)
            mapping[len(lines)] = (number, i)
    return Source("".join(lines), str(path), mapping, broken)


def load(path: Path) -> Source:
    if path.suffix == ".ipynb":
        return load_notebook(path)
    return Source(path.read_text(encoding="utf-8", errors="replace"), str(path))


def suppressed(text_lines: list[str], finding: Finding) -> bool:
    if not 0 < finding.line <= len(text_lines):
        return False
    m = NOQA.search(text_lines[finding.line - 1])
    if not m:
        return False
    codes = m.group("codes")
    return codes is None or finding.code in {c.strip().upper() for c in codes.split(",")}


def run(source: Source) -> list[Finding]:
    from .rules import analyse

    out: list[Finding] = []
    for cell in source.broken_cells:
        out.append(Finding("LS000", "info", 0, f"cell {cell} is not valid Python and was skipped",
                           "Fix the syntax error to have this cell checked.", source.path, cell, 1))
    try:
        tree = ast.parse(source.text)
    except SyntaxError as err:
        return out + [Finding("LS000", "info", err.lineno or 0, f"could not parse file: {err.msg}",
                              "Fix the syntax error to have this file checked.", source.path)]
    text_lines = source.text.splitlines()
    for f in analyse(tree):
        if suppressed(text_lines, f):
            continue
        cell, cell_line = source.cell_of_line.get(f.line, (None, None))
        out.append(Finding(f.code, f.level, f.line, f.message, f.hint, source.path, cell, cell_line))
    return sorted(out, key=lambda f: (f.line, f.code))


def check_source(text: str, path: str = "<string>") -> list[Finding]:
    return run(Source(text, path))


def check_path(path: str | Path) -> list[Finding]:
    try:
        source = load(Path(path))
    except (OSError, UnicodeDecodeError, ValueError, AttributeError) as err:
        return [Finding("LS000", "info", 0, f"could not read file: {type(err).__name__}",
                        "Make sure it is a valid .py file or Jupyter notebook.", str(path))]
    return run(source)
