"""Tidy Jupyter notebooks for publication without touching results.

What it does
- Removes `metadata.widgets` (Colab widget state breaks GitHub's renderer).
- Collapses very long stream outputs (pip logs, progress bars, repeated
  "Wrote file" lines) to their first and last lines.
- Keeps every figure, table and metric output untouched.

Usage:  python tools/clean_notebooks.py path/to/*.ipynb
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

MAX_STREAM_LINES = 40
KEEP_HEAD = 8
KEEP_TAIL = 12
NOISY_PREFIXES = ("Requirement already satisfied", "Collecting ", "Downloading ", "  Downloading", "  Using cached")


def _collapse(lines: list[str]) -> list[str]:
    text = "".join(lines).splitlines(keepends=True)
    text = [ln for ln in text if not ln.startswith(NOISY_PREFIXES)] or text[-1:]
    if len(text) <= MAX_STREAM_LINES:
        return text
    hidden = len(text) - KEEP_HEAD - KEEP_TAIL
    return text[:KEEP_HEAD] + [f"... [{hidden} lines of log output omitted for readability] ...\n"] + text[-KEEP_TAIL:]


def clean(path: Path) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    nb.get("metadata", {}).pop("widgets", None)
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        new_outputs = []
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream":
                out["text"] = _collapse(out["text"] if isinstance(out["text"], list) else [out["text"]])
            data = out.get("data", {})
            # Drop live widget views (they render as raw JSON on GitHub).
            if "application/vnd.jupyter.widget-view+json" in data:
                data.pop("application/vnd.jupyter.widget-view+json")
                if not data:
                    continue
            new_outputs.append(out)
        cell["outputs"] = new_outputs
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"cleaned {path}  ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        clean(Path(arg))
