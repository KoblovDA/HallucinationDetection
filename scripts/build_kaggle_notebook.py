"""Assemble cells/*.py and cells/*.md from a folder into a single ipynb.

Cells are ordered lexicographically by filename. `.py` files become code cells, `.md` files
become markdown cells.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build(cells_dir: Path, out_path: Path) -> None:
    cells = []
    for f in sorted(cells_dir.iterdir()):
        if f.suffix == ".py":
            cells.append({
                "cell_type": "code",
                "metadata": {},
                "source": f.read_text(),
                "outputs": [],
                "execution_count": None,
            })
        elif f.suffix == ".md":
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": f.read_text(),
            })
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out_path.write_text(json.dumps(notebook, indent=1))
    print(f"Wrote {out_path} ({len(cells)} cells)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    build(Path(args.cells), Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
