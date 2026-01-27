#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
Compare two embedd stats artifacts (JSON or JSONL) and print a compact delta.

Design goals:
- No network
- No secrets
- Stable-ish output: focuses on numeric/stat fields and ignores timestamps unless requested.

Usage:
  uv run scripts/stats_diff.py a.json b.json
  uv run scripts/stats_diff.py a.jsonl b.jsonl
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


def load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        out: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
        return out
    if path.suffix.lower() == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            return [obj]
        raise TypeError(f"unexpected JSON root type: {type(obj)}")
    raise ValueError(f"unsupported file extension: {path.suffix} (want .json or .jsonl)")


def key_of(r: dict[str, Any]) -> tuple[str, str, str, str]:
    # Include modality so multimodal logs don't collide.
    return (
        str(r.get("modality", "text")),
        str(r.get("backend", "")),
        str(r.get("mode", "")),
        str(r.get("model_id", "")),
    )


NUM_FIELDS = [
    "dim",
    "wrong_dim",
    "non_finite",
    "n_valid",
    "n_texts",
    "n_inputs",
    "n_embs",
    "corpus_n_lines",
    "embed_ms_total",
    "embed_ms_per_text",
    "embed_ms_per_input",
    "embed_texts_per_s",
    "l2_norm_min",
    "l2_norm_max",
    "l2_norm_mean",
    "l2_norm_std",
    "truncation_max_len",
    "output_dim",
]

STR_FIELDS = [
    "embedd_version",
    "modality",
    "corpus_hash_fnv1a64",
    "corpus_source",
    "corpus_path",
    "prompt_hash_fnv1a64",
    "prompt_apply",
    "prompt_name",
    "normalization",
    "truncation_policy",
    "truncation_direction",
]


@dataclass(frozen=True)
class Delta:
    key: tuple[str, str, str, str]
    field: str
    a: Any
    b: Any


def iter_deltas(a: dict[str, Any], b: dict[str, Any]) -> Iterable[Delta]:
    k = key_of(a)
    for f in STR_FIELDS + NUM_FIELDS:
        av = a.get(f)
        bv = b.get(f)
        if av != bv:
            yield Delta(k, f, av, bv)


def print_header(title: str) -> None:
    sys.stdout.write(title + "\n")
    sys.stdout.write("-" * len(title) + "\n")


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        sys.stderr.write("usage: stats_diff.py <a.json|a.jsonl> <b.json|b.jsonl>\n")
        return 2

    a_path = Path(argv[1])
    b_path = Path(argv[2])
    a_recs = load_records(a_path)
    b_recs = load_records(b_path)

    a_map = {key_of(r): r for r in a_recs}
    b_map = {key_of(r): r for r in b_recs}

    keys = sorted(set(a_map) | set(b_map))
    missing_a = [k for k in keys if k not in a_map]
    missing_b = [k for k in keys if k not in b_map]

    print_header("embedd stats diff")
    print(f"a={a_path}")
    print(f"b={b_path}")
    print(f"n_a={len(a_recs)} n_b={len(b_recs)}")

    if missing_a:
        print("\nmissing in a:")
        for k in missing_a:
            print(f"  key={k}")
    if missing_b:
        print("\nmissing in b:")
        for k in missing_b:
            print(f"  key={k}")

    any_delta = False
    for k in keys:
        if k not in a_map or k not in b_map:
            continue
        deltas = list(iter_deltas(a_map[k], b_map[k]))
        if not deltas:
            continue
        any_delta = True
        print("\nkey=" + repr(k))
        for d in deltas:
            print(f"  {d.field}: {d.a!r} -> {d.b!r}")

    if not any_delta and not missing_a and not missing_b:
        print("\n(no differences)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

