#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
Render a tiny HTML report from an embedd stats JSONL log.

Usage:
  uv run scripts/stats_report.py /path/to/stats.jsonl /path/to/report.html
"""

from __future__ import annotations

import html
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Row:
    modality: str
    backend: str
    mode: str
    model_id: str
    embedd_version: str
    generated_at_unix_s: int | None
    corpus_hash: str
    dim: int | None
    non_finite: int | None
    wrong_dim: int | None
    l2_min: float | None
    l2_max: float | None
    l2_mean: float | None
    l2_std: float | None
    embed_ms_total: float | None
    embed_ms_per_input: float | None
    embed_inputs_per_s: float | None
    prompt_hash: str
    prompt_apply: str
    prompt_name: str
    normalization: str
    truncation_policy: str
    output_dim: int | None


def load_rows(path: Path) -> list[Row]:
    rows: list[Row] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r: dict[str, Any] = json.loads(line)
        rows.append(
            Row(
                modality=str(r.get("modality", "text")),
                backend=str(r.get("backend", "")),
                mode=str(r.get("mode", "")),
                model_id=str(r.get("model_id", "")),
                embedd_version=str(r.get("embedd_version", "")),
                generated_at_unix_s=r.get("generated_at_unix_s"),
                corpus_hash=str(r.get("corpus_hash_fnv1a64", "")),
                dim=r.get("dim"),
                non_finite=r.get("non_finite"),
                wrong_dim=r.get("wrong_dim"),
                l2_min=r.get("l2_norm_min"),
                l2_max=r.get("l2_norm_max"),
                l2_mean=r.get("l2_norm_mean"),
                l2_std=r.get("l2_norm_std"),
                embed_ms_total=r.get("embed_ms_total"),
                embed_ms_per_input=(r.get("embed_ms_per_input") or r.get("embed_ms_per_text")),
                embed_inputs_per_s=(r.get("embed_inputs_per_s") or r.get("embed_texts_per_s")),
                prompt_hash=str(r.get("prompt_hash_fnv1a64", "")),
                prompt_apply=str(r.get("prompt_apply", "")),
                prompt_name=str(r.get("prompt_name", "")),
                normalization=str(r.get("normalization", "")),
                truncation_policy=str(r.get("truncation_policy", "")),
                output_dim=r.get("output_dim"),
            )
        )
    return rows


def fmt(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.6f}"
    return str(x)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        sys.stderr.write("usage: stats_report.py <stats.jsonl> <report.html>\n")
        return 2

    in_path = Path(argv[1])
    out_path = Path(argv[2])
    rows = load_rows(in_path)

    # Group by (backend, mode, model_id, corpus_hash); keep the newest (max ts).
    latest: dict[tuple[str, str, str, str, str], Row] = {}
    for r in rows:
        key = (r.modality, r.backend, r.mode, r.model_id, r.corpus_hash)
        cur = latest.get(key)
        if cur is None:
            latest[key] = r
            continue
        if (r.generated_at_unix_s or 0) >= (cur.generated_at_unix_s or 0):
            latest[key] = r

    latest_rows = sorted(
        latest.values(),
        key=lambda r: (r.modality, r.backend, r.mode, r.model_id, r.corpus_hash),
    )

    title = f"embedd stats report: {in_path.name}"

    def td(s: str) -> str:
        return f"<td>{html.escape(s)}</td>"

    table_rows = []
    for r in latest_rows:
        table_rows.append(
            "<tr>"
            + td(r.modality)
            + td(r.backend)
            + td(r.mode)
            + td(r.model_id)
            + td(r.embedd_version)
            + td(r.corpus_hash)
            + td(r.prompt_hash)
            + td(r.prompt_apply)
            + td(r.prompt_name)
            + td(r.normalization)
            + td(r.truncation_policy)
            + td(fmt(r.output_dim))
            + td(fmt(r.dim))
            + td(fmt(r.wrong_dim))
            + td(fmt(r.non_finite))
            + td(fmt(r.embed_ms_total))
            + td(fmt(r.embed_ms_per_input))
            + td(fmt(r.embed_inputs_per_s))
            + td(fmt(r.l2_min))
            + td(fmt(r.l2_max))
            + td(fmt(r.l2_mean))
            + td(fmt(r.l2_std))
            + "</tr>"
        )

    html_doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; padding: 16px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; vertical-align: top; }}
    th {{ background: #f5f5f5; position: sticky; top: 0; }}
    .muted {{ color: #666; }}
  </style>
</head>
<body>
  <h2>{html.escape(title)}</h2>
  <p class="muted">Input: {html.escape(str(in_path))} &nbsp; Rows: {len(rows)} &nbsp; Latest rows: {len(latest_rows)}</p>

  <table>
    <thead>
      <tr>
        <th>modality</th>
        <th>backend</th>
        <th>mode</th>
        <th>model_id</th>
        <th>embedd_version</th>
        <th>corpus_hash</th>
        <th>prompt_hash</th>
        <th>prompt_apply</th>
        <th>prompt_name</th>
        <th>normalization</th>
        <th>truncation_policy</th>
        <th>output_dim</th>
        <th>dim</th>
        <th>wrong_dim</th>
        <th>non_finite</th>
        <th>embed_ms_total</th>
        <th>embed_ms_per_input</th>
        <th>embed_inputs_per_s</th>
        <th>l2_min</th>
        <th>l2_max</th>
        <th>l2_mean</th>
        <th>l2_std</th>
      </tr>
    </thead>
    <tbody>
      {"".join(table_rows)}
    </tbody>
  </table>
</body>
</html>
"""

    out_path.write_text(html_doc, encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

