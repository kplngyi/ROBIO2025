#!/usr/bin/env python3
"""Aggregate same-side channel scores across subjects and export to CSV.

Output columns:
- subject (always ALL)
- file (A1L or A1R)
- selected_channel_names
- selected_channel_scores
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


def extract_side(file_value: object) -> str | None:
    text = str(file_value)
    if "A1L" in text:
        return "A1L"
    if "A1R" in text:
        return "A1R"
    return None


def parse_list(value: object) -> list:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    return []


def aggregate_side_scores(df: pd.DataFrame) -> pd.DataFrame:
    required = {"file", "selected_channel_names", "selected_channel_scores"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    side_channel_scores: dict[str, defaultdict[str, float]] = {
        "A1L": defaultdict(float),
        "A1R": defaultdict(float),
    }
    total_channel_scores: defaultdict[str, float] = defaultdict(float)

    for _, row in df.iterrows():
        side = extract_side(row.get("file"))
        if side is None:
            continue

        names = parse_list(row.get("selected_channel_names"))
        scores = parse_list(row.get("selected_channel_scores"))

        if not names or not scores:
            continue

        for name, score in zip(names, scores):
            channel_name = str(name).strip()
            if not channel_name:
                continue
            try:
                score_value = float(score)
            except (TypeError, ValueError):
                continue
            side_channel_scores[side][channel_name] += score_value
            total_channel_scores[channel_name] += score_value

    rows: list[dict[str, object]] = []
    for side in ["A1L", "A1R"]:
        channel_dict = side_channel_scores[side]
        ranked = sorted(channel_dict.items(), key=lambda item: item[1], reverse=True)
        names = [name for name, _ in ranked]
        scores = [round(score, 10) for _, score in ranked]

        rows.append(
            {
                "subject": "ALL",
                "file": side,
                "selected_channel_names": str(names),
                "selected_channel_scores": json.dumps(scores),
            }
        )

    ranked_total = sorted(total_channel_scores.items(), key=lambda item: item[1], reverse=True)
    total_names = [name for name, _ in ranked_total]
    total_scores = [round(score, 10) for _, score in ranked_total]
    rows.append(
        {
            "subject": "ALL",
            "file": "ALL",
            "selected_channel_names": str(total_names),
            "selected_channel_scores": json.dumps(total_scores),
        }
    )

    return pd.DataFrame(
        rows,
        columns=["subject", "file", "selected_channel_names", "selected_channel_scores"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate channel scores across subjects by side (A1L/A1R)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("ResEEG/32_300_summary_63_results.csv"),
        help="Input summary CSV path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ResEEG/32_300_summary_63_results_side_aggregated.csv"),
        help="Output CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input)
    result = aggregate_side_scores(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    print(f"Saved: {args.output}")
    print(result[["subject", "file"]])


if __name__ == "__main__":
    main()
