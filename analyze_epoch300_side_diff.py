#!/usr/bin/env python3
"""Analyze left-vs-right accuracy differences for epoch=300 across channel counts."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SUMMARY_PATTERN = re.compile(r"^(?P<batch>\d+)_(?P<epochs>\d+)_summary_(?P<topk>\d+)_results\.csv$")


def extract_side(file_name: str) -> str:
    text = str(file_name)
    if "A1L" in text:
        return "L"
    if "A1R" in text:
        return "R"
    return "unknown"


def parse_name_metadata(path: Path) -> dict[str, int | None]:
    match = SUMMARY_PATTERN.match(path.name)
    if match:
        return {
            "run_batch": int(match.group("batch")),
            "run_epochs": int(match.group("epochs")),
            "run_top_k": int(match.group("topk")),
        }

    digits = re.findall(r"\d+", path.stem)
    return {
        "run_batch": int(digits[0]) if len(digits) >= 1 else None,
        "run_epochs": int(digits[1]) if len(digits) >= 2 else None,
        "run_top_k": int(digits[2]) if len(digits) >= 3 else None,
    }


def find_accuracy_column(columns: list[str]) -> str:
    candidates = ["test_acc", "test_accuracy", "acc", "accuracy"]
    lowered = {c.lower(): c for c in columns}
    for key in candidates:
        if key in lowered:
            return lowered[key]
    raise ValueError(f"No accuracy column found in: {columns}")


def load_epoch300_rows(root: Path, modality_map: dict[str, str], epoch: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for modality, folder in modality_map.items():
        folder_path = root / folder
        if not folder_path.exists():
            continue

        for file_path in sorted(folder_path.glob("*summary*_results.csv")):
            meta = parse_name_metadata(file_path)
            if meta["run_epochs"] != epoch:
                continue

            try:
                df = pd.read_csv(file_path)
            except Exception:
                continue
            if df.empty:
                continue

            try:
                acc_col = find_accuracy_column(list(df.columns))
            except ValueError:
                continue

            df = df.copy()
            df["modality"] = modality
            df["source_file"] = file_path.name
            df["run_batch"] = meta["run_batch"]
            df["run_epochs"] = meta["run_epochs"]
            df["run_top_k"] = meta["run_top_k"]
            df["accuracy"] = pd.to_numeric(df[acc_col], errors="coerce")

            if "subject" not in df.columns:
                df["subject"] = "unknown"
            if "file" not in df.columns:
                df["file"] = "unknown"

            df["side"] = df["file"].astype(str).map(extract_side)
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["accuracy", "run_top_k"])
    out = out[out["side"].isin(["L", "R"])].copy()
    out["run_top_k"] = pd.to_numeric(out["run_top_k"], errors="coerce")
    out = out.dropna(subset=["run_top_k"])
    return out


def compute_side_diff(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["modality", "subject", "run_top_k", "side"])["accuracy"]
        .mean()
        .reset_index()
    )

    pivot = grouped.pivot_table(
        index=["modality", "subject", "run_top_k"],
        columns="side",
        values="accuracy",
        aggfunc="mean",
    ).reset_index()

    if "L" not in pivot.columns:
        pivot["L"] = pd.NA
    if "R" not in pivot.columns:
        pivot["R"] = pd.NA

    pivot = pivot.rename(columns={"L": "left_acc", "R": "right_acc"})
    pivot = pivot.dropna(subset=["left_acc", "right_acc"]).copy()
    pivot["diff_left_minus_right"] = pivot["left_acc"] - pivot["right_acc"]
    pivot["diff_abs"] = pivot["diff_left_minus_right"].abs()
    return pivot.sort_values(["modality", "subject", "run_top_k"]).reset_index(drop=True)


def save_plots(detail: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for modality in sorted(detail["modality"].unique()):
        mod = detail[detail["modality"] == modality]
        if mod.empty:
            continue

        # Subject-wise lines: signed difference over top_k.
        fig, ax = plt.subplots(figsize=(10, 6))
        for subject in sorted(mod["subject"].astype(str).unique()):
            sub = mod[mod["subject"].astype(str) == subject].sort_values("run_top_k")
            ax.plot(
                sub["run_top_k"],
                sub["diff_left_minus_right"],
                marker="o",
                linewidth=1.8,
                markersize=3.5,
                label=subject,
            )
        ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.set_title(f"{modality} (epoch=300): Left-Right Accuracy Difference")
        ax.set_xlabel("Channel Count (top_k)")
        ax.set_ylabel("Left - Right Accuracy")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="best", fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{modality.lower()}_epoch300_side_diff_line.png", dpi=200)
        plt.close(fig)

        # Heatmap: subject x top_k with signed difference.
        pivot = mod.pivot_table(
            index="subject",
            columns="run_top_k",
            values="diff_left_minus_right",
            aggfunc="mean",
        ).sort_index()
        if pivot.empty:
            continue

        fig, ax = plt.subplots(figsize=(12, max(4, 0.6 * len(pivot.index))))
        vmax = max(abs(pivot.min().min()), abs(pivot.max().max()))
        im = ax.imshow(pivot.values, aspect="auto", interpolation="nearest", vmin=-vmax, vmax=vmax, cmap="coolwarm")
        ax.set_title(f"{modality} (epoch=300): Left-Right Difference Heatmap")
        ax.set_xlabel("Channel Count (top_k)")
        ax.set_ylabel("Subject")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(int(c)) for c in pivot.columns], rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(i) for i in pivot.index])
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Left - Right Accuracy")
        fig.tight_layout()
        fig.savefig(fig_dir / f"{modality.lower()}_epoch300_side_diff_heatmap.png", dpi=200)
        plt.close(fig)


def save_side_accuracy_whisker_plots(rows: pd.DataFrame, out_dir: Path, epoch: int) -> None:
    """Plot L/R accuracy with whiskers (min-max) and mean marker at each top_k."""
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    stats = (
        rows.groupby(["modality", "side", "run_top_k"])["accuracy"]
        .agg(mean="mean", min="min", max="max", count="count")
        .reset_index()
        .sort_values(["modality", "side", "run_top_k"])
    )
    if stats.empty:
        return

    for modality in sorted(stats["modality"].unique()):
        mod = stats[stats["modality"] == modality].copy()
        if mod.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        for side, color, offset in [("L", "#1f77b4", -0.35), ("R", "#d62728", 0.35)]:
            cur = mod[mod["side"] == side].copy()
            if cur.empty:
                continue
            cur = cur.sort_values("run_top_k")
            x = cur["run_top_k"].to_numpy(dtype=float) + offset
            y = cur["mean"].to_numpy(dtype=float)
            yerr_lower = (cur["mean"] - cur["min"]).to_numpy(dtype=float)
            yerr_upper = (cur["max"] - cur["mean"]).to_numpy(dtype=float)

            ax.errorbar(
                x,
                y,
                yerr=[yerr_lower, yerr_upper],
                fmt="o",
                color=color,
                ecolor=color,
                elinewidth=1.7,
                capsize=3,
                markersize=4,
                label=f"Side {side} mean (whisker=min/max)",
            )

        x_ticks = sorted(mod["run_top_k"].unique())
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(int(v)) for v in x_ticks], rotation=45, ha="right")
        ax.set_title(f"{modality} (epoch={epoch}): L/R Accuracy with Whiskers")
        ax.set_xlabel("Channel Count (top_k)")
        ax.set_ylabel("Accuracy")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{modality.lower()}_epoch{epoch}_side_accuracy_whisker.png", dpi=200)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze epoch=300 left/right accuracy differences.")
    parser.add_argument("--root", type=str, default=".", help="Project root path")
    parser.add_argument("--out", type=str, default="analysis_epoch300_side", help="Output folder")
    parser.add_argument("--epoch", type=int, default=300, help="Target epoch value")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    modality_map = {
        "EEG": "ResEEG",
        "fNIRS": "ResfNIRS",
        "Fusion": "ResFusion",
    }

    rows = load_epoch300_rows(root, modality_map, epoch=args.epoch)
    if rows.empty:
        print(f"No valid rows found for epoch={args.epoch}.")
        return

    detail = compute_side_diff(rows)
    if detail.empty:
        print("No L/R pairs found after grouping.")
        return

    detail.to_csv(out_dir / f"epoch{args.epoch}_side_diff_detailed.csv", index=False)

    by_subject = (
        detail.groupby(["modality", "subject"]).agg(
            mean_signed=("diff_left_minus_right", "mean"),
            mean_abs=("diff_abs", "mean"),
            max_abs=("diff_abs", "max"),
            n_topk=("diff_abs", "count"),
        )
        .reset_index()
        .sort_values(["modality", "mean_abs"], ascending=[True, False])
    )
    by_subject.to_csv(out_dir / f"epoch{args.epoch}_side_diff_by_subject.csv", index=False)

    by_topk = (
        detail.groupby(["modality", "run_top_k"]).agg(
            mean_signed=("diff_left_minus_right", "mean"),
            mean_abs=("diff_abs", "mean"),
            std_signed=("diff_left_minus_right", "std"),
            n_subjects=("diff_abs", "count"),
        )
        .reset_index()
        .sort_values(["modality", "run_top_k"])
    )
    by_topk.to_csv(out_dir / f"epoch{args.epoch}_side_diff_by_topk.csv", index=False)

    modality_summary = (
        detail.groupby("modality").agg(
            mean_signed=("diff_left_minus_right", "mean"),
            mean_abs=("diff_abs", "mean"),
            max_abs=("diff_abs", "max"),
            n_pairs=("diff_abs", "count"),
        )
        .reset_index()
        .sort_values("mean_abs", ascending=False)
    )
    modality_summary.to_csv(out_dir / f"epoch{args.epoch}_side_diff_modality_summary.csv", index=False)

    save_plots(detail, out_dir)
    save_side_accuracy_whisker_plots(rows, out_dir, epoch=args.epoch)

    report = [
        f"# Epoch {args.epoch} Left-Right Difference Report",
        "",
        f"Rows with L/R pairs: {len(detail)}",
        "",
        "## Modality Summary",
    ]
    for _, row in modality_summary.iterrows():
        report.append(
            f"- {row['modality']}: mean_signed={row['mean_signed']:.4f}, "
            f"mean_abs={row['mean_abs']:.4f}, max_abs={row['max_abs']:.4f}, n={int(row['n_pairs'])}"
        )
    (out_dir / f"epoch{args.epoch}_side_diff_report.md").write_text("\n".join(report), encoding="utf-8")

    print(f"Done. Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
