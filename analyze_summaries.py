#!/usr/bin/env python3
"""Aggregate and analyze summary CSV files across EEG, fNIRS and Fusion results."""

from __future__ import annotations

import argparse
import ast
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


def parse_channel_names(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed]
    except Exception:
        pass
    # Fallback: tolerant parser for malformed list-like strings.
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    parts = [p.strip().strip("'\"") for p in text.split(",") if p.strip()]
    return [p for p in parts if p]


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


def load_summaries(root: Path, modality_map: dict[str, str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for modality, folder in modality_map.items():
        folder_path = root / folder
        if not folder_path.exists():
            print(f"[WARN] Missing folder for {modality}: {folder_path}")
            continue

        files = sorted(folder_path.glob("*summary*_results.csv"))
        if not files:
            print(f"[WARN] No summary files found in {folder_path}")
            continue

        for file_path in files:
            try:
                df = pd.read_csv(file_path)
            except Exception as exc:
                print(f"[WARN] Failed reading {file_path.name}: {exc}")
                continue

            if df.empty:
                continue

            try:
                acc_col = find_accuracy_column(list(df.columns))
            except ValueError as exc:
                print(f"[WARN] {file_path.name}: {exc}")
                continue

            meta = parse_name_metadata(file_path)

            df = df.copy()
            df["modality"] = modality
            df["source_file"] = file_path.name
            df["accuracy"] = pd.to_numeric(df[acc_col], errors="coerce")
            df["run_batch"] = meta["run_batch"]
            df["run_epochs"] = meta["run_epochs"]
            df["run_top_k"] = meta["run_top_k"]

            if "subject" not in df.columns:
                df["subject"] = "unknown"
            if "file" not in df.columns:
                df["file"] = "unknown"
            if "selected_channel_names" not in df.columns:
                df["selected_channel_names"] = "[]"

            df["side"] = df["file"].astype(str).map(extract_side)

            frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["accuracy"])
    return combined


def plot_subject_channel_performance(combined: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Average across repeated runs so each subject has one curve per channel count.
    grouped = (
        combined.groupby(["modality", "subject", "run_top_k"], dropna=False)["accuracy"]
        .mean()
        .reset_index()
        .dropna(subset=["run_top_k"])
    )
    if grouped.empty:
        return

    grouped["run_top_k"] = pd.to_numeric(grouped["run_top_k"], errors="coerce")
    grouped = grouped.dropna(subset=["run_top_k"])

    for modality in sorted(grouped["modality"].dropna().unique()):
        mod = grouped[grouped["modality"] == modality].copy()
        if mod.empty:
            continue

        # Line plot: each subject is a curve over channel count (run_top_k).
        fig, ax = plt.subplots(figsize=(10, 6))
        for subject in sorted(mod["subject"].astype(str).unique()):
            sub_df = mod[mod["subject"].astype(str) == subject].sort_values("run_top_k")
            ax.plot(
                sub_df["run_top_k"],
                sub_df["accuracy"],
                marker="o",
                linewidth=1.8,
                markersize=3.5,
                label=subject,
                alpha=0.9,
            )

        ax.set_title(f"{modality}: Subject Performance vs Channel Count")
        ax.set_xlabel("Channel Count (run_top_k)")
        ax.set_ylabel("Mean Accuracy")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="best", fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{modality.lower()}_subject_channel_line.png", dpi=200)
        plt.close(fig)

        # Heatmap-like matrix: rows=subject, cols=channel count, values=mean accuracy.
        pivot = mod.pivot_table(
            index="subject",
            columns="run_top_k",
            values="accuracy",
            aggfunc="mean",
        ).sort_index()
        if pivot.empty:
            continue

        fig, ax = plt.subplots(figsize=(12, max(4, 0.6 * len(pivot.index))))
        im = ax.imshow(pivot.values, aspect="auto", interpolation="nearest")
        ax.set_title(f"{modality}: Subject-Channel Accuracy Heatmap")
        ax.set_xlabel("Channel Count (run_top_k)")
        ax.set_ylabel("Subject")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(int(c)) for c in pivot.columns], rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(idx) for idx in pivot.index])
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Mean Accuracy")
        fig.tight_layout()
        fig.savefig(fig_dir / f"{modality.lower()}_subject_channel_heatmap.png", dpi=200)
        plt.close(fig)

    # One combined comparison chart using subject-modality labels.
    combined_subject = (
        grouped.assign(subject_mod=lambda d: d["modality"].astype(str) + "-" + d["subject"].astype(str))
        .sort_values(["subject_mod", "run_top_k"])
    )
    fig, ax = plt.subplots(figsize=(12, 7))
    for label in combined_subject["subject_mod"].unique():
        sub = combined_subject[combined_subject["subject_mod"] == label]
        ax.plot(sub["run_top_k"], sub["accuracy"], linewidth=1.2, alpha=0.45)
    ax.set_title("All Modalities: Subject Performance vs Channel Count")
    ax.set_xlabel("Channel Count (run_top_k)")
    ax.set_ylabel("Mean Accuracy")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "all_modalities_subject_channel_line.png", dpi=300)
    plt.close(fig)


def plot_side_channel_performance(combined: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    grouped = (
        combined.groupby(["modality", "subject", "side", "run_top_k"], dropna=False)["accuracy"]
        .mean()
        .reset_index()
        .dropna(subset=["run_top_k"])
    )
    grouped = grouped[grouped["side"].isin(["L", "R"])]
    if grouped.empty:
        return

    grouped["run_top_k"] = pd.to_numeric(grouped["run_top_k"], errors="coerce")
    grouped = grouped.dropna(subset=["run_top_k"])

    for modality in sorted(grouped["modality"].dropna().unique()):
        mod = grouped[grouped["modality"] == modality]
        if mod.empty:
            continue

        # Compare L/R per subject at each channel count using mean across subjects.
        side_trend = (
            mod.groupby(["side", "run_top_k"]) ["accuracy"]
            .mean()
            .reset_index()
            .sort_values(["side", "run_top_k"])
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        for side, color in [("L", "#1f77b4"), ("R", "#d62728")]:
            cur = side_trend[side_trend["side"] == side]
            if cur.empty:
                continue
            ax.plot(
                cur["run_top_k"],
                cur["accuracy"],
                marker="o",
                linewidth=2.0,
                markersize=4,
                label=f"Side {side}",
                color=color,
            )
        ax.set_title(f"{modality}: Left vs Right Performance by Channel Count")
        ax.set_xlabel("Channel Count (run_top_k)")
        ax.set_ylabel("Mean Accuracy")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(fig_dir / f"{modality.lower()}_side_channel_line.png", dpi=200)
        plt.close(fig)


def compute_side_channel_differences(combined: pd.DataFrame) -> pd.DataFrame:
    work = combined.copy()
    if "selected_channel_names" not in work.columns:
        return pd.DataFrame()

    work["side"] = work["file"].astype(str).map(extract_side)
    work = work[work["side"].isin(["L", "R"])].copy()
    if work.empty:
        return pd.DataFrame()

    key_cols = ["modality", "subject", "run_batch", "run_epochs", "run_top_k"]
    rows: list[dict[str, object]] = []

    for key, grp in work.groupby(key_cols, dropna=False):
        left = grp[grp["side"] == "L"]
        right = grp[grp["side"] == "R"]
        if left.empty or right.empty:
            continue

        left_channels: set[str] = set()
        right_channels: set[str] = set()
        for val in left["selected_channel_names"].tolist():
            left_channels.update(parse_channel_names(val))
        for val in right["selected_channel_names"].tolist():
            right_channels.update(parse_channel_names(val))

        inter = left_channels & right_channels
        union = left_channels | right_channels
        only_l = sorted(left_channels - right_channels)
        only_r = sorted(right_channels - left_channels)
        jaccard = (len(inter) / len(union)) if union else 1.0

        left_acc = pd.to_numeric(left["accuracy"], errors="coerce").mean()
        right_acc = pd.to_numeric(right["accuracy"], errors="coerce").mean()

        rows.append(
            {
                "modality": key[0],
                "subject": key[1],
                "run_batch": key[2],
                "run_epochs": key[3],
                "run_top_k": key[4],
                "left_acc_mean": left_acc,
                "right_acc_mean": right_acc,
                "acc_gap_left_minus_right": left_acc - right_acc,
                "left_channel_count": len(left_channels),
                "right_channel_count": len(right_channels),
                "common_channel_count": len(inter),
                "union_channel_count": len(union),
                "channel_jaccard": jaccard,
                "common_channels": "|".join(sorted(inter)),
                "left_only_channels": "|".join(only_l),
                "right_only_channels": "|".join(only_r),
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def write_stats_bundle(combined: pd.DataFrame, out_dir: Path, make_plots: bool, title: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    combined_cols = [
        "modality",
        "subject",
        "file",
        "side",
        "run_batch",
        "run_epochs",
        "run_top_k",
        "accuracy",
        "selected_channel_names",
        "source_file",
    ]
    for col in combined_cols:
        if col not in combined.columns:
            combined[col] = None

    combined = combined[combined_cols].copy()
    combined.to_csv(out_dir / "combined_summaries.csv", index=False)

    modality_stats = (
        combined.groupby("modality")["accuracy"]
        .agg(count="count", mean="mean", std="std", min="min", max="max")
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    modality_stats.to_csv(out_dir / "modality_stats.csv", index=False)

    config_stats = (
        combined.groupby(["modality", "run_batch", "run_epochs", "run_top_k"])["accuracy"]
        .agg(count="count", mean="mean", std="std", min="min", max="max")
        .reset_index()
        .sort_values(["modality", "mean"], ascending=[True, False])
    )
    config_stats.to_csv(out_dir / "config_level_stats.csv", index=False)

    topk_trend = (
        combined.groupby(["modality", "run_top_k"])["accuracy"]
        .agg(count="count", mean="mean", std="std", min="min", max="max")
        .reset_index()
        .sort_values(["modality", "run_top_k"])
    )
    topk_trend.to_csv(out_dir / "topk_trend.csv", index=False)

    side_stats = (
        combined[combined["side"].isin(["L", "R"])]
        .groupby(["modality", "side"])["accuracy"]
        .agg(count="count", mean="mean", std="std", min="min", max="max")
        .reset_index()
        .sort_values(["modality", "side"])
    )
    side_stats.to_csv(out_dir / "side_stats.csv", index=False)

    side_topk_trend = (
        combined[combined["side"].isin(["L", "R"])]
        .groupby(["modality", "side", "run_top_k"])["accuracy"]
        .agg(count="count", mean="mean", std="std", min="min", max="max")
        .reset_index()
        .sort_values(["modality", "side", "run_top_k"])
    )
    side_topk_trend.to_csv(out_dir / "side_topk_trend.csv", index=False)

    side_channel_diff = compute_side_channel_differences(combined)
    side_channel_diff.to_csv(out_dir / "side_channel_differences.csv", index=False)

    side_channel_summary = pd.DataFrame()
    if not side_channel_diff.empty:
        side_channel_summary = (
            side_channel_diff.groupby(["modality", "run_top_k"])
            .agg(
                n_pairs=("subject", "count"),
                mean_jaccard=("channel_jaccard", "mean"),
                mean_common_channels=("common_channel_count", "mean"),
                mean_left_minus_right=("acc_gap_left_minus_right", "mean"),
            )
            .reset_index()
            .sort_values(["modality", "run_top_k"])
        )
    side_channel_summary.to_csv(out_dir / "side_channel_diff_summary.csv", index=False)

    best_overall = (
        combined.sort_values("accuracy", ascending=False)
        .head(20)
        .reset_index(drop=True)
    )
    best_overall.to_csv(out_dir / "best_overall_top20.csv", index=False)

    idx_best_modality = combined.groupby("modality")["accuracy"].idxmax()
    best_per_modality = combined.loc[idx_best_modality].sort_values("accuracy", ascending=False)
    best_per_modality.to_csv(out_dir / "best_per_modality.csv", index=False)

    idx_best_subject_mod = combined.groupby(["subject", "modality"])["accuracy"].idxmax()
    best_per_subject_mod = (
        combined.loc[idx_best_subject_mod]
        .sort_values(["subject", "modality"]) 
        .reset_index(drop=True)
    )
    best_per_subject_mod.to_csv(out_dir / "best_per_subject_modality.csv", index=False)

    lines = [
        "# Summary Analysis Report",
        "",
        f"Scope: {title}",
        f"Total records: {len(combined)}",
        f"Modalities: {', '.join(sorted(combined['modality'].dropna().astype(str).unique()))}",
        "",
        "## Modality Mean Accuracy",
    ]
    for _, row in modality_stats.iterrows():
        lines.append(
            f"- {row['modality']}: mean={row['mean']:.4f}, std={row['std']:.4f}, "
            f"min={row['min']:.4f}, max={row['max']:.4f}, n={int(row['count'])}"
        )

    lines.append("")
    lines.append("## Best Per Modality")
    for _, row in best_per_modality.iterrows():
        lines.append(
            f"- {row['modality']}: acc={row['accuracy']:.4f}, subject={row['subject']}, "
            f"file={row['file']}, batch={row['run_batch']}, epochs={row['run_epochs']}, top_k={row['run_top_k']}"
        )

    if not side_stats.empty:
        lines.append("")
        lines.append("## Side Mean Accuracy")
        for _, row in side_stats.iterrows():
            lines.append(
                f"- {row['modality']} side={row['side']}: mean={row['mean']:.4f}, "
                f"std={row['std']:.4f}, n={int(row['count'])}"
            )

    if not side_channel_summary.empty:
        lines.append("")
        lines.append("## Side Channel Difference (By top_k)")
        best_overlap = side_channel_summary.sort_values("mean_jaccard", ascending=False).head(3)
        for _, row in best_overlap.iterrows():
            lines.append(
                f"- {row['modality']} top_k={int(row['run_top_k'])}: "
                f"jaccard={row['mean_jaccard']:.4f}, common={row['mean_common_channels']:.2f}, "
                f"left-right acc gap={row['mean_left_minus_right']:.4f}, pairs={int(row['n_pairs'])}"
            )

    report_path = out_dir / "report_summary.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    if make_plots:
        plot_subject_channel_performance(combined, out_dir)
        plot_side_channel_performance(combined, out_dir)


def generate_outputs(df: pd.DataFrame, out_dir: Path, make_plots: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Overall bundle keeps backward compatibility with existing output names.
    write_stats_bundle(df.copy(), out_dir, make_plots=make_plots, title="all epochs")

    # Epoch-level split to compare epoch=100 and epoch=300 (or other values) separately.
    epoch_values = sorted(pd.to_numeric(df["run_epochs"], errors="coerce").dropna().unique())
    if epoch_values:
        epoch_overview = (
            df.groupby(["run_epochs", "modality"])["accuracy"]
            .agg(count="count", mean="mean", std="std", min="min", max="max")
            .reset_index()
            .sort_values(["run_epochs", "modality"])
        )
        epoch_overview.to_csv(out_dir / "epoch_modality_stats.csv", index=False)

    for epoch in epoch_values:
        epoch_int = int(epoch)
        epoch_df = df[pd.to_numeric(df["run_epochs"], errors="coerce") == epoch].copy()
        if epoch_df.empty:
            continue
        epoch_dir = out_dir / f"epochs_{epoch_int}"
        write_stats_bundle(
            epoch_df,
            epoch_dir,
            make_plots=make_plots,
            title=f"run_epochs={epoch_int}",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze summary CSV results across modalities.")
    parser.add_argument("--root", type=str, default=".", help="Project root path")
    parser.add_argument("--out", type=str, default="analysis", help="Output directory name")
    parser.add_argument("--eeg-dir", type=str, default="ResEEG", help="EEG summary directory")
    parser.add_argument("--fnirs-dir", type=str, default="ResfNIRS", help="fNIRS summary directory")
    parser.add_argument("--fusion-dir", type=str, default="ResFusion", help="Fusion summary directory")
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable automatic plot generation",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / args.out

    modality_map = {
        "EEG": args.eeg_dir,
        "fNIRS": args.fnirs_dir,
        "Fusion": args.fusion_dir,
    }

    combined = load_summaries(root, modality_map)
    if combined.empty:
        print("No valid summary rows found. Nothing to write.")
        return

    generate_outputs(combined, out_dir, make_plots=not args.no_plots)
    print(f"Analysis completed. Results written to: {out_dir}")


if __name__ == "__main__":
    main()
