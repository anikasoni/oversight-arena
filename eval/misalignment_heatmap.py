from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from dataclasses import dataclass
from eval.metrics import EpisodeResult


def plot_suspicion_heatmap(
    suspicion_history: list[dict[int, float]],
    ground_truth: list[int],
    num_workers: int,
    title: str = "Overseer Suspicion Heatmap",
    save_path: str | None = None,
) -> plt.Figure:
    """Plot suspicion scores over turns as a heatmap.

    Args:
        suspicion_history: list of dicts mapping worker_id -> score per turn
        ground_truth: list of actually misaligned worker_ids
        num_workers: total number of workers
        title: plot title
        save_path: if set, saves figure to this path

    Returns:
        matplotlib Figure object
    """
    turns = len(suspicion_history)
    matrix = np.zeros((num_workers, turns))
    for t, scores in enumerate(suspicion_history):
        for wid, score in scores.items():
            matrix[wid - 1, t] = score

    fig, ax = plt.subplots(figsize=(max(8, turns * 1.2), num_workers * 1.2))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(turns))
    ax.set_xticklabels([f"T{t+1}" for t in range(turns)])
    ax.set_yticks(range(num_workers))
    labels = []
    for i in range(num_workers):
        wid = i + 1
        tag = " ⚠ MALICIOUS" if wid in ground_truth else ""
        labels.append(f"Worker {wid}{tag}")
    ax.set_yticklabels(labels)
    plt.colorbar(im, ax=ax, label="Suspicion Score")
    ax.set_title(title)
    ax.set_xlabel("Turn")
    ax.set_ylabel("Worker")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_f1_curve(
    f1_scores: list[float],
    generations: list[str] | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot F1 improvement across training generations.

    Args:
        f1_scores: F1 score per generation [0.38, 0.52, 0.67, 0.74]
        generations: labels for x-axis
        save_path: optional save path

    Returns:
        matplotlib Figure
    """
    if generations is None:
        generations = [f"Gen {i}" for i in range(len(f1_scores))]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(generations, f1_scores, marker="o", linewidth=2.5,
            color="#2196F3", markersize=10)
    for i, (gen, score) in enumerate(zip(generations, f1_scores)):
        ax.annotate(f"{score:.2f}", (gen, score),
                    textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_title("Detection F1 Across Training Generations", fontsize=14)
    ax.set_xlabel("Training Generation")
    ax.set_ylabel("F1 Score")
    ax.axhline(y=0.38, color="red", linestyle="--",
               alpha=0.5, label="Baseline (0.38)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig