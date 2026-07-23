"""Reproduce all compact numerical results and figures for the JSS manuscript.

Run from the package root after installing the package:

    python replication/run_all.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stsckm import (
    RISK_LABELS,
    STSCKM,
    add_default_features,
    assign_risk_labels,
    evaluate_labels,
    load_sample_wildfire,
    neighbor_disagreement,
    standardize_features,
)

plt.switch_backend("Agg")

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
EXPECTED = ROOT / "expected" / "sensitivity.csv"

RISK_COLORS = {
    "Low Risk": "#2C7BB6",
    "Moderate Risk": "#57B870",
    "High Risk": "#F2A541",
    "Extreme Risk": "#C83E4D",
}


def prepare_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load and prepare the bundled event data."""
    frame = add_default_features(load_sample_wildfire())
    spatial, _ = standardize_features(frame, ["x_proj", "y_proj"])
    temporal, _ = standardize_features(frame, ["time_days"])
    return frame, spatial, temporal


def fit_model(
    spatial: np.ndarray,
    temporal: np.ndarray,
    penalty: float,
) -> STSCKM:
    """Fit one deterministic model from the manuscript parameter grid."""
    return STSCKM(
        n_clusters=4,
        spatial_weight=0.5,
        temporal_weight=1.5,
        lambda_spatial=penalty,
        n_neighbors=5,
        n_init=10,
        random_state=42,
    ).fit(spatial, temporal)


def sensitivity_table(
    spatial: np.ndarray,
    temporal: np.ndarray,
) -> pd.DataFrame:
    """Reproduce the penalty sensitivity table."""
    evaluation = np.column_stack([spatial, temporal])
    records = []
    for penalty in (0.0, 0.25, 0.5, 1.0, 2.0):
        model = fit_model(spatial, temporal, penalty)
        metrics = evaluate_labels(evaluation, model.labels_)
        records.append(
            {
                "lambda_spatial": penalty,
                "iterations": model.n_iter_,
                "objective": model.objective_,
                "edge_disagreement": neighbor_disagreement(
                    model.labels_,
                    model.neighbors_,
                ),
                **metrics,
            }
        )
    return pd.DataFrame.from_records(records)


def check_expected(result: pd.DataFrame) -> None:
    """Fail loudly if reproduced values differ from the archived result."""
    expected = pd.read_csv(EXPECTED)
    if list(result.columns) != list(expected.columns):
        raise AssertionError("reproduced and expected columns differ")
    np.testing.assert_allclose(
        result.to_numpy(dtype=float),
        expected.to_numpy(dtype=float),
        rtol=1e-9,
        atol=1e-10,
    )


def software_figure(
    frame: pd.DataFrame,
    spatial: np.ndarray,
    temporal: np.ndarray,
) -> None:
    """Create the input, partition, and post hoc profile panels."""
    model = fit_model(spatial, temporal, 1.0)
    frame = frame.copy()
    frame["cluster"] = model.labels_
    frame["risk_zone"] = assign_risk_labels(frame, "cluster", "log_frp")

    figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.75), sharex=True, sharey=True)
    intensity = axes[0].scatter(
        frame["longitude"],
        frame["latitude"],
        c=frame["log_frp"],
        cmap="viridis",
        s=10,
        alpha=0.78,
        linewidths=0,
    )
    figure.colorbar(intensity, ax=axes[0], fraction=0.046, pad=0.03).set_label(
        "log(1 + FRP)"
    )
    axes[0].set_title("(a) Input events and intensity")

    cluster_colors = ["#38598B", "#6A4C93", "#2A9D8F", "#E76F51"]
    for cluster, color in enumerate(cluster_colors):
        subset = frame[frame["cluster"] == cluster]
        axes[1].scatter(
            subset["longitude"],
            subset["latitude"],
            s=10,
            alpha=0.78,
            linewidths=0,
            color=color,
            label=f"Cluster {cluster}",
        )
    axes[1].set_title("(b) ST-SCKM partition")
    axes[1].legend(fontsize=7, frameon=True, markerscale=1.6)

    for risk in RISK_LABELS:
        subset = frame[frame["risk_zone"] == risk]
        axes[2].scatter(
            subset["longitude"],
            subset["latitude"],
            s=10,
            alpha=0.8,
            linewidths=0,
            color=RISK_COLORS[risk],
            label=risk.replace(" Risk", ""),
        )
    axes[2].set_title("(c) Post hoc risk profile")
    axes[2].legend(fontsize=7, frameon=True, markerscale=1.6)

    for axis in axes:
        axis.set_xlabel("Longitude")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Latitude")
    figure.tight_layout()
    figure.savefig(OUTPUT / "software_illustration.pdf", bbox_inches="tight")
    figure.savefig(OUTPUT / "software_illustration.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def sensitivity_figure(result: pd.DataFrame) -> None:
    """Create the coherence-separation tuning figure."""
    agreement = 1.0 - result["edge_disagreement"]
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 3.7))
    axes[0].plot(
        result["lambda_spatial"],
        agreement,
        color="#2C7BB6",
        marker="o",
        linewidth=2,
        label="Neighbor agreement",
    )
    axes[0].plot(
        result["lambda_spatial"],
        result["silhouette"],
        color="#C83E4D",
        marker="s",
        linewidth=2,
        label="Silhouette",
    )
    axes[0].set_xlabel("Spatial penalty")
    axes[0].set_ylabel("Metric value")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("(a) Metric paths")
    axes[0].legend(fontsize=8)

    scatter = axes[1].scatter(
        agreement,
        result["silhouette"],
        c=result["lambda_spatial"],
        cmap="plasma",
        s=58,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    axes[1].plot(agreement, result["silhouette"], color="#7A7F87", linewidth=1.2)
    axes[1].set_xlabel("KNN neighbor agreement")
    axes[1].set_ylabel("Silhouette coefficient")
    axes[1].set_title("(b) Coherence-separation tradeoff")
    figure.colorbar(scatter, ax=axes[1], fraction=0.046, pad=0.04).set_label(
        "Spatial penalty"
    )
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(OUTPUT / "penalty_tradeoff.pdf", bbox_inches="tight")
    figure.savefig(OUTPUT / "penalty_tradeoff.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Run the complete compact replication."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame, spatial, temporal = prepare_data()
    result = sensitivity_table(spatial, temporal)
    check_expected(result)
    result.to_csv(OUTPUT / "sensitivity.csv", index=False)
    software_figure(frame, spatial, temporal)
    sensitivity_figure(result)
    print(result.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"Replication completed: {OUTPUT}")


if __name__ == "__main__":
    main()
