"""Reproduce every numerical result and data-driven figure in the manuscript.

Run this commented standalone script from the replication directory:

    python run_all.py

The script uses only the public ``stsckm`` API and writes all regenerated
artifacts to ``output/``. Numerical tables are checked against ``expected/``.
"""

from __future__ import annotations

from contextlib import redirect_stdout
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from manuscript_examples import (
    evaluate_example,
    fit_example,
    graph_examples,
    prepare_example,
    tuning_example,
)
from session_info import session_information
from sklearn.cluster import AgglomerativeClustering, KMeans

from stsckm import (
    RISK_LABELS,
    STSCKM,
    assign_risk_labels,
    evaluate_labels,
    fit_stability,
    graph_diagnostics,
    spatial_graph,
)

plt.switch_backend("Agg")

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
EXPECTED = ROOT / "expected"
RISK_COLORS = {
    "Low Risk": "#2C7BB6",
    "Moderate Risk": "#57B870",
    "High Risk": "#F2A541",
    "Extreme Risk": "#C83E4D",
}


def fit_model(X_spatial, X_temporal, penalty):
    """Fit one deterministic model from the manuscript sensitivity grid."""
    return STSCKM(
        n_clusters=4,
        spatial_weight=0.5,
        temporal_weight=1.5,
        lambda_spatial=penalty,
        graph_type="knn",
        n_neighbors=5,
        graph_symmetrize="union",
        n_init=10,
        random_state=42,
    ).fit(X_spatial, X_temporal)


def sensitivity_table(X_spatial, X_temporal):
    """Reproduce the graph-penalty sensitivity analysis."""
    evaluation = np.column_stack([X_spatial, X_temporal])
    records = []
    for penalty in (0.0, 0.25, 0.5, 1.0, 2.0):
        model = fit_model(X_spatial, X_temporal, penalty)
        records.append(
            {
                "lambda_spatial": penalty,
                "iterations": model.n_iter_,
                "objective": model.objective_,
                **evaluate_labels(evaluation, model.labels_),
                **graph_diagnostics(model.labels_, model.adjacency_),
            }
        )
    return pd.DataFrame.from_records(records)


def method_comparison(X_spatial, X_temporal):
    """Compare three estimators under a shared feature and graph definition."""
    weighted = np.column_stack(
        [X_spatial * np.sqrt(0.5), X_temporal * np.sqrt(1.5)]
    )
    adjacency = spatial_graph(
        X_spatial, graph_type="knn", n_neighbors=5, symmetrize="union"
    )

    labels = {
        "K-means": KMeans(n_clusters=4, n_init=10, random_state=42).fit_predict(
            weighted
        ),
        "Connectivity-constrained Ward": AgglomerativeClustering(
            n_clusters=4,
            linkage="ward",
            connectivity=adjacency,
        ).fit_predict(weighted),
        "ST-SCKM": STSCKM(
            n_clusters=4,
            spatial_weight=0.5,
            temporal_weight=1.5,
            lambda_spatial=1.0,
            graph_symmetrize="union",
            n_neighbors=5,
            n_init=10,
            random_state=42,
        ).fit_predict(X_spatial, X_temporal),
    }

    records = []
    for method, partition in labels.items():
        records.append(
            {
                "method": method,
                "n_clusters": len(np.unique(partition)),
                **evaluate_labels(weighted, partition),
                **graph_diagnostics(partition, adjacency),
            }
        )
    return pd.DataFrame.from_records(records), labels


def graph_variant_table(X_spatial, X_temporal):
    """Exercise built-in and caller-supplied sparse graph interfaces."""
    mutual = spatial_graph(
        X_spatial, graph_type="knn", n_neighbors=5, symmetrize="mutual"
    )
    specifications = {
        "directed KNN": (
            STSCKM(graph_symmetrize="none", random_state=42),
            None,
        ),
        "union KNN": (
            STSCKM(graph_symmetrize="union", random_state=42),
            None,
        ),
        "radius": (
            STSCKM(graph_type="radius", radius=0.20, random_state=42),
            None,
        ),
        "custom mutual KNN": (STSCKM(random_state=42), mutual),
    }
    records = []
    n_samples = len(X_spatial)
    for graph_name, (estimator, adjacency) in specifications.items():
        fitted = estimator.fit(X_spatial, X_temporal, adjacency=adjacency)
        records.append(
            {
                "graph": graph_name,
                "directed_edges": fitted.adjacency_.nnz,
                "density": fitted.adjacency_.nnz / (n_samples * (n_samples - 1)),
                **fitted.graph_diagnostics_,
            }
        )
    return pd.DataFrame.from_records(records)


def stability_table(X_spatial, X_temporal):
    """Archive pairwise adjusted Rand stability across fixed seeds."""
    result = fit_stability(
        STSCKM(
            n_clusters=4,
            lambda_spatial=1.0,
            graph_symmetrize="union",
        ),
        X_spatial,
        X_temporal,
        seeds=(0, 1, 2, 3, 4),
    )
    records = [
        {"comparison": index + 1, "adjusted_rand": score}
        for index, score in enumerate(result.pairwise_adjusted_rand)
    ]
    table = pd.DataFrame.from_records(records)
    table["mean_adjusted_rand"] = result.mean_adjusted_rand
    return table


def software_figure(events, X_spatial, X_temporal):
    """Create the input, fitted partition, and post hoc profile panels."""
    model = fit_model(X_spatial, X_temporal, 1.0)
    frame = events.copy()
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
        "log(1 + intensity)"
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
    axes[2].set_title("(c) Post hoc profile")
    axes[2].legend(fontsize=7, frameon=True, markerscale=1.6)

    for axis in axes:
        axis.set_xlabel("Longitude")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Latitude")
    figure.tight_layout()
    figure.savefig(OUTPUT / "software_illustration.pdf", bbox_inches="tight")
    figure.savefig(OUTPUT / "software_illustration.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def comparison_figure(events, labels):
    """Plot the partitions from the empirical software comparison."""
    figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.7), sharex=True, sharey=True)
    for axis, (method, partition) in zip(axes, labels.items(), strict=True):
        axis.scatter(
            events["longitude"],
            events["latitude"],
            c=partition,
            cmap="tab10",
            s=10,
            alpha=0.78,
            linewidths=0,
        )
        axis.set_title(method)
        axis.set_xlabel("Longitude")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Latitude")
    figure.tight_layout()
    figure.savefig(OUTPUT / "method_comparison.pdf", bbox_inches="tight")
    figure.savefig(OUTPUT / "method_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def sensitivity_figure(result):
    """Create the graph-coherence and feature-separation tradeoff figure."""
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 3.7))
    axes[0].plot(
        result["lambda_spatial"],
        result["neighbor_agreement"],
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
    axes[0].set(xlabel="Graph penalty", ylabel="Metric value", ylim=(0, 1))
    axes[0].set_title("(a) Metric paths")
    axes[0].legend(fontsize=8)

    scatter = axes[1].scatter(
        result["neighbor_agreement"],
        result["silhouette"],
        c=result["lambda_spatial"],
        cmap="plasma",
        s=58,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    axes[1].plot(
        result["neighbor_agreement"],
        result["silhouette"],
        color="#7A7F87",
        linewidth=1.2,
    )
    axes[1].set(
        xlabel="Neighbor agreement",
        ylabel="Silhouette coefficient",
        title="(b) Coherence-separation tradeoff",
    )
    figure.colorbar(scatter, ax=axes[1], fraction=0.046, pad=0.04).set_label(
        "Graph penalty"
    )
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(OUTPUT / "penalty_tradeoff.pdf", bbox_inches="tight")
    figure.savefig(OUTPUT / "penalty_tradeoff.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def check_expected(name, result, *, tolerance=1e-8):
    """Fail if a regenerated table differs materially from its archive."""
    expected_path = EXPECTED / f"{name}.csv"
    if not expected_path.exists():
        raise FileNotFoundError(f"missing archived result: {expected_path}")
    expected = pd.read_csv(expected_path)
    if list(result.columns) != list(expected.columns):
        raise AssertionError(f"column mismatch for {name}")
    if len(result) != len(expected):
        raise AssertionError(f"row-count mismatch for {name}")
    for column in result.columns:
        if pd.api.types.is_numeric_dtype(result[column]):
            np.testing.assert_allclose(
                result[column].to_numpy(dtype=float),
                expected[column].to_numpy(dtype=float),
                rtol=tolerance,
                atol=tolerance,
                err_msg=f"numeric mismatch in {name}.{column}",
            )
        elif not result[column].astype(str).equals(expected[column].astype(str)):
            raise AssertionError(f"text mismatch in {name}.{column}")


def run_complete_replication():
    """Execute listings, tables, checks, figures, and environment capture."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    events, X_spatial, X_temporal, _, _ = prepare_example()

    # Execute the same modular examples printed in the manuscript.
    model, _ = fit_example(X_spatial, X_temporal)
    internal, graph, connectivity, profiled = evaluate_example(
        events, X_spatial, X_temporal, model
    )
    radius_model, custom_model = graph_examples(X_spatial, X_temporal)
    tuning = tuning_example(X_spatial, X_temporal)
    profiled.to_csv(OUTPUT / "example_predictions.csv", index=False)
    connectivity.to_csv(OUTPUT / "example_connectivity.csv", index=False)
    tuning.to_csv(OUTPUT / "parameter_search.csv", index=False)

    tables = {
        "sensitivity": sensitivity_table(X_spatial, X_temporal),
        "method_comparison": method_comparison(X_spatial, X_temporal)[0],
        "graph_variants": graph_variant_table(X_spatial, X_temporal),
        "stability": stability_table(X_spatial, X_temporal),
    }
    for name, table in tables.items():
        check_expected(name, table)
        table.to_csv(OUTPUT / f"{name}.csv", index=False)

    _, comparison_labels = method_comparison(X_spatial, X_temporal)
    software_figure(events, X_spatial, X_temporal)
    comparison_figure(events, comparison_labels)
    sensitivity_figure(tables["sensitivity"])

    information = session_information()
    (OUTPUT / "session_info.txt").write_text(information, encoding="utf-8")

    print("Internal metrics:", internal)
    print("Graph diagnostics:", graph)
    print("Radius graph diagnostics:", radius_model.graph_diagnostics_)
    print("Custom graph diagnostics:", custom_model.graph_diagnostics_)
    for name, table in tables.items():
        print(f"\n{name}:\n{table.to_string(index=False)}")
    print("\n" + information, end="")
    print(f"Replication completed: {OUTPUT}")


def main():
    """Write a readable log while also reporting completion to the terminal."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT / "run_all.log"
    with log_path.open("w", encoding="utf-8") as log, redirect_stdout(log):
        run_complete_replication()
    print(f"Replication completed successfully. See {log_path}")


if __name__ == "__main__":
    main()
