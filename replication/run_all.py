"""Reproduce every numerical result and data-driven figure in the manuscript.

Run this commented standalone script from the replication directory:

    python run_all.py

The script uses only the public ``stsckm`` API and writes all regenerated
artifacts to ``output/``. Numerical tables are checked against ``expected/``.
"""

from __future__ import annotations

from contextlib import redirect_stdout
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from manuscript_examples import (
    custom_weighted_example,
    evaluate_example,
    fit_example,
    fitted_summary_example,
    general_earthquake_example,
    graph_examples,
    prepare_example,
    transform_example,
    tuning_example,
)
from session_info import session_information
from sklearn.base import clone
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score
from worked_analysis import run_workflow

from stsckm import (
    RISK_LABELS,
    STSCKM,
    GraphRegularizedKMeans,
    add_point_event_features,
    assign_risk_labels,
    combine_adjacencies,
    evaluate_labels,
    fit_stability,
    graph_diagnostics,
    load_sample_earthquakes,
    spatial_graph,
    standardize_features,
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


def method_agreement_table(labels):
    """Compare partitions with a label-invariant pairwise index."""
    methods = list(labels)
    records = []
    for left_index, left in enumerate(methods):
        for right in methods[left_index + 1 :]:
            records.append(
                {
                    "method_1": left,
                    "method_2": right,
                    "adjusted_rand": adjusted_rand_score(labels[left], labels[right]),
                }
            )
    return pd.DataFrame.from_records(records)


def earthquake_comparison():
    """Compare three formulations on a non-wildfire event catalog."""
    events = add_point_event_features(
        load_sample_earthquakes(),
        time_column="time",
        intensity_column="mag",
    )
    spatial, _ = standardize_features(events, ["x_proj", "y_proj"])
    temporal, _ = standardize_features(events, ["time_days"])
    features, _ = standardize_features(events, ["mag", "depth", "time_days"])
    feature_weights = np.array([1.0, 0.5, 0.5])
    weighted_features = features * np.sqrt(feature_weights)

    spatial_layer = spatial_graph(spatial, n_neighbors=6, symmetrize="union")
    temporal_layer = spatial_graph(temporal, n_neighbors=2, symmetrize="union")
    adjacency = combine_adjacencies(
        [spatial_layer, temporal_layer],
        weights=[0.8, 0.2],
        normalize="max",
        symmetrize="union",
    )

    labels = {
        "K-means": KMeans(n_clusters=5, n_init=10, random_state=42).fit_predict(
            weighted_features
        ),
        "Connectivity-constrained Ward": AgglomerativeClustering(
            n_clusters=5,
            linkage="ward",
            connectivity=adjacency,
        ).fit_predict(weighted_features),
        "Graph-regularized K-means": GraphRegularizedKMeans(
            n_clusters=5,
            graph_penalty=0.75,
            feature_weights=feature_weights.tolist(),
            graph_symmetrize="union",
            random_state=42,
        ).fit_predict(features, adjacency=adjacency),
    }

    records = []
    for method, partition in labels.items():
        records.append(
            {
                "method": method,
                "n_clusters": len(np.unique(partition)),
                **evaluate_labels(weighted_features, partition),
                **graph_diagnostics(partition, adjacency),
            }
        )
    table = pd.DataFrame.from_records(records)
    return events, table, labels


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


def order_sensitivity_table(X_spatial, X_temporal):
    """Compare row permutations for sequential and synchronous updates."""
    rng = np.random.default_rng(2026)
    selected = np.sort(rng.choice(len(X_spatial), size=400, replace=False))
    spatial = X_spatial[selected]
    temporal = X_temporal[selected]
    records = []
    for scheme in ("sequential", "synchronous"):
        estimator = STSCKM(
            n_clusters=4,
            spatial_weight=0.5,
            temporal_weight=1.5,
            lambda_spatial=1.0,
            graph_symmetrize="union",
            update_scheme=scheme,
            n_init=10,
            random_state=42,
        )
        reference = clone(estimator).fit(spatial, temporal)
        reference_labels = reference.labels_.copy()
        reference_graph = reference.adjacency_.copy()
        for permutation_id in range(1, 5):
            permutation = np.random.default_rng(100 + permutation_id).permutation(400)
            fitted = clone(estimator).fit(spatial[permutation], temporal[permutation])
            restored = np.empty(400, dtype=int)
            restored[permutation] = fitted.labels_
            diagnostics = graph_diagnostics(restored, reference_graph)
            records.append(
                {
                    "update_scheme": scheme,
                    "permutation": permutation_id,
                    "adjusted_rand_to_reference": adjusted_rand_score(
                        reference_labels, restored
                    ),
                    "neighbor_agreement": diagnostics["neighbor_agreement"],
                    "n_components_total": diagnostics["n_components_total"],
                }
            )
    return pd.DataFrame.from_records(records)


def scaling_benchmark(X_spatial, X_temporal):
    """Record illustrative wall times and sparse graph sizes."""
    records = []
    for n_samples in (200, 400, 800, 1200):
        indices = np.linspace(0, len(X_spatial) - 1, n_samples, dtype=int)
        spatial = X_spatial[indices]
        temporal = X_temporal[indices]
        timings = []
        fitted = None
        for _ in range(3):
            start = perf_counter()
            fitted = STSCKM(
                n_clusters=4,
                spatial_weight=0.5,
                temporal_weight=1.5,
                lambda_spatial=1.0,
                n_neighbors=5,
                graph_symmetrize="union",
                n_init=1,
                max_iter=10,
                random_state=42,
            ).fit(spatial, temporal)
            timings.append(perf_counter() - start)
        records.append(
            {
                "n_samples": n_samples,
                "directed_edges": fitted.adjacency_.nnz,
                "graph_megabytes": (
                    fitted.adjacency_.data.nbytes
                    + fitted.adjacency_.indices.nbytes
                    + fitted.adjacency_.indptr.nbytes
                )
                / 1_000_000,
                "median_seconds": float(np.median(timings)),
                "n_iter": fitted.n_iter_,
            }
        )
    return pd.DataFrame.from_records(records)


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


def earthquake_figure(events, labels):
    """Plot the non-wildfire comparison under a shared event catalog."""
    methods = list(labels)
    figure, axes = plt.subplots(
        1,
        len(methods),
        figsize=(12.6, 3.8),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    for axis, method in zip(axes, methods, strict=True):
        axis.scatter(
            events["longitude"],
            events["latitude"],
            c=labels[method],
            cmap="tab10",
            s=10,
            alpha=0.75,
            linewidths=0,
        )
        axis.set_title(method, fontsize=9)
        axis.set_xlabel("Longitude")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Latitude")
    figure.savefig(OUTPUT / "earthquake_comparison.pdf", bbox_inches="tight")
    figure.savefig(OUTPUT / "earthquake_comparison.png", dpi=300, bbox_inches="tight")
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


def tuning_figure(result):
    """Plot two diagnostics over the K and graph-penalty grid."""
    penalties = sorted(result["lambda_spatial"].unique())
    clusters = sorted(result["n_clusters"].unique())
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 3.45), constrained_layout=True)
    for axis, metric, title in (
        (axes[0], "silhouette", "(a) Feature-space silhouette"),
        (axes[1], "neighbor_agreement", "(b) Neighbor agreement"),
    ):
        matrix = (
            result.pivot(index="n_clusters", columns="lambda_spatial", values=metric)
            .loc[clusters, penalties]
            .to_numpy()
        )
        image = axis.imshow(matrix, cmap="viridis", aspect="auto", vmin=0, vmax=1)
        axis.set_xticks(range(len(penalties)), [str(value) for value in penalties])
        axis.set_yticks(range(len(clusters)), [str(value) for value in clusters])
        axis.set(xlabel="Graph penalty", ylabel="Number of clusters", title=title)
        for row in range(len(clusters)):
            for column in range(len(penalties)):
                color = "white" if matrix[row, column] < 0.55 else "black"
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:.3f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=8,
                )
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.savefig(OUTPUT / "tuning_heatmap.pdf", bbox_inches="tight")
    figure.savefig(OUTPUT / "tuning_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def custom_graph_figure():
    """Visualize supplied and regularization graphs for the worked example."""
    spatial, _, adjacency, model = custom_weighted_example()
    regularization = model.regularization_adjacency_.toarray()
    figure, axes = plt.subplots(1, 3, figsize=(10.8, 3.2), constrained_layout=True)

    def draw_graph(axis, matrix, directed):
        for i, j in zip(*np.nonzero(matrix), strict=True):
            if not directed and j <= i:
                continue
            start, end = spatial[i], spatial[j]
            if directed:
                axis.annotate(
                    "",
                    xy=end,
                    xytext=start,
                    arrowprops={"arrowstyle": "->", "color": "#7A7F87", "lw": 1.2},
                )
            else:
                axis.plot([start[0], end[0]], [start[1], end[1]], color="#7A7F87", lw=1.2)
            midpoint = (start + end) / 2
            axis.text(
                midpoint[0],
                midpoint[1] + 0.035,
                f"{matrix[i, j]:.2g}",
                fontsize=7,
                ha="center",
            )
        axis.scatter(
            spatial[:, 0],
            spatial[:, 1],
            s=52,
            color="#2C7BB6",
            edgecolor="white",
            zorder=3,
        )
        for index, point in enumerate(spatial):
            axis.text(point[0], point[1] - 0.07, str(index), ha="center", fontsize=8)
        axis.set(xlim=(-0.15, 2.35), ylim=(-0.12, 0.22), yticks=[])
        axis.grid(alpha=0.2)

    draw_graph(axes[0], adjacency, True)
    axes[0].set_title("(a) Supplied directed weights")
    draw_graph(axes[1], regularization, False)
    axes[1].set_title("(b) Symmetric penalty weights")
    axes[2].scatter(
        spatial[:, 0], spatial[:, 1], c=model.labels_, cmap="tab10", s=72, edgecolor="white"
    )
    for index, point in enumerate(spatial):
        axes[2].text(point[0], point[1] - 0.07, str(index), ha="center", fontsize=8)
    axes[2].set(xlim=(-0.15, 2.35), ylim=(-0.12, 0.22), yticks=[])
    axes[2].grid(alpha=0.2)
    axes[2].set_title("(c) Fitted labels")
    for axis in axes:
        axis.set_xlabel("Spatial coordinate 1")
    figure.savefig(OUTPUT / "custom_weighted_graph.pdf", bbox_inches="tight")
    figure.savefig(OUTPUT / "custom_weighted_graph.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def scaling_figure(result):
    """Plot illustrative fit time and sparse graph memory."""
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), constrained_layout=True)
    axes[0].plot(
        result["n_samples"],
        result["median_seconds"],
        marker="o",
        color="#38598B",
        linewidth=2,
    )
    axes[0].set(
        xlabel="Number of observations",
        ylabel="Median fit time (seconds)",
        title="(a) End-to-end fit time",
    )
    axes[1].plot(
        result["n_samples"],
        result["graph_megabytes"],
        marker="s",
        color="#2A9D8F",
        linewidth=2,
    )
    axes[1].set(
        xlabel="Number of observations",
        ylabel="CSR graph storage (MB)",
        title="(b) Stored adjacency",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.savefig(OUTPUT / "scaling_benchmark.pdf", bbox_inches="tight")
    figure.savefig(OUTPUT / "scaling_benchmark.png", dpi=300, bbox_inches="tight")
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
    fitted_summary = fitted_summary_example(events, model)
    transformed = transform_example(X_spatial, X_temporal, model)
    general_earthquake_example()
    profiled.to_csv(OUTPUT / "example_predictions.csv", index=False)
    connectivity.to_csv(OUTPUT / "example_connectivity.csv", index=False)
    tuning.to_csv(OUTPUT / "parameter_search.csv", index=False)
    fitted_summary.to_csv(OUTPUT / "fitted_summary.csv", index=False)
    transformed.to_csv(OUTPUT / "transform_example.csv", index=False)

    earthquake_events, earthquake_table, earthquake_labels = earthquake_comparison()
    tables = {
        "sensitivity": sensitivity_table(X_spatial, X_temporal),
        "method_comparison": method_comparison(X_spatial, X_temporal)[0],
        "method_agreement": method_agreement_table(
            method_comparison(X_spatial, X_temporal)[1]
        ),
        "graph_variants": graph_variant_table(X_spatial, X_temporal),
        "stability": stability_table(X_spatial, X_temporal),
        "order_sensitivity": order_sensitivity_table(X_spatial, X_temporal),
        "earthquake_comparison": earthquake_table,
    }
    for name, table in tables.items():
        check_expected(name, table)
        table.to_csv(OUTPUT / f"{name}.csv", index=False)

    _, comparison_labels = method_comparison(X_spatial, X_temporal)
    software_figure(events, X_spatial, X_temporal)
    comparison_figure(events, comparison_labels)
    earthquake_figure(earthquake_events, earthquake_labels)
    sensitivity_figure(tables["sensitivity"])
    tuning_figure(tuning)
    custom_graph_figure()
    benchmark = scaling_benchmark(X_spatial, X_temporal)
    benchmark.to_csv(OUTPUT / "scaling_benchmark.csv", index=False)
    scaling_figure(benchmark)
    run_workflow(OUTPUT / "worked_analysis")

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
