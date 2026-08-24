"""A complete, compact analysis using only the public stsckm interface."""

from __future__ import annotations

from pathlib import Path

from manuscript_examples import (
    evaluate_example,
    fit_example,
    fitted_summary_example,
    prepare_example,
    transform_example,
    tuning_example,
)


def run_workflow(output_dir: str | Path):
    """Fit, diagnose, screen parameters, and save auditable outputs."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    events, X_spatial, X_temporal, spatial_scaler, temporal_scaler = (
        prepare_example()
    )
    model, labels = fit_example(X_spatial, X_temporal)
    internal, graph, connectivity, profiled = evaluate_example(
        events, X_spatial, X_temporal, model
    )
    fitted_summary = fitted_summary_example(events, model)
    centroid_costs = transform_example(X_spatial, X_temporal, model)
    grid = tuning_example(X_spatial, X_temporal)

    # This transparent screen is illustrative, not a universal selection rule.
    candidates = grid.loc[
        (grid["neighbor_agreement"] >= 0.65) & (grid["silhouette"] >= 0.20)
    ].sort_values(["n_components_total", "silhouette"], ascending=[True, False])

    profiled.assign(cluster=labels).to_csv(output / "labels_and_profiles.csv", index=False)
    connectivity.to_csv(output / "connectivity.csv", index=False)
    fitted_summary.to_csv(output / "fitted_summary.csv", index=False)
    centroid_costs.to_csv(output / "centroid_costs.csv", index=False)
    grid.to_csv(output / "parameter_grid.csv", index=False)
    candidates.to_csv(output / "screened_candidates.csv", index=False)

    metadata = {
        "n_observations": len(events),
        "n_iterations": model.n_iter_,
        "objective": model.objective_,
        "spatial_scaler_mean": spatial_scaler.mean_.tolist(),
        "spatial_scaler_scale": spatial_scaler.scale_.tolist(),
        "temporal_scaler_mean": temporal_scaler.mean_.tolist(),
        "temporal_scaler_scale": temporal_scaler.scale_.tolist(),
        **internal,
        **graph,
    }
    (output / "analysis_summary.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in metadata.items()) + "\n",
        encoding="utf-8",
    )
    return model, candidates, metadata


def main() -> None:
    """Run the worked analysis from the replication folder."""
    model, candidates, metadata = run_workflow(Path("output") / "worked_analysis")
    print("Worked analysis complete")
    print(f"Iterations: {model.n_iter_}")
    print(f"Neighbor agreement: {metadata['neighbor_agreement']:.3f}")
    print(candidates[["n_clusters", "lambda_spatial", "silhouette", "neighbor_agreement"]])


if __name__ == "__main__":
    main()
