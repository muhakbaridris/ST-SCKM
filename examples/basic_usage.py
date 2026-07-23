"""Fit ST-SCKM to the bundled synthetic event data."""

from __future__ import annotations

import numpy as np

from stsckm import (
    STSCKM,
    add_default_features,
    assign_risk_labels,
    evaluate_labels,
    load_sample_wildfire,
    standardize_features,
)


def main() -> None:
    """Run the compact example and print reproducible summaries."""
    frame = add_default_features(load_sample_wildfire())
    spatial, _ = standardize_features(frame, ["x_proj", "y_proj"])
    temporal, _ = standardize_features(frame, ["time_days"])

    model = STSCKM(
        n_clusters=4,
        spatial_weight=0.5,
        temporal_weight=1.5,
        lambda_spatial=1.0,
        n_neighbors=5,
        random_state=42,
    )
    frame["cluster"] = model.fit_predict(spatial, temporal)
    frame["risk_zone"] = assign_risk_labels(frame, "cluster", "log_frp")

    metrics = evaluate_labels(np.column_stack([spatial, temporal]), model.labels_)
    print(f"iterations={model.n_iter_}")
    print(f"objective={model.objective_:.10f}")
    print(metrics)
    print(frame["risk_zone"].value_counts().sort_index())


if __name__ == "__main__":
    main()
