"""Run a compact ST-SCKM example on the bundled sample dataset."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import STSCKM
from src.clustering import assign_risk_labels, evaluate_labels
from src.utils import add_default_features, standardize_features


DATA_PATH = ROOT / "data" / "sample_data.csv"
OUTPUT_DIR = ROOT / "examples" / "output"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df = add_default_features(df)

    X_spatial, _ = standardize_features(df, ["x_proj", "y_proj"])
    X_temporal, _ = standardize_features(df, ["time_days"])

    model = STSCKM(
        n_clusters=4,
        spatial_weight=0.5,
        temporal_weight=1.5,
        lambda_spatial=1.0,
        n_neighbors=5,
        n_init=10,
        random_state=42,
    )
    df["cluster"] = model.fit_predict(X_spatial, X_temporal)
    df["risk_zone"] = assign_risk_labels(df, "cluster", "log_frp")

    X_eval = pd.concat(
        [
            pd.DataFrame(X_spatial, columns=["x_scaled", "y_scaled"]),
            pd.DataFrame(X_temporal, columns=["t_scaled"]),
        ],
        axis=1,
    ).to_numpy()
    metrics = evaluate_labels(X_eval, df["cluster"].to_numpy())

    result_path = OUTPUT_DIR / "sample_predictions.csv"
    df.to_csv(result_path, index=False)

    palette = {
        "Low Risk": "#1f77b4",
        "Moderate Risk": "#2ca02c",
        "High Risk": "#ff7f0e",
        "Extreme Risk": "#d62728",
    }
    fig, ax = plt.subplots(figsize=(7, 8))
    for risk, group in df.groupby("risk_zone"):
        ax.scatter(
            group["longitude"],
            group["latitude"],
            s=14,
            alpha=0.72,
            color=palette.get(risk, "gray"),
            label=risk,
        )
    ax.set_title("ST-SCKM Sample Wildfire Risk Zones")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "sample_risk_zones.png", dpi=200)

    print("ST-SCKM example complete")
    print(f"Iterations: {model.n_iter_}")
    print(f"Metrics: {metrics}")
    print(f"Predictions: {result_path}")


if __name__ == "__main__":
    main()
