"""Run the complete introductory example from inside ``replication/``.

Usage
-----
python examples/run_example.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPLICATION = Path(__file__).resolve().parents[1]
if str(REPLICATION) not in sys.path:
    sys.path.insert(0, str(REPLICATION))

from manuscript_examples import (  # noqa: E402
    evaluate_example,
    fit_example,
    prepare_example,
)


def main():
    """Fit, summarize, and plot the documented introductory example."""
    output = REPLICATION / "output"
    output.mkdir(parents=True, exist_ok=True)

    events, X_spatial, X_temporal, _, _ = prepare_example()
    model, labels = fit_example(X_spatial, X_temporal)
    internal, graph, connectivity, profiled = evaluate_example(
        events, X_spatial, X_temporal, model
    )

    profiled.to_csv(output / "example_predictions.csv", index=False)
    connectivity.to_csv(output / "example_connectivity.csv", index=False)

    figure, axis = plt.subplots(figsize=(6.4, 4.8))
    scatter = axis.scatter(
        profiled["longitude"],
        profiled["latitude"],
        c=labels,
        cmap="tab10",
        s=12,
        alpha=0.8,
        linewidths=0,
    )
    axis.set(xlabel="Longitude", ylabel="Latitude", title="ST-SCKM partition")
    axis.grid(alpha=0.2)
    figure.colorbar(scatter, ax=axis, label="Cluster")
    figure.tight_layout()
    figure.savefig(output / "run_example.png", dpi=200)
    plt.close(figure)

    print("Internal metrics:", internal)
    print("Graph diagnostics:", graph)
    print(f"Outputs written to {output}")


if __name__ == "__main__":
    main()
