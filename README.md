# ST-SCKM

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Reference implementation of **Spatio-Temporal Spatially Constrained K-Means
(ST-SCKM)** for point-based spatio-temporal risk zoning.

ST-SCKM extends k-means with weighted spatial and temporal distances plus a
K-nearest-neighbor soft contiguity penalty. The method is designed for wildfire
risk zoning, environmental monitoring, epidemiology, transportation analytics,
and other GeoAI workflows where clusters should be interpretable as coherent
spatial management zones.

## Related Publication

**Performance Evaluation of ST-DBSCAN and Spatio-Temporal Spatially Constrained
K-Means (ST-SCKM) for Wildfire Risk Zoning and Resilience Analysis**

Accepted in **Journal of Safety Science and Resilience**.

## Repository Structure

```text
ST-SCKM/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── DESCRIPTION
├── data/
│   ├── sample_data.csv
│   └── README.md
├── src/
│   ├── __init__.py
│   ├── st_sckm.py
│   ├── utils.py
│   ├── distance.py
│   ├── graph.py
│   └── clustering.py
├── notebooks/
│   ├── Example_California.ipynb
│   └── Parameter_Tuning.ipynb
├── figures/
│   ├── workflow.png
│   └── architecture.png
├── docs/
│   └── algorithm.pdf
├── examples/
│   ├── run_example.py
│   └── output/
└── paper/
    ├── citation.bib
    └── accepted_manuscript.pdf
```

## Installation

```bash
git clone https://github.com/muhakbaridris/ST-SCKM.git
cd ST-SCKM
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

```python
import pandas as pd

from src import STSCKM
from src.clustering import assign_risk_labels
from src.utils import add_default_features, standardize_features

df = pd.read_csv("data/sample_data.csv")
df = add_default_features(df)

X_spatial, _ = standardize_features(df, ["x_proj", "y_proj"])
X_temporal, _ = standardize_features(df, ["time_days"])

model = STSCKM(
    n_clusters=4,
    spatial_weight=0.5,
    temporal_weight=1.5,
    lambda_spatial=1.0,
    n_neighbors=5,
    random_state=42,
)

df["cluster"] = model.fit_predict(X_spatial, X_temporal)
df["risk_zone"] = assign_risk_labels(df, "cluster", "log_frp")
```

Or run the bundled example:

```bash
python examples/run_example.py
```

The script writes predictions and a sample risk-zone plot to `examples/output/`.

## Algorithm

For each observation \(x_i = (p_i, z_i)\), ST-SCKM minimizes:

```text
J = sum_i [ w_s ||p_i - mu_k^p||^2 + w_t ||z_i - mu_k^z||^2 ]
    + lambda * sum_i sum_{j in N(i)} I(label_i != label_j)
```

where `N(i)` is the spatial KNN neighborhood of observation `i`. The first term
encourages spatial-temporal compactness; the second term discourages fragmented
cluster assignments among nearby points.

## Data

`data/sample_data.csv` is synthetic and included only for demonstration. The
accepted manuscript uses MODIS FIRMS active fire detections and ERA5
meteorological variables for California from 2019 to 2024.

## Citation

If you use this repository, please cite the manuscript and software metadata in
`CITATION.cff` or `paper/citation.bib`.

## License

This project is released under the MIT License.
