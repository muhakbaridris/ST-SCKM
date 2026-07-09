````markdown
# ST-SCKM: Spatio-Temporal Spatially Constrained K-Means

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Status](https://img.shields.io/badge/Status-Accepted-success)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![GeoAI](https://img.shields.io/badge/GeoAI-Spatial%20Clustering-orange)

Official reference implementation of **Spatio-Temporal Spatially Constrained K-Means (ST-SCKM)**, a novel clustering algorithm for point-based spatio-temporal data.

ST-SCKM extends the conventional K-Means algorithm by jointly optimizing:

- attribute similarity,
- spatial proximity,
- temporal proximity, and
- graph-based spatial contiguity through a K-nearest-neighbor (KNN) soft penalty.

The resulting clusters are spatially coherent, temporally consistent, and directly interpretable as management or risk zones.

---

# Related Publication

**Performance Evaluation of ST-DBSCAN and Spatio-Temporal Spatially Constrained K-Means (ST-SCKM) for Wildfire Risk Zoning and Resilience Analysis**

✅ Accepted for publication in

**Journal of Safety Science and Resilience (Scopus Q1)**

DOI: *To be added after publication.*

---

# Applications

ST-SCKM is designed for spatio-temporal clustering problems including

- Wildfire Risk Mapping
- Environmental Monitoring
- Climate Hazard Analysis
- Disease Surveillance
- Transportation Safety
- Remote Sensing
- GeoAI
- Spatial Statistics
- Smart Cities

---

# Repository Structure

```text
ST-SCKM/
│
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── DESCRIPTION
│
├── data/
│   ├── sample_data.csv
│   └── README.md
│
├── src/
│   ├── __init__.py
│   ├── st_sckm.py
│   ├── clustering.py
│   ├── graph.py
│   ├── distance.py
│   └── utils.py
│
├── notebooks/
│   ├── Example_California.ipynb
│   └── Parameter_Tuning.ipynb
│
├── figures/
│   ├── workflow.png
│   └── architecture.png
│
├── docs/
│   └── algorithm.pdf
│
├── examples/
│   ├── run_example.py
│   └── output/
│
└── paper/
    ├── citation.bib
    └── accepted_manuscript.pdf
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/muhakbaridris/ST-SCKM.git

cd ST-SCKM
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate

Linux / macOS

```bash
source .venv/bin/activate
```

Windows

```bash
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Quick Start

```python
import pandas as pd

from src import STSCKM
from src.clustering import assign_risk_labels
from src.utils import (
    add_default_features,
    standardize_features,
)

df = pd.read_csv("data/sample_data.csv")

df = add_default_features(df)

X_spatial, _ = standardize_features(
    df,
    ["x_proj", "y_proj"]
)

X_temporal, _ = standardize_features(
    df,
    ["time_days"]
)

model = STSCKM(
    n_clusters=4,
    spatial_weight=0.5,
    temporal_weight=1.5,
    lambda_spatial=1.0,
    n_neighbors=5,
    random_state=42,
)

df["cluster"] = model.fit_predict(
    X_spatial,
    X_temporal,
)

df["risk_zone"] = assign_risk_labels(
    df,
    "cluster",
    "log_frp",
)
```

or simply run

```bash
python examples/run_example.py
```

---

# Mathematical Formulation

For each observation

\[
x_i=(p_i,z_i)
\]

ST-SCKM minimizes

\[
J=
\sum_i
\left(
w_s
\|p_i-\mu_k^p\|^2
+
w_t
\|z_i-\mu_k^z\|^2
\right)
+
\lambda
\sum_i
\sum_{j\in N(i)}
I(c_i\neq c_j)
\]

where

- \(w_s\) = spatial weight
- \(w_t\) = temporal weight
- \(N(i)\) = spatial K-nearest-neighbor graph
- \(\lambda\) = spatial contiguity penalty

The objective jointly promotes spatial compactness, temporal consistency, and geographically contiguous clusters.

---

# Data

The repository includes a synthetic dataset for demonstration purposes.

The experiments reported in the associated publication use

- NASA FIRMS active fire detections
- ERA5 meteorological variables

covering California, USA (2019–2024).

---

# Citation

If you use ST-SCKM in your research, please cite both the software and the associated publication.

Software citation:

See

```
CITATION.cff
```

BibTeX:

```
paper/citation.bib
```

---

# License

This project is distributed under the MIT License.

---

# Roadmap

Future developments include

- Graph ST-SCKM
- Adaptive ST-SCKM
- Deep ST-SCKM
- GPU acceleration
- R package
- PyPI package

---

# Author

**Muh. Akbar Idris**

School of Data Science, Mathematics and Informatics

IPB University

Indonesia

Email:

muhakbaridris@apps.ipb.ac.id

GitHub:

https://github.com/muhakbaridris
````
