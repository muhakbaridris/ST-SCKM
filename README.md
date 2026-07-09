# ST-SCKM

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Official implementation of **Spatio-Temporal Spatially Constrained K-Means (ST-SCKM)**.

ST-SCKM is a novel clustering algorithm designed for spatio-temporal data by jointly considering attribute similarity, spatial contiguity, and temporal proximity. The method is particularly suitable for environmental monitoring, wildfire risk zoning, epidemiology, transportation, and other geospatial applications.

---

## 📄 Related Publication

**Performance Evaluation of ST-DBSCAN and Spatio-Temporal Spatially Constrained K-Means (ST-SCKM) for Wildfire Risk Zoning and Resilience Analysis**

Accepted for publication in

**Journal of Safety Science and Resilience (Scopus Q1)**

DOI: *(To be added after publication)*

---

## Features

- Spatially constrained clustering
- Temporal weighting
- Graph-based neighborhood constraints
- Flexible distance metrics
- Wildfire risk zoning
- Spatio-temporal geospatial analysis

---

## Repository Structure

```
ST-SCKM/
│
├── data/
├── src/
├── notebooks/
├── figures/
├── docs/
├── examples/
├── README.md
├── LICENSE
├── CITATION.cff
└── requirements.txt
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/muhakbaridris/ST-SCKM.git
cd ST-SCKM
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Example

```python
from stsckm import STSCKM

model = STSCKM(
    n_clusters=10,
    lambda_spatial=0.3,
    lambda_temporal=0.2
)

model.fit(X)

labels = model.labels_
```

---

## Applications

- Wildfire Risk Mapping
- Environmental Monitoring
- Disease Surveillance
- Urban Analytics
- Remote Sensing
- GeoAI
- Spatial Statistics

---

## Citation

If you use ST-SCKM in your research, please cite:

*(Citation will be updated after publication.)*

---

## License

MIT License

---

## Contact

Muh. Akbar Idris

School of Data Science, Mathematics and Informatics

IPB University

Email:
muhakbaridris@apps.ipb.ac.id
