# Replication materials

This folder reproduces every numerical table, empirical comparison, and
data-driven figure in the manuscript. All commands below are run **from this
`replication` folder**, not from its parent directory.

## Review installation

The submission archive contains both this folder and the package source
distribution. Create an isolated environment and install the exact submitted
software archive first:

```bash
cd replication
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --no-deps ../software/stsckm-2.1.0.tar.gz
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`. The
`stsckm==2.1.0` line in `requirements.txt` is satisfied by the installed local
archive. After publication, the same environment can be created directly from
PyPI with `python -m pip install -r requirements.txt`.

## Complete replication

Run the commented standalone entry point:

```bash
python run_all.py
```

It performs the following tasks:

1. executes the code listings imported by the manuscript;
2. executes `python examples/run_example.py` through the same public API;
3. reproduces the spatial-penalty sensitivity table;
4. compares ST-SCKM with unconstrained K-means and graph-constrained Ward
   clustering on the bundled illustration;
5. applies the general graph-regularized estimator to an archived USGS
   earthquake catalog and compares it with K-means and graph-constrained Ward;
6. evaluates KNN, radius, custom, and layered graph interfaces;
7. assesses repeated-fit stability across five fixed random seeds;
8. checks sequential and synchronous updates under four fixed row
   permutations;
9. writes fitted-state, centroid-cost, and parameter-grid tables;
10. runs a small three-repeat scaling benchmark whose timings are recorded but
   not equality-checked;
11. compares the three empirical partitions with pairwise adjusted Rand
    indices;
12. executes `worked_analysis.py`, the complete end-to-end manuscript example;
13. recreates all data-driven manuscript figures; and
14. writes interpreter and dependency versions to `output/session_info.txt`.

The script checks platform-invariant numerical CSV files against the archived
files in `expected/` and exits with an error if the values differ beyond the
recorded floating-point tolerance. Wall time is regenerated but is not checked
for equality because it depends on the review computer. A complete run takes
less than a few minutes on a regular laptop.

The introductory example can also be run independently, from this folder:

```bash
python examples/run_example.py
```

## Folder contents

- `run_all.py`: single complete replication entry point.
- `manuscript_examples.py`: exact executable counterparts of manuscript code.
- `worked_analysis.py`: complete preparation, fit, diagnosis, screening, and
  output workflow printed in the manuscript.
- `examples/run_example.py`: independently runnable introductory example.
- `requirements.txt`: exact package versions for the archived results.
- `expected/`: archived numerical results used for verification.
- `output/`: regenerated tables, figures, predictions, log, and session data.

The principal output files are `parameter_search.csv`,
`fitted_summary.csv`, `transform_example.csv`, `method_agreement.csv`,
`order_sensitivity.csv`, `earthquake_comparison.csv`,
`scaling_benchmark.csv`, and the PDF and PNG figures. `run_all.log` prints all
checked tables and the active environment.

The wildfire illustration is synthetic. The second illustration is an archived
924-event response from the USGS Earthquake Catalog API, with its fixed query,
retrieval date, and SHA-256 digest recorded in `data/README.md`. Both datasets
are distributed inside the `stsckm` package, so replication does not require
network access.
