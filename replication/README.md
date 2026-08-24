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
python -m pip install --no-deps ../software/stsckm-2.0.0.tar.gz
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`. The
`stsckm==2.0.0` line in `requirements.txt` is satisfied by the installed local
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
5. evaluates KNN, radius, and custom graph interfaces;
6. assesses repeated-fit stability across five fixed random seeds;
7. checks sequential and synchronous updates under four fixed row
   permutations;
8. writes fitted-state, centroid-cost, and parameter-grid tables;
9. runs a small three-repeat scaling benchmark whose timings are recorded but
   not equality-checked;
10. recreates all data-driven manuscript figures; and
11. writes interpreter and dependency versions to `output/session_info.txt`.

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
- `examples/run_example.py`: independently runnable introductory example.
- `requirements.txt`: exact package versions for the archived results.
- `expected/`: archived numerical results used for verification.
- `output/`: regenerated tables, figures, predictions, log, and session data.

The principal output files are `parameter_search.csv`,
`fitted_summary.csv`, `transform_example.csv`, `order_sensitivity.csv`,
`scaling_benchmark.csv`, and the PDF and PNG figures. `run_all.log` prints all
checked tables and the active environment.

The bundled event data are synthetic and distributed inside the `stsckm`
package. No network access or external data download is required.
