# Replication

From an installed source checkout, run:

```bash
python replication/run_all.py
```

The script:

1. loads the bundled 1,200-row synthetic dataset;
2. reproduces the five-value spatial-penalty analysis;
3. checks the table against `expected/sensitivity.csv`;
4. writes the reproduced CSV, a software-illustration figure, and a
   coherence-separation figure to `output/`.

The compact replication is designed to complete within a few minutes on a
regular laptop.

`environment.txt` records the environment used for the archived outputs.
The numerical check uses tight floating-point tolerances, so a materially
different dependency stack may produce small platform-dependent differences.
