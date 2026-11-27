"""Prediction figures directly verifiable from the saved CSV."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_prediction_file(csv_file):
    path = Path(csv_file)
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return
    x = [float(r["observed_cycle"]) for r in rows]
    specs = [
        ("true_next_soh_pct", "pred_next_soh_pct", "Next-cycle SOH (%)", "soh_prediction.png"),
        ("true_rul_cycles", "pred_rul_cycles_raw", "RUL (discharge cycles)", "rul_prediction.png")]
    for truth, estimate, ylabel, filename in specs:
        fig, axis = plt.subplots(figsize=(9, 4.3))
        axis.plot(x, [float(r[truth]) for r in rows], label="Observed target", lw=2)
        axis.plot(x, [float(r[estimate]) for r in rows], label="Model prediction", lw=1.5)
        axis.set_xlabel("Completed discharge cycle at prediction time")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(str(path.parent / filename), dpi=160)
        plt.close(fig)
