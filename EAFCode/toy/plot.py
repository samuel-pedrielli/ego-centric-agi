"""
Plot utilities for the toy model (reads results/nu_sweep.csv).
Generates PNGs: stability_curve.png (empirical slope vs nu).
"""

import csv, os
import matplotlib.pyplot as plt

def plot_nu_sweep(csv_path="results/nu_sweep.csv", out_path="results/stability_curve.png"):
    x, slope, pred, emp = [], [], [], []
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            x.append(float(row["nu"]))
            slope.append(float(row["emp_slope"]))
            pred.append(int(row["pred_stable"]))
            emp.append(int(row["emp_stable"]))

    plt.figure()
    plt.axhline(0.0)
    plt.plot(x, slope, marker="o")
    plt.xlabel("nu")
    plt.ylabel("slope of log V (last segment)")
    plt.title("Empirical stability (negative = contracting)")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"[plot] saved {out_path}")

if __name__ == "__main__":
    plot_nu_sweep()
