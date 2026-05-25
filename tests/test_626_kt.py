"""Read the first CSV under pv_kt and print column keys."""

import csv
import os
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = os.path.expanduser("~/datasets/luoyang_SPMF/pv_ktuni")


def main() -> None:
    if not os.path.isdir(DATA_DIR):
        print(f"Directory not found: {DATA_DIR}")
        return

    names = sorted(n for n in os.listdir(DATA_DIR) if n.lower().endswith(".csv"))
    if not names:
        print(f"No CSV files under: {DATA_DIR}")
        return

    for name in names:
        path = os.path.join(DATA_DIR, name)
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        active_power = np.array([float(r["active_power"]) for r in rows], dtype=np.float64)
        p_cs = np.array([float(r["p_cs"]) for r in rows], dtype=np.float64)
        kt = np.array([float(r["kt"]) for r in rows], dtype=np.float64)
        kt_mask = np.array([float(r["kt_mask"]) for r in rows], dtype=np.float64)
        p_mean = np.array([float(r["p_mean"]) for r in rows], dtype=np.float64)

        plt.plot(kt[40000:40300], label="kt")
        plt.legend()
        plt.savefig("kt_mask_p_mean.png")
        break

if __name__ == "__main__":
    main()
