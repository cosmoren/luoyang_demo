"""Read Himawari (or other) NetCDF satellite .nc file and display the image."""

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset


def read_nc(file):
    nc = Dataset(file)
    print(nc)
    v = nc.variables["CLER_23"]
    raw = v[:]
    nc.close()

    print("raw min/max:", np.nanmin(raw), np.nanmax(raw))

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    im = ax.imshow(raw, vmin=0, vmax=300, cmap="viridis", aspect="auto")
    plt.colorbar(im, ax=ax, label="CLOT")
    ax.set_title("CLOT (0–150)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    file = "/home/hw1/workspace/himawari/NC_H09_20260304_0650_L2CLP010_FLDK.02401_02401.nc"
    read_nc(file)
    