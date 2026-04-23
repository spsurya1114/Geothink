# visualize.py
# Run this after executing a workflow to generate a PNG map
# Usage: python visualize.py

import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

def visualize_flood_risk():
    # Try clipped first, fall back to full
    risk_path = Path("outputs/flood_risk_clipped.tif")
    if not risk_path.exists():
        risk_path = Path("outputs/flood_risk.tif")

    if not risk_path.exists():
        print("No flood risk file found. Run the pipeline first.")
        return

    print(f"Loading: {risk_path}")

    with rasterio.open(risk_path) as src:
        risk = src.read(1)
        bounds = src.bounds

    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Color map: 0=nodata(white), 1=low(green), 2=medium(yellow), 3=high(red)
    colors = ["white", "green", "yellow", "red"]
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(colors)

    img = ax.imshow(
        risk,
        cmap=cmap,
        vmin=0,
        vmax=3,
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top]
    )

    # Legend
    patches = [
        mpatches.Patch(color="red",    label="High risk   (< 75m elevation)"),
        mpatches.Patch(color="yellow", label="Medium risk (75–100m elevation)"),
        mpatches.Patch(color="green",  label="Low risk    (> 100m elevation)"),
    ]
    ax.legend(handles=patches, loc="lower left", fontsize=11)

    # Labels
    ax.set_title(
        "GeoThink — Flood Risk Map: Tiruchirappalli (Trichy), Tamil Nadu",
        fontsize=14, fontweight="bold", pad=15
    )
    ax.set_xlabel("Longitude (UTM Zone 44N)", fontsize=10)
    ax.set_ylabel("Latitude (UTM Zone 44N)",  fontsize=10)

    # Stats
    high   = int((risk == 3).sum())
    medium = int((risk == 2).sum())
    low    = int((risk == 1).sum())
    total  = high + medium + low

    stats_text = (
        f"High risk:   {high:>10,} cells ({100*high/total:.1f}%)\n"
        f"Medium risk: {medium:>10,} cells ({100*medium/total:.1f}%)\n"
        f"Low risk:    {low:>10,} cells ({100*low/total:.1f}%)"
    )
    ax.text(
        0.02, 0.97, stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    plt.tight_layout()

    out_path = Path("outputs/flood_risk_map.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Map saved to: {out_path}")
    plt.show()


if __name__ == "__main__":
    visualize_flood_risk()