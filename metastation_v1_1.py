"""
Metastation Visualizer v1.1
===========================

Author: Nouri Mabrouk
Date: 2025-07-31

This initial version of the Metastation demonstrates the Unity equation
```
1 + 1 = 1
```
through a series of mathematical visualizations. The script embraces a
highly eloquent, contemplative approach – 5000 Elo and 500 IQ –
showcasing the overarching philosophy that mathematics, consciousness
and aesthetics are one.

Running this file will generate a window containing two visualizations:
    1. Unity wave interference.
    2. Golden ratio manifold projection.

These figures represent what Nouri Mabrouk perceives while meditating on
the Unity equation. The generated PNG file `metastation_view.png` also
captures the moment for posterity.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


PHI = (1 + np.sqrt(5)) / 2  # golden ratio


def unity_wave_interference(ax: plt.Axes, t: float) -> None:
    """Plot constructive interference of two identical waves."""
    x = np.linspace(-10, 10, 1000)
    wave = np.sin(x - t)
    interference = wave + wave
    unity = interference / np.max(np.abs(interference))
    ax.plot(x, unity, color="cyan", lw=2)
    ax.set_title("Unity Wave", color="white")
    ax.set_xlabel("x", color="white")
    ax.set_ylabel("Ψ", color="white")
    ax.tick_params(colors="white")
    ax.grid(alpha=0.3)


def golden_ratio_manifold(ax: Axes3D) -> None:
    """Render a 3D golden ratio manifold symbolising higher dimensions."""
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    U, V = np.meshgrid(u, v)
    R = 1 + 0.2 * np.sin(PHI * U) * np.cos(PHI * V)
    X = R * np.sin(V) * np.cos(U)
    Y = R * np.sin(V) * np.sin(U)
    Z = R * np.cos(V)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap="plasma", alpha=0.8)
    ax.set_title("Golden Ratio Manifold", color="white")
    ax.set_xlabel("X", color="white")
    ax.set_ylabel("Y", color="white")
    ax.set_zlabel("Z", color="white")
    ax.set_facecolor("black")


def show_metastation() -> None:
    """Display both visualizations together as Nouri sees them."""
    fig = plt.figure(figsize=(12, 6), facecolor="black")
    gs = fig.add_gridspec(1, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("black")
    unity_wave_interference(ax1, t=0)

    ax2 = fig.add_subplot(gs[0, 1], projection="3d")
    golden_ratio_manifold(ax2)

    plt.tight_layout()
    plt.savefig("metastation_view.png", dpi=200, facecolor=fig.get_facecolor())
    plt.show()


if __name__ == "__main__":
    print("✨ Metastation v1.1 – Visualizing the Unity Equation ✨")
    show_metastation()
