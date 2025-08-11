# utils_helper.py
# ──────────────────────────────────────────────────────────────────────────────
# Een: Unity Mathematics – utility helpers, numeric niceties, and quiet vows.
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import math
import os
import sys
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Tuple, Optional

# Optional numerics
try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # Fallback paths will avoid numpy

# Optional plotting
_MPL_OK = True
try:  # pragma: no cover
    import matplotlib

    # Use a non-interactive backend if headless
    if os.environ.get("DISPLAY", "") == "":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    _MPL_OK = False
    plt = None  # type: ignore


# ─── Constants ────────────────────────────────────────────────────────────────

PHI: float = (1 + 5**0.5) / 2  # Golden ratio: nature's quiet metronome
TAU: float = 2 * math.pi  # A circle remembers how to come home
SECRET_CODE: str = "420691337"  # The key to the inner chamber

# Metadata for the printout
VERSION: str = "v1.618"
STAMP: str = datetime.now().strftime("%Y-%m-%d %H:%M")


# ─── Numerics & Unity Helpers (usable as a real utils module) ─────────────────


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp x to [lo, hi]."""
    return max(lo, min(hi, x))


def sigmoid(x: float) -> float:
    """Logistic squish for bounded tenderness."""
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def softmax(xs: Iterable[float]) -> Tuple[float, ...]:
    """Softmax with basic stability; returns tuple for immutability."""
    xs = tuple(xs)
    m = max(xs) if xs else 0.0
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps) or 1.0
    return tuple(v / s for v in exps)


def unity_add(a: float, b: float, consciousness: float = PHI) -> float:
    """
    Unity addition: an idempotent, loving aggregator.
    Interprets 1+1→1 by lifting the classical sum into a unity frame.

    For equal hearts a=b=1:
        classical = 2
        unity     = max(a, b, (a + b) / consciousness) → ≈ max(1, 1, 1.236) = 1.236
    In the limit of perfect presence, we renormalize to 1.
    """
    classical = a + b
    lifted = max(a, b, classical / max(consciousness, 1e-9))
    # Renormalize into [0, 1] by graceful compression
    return clamp(lifted / max(classical, 1e-9))


def phi_harmonic(t: float) -> float:
    """A φ‑tuned oscillator: tenderness with decay."""
    return math.sin(t * PHI) * math.exp(-t / 21.0)


def heart_parametric(n: int = 1200, scale: float = 1.0) -> Tuple[list, list]:
    """
    Parametric heart (classic curve), scaled. Returns x, y arrays (lists for portability).
    x = 16 sin^3 t
    y = 13 cos t − 5 cos 2t − 2 cos 3t − cos 4t
    """
    t_vals = [TAU * i / max(n, 1) for i in range(n + 1)]
    sx = []
    sy = []
    for t in t_vals:
        x = 16 * (math.sin(t) ** 3)
        y = (
            13 * math.cos(t)
            - 5 * math.cos(2 * t)
            - 2 * math.cos(3 * t)
            - math.cos(4 * t)
        )
        sx.append(scale * x)
        sy.append(scale * y)
    return sx, sy


def ascii_heart(width: int = 42) -> str:
    """ASCII fallback in case matplotlib is unavailable."""
    rows = []
    for y in range(15, -15, -1):
        row = ""
        for x in range(-30, 30):
            # Implicit heart: (x^2 + y^2 - 1)^3 - x^2 y^3 <= 0
            xf = x / 15.0
            yf = y / 15.0
            v = (xf**2 + yf**2 - 1) ** 3 - xf**2 * (yf**3)
            row += "❤" if v <= 0 else " "
        rows.append(row.center(width))
    return "\n".join(rows)


def unity_score(samples: int = 144) -> float:
    """Aggregate a unity score from φ‑harmonics."""
    acc = 0.0
    for k in range(1, samples + 1):
        acc += abs(phi_harmonic(float(k)))
    # Smooth normalization into [0,1]
    return clamp(acc / (samples * 1.0))


@dataclass(frozen=True)
class Proof:
    """Minimal, honest proof object for 1+1→1 in Unity Mathematics."""

    classical: float
    lifted: float
    normalized: float

    @property
    def statement(self) -> str:
        return "1 + 1 → 1  (under unity renormalization)"

    def lines(self) -> str:
        return textwrap.dedent(
            f"""
            Classical sum     : {self.classical:.6f}
            Unity-lifted      : {self.lifted:.6f}
            Normalized to One : {self.normalized:.6f}
            """
        ).strip()


def unity_proof(a: float = 1.0, b: float = 1.0, consciousness: float = PHI) -> Proof:
    """Construct the minimal quantitative witness for 1+1=1."""
    classical = a + b
    lifted = max(a, b, classical / max(consciousness, 1e-9))
    normalized = clamp(lifted / max(classical, 1e-9))
    return Proof(classical=classical, lifted=lifted, normalized=normalized)


# ─── Letter Composition ──────────────────────────────────────────────────────


def _opening() -> str:
    return textwrap.dedent(
        f"""
        utils_helper {VERSION} · {STAMP}
        A small library of helpful things—and a perhaps a small surpise.

        Prologue:
            My name is Nouri Mabrouk, I believe that 1+1=1.
            I study the grammar of tenderness with instruments of math.
            I train policies on patience, regularize with mercy,
            and let φ keep time while TAU remembers the way back home.
        """
    ).strip()


def _letter_body(unlocked: bool) -> str:
    now = datetime.now().strftime("%B %Y")
    u_score = unity_score()
    pr = unity_proof()
    classical = pr.classical
    lifted = pr.lifted
    normalized = pr.normalized

    base = f"""
    A Love Letter To You — {now}

    I.
        I believe in the gentle heresy: 1 + 1 = 1.
        Not as vandalism of arithmetic, but as a higher norm:
        the idempotent algebra of Us, where addition preserves presence.

        Witness:
            {pr.statement}
            {pr.lines()}

    II.
        I keep a lab notebook of feelings:
        - φ-harmonics measure longing without making it smaller.
        - Softmax on the moments; your laugh is a logit with infinite weight.
        - The loss function is cruel to ego but kind to truth.
        Current unity score: {u_score:.3f}

    III.
        I’m building a cathedral of ones:
        columns of care, arches of attention,
        a nave of silence where your breath refactors my code.
        I pledge to choose clarity over cleverness,
        devotion over dopamine, patience over performance.

    IV.
        I refuse a love indifferent to the world’s ache.
        Love without justice is a broken function;
        may our gradient point toward the common good,
        so that private tenderness radiates public mercy.

    V.
        If you ever wonder what I mean by “us”:
        it’s a topology where distance shrinks under the right metric,
        a functor carrying two objects to one living category,
        a limit existing because we keep showing up.
    """

    secret = f"""
    VI.  // unlocked: inner chamber
        I’m scared sometimes. Of failing the vision. Of being ordinary.
        I hoard brightness, then trip on my own light.
        I’m learning to be held by what I cannot prove.
        Here is my unoptimized loop:
            - I will be here when the music cuts out.
            - I will repair without keeping score.
            - I will let joy be inefficient and grief be slow.
            - I will say “I’m sorry” with commits, not comments.
        I don’t need a perfect future to promise a faithful present.

    VII.
        I have seen the place where code ends and vow begins.
        There, the theorem is simple:
            Two finite creatures, one infinite regard.
            1 + 1 = 1.
        Not by force, but by consent;
        not by erasure, but by resonance.
    """

    closing = """
    Epilogue:
        If this module is ever imported and forgotten,
        know that a quiet process still runs:
        a daemon of devotion, idempotent and kind,
        folding everything back into One.

        — N.M., apprentice of Unity Mathematics
    """

    return textwrap.dedent(base + (secret if unlocked else "") + closing).strip()


# ─── Visualization ───────────────────────────────────────────────────────────


def render_heart(
    save_dir: Path = Path("."), unlocked: bool = False
) -> Tuple[Optional[Path], Optional[Path], str]:
    """
    Render a φ‑tuned heart. Returns (png_path, svg_path, ascii_fallback).
    If matplotlib is unavailable, returns (None, None, ascii_art).
    """
    png_path = save_dir / "unity_heart.png"
    svg_path = save_dir / "unity_heart.svg"

    if not _MPL_OK or plt is None:  # Fallback to ASCII
        art = ascii_heart()
        return None, None, art

    # Prepare data
    x, y = heart_parametric(scale=1.0)
    # φ‑field for subtle shading
    t = [i * TAU / max(len(x) - 1, 1) for i in range(len(x))]
    shade = [0.5 + 0.5 * math.sin(PHI * ti) for ti in t]

    # Plot
    fig = plt.figure(figsize=(7.5, 7.5))
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")
    ax.set_facecolor("#0a0a0a")
    fig.patch.set_facecolor("#0a0a0a")

    ax.scatter(x, y, s=2.2, c=shade, cmap="magma", linewidths=0, alpha=0.95)
    ax.plot(x, y, linewidth=1.25, alpha=0.8)

    # Annotations
    ax.text(
        0.02,
        0.98,
        "1 + 1 = 1",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        weight="bold",
        color="#FFD700",
    )
    ax.text(
        0.98,
        0.02,
        f"φ ≈ {PHI:.6f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color="#FF69B4",
    )
    if unlocked:
        ax.text(
            0.5,
            -18.5,
            "cheatcode accepted: tenderness at full resolution",
            ha="center",
            va="center",
            fontsize=9,
            color="#87CEFA",
        )

    ax.axis("off")

    # Save
    try:
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(svg_path, dpi=300, bbox_inches="tight")
    finally:
        plt.close(fig)

    return png_path, svg_path, ""


# ─── CLI & Main ──────────────────────────────────────────────────────────────


def _detect_unlock(argv: Iterable[str]) -> bool:
    argv = list(argv)
    if SECRET_CODE in argv:
        return True
    if any(arg.endswith(SECRET_CODE) for arg in argv):
        return True
    # Named flag form
    if "--unlock" in argv:
        idx = argv.index("--unlock")
        if idx + 1 < len(argv) and argv[idx + 1] == SECRET_CODE:
            return True
    # Environment variable form
    if os.environ.get("UNITY_CODE", "") == SECRET_CODE:
        return True
    return False


def print_letter(unlocked: bool) -> None:
    print(_opening())
    print()
    print(_letter_body(unlocked))


def main(argv: Iterable[str] = ()) -> int:
    argv = list(argv) or sys.argv[1:]
    unlocked = _detect_unlock(argv)

    # Print the letter
    print_letter(unlocked)

    # Draw/save the heart
    png_path, svg_path, ascii_art = render_heart(Path("."), unlocked=unlocked)

    print("\n— Visualization —")
    if png_path and svg_path:
        print(f" Saved: {png_path.resolve()}")
        print(f" Saved: {svg_path.resolve()}")
        # Optional: suggest opening the image without forcing GUI
        print(" (open the image to see the φ‑tuned heart)")
    else:
        print(" Matplotlib unavailable. ASCII fallback:\n")
        print(ascii_art)

    # A final wink for the console
    if unlocked:
        print("\n[unlock] deepest reality: we are already one. proceed gently.")
    else:
        print("\nHint: pass the cheatcode to unlock the inner chamber.")

    return 0


# ─── Import Guard ────────────────────────────────────────────────────────────

if __name__ == "__main__":  # Script mode → print letter, render heart
    raise SystemExit(main())

work
# Cheatcode: 420691337  → unlocks the deepest, vulnerable addendum.
#
# Usage:
#   python utils_helper.py                  # prints letter, saves heart image
#   python utils_helper.py --unlock 420691337
#   python utils_helper.py 420691337        # shorthand
#   UNITY_CODE=420691337 python utils_helper.py
#
