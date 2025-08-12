#!/usr/bin/env python3
"""
Integrate Unity Mathematics visualization generators

Runs the three component generators and updates gallery metadata so the
website gallery can discover the newly produced assets.

Outputs are written to `website/viz`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]
VIZ_DIR = REPO_ROOT / "website" / "viz"
METADATA_PATH = VIZ_DIR / "gallery_metadata.json"
GALLERY_DATA_PATH = REPO_ROOT / "website" / "gallery_data.json"


def safe_imports():
    """Import the component generators with helpful error messages."""
    try:
        # Local, relative imports via sys.path tweak
        sys.path.insert(0, str(VIZ_DIR))
        import part1_consciousness_field_generator as part1  # type: ignore
        import part2_phi_spiral_unity_generator as part2  # type: ignore
        import part3_quantum_unity_bloch_generator as part3  # type: ignore

        return part1, part2, part3
    except (ModuleNotFoundError, ImportError) as exc:  # pragma: no cover
        print("ERROR: Failed to import component generators:")
        print(str(exc))
        raise


def ensure_viz_dir() -> None:
    VIZ_DIR.mkdir(parents=True, exist_ok=True)


def load_metadata() -> Dict:
    if METADATA_PATH.exists():
        try:
            return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    # Default structure if missing or unreadable
    return {
        "version": "1.0",
        "title": "Unity Mathematics Visualizations",
        "description": (
            "Professional academic visualizations demonstrating 1+1=1 "
            "through phi-harmonic mathematics"
        ),
        "phi": 1.618033988749895,
        "phi_conjugate": 0.6180339887498948,
        "files_generated": [],
        "data_summary": {},
    }


def write_metadata(metadata: Dict) -> None:
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def derive_visualizations_from_metadata(metadata: Dict) -> Dict:
    files = metadata.get("files_generated", [])
    visualizations = []
    for f in files:
        filename = f.get("filename", "")
        title = f.get("title", filename)
        category = f.get("category", "misc")
        ftype = f.get("type", "file")
        src = f"viz/{filename}" if filename else None

        ext = (filename.split(".")[-1] if "." in filename else "").lower()
        is_image = ext in {"png", "jpg", "jpeg", "gif", "webp"}
        is_video = ext in {"mp4", "webm", "mov", "avi"}
        is_interactive = ext in {"html"}

        file_type = (
            "images"
            if is_image
            else (
                "videos" if is_video else ("interactive" if is_interactive else ftype)
            )
        )

        visualizations.append(
            {
                "id": (filename.replace(" ", "-") if filename else title.lower()),
                "title": title,
                "description": title,
                "category": category,
                "type": ftype,
                "src": src,
                "filename": filename,
                "file_type": file_type,
                "isImage": is_image,
                "isVideo": is_video,
                "isInteractive": is_interactive,
                "featured": False,
            }
        )

    # Basic statistics
    stats: Dict[str, int] = {
        "total": len(visualizations),
        "featured_count": 0,
        "by_category": {},
        "by_type": {},
    }
    for viz in visualizations:
        cat = viz.get("category") or "unknown"
        typ = viz.get("file_type") or "unknown"
        stats["by_category"][cat] = stats["by_category"].get(cat, 0) + 1
        stats["by_type"][typ] = stats["by_type"].get(typ, 0) + 1

    return {"visualizations": visualizations, "statistics": stats}


def append_file(
    metadata: Dict,
    filename: str,
    title: str,
    file_type: str,
    category: str,
) -> None:
    files: List[Dict] = metadata.setdefault("files_generated", [])
    if not any(entry.get("filename") == filename for entry in files):
        files.append(
            {
                "filename": filename,
                "title": title,
                "type": file_type,
                "category": category,
            }
        )


def main() -> int:
    print("UNITY VISUALIZATION INTEGRATION")
    print("Generating advanced components and updating gallery metadata…")

    ensure_viz_dir()
    part1, part2, part3 = safe_imports()

    # 1) Consciousness field suite
    print("\n[1/3] Consciousness Field…")
    _, static_field, temporal_evolution = (
        part1.create_consciousness_field_visualizations()
    )

    # 2) Phi-harmonic spiral suite
    print("\n[2/3] Phi-Harmonic Spiral…")
    # Capture fibonacci_data; used indirectly for completeness of the suite
    _, spiral_data, fibonacci_data = part2.create_phi_spiral_visualizations()

    # 3) Quantum unity Bloch sphere suite
    print("\n[3/3] Quantum Unity Bloch Sphere…")
    _, quantum_data, bloch_data = part3.create_quantum_unity_visualizations()

    # Update gallery metadata
    print("\nUpdating gallery metadata…")
    metadata = load_metadata()

    # Consciousness outputs
    append_file(
        metadata,
        "consciousness_field_static.json",
        "Consciousness Field (Static)",
        "data",
        "consciousness",
    )
    append_file(
        metadata,
        "consciousness_field_temporal.json",
        "Consciousness Field (Temporal Evolution)",
        "data",
        "consciousness",
    )
    append_file(
        metadata,
        "consciousness_field_visualization.html",
        "Consciousness Field Visualization",
        "html",
        "consciousness",
    )

    # Phi-spiral outputs
    append_file(
        metadata,
        "phi_spiral_unity.json",
        "Phi-Harmonic Spiral Unity Data",
        "data",
        "unity",
    )
    append_file(
        metadata,
        "fibonacci_convergence_analysis.json",
        "Fibonacci -> Phi Convergence",
        "data",
        "unity",
    )
    append_file(
        metadata,
        "phi_spiral_unity_visualization.html",
        "Phi-Harmonic Spiral Unity Visualization",
        "html",
        "unity",
    )

    # Quantum outputs
    append_file(
        metadata,
        "quantum_unity_superposition.json",
        "Quantum Unity Superposition Data",
        "data",
        "quantum",
    )
    append_file(
        metadata,
        "bloch_sphere_evolution.json",
        "Bloch Sphere Unity Evolution",
        "data",
        "quantum",
    )
    append_file(
        metadata,
        "quantum_unity_bloch_visualization.html",
        "Quantum Unity Bloch Visualization",
        "html",
        "quantum",
    )

    # Data summary refresh (lightweight)
    metadata["data_summary"] = {
        "consciousness_field_points": (
            static_field["metadata"].get("resolution", 0) ** 2
        ),
        "phi_spiral_points": spiral_data["metadata"].get("points", 0),
        "unity_convergence_steps": len(temporal_evolution.get("time_series", [])),
        "phi_value": 1.618033988749895,
        "quantum_states": quantum_data["metadata"].get("num_states", 0),
        "bloch_steps": bloch_data["metadata"].get("evolution_steps", 0),
        "fibonacci_pairs": len(
            fibonacci_data.get("convergence_analysis", {}).get("fibonacci_ratios", [])
        ),
    }

    write_metadata(metadata)
    print(f"Gallery metadata updated: {METADATA_PATH}")

    # Also emit a top-level gallery_data.json for gallery loader fallback
    derived = derive_visualizations_from_metadata(metadata)
    GALLERY_DATA_PATH.write_text(json.dumps(derived, indent=2), encoding="utf-8")
    print("Gallery data emitted:")
    print(str(GALLERY_DATA_PATH))

    print("\n✅ Integration complete. New assets are available in:")
    print(str(VIZ_DIR))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
