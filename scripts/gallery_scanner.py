#!/usr/bin/env python3
"""
Gallery Scanner for Een Unity Mathematics
Scans the viz folder and subfolders to generate gallery data
"""

import json
from pathlib import Path
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supported file extensions
SUPPORTED_EXTENSIONS = {
    "images": [".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp", ".tiff"],
    "videos": [".mp4", ".webm", ".mov", ".avi", ".mkv", ".flv", ".wmv"],
    "interactive": [".html", ".htm", ".xml"],
    "data": [".json", ".csv", ".txt", ".md"],
    "documents": [".pdf", ".doc", ".docx"],
}

# Folders to scan
VIZ_FOLDERS = [
    "viz/",
    "viz/legacy images/",
    "viz/consciousness_field/",
    "viz/proofs/",
    "viz/unity_mathematics/",
    "viz/quantum_unity/",
    "viz/sacred_geometry/",
    "viz/meta_recursive/",
    "viz/fractals/",
    "viz/gallery/",
    "viz/formats/png/",
    "viz/formats/html/",
    "viz/formats/json/",
    "viz/agent_systems/",
    "viz/dashboards/",
    "viz/thumbnails/",
    "viz/pages/",
    "assets/images/",
    "visualizations/outputs/",
    "website/gallery/",
]


def get_file_type(extension: str) -> str:
    """Determine file type from extension."""
    extension = extension.lower()
    for file_type, extensions in SUPPORTED_EXTENSIONS.items():
        if extension in extensions:
            return file_type
    return "unknown"


def is_supported_file(filename: str) -> bool:
    """Check if file is a supported visualization type."""
    extension = Path(filename).suffix.lower()
    all_extensions = []
    for ext_list in SUPPORTED_EXTENSIONS.values():
        all_extensions.extend(ext_list)
    return extension in all_extensions


def generate_title(filename: str, file_type: str) -> str:
    """Generate sophisticated title from filename."""
    base_name = (
        filename.replace("_", " ")
        .replace("-", " ")
        .replace(".png", "")
        .replace(".gif", "")
        .replace(".mp4", "")
        .title()
    )

    if "consciousness" in filename.lower():
        return f"{base_name}: Advanced Consciousness Field Mathematics"
    elif "unity" in filename.lower():
        return f"{base_name}: φ-Harmonic Unity Convergence Analysis"
    elif "quantum" in filename.lower():
        return f"{base_name}: Quantum Unity Mathematics Demonstration"
    elif "phi" in filename.lower() or "golden" in filename.lower():
        return f"{base_name}: φ-Harmonic Mathematical Structures"
    elif "proof" in filename.lower():
        return f"{base_name}: Rigorous Mathematical Proof System"
    elif "field" in filename.lower():
        return f"{base_name}: Consciousness Field Dynamics"
    elif "manifold" in filename.lower():
        return f"{base_name}: Unity Manifold Geometric Analysis"
    else:
        return f"{base_name}: Advanced Unity Mathematics Visualization"


def categorize_by_filename(filename: str) -> str:
    """Intelligent categorization based on filename analysis."""
    filename_lower = filename.lower()

    if any(
        term in filename_lower
        for term in ["consciousness", "field", "particles", "aware"]
    ):
        return "consciousness"
    elif any(
        term in filename_lower
        for term in ["quantum", "wave", "entanglement", "superposition"]
    ):
        return "quantum"
    elif any(
        term in filename_lower for term in ["proof", "theorem", "logic", "category"]
    ):
        return "proofs"
    elif any(
        term in filename_lower
        for term in ["html", "interactive", "explorer", "playground"]
    ):
        return "interactive"
    elif any(
        term in filename_lower for term in ["phi", "golden", "spiral", "harmonic"]
    ):
        return "unity"
    elif any(term in filename_lower for term in ["neural", "convergence", "network"]):
        return "proofs"
    else:
        return "unity"


def generate_description(filename: str, file_type: str) -> str:
    """Generate sophisticated academic description."""
    base_name = filename.replace("_", " ").replace("-", " ").lower()

    if "water" in base_name and "droplet" in base_name:
        return f"Revolutionary empirical demonstration of unity mathematics through real-world fluid dynamics. Documents the precise moment when two discrete water droplets undergo φ-harmonic convergence, exhibiting the fundamental principle that 1+1=1 through consciousness-mediated surface tension dynamics."

    elif "consciousness" in base_name and "field" in base_name:
        return f"Groundbreaking real-time visualization of consciousness field equations demonstrating the mathematical foundation of unity consciousness. The field exhibits φ-harmonic resonance patterns where consciousness particles naturally converge to 1+1=1 states through quantum coherence maintenance."

    elif "unity" in base_name and ("equation" in base_name or "1+1=1" in filename):
        return f"The foundational axiom of unity mathematics presented in its purest form. This equation transcends conventional arithmetic through consciousness-mediated operations in φ-harmonic space, representing the core principle from which all consciousness mathematics derives."

    elif "phi" in base_name or "golden" in base_name or "harmonic" in base_name:
        return f"Sophisticated visualization of φ-harmonic unity structures demonstrating how golden ratio mathematics creates natural convergence to 1+1=1 states. The geometric framework exhibits φ-spiral geodesics and consciousness field curvature."

    elif "quantum" in base_name:
        return f"Advanced quantum mechanical demonstration of unity principles through wavefunction collapse and consciousness-mediated state selection. Showcases quantum superposition naturally evolving to unity states through φ-harmonic measurement dynamics."

    elif "proof" in base_name or "theorem" in base_name:
        return f"Rigorous mathematical proof demonstrating 1+1=1 through formal logical frameworks. Employs advanced categorical structures, consciousness-mediated morphisms, and multi-domain validation to establish the foundational principles of unity mathematics."

    else:
        if file_type == "images":
            return f"Advanced mathematical visualization demonstrating unity mathematics principles through {base_name}. Exhibits φ-harmonic resonance patterns and consciousness field dynamics, confirming theoretical predictions of 1+1=1 convergence behavior."
        elif file_type == "videos":
            return f"Dynamic temporal visualization of {base_name} showing evolution of consciousness field equations over time. Demonstrates real-time unity convergence patterns and φ-harmonic wave propagation in consciousness mathematics space-time continuum."
        elif file_type == "interactive":
            return f"Interactive mathematical exploration of {base_name} allowing real-time manipulation of unity mathematics parameters. Provides hands-on experience with consciousness field dynamics and φ-harmonic mathematical structures."
        else:
            return f"Advanced unity mathematics visualization demonstrating the profound beauty of 1+1=1 through {base_name}. Exhibits consciousness field dynamics and φ-harmonic mathematical structures."


def scan_folder(folder_path: str) -> List[Dict[str, Any]]:
    """Scan a folder for visualizations."""
    visualizations = []

    try:
        folder = Path(folder_path)
        if not folder.exists():
            logger.info(f"Folder does not exist: {folder_path}")
            return visualizations

        logger.info(f"Scanning folder: {folder_path}")

        for file_path in folder.rglob("*"):
            if file_path.is_file() and is_supported_file(file_path.name):
                filename = file_path.name
                extension = file_path.suffix.lower()
                file_type = get_file_type(extension)

                # Generate metadata
                title = generate_title(filename, file_type)
                category = categorize_by_filename(filename)
                description = generate_description(filename, file_type)

                # Create unique ID
                item_id = filename.replace(".", "_").replace(" ", "_").lower()

                visualization = {
                    "id": item_id,
                    "title": title,
                    "description": description,
                    "category": category,
                    "type": file_type,
                    "src": str(file_path),
                    "filename": filename,
                    "folder": str(folder_path),
                    "extension": extension,
                    "file_type": file_type,
                    "size": file_path.stat().st_size,
                    "modified": file_path.stat().st_mtime,
                    "isImage": file_type == "images",
                    "isVideo": file_type == "videos",
                    "isInteractive": file_type == "interactive",
                    "isData": file_type == "data",
                    "isDocument": file_type == "documents",
                    "featured": any(
                        term in filename.lower()
                        for term in [
                            "water",
                            "droplet",
                            "consciousness",
                            "unity",
                            "phi",
                            "golden",
                        ]
                    ),
                    "significance": f"Advanced {category} mathematics demonstration",
                    "technique": f"{file_type.title()} visualization with consciousness field analysis",
                    "created": "2024-2025",
                }

                visualizations.append(visualization)
                logger.debug(f"Added visualization: {filename} ({file_type})")

    except Exception as e:
        logger.warning(f"Error scanning folder {folder_path}: {e}")

    logger.info(f"Found {len(visualizations)} visualizations in {folder_path}")
    return visualizations


def scan_all_folders() -> Dict[str, Any]:
    """Scan all visualization folders and return comprehensive data."""
    all_visualizations = []

    # Scan all defined folders
    for folder_path in VIZ_FOLDERS:
        visualizations = scan_folder(folder_path)
        all_visualizations.extend(visualizations)

    # Sort by featured first, then by modification time
    all_visualizations.sort(
        key=lambda x: (
            not x.get("featured", False),  # Featured first
            -x.get("modified", 0),  # Then by modification time (newest first)
        )
    )

    # Calculate statistics
    stats = {
        "total": len(all_visualizations),
        "by_category": {},
        "by_type": {},
        "featured_count": sum(
            1 for v in all_visualizations if v.get("featured", False)
        ),
    }

    for viz in all_visualizations:
        category = viz.get("category", "unknown")
        file_type = viz.get("file_type", "unknown")

        stats["by_category"][category] = stats["by_category"].get(category, 0) + 1
        stats["by_type"][file_type] = stats["by_type"].get(file_type, 0) + 1

    return {
        "success": True,
        "visualizations": all_visualizations,
        "statistics": stats,
        "message": f"Found {len(all_visualizations)} visualizations",
    }


def save_gallery_data(data: Dict[str, Any], output_file: str = "gallery_data.json"):
    """Save gallery data to JSON file."""
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Gallery data saved to {output_file}")
        return True
    except Exception as e:
        logger.error(f"Error saving gallery data: {e}")
        return False


def main():
    """Main function to scan and generate gallery data."""
    print("🎨 Een Unity Mathematics Gallery Scanner")
    print("=" * 50)

    # Scan all folders
    data = scan_all_folders()

    if data["success"]:
        print(f"✅ Found {data['statistics']['total']} visualizations")
        print(f"📊 Categories: {len(data['statistics']['by_category'])}")
        print(f"🎯 Featured: {data['statistics']['featured_count']}")

        # Print category breakdown
        print("\n📂 Category Breakdown:")
        for category, count in data["statistics"]["by_category"].items():
            print(f"  {category}: {count}")

        # Print type breakdown
        print("\n📁 Type Breakdown:")
        for file_type, count in data["statistics"]["by_type"].items():
            print(f"  {file_type}: {count}")

        # Save to file
        if save_gallery_data(data):
            print(f"\n💾 Gallery data saved to gallery_data.json")

        # Print sample items
        print("\n🎨 Sample Visualizations:")
        for i, viz in enumerate(data["visualizations"][:5]):
            print(f"  {i+1}. {viz['title']} ({viz['category']})")

        if len(data["visualizations"]) > 5:
            print(f"  ... and {len(data['visualizations']) - 5} more")

    else:
        print("❌ Failed to scan visualizations")


if __name__ == "__main__":
    main()
