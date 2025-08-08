"""Export unity proof artifacts into website/ for publication.
Run:
  python scripts/export_unity_proof_artifacts.py
"""

# pylint: disable=C0301
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEBSITE = ROOT / "website"

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.proofs.unity_master_proof import export_artifacts


def main() -> None:
    WEBSITE.mkdir(parents=True, exist_ok=True)
    html_path = WEBSITE / "unity_proof.html"
    json_path = WEBSITE / "unity_proof.json"
    export_artifacts(str(html_path), str(json_path))
    print({"html": str(html_path), "json": str(json_path)})


if __name__ == "__main__":
    main()


