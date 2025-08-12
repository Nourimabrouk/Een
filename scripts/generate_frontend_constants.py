"""
Generate website JS constants from core/constants.py

Writes website/js/unity-constants.js and website/js/unity-invariants.js.
Run: python scripts/generate_frontend_constants.py
"""

from __future__ import annotations

import json
from pathlib import Path

from src.core.constants import get_constants_dict


JS_HEADER = "// Auto-generated from core/constants.py — do not edit manually\n"


def to_js_constants(constants: dict) -> str:
    lines = [JS_HEADER, "export const UNITY_CONSTANTS = Object.freeze({"]
    for k, v in constants.items():
        if isinstance(v, str):
            value = json.dumps(v)
        else:
            value = str(v).lower() if isinstance(v, bool) else repr(v)
        lines.append(f"  {k}: {value},")
    lines.append("});\n")
    return "\n".join(lines)


def to_js_invariants(constants: dict) -> str:
    content = [JS_HEADER]
    content.append("import { UNITY_CONSTANTS } from './unity-constants.js';")
    content.append(
        (
            "\nexport function checkPhiIdentity(tol = 1e-12) {\n"
            "  const { PHI } = UNITY_CONSTANTS;\n"
            "  return Math.abs(PHI * PHI - (PHI + 1)) < tol;\n}\n"
        )
    )
    content.append("export function getUnityConstants() { return UNITY_CONSTANTS; }\n")
    return "\n".join(content)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    website_js = repo_root / "website" / "js"
    website_js.mkdir(parents=True, exist_ok=True)

    constants = get_constants_dict()
    constants_js = to_js_constants(constants)
    (website_js / "unity-constants.js").write_text(constants_js, encoding="utf-8")
    invariants_js = to_js_invariants(constants)
    (website_js / "unity-invariants.js").write_text(
        invariants_js,
        encoding="utf-8",
    )
    print("Generated website/js/unity-constants.js and unity-invariants.js")


if __name__ == "__main__":
    main()
