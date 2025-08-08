#!/usr/bin/env python3
"""
Vercel Serverless Function: /api/consciousness-field/metrics
Lightweight, dependency-free metrics suitable for serverless runtime.
"""

import json
import math
import time


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def handler(request, response):
    # CORS
    response["Access-Control-Allow-Origin"] = "*"
    response["Access-Control-Allow-Methods"] = "GET, OPTIONS"
    response["Access-Control-Allow-Headers"] = "Content-Type"

    if request.get("method") == "OPTIONS":
        return {"statusCode": 200}

    if request.get("method") != "GET":
        return {"statusCode": 405, "body": json.dumps({"error": "Method not allowed"})}

    try:
        phi = 1.618033988749895
        now = time.time()

        # Smooth pseudo-dynamics
        density = _clamp(0.65 + 0.2 * math.sin(now * 0.11))
        convergence = _clamp(0.8 + 0.15 * math.sin(now * 0.07 + 1.0))
        resonance = _clamp(0.12 + 0.08 * math.sin(now * phi * 0.1))
        field_strength = _clamp(0.5 * density + 0.3 * convergence + 0.2 * resonance)

        data = {
            "consciousness_density": round(density, 4),
            "unity_convergence_rate": round(convergence, 4),
            "resonance_frequency": round(resonance, 4),
            "field_strength": round(field_strength, 4),
            "timestamp": now,
        }

        return {"statusCode": 200, "body": json.dumps(data)}

    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": f"metrics error: {str(e)}"})}


def main(request):
    return handler(request, {})
