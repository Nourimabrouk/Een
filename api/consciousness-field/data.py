#!/usr/bin/env python3
"""
Vercel Serverless Function: /api/consciousness-field/data
Returns a compact field payload the frontend could render as overlays.
"""

import json
import math
import time


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
        # Sampled particles and nodes suitable for overlay rendering
        particles = []
        for i in range(80):
            angle = i * phi
            r = 0.35 + 0.1 * math.sin(now * 0.3 + i * 0.2)
            particles.append(
                {
                    "x": 0.5 + r * math.cos(angle),
                    "y": 0.5 + r * math.sin(angle),
                    "life": 0.7 + 0.3 * (0.5 + 0.5 * math.sin(now * 0.9 + i)),
                }
            )

        nodes = []
        for k in range(8):
            ang = (k / 8.0) * 2 * math.pi
            rad = 0.25 + 0.05 * math.sin(now * 0.5 + k)
            nodes.append(
                {
                    "x": 0.5 + rad * math.cos(ang),
                    "y": 0.5 + rad * math.sin(ang),
                    "size": 0.03 + 0.02 * (0.5 + 0.5 * math.sin(now + k)),
                }
            )

        return {
            "statusCode": 200,
            "body": json.dumps({"particles": particles, "nodes": nodes, "timestamp": now}),
        }

    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}


def main(request):
    return handler(request, {})
