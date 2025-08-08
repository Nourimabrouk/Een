#!/usr/bin/env python3
"""
Vercel Serverless Function: /api/consciousness-field/density
Sets the frontend engine density (stateless serverless shim).
"""

import json


def handler(request, response):
    # CORS
    response["Access-Control-Allow-Origin"] = "*"
    response["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    response["Access-Control-Allow-Headers"] = "Content-Type"

    method = request.get("method")
    if method == "OPTIONS":
        return {"statusCode": 200}

    if method != "POST":
        return {"statusCode": 405, "body": json.dumps({"error": "Method not allowed"})}

    try:
        body_raw = request.get("body") or "{}"
        payload = json.loads(body_raw)
        density = float(payload.get("density", 0.0))
        density = max(0.0, min(1.0, density))

        # Echo back (frontend applies to engine state)
        return {"statusCode": 200, "body": json.dumps({"status": "success", "density": density})}
    except Exception as e:
        return {"statusCode": 400, "body": json.dumps({"error": str(e)})}


def main(request):
    return handler(request, {})
