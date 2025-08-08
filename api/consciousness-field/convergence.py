#!/usr/bin/env python3
"""
Vercel Serverless Function: /api/consciousness-field/convergence
Sets unity convergence (stateless serverless shim).
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
        rate = float(payload.get("rate", 0.0))
        rate = max(0.0, min(1.0, rate))

        # Echo back
        return {"statusCode": 200, "body": json.dumps({"status": "success", "rate": rate})}
    except Exception as e:
        return {"statusCode": 400, "body": json.dumps({"error": str(e)})}


def main(request):
    return handler(request, {})
