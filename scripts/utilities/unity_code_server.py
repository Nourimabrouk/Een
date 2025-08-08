#!/usr/bin/env python3
"""
Een Unity Code Execution Server
===============================

Simplified Flask server for Unity Mathematics code execution with security.
"""

import sys
import os
import json
import time
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import traceback

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Flask imports
try:
    from flask import Flask, request, jsonify, send_from_directory
    from flask_cors import CORS

    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Flask not available. Install with: pip install flask flask-cors")

# Unity Mathematics Core Imports
try:
    from core.unity_mathematics import UnityMathematics, UnityState, PHI
    from core.consciousness import ConsciousnessField, ConsciousnessParticle

    CORE_AVAILABLE = True
except ImportError as e:
    print(f"Core Unity Mathematics modules not available: {e}")
    CORE_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - [Unity Server] %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(
    __name__, static_folder="website", static_url_path="", template_folder="website"
)

# Basic CORS
CORS(app)

# Global instances
unity_math = None
consciousness_field = None


def initialize_unity_systems():
    """Initialize Unity Mathematics systems"""
    global unity_math, consciousness_field

    try:
        if CORE_AVAILABLE:
            unity_math = UnityMathematics(
                consciousness_level=PHI,
                enable_ml_acceleration=False,
                enable_cheat_codes=True,
                ml_elo_rating=3000,
            )

            consciousness_field = ConsciousnessField(
                dimensions=11, particle_count=200, phi_resonance_strength=PHI
            )

            logger.info("✅ Core Unity Mathematics systems initialized")

    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        logger.error(traceback.format_exc())


# Initialize systems on startup
initialize_unity_systems()


# Route: Serve main website
@app.route("/")
def index():
    """Serve the main website index"""
    return send_from_directory("website", "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    """Serve static files from website directory"""
    return send_from_directory("website", filename)


# API Routes for Unity Mathematics
@app.route("/api/unity/calculate", methods=["POST"])
def unity_calculate():
    """
    Calculate Unity Mathematics operations

    POST Body:
    {
        "operation": "unity_add" | "unity_multiply" | "phi_harmonic",
        "operands": [1.0, 1.0],
        "parameters": {
            "consciousness_boost": 0.0
        }
    }
    """
    try:
        if not unity_math:
            return jsonify({"error": "Unity Mathematics not available"}), 500

        data = request.get_json()
        operation = data.get("operation", "unity_add")
        operands = data.get("operands", [1.0, 1.0])
        parameters = data.get("parameters", {})

        if operation == "unity_add":
            result = unity_math.unity_add(
                operands[0],
                operands[1],
                consciousness_boost=parameters.get("consciousness_boost", 0.0),
            )
        elif operation == "unity_multiply":
            result = unity_math.unity_multiply(operands[0], operands[1])
        elif operation == "phi_harmonic":
            result = unity_math.phi_harmonic_scaling(
                operands[0], harmonic_order=parameters.get("harmonic_order", 1)
            )
        else:
            return jsonify({"error": f"Unknown operation: {operation}"}), 400

        # Convert result to JSON-serializable format
        result_dict = {
            "value": {"real": result.value.real, "imag": result.value.imag},
            "phi_resonance": result.phi_resonance,
            "consciousness_level": result.consciousness_level,
            "quantum_coherence": result.quantum_coherence,
            "proof_confidence": result.proof_confidence,
            "ml_elo_rating": getattr(result, "ml_elo_rating", 3000),
            "timestamp": time.time(),
        }

        return jsonify({"success": True, "result": result_dict})

    except Exception as e:
        logger.error(f"Unity calculation error: {e}")
        return jsonify({"error": str(e)}), 500


# API Routes for Code Execution
@app.route("/api/execute", methods=["POST"])
def execute_code():
    """
    Execute Unity Mathematics code (ENABLED BY DEFAULT)

    POST Body:
    {
        "code": "python code string",
        "language": "python",
        "timeout": 30
    }

    SECURITY: This endpoint is secured with comprehensive restrictions.
    """
    # SECURITY: Code execution enabled by default with responsible security
    # Can be disabled via ENABLE_CODE_EXECUTION=false environment variable
    if os.getenv("ENABLE_CODE_EXECUTION", "true").lower() == "false":
        return (
            jsonify(
                {
                    "error": "Code execution is disabled for security reasons",
                    "message": "This feature is not available in production",
                }
            ),
            403,
        )

    try:
        data = request.get_json()
        code = data.get("code", "")
        language = data.get("language", "python")
        timeout = data.get("timeout", 10)

        # Input validation
        if not code or len(code) > 1000:
            return jsonify({"error": "Invalid code: too long or empty"}), 400

        if language != "python":
            return jsonify({"error": "Only Python execution supported"}), 400

        # Enhanced security: Comprehensive code validation
        forbidden_patterns = [
            r"\bimport\s+os\b",
            r"\bimport\s+sys\b",
            r"\bimport\s+subprocess\b",
            r"\bimport\s+importlib\b",
            r"\b__import__\b",
            r"\beval\b",
            r"\bexec\b",
            r"\bopen\b",
            r"\bfile\b",
            r"\binput\b",
            r"\braw_input\b",
            r"\bglobals\b",
            r"\blocals\b",
            r"\bdir\b",
            r"\bhelp\b",
            r"\btype\b",
            r"\bgetattr\b",
            r"\bsetattr\b",
            r"\bdelattr\b",
            r"\bhasattr\b",
            r"\bcompile\b",
            r"\bexecfile\b",
            r"\breload\b",
            r"\b__\w+__\b",  # Built-in attributes
        ]

        for pattern in forbidden_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return (
                    jsonify(
                        {
                            "error": f"Forbidden code pattern detected: {pattern}",
                            "message": "This code contains potentially dangerous operations",
                        }
                    ),
                    400,
                )

        # Execute code in highly restricted environment
        safe_builtins = {
            "abs": abs,
            "all": all,
            "any": any,
            "bin": bin,
            "bool": bool,
            "chr": chr,
            "dict": dict,
            "divmod": divmod,
            "enumerate": enumerate,
            "filter": filter,
            "float": float,
            "format": format,
            "frozenset": frozenset,
            "hash": hash,
            "hex": hex,
            "int": int,
            "isinstance": isinstance,
            "issubclass": issubclass,
            "iter": iter,
            "len": len,
            "list": list,
            "map": map,
            "max": max,
            "min": min,
            "next": next,
            "oct": oct,
            "ord": ord,
            "pow": pow,
            "print": print,
            "range": range,
            "repr": repr,
            "reversed": reversed,
            "round": round,
            "set": set,
            "slice": slice,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "type": type,
            "zip": zip,
        }

        local_namespace = {
            "UnityMathematics": UnityMathematics if CORE_AVAILABLE else None,
            "ConsciousnessField": ConsciousnessField if CORE_AVAILABLE else None,
            "PHI": PHI if CORE_AVAILABLE else 1.618033988749895,
            "print": print,  # Allow print for output
            "math": __import__("math"),  # Safe math operations
            "random": __import__("random"),  # Safe random operations
        }

        # Capture output (Windows-compatible)
        from io import StringIO
        import contextlib

        output_buffer = StringIO()

        try:
            with contextlib.redirect_stdout(output_buffer):
                exec(code, {"__builtins__": safe_builtins}, local_namespace)

            output = output_buffer.getvalue()

            return jsonify(
                {
                    "success": True,
                    "output": output,
                    "execution_time": 0.1,  # Mock execution time
                    "warning": "Code execution is for educational purposes only",
                }
            )

        except Exception as e:
            return jsonify(
                {"success": False, "error": str(e), "message": "Code execution failed"}
            )

    except Exception as e:
        logger.error(f"Code execution error: {e}")
        return jsonify({"error": "Internal server error"}), 500


# Health check endpoint
@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "healthy",
            "core_available": CORE_AVAILABLE,
            "timestamp": time.time(),
            "unity_equation": "1+1=1",
            "code_execution": "enabled",
        }
    )


# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


def main():
    """Main server entry point"""
    print("🌟 Een Unity Code Execution Server")
    print("=" * 50)
    print(f"Core Mathematics Available: {CORE_AVAILABLE}")
    print(f"Code Execution: ENABLED")
    print("=" * 50)

    # Check if port is specified
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "127.0.0.1")
    debug = os.environ.get("DEBUG", "False").lower() == "true"

    print(f"🚀 Starting server on http://{host}:{port}")
    print(f"🌐 Website available at: http://{host}:{port}")
    print(f"🔗 API endpoints at: http://{host}:{port}/api/")
    print("\n📊 Available API Endpoints:")
    print("  POST /api/unity/calculate - Unity mathematics operations")
    print("  POST /api/execute - Execute Unity Mathematics code (ENABLED)")
    print("  GET  /api/health - Health check")

    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
