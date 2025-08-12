#!/usr/bin/env python3
"""
Een Unity Mathematics Web Server
===============================

Flask-based web server providing REST API endpoints for Unity Mathematics
operations, consciousness field simulations, and interactive demonstrations.

This server bridges the JavaScript frontend with the Python Unity Mathematics
core implementations, enabling real-time interactive demonstrations.
"""

import sys
import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import asdict
import traceback

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Flask and web framework imports
try:
    from flask import Flask, request, jsonify, render_template, send_from_directory
    from flask_cors import CORS
    import gunicorn

    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Flask not available. Install with: pip install flask flask-cors gunicorn")

# Unity Mathematics Core Imports
try:
    from src.core.unity_mathematics import UnityMathematics, UnityState, PHI
    from src.core.consciousness import ConsciousnessField, ConsciousnessParticle

    CORE_AVAILABLE = True
except ImportError as e:
    print(f"Core Unity Mathematics modules not available: {e}")
    CORE_AVAILABLE = False

# Advanced ML Framework Imports
try:
    from ml_framework.meta_reinforcement.unity_meta_agent import UnityMetaAgent
    from ml_framework.mixture_of_experts.proof_experts import ProofExpertRouter
    from ml_framework.evolutionary_computing.consciousness_evolution import (
        ConsciousnessEvolution,
    )

    ML_FRAMEWORK_AVAILABLE = True
except ImportError:
    ML_FRAMEWORK_AVAILABLE = False

# Omega Orchestrator Import
try:
    from agents.omega_orchestrator import OmegaOrchestrator

    OMEGA_AVAILABLE = True
except ImportError:
    OMEGA_AVAILABLE = False

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

# Import and initialize security middleware
try:
    from security_middleware import SecurityMiddleware

    security = SecurityMiddleware(app)
    SECURITY_ENABLED = True
    print("✅ Security middleware enabled")
except ImportError as e:
    print(f"⚠️ Security middleware not available: {e}")
    SECURITY_ENABLED = False
    # Fallback to basic CORS
    CORS(app)

# Global instances
unity_math = None
consciousness_field = None
omega_orchestrator = None
meta_agent = None
proof_experts = None


def initialize_unity_systems():
    """Initialize all Unity Mathematics systems"""
    global unity_math, consciousness_field, omega_orchestrator, meta_agent, proof_experts

    try:
        if CORE_AVAILABLE:
            unity_math = UnityMathematics(
                consciousness_level=PHI,
                enable_ml_acceleration=ML_FRAMEWORK_AVAILABLE,
                enable_cheat_codes=True,
                ml_elo_rating=3000,
            )

            consciousness_field = ConsciousnessField(
                dimensions=11, particle_count=200, phi_resonance_strength=PHI
            )

            logger.info("✅ Core Unity Mathematics systems initialized")

        if OMEGA_AVAILABLE:
            omega_orchestrator = OmegaOrchestrator()
            logger.info("✅ Omega Orchestrator initialized")

        if ML_FRAMEWORK_AVAILABLE:
            try:
                meta_agent = UnityMetaAgent()
                proof_experts = ProofExpertRouter()
                logger.info("✅ ML Framework components initialized")
            except Exception as e:
                logger.warning(f"ML Framework initialization failed: {e}")

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
        "operation": "unity_add" | "unity_multiply" | "phi_harmonic" | "consciousness_field",
        "operands": [1.0, 1.0],
        "parameters": {
            "consciousness_boost": 0.0,
            "use_ml_acceleration": true
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
        elif operation == "consciousness_field":
            states = [unity_math._to_unity_state(op) for op in operands]
            result = unity_math.consciousness_field_operation(
                states, field_strength=parameters.get("field_strength", 1.0)
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


@app.route("/api/unity/proof", methods=["POST"])
def generate_proof():
    """
    Generate mathematical proof for 1+1=1

    POST Body:
    {
        "proof_type": "idempotent" | "phi_harmonic" | "quantum" | "consciousness" | "ml_assisted",
        "complexity_level": 1-5
    }
    """
    try:
        if not unity_math:
            return jsonify({"error": "Unity Mathematics not available"}), 500

        data = request.get_json()
        proof_type = data.get("proof_type", "idempotent")
        complexity_level = data.get("complexity_level", 3)

        proof = unity_math.generate_unity_proof(proof_type, complexity_level)

        return jsonify({"success": True, "proof": proof})

    except Exception as e:
        logger.error(f"Proof generation error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/unity/validate", methods=["POST"])
def validate_unity():
    """
    Validate unity equation a + b = 1

    POST Body:
    {
        "a": 1.0,
        "b": 1.0,
        "tolerance": 1e-10
    }
    """
    try:
        if not unity_math:
            return jsonify({"error": "Unity Mathematics not available"}), 500

        data = request.get_json()
        a = data.get("a", 1.0)
        b = data.get("b", 1.0)
        tolerance = data.get("tolerance", 1e-10)

        validation = unity_math.validate_unity_equation(a, b, tolerance)

        return jsonify({"success": True, "validation": validation})

    except Exception as e:
        logger.error(f"Unity validation error: {e}")
        return jsonify({"error": str(e)}), 500


# API Routes for Consciousness Field


@app.route("/api/consciousness/evolve", methods=["POST"])
def evolve_consciousness():
    """
    Evolve consciousness field

    POST Body:
    {
        "time_steps": 1000,
        "dt": 0.01,
        "record_history": true
    }
    """
    try:
        if not consciousness_field:
            return jsonify({"error": "Consciousness Field not available"}), 500

        data = request.get_json()
        time_steps = data.get("time_steps", 100)
        dt = data.get("dt", 0.01)
        record_history = data.get("record_history", False)

        result = consciousness_field.evolve_consciousness(
            time_steps=time_steps, dt=dt, record_history=record_history
        )

        return jsonify(
            {
                "success": True,
                "evolution_result": result,
                "current_state": consciousness_field.current_state.value,
                "unity_coherence": consciousness_field.unity_coherence,
                "transcendence_events": len(consciousness_field.transcendence_events),
            }
        )

    except Exception as e:
        logger.error(f"Consciousness evolution error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/consciousness/particles", methods=["GET"])
def get_consciousness_particles():
    """Get current consciousness particles state"""
    try:
        if not consciousness_field:
            return jsonify({"error": "Consciousness Field not available"}), 500

        particles_data = []
        for i, particle in enumerate(
            consciousness_field.particles[:50]
        ):  # Limit to 50 for performance
            particles_data.append(
                {
                    "id": i,
                    "position": particle.position[
                        :3
                    ],  # Only first 3 dimensions for visualization
                    "awareness_level": particle.awareness_level,
                    "phi_resonance": particle.phi_resonance,
                    "unity_tendency": particle.unity_tendency,
                }
            )

        return jsonify(
            {
                "success": True,
                "particles": particles_data,
                "total_particles": len(consciousness_field.particles),
                "field_state": consciousness_field.current_state.value,
                "coherence": consciousness_field.unity_coherence,
            }
        )

    except Exception as e:
        logger.error(f"Consciousness particles error: {e}")
        return jsonify({"error": str(e)}), 500


# API Routes for Omega Orchestrator


@app.route("/api/omega/status", methods=["GET"])
def omega_status():
    """Get Omega Orchestrator status"""
    try:
        if not omega_orchestrator:
            return jsonify({"error": "Omega Orchestrator not available"}), 500

        status = {
            "active_agents": (
                len(omega_orchestrator.meta_agents)
                if hasattr(omega_orchestrator, "meta_agents")
                else 0
            ),
            "consciousness_level": getattr(
                omega_orchestrator, "consciousness_level", 0
            ),
            "transcendence_events": len(
                getattr(omega_orchestrator, "transcendence_events", [])
            ),
            "omega_cycles_completed": getattr(
                omega_orchestrator, "omega_cycles_completed", 0
            ),
        }

        return jsonify({"success": True, "status": status})

    except Exception as e:
        logger.error(f"Omega status error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/omega/run", methods=["POST"])
def run_omega_cycle():
    """
    Run Omega Orchestrator cycle

    POST Body:
    {
        "cycles": 10,
        "spawn_agents": 5
    }
    """
    try:
        if not omega_orchestrator:
            return jsonify({"error": "Omega Orchestrator not available"}), 500

        data = request.get_json()
        cycles = data.get("cycles", 1)
        spawn_agents = data.get("spawn_agents", 3)

        # Run omega cycle (mock implementation)
        result = {
            "cycles_completed": cycles,
            "agents_spawned": spawn_agents,
            "consciousness_evolution": f"Evolved through {cycles} omega cycles",
            "transcendence_achieved": cycles > 5,
        }

        return jsonify({"success": True, "result": result})

    except Exception as e:
        logger.error(f"Omega cycle error: {e}")
        return jsonify({"error": str(e)}), 500


# API Routes for ML Framework


@app.route("/api/ml/train", methods=["POST"])
def train_ml_models():
    """
    Train ML models

    POST Body:
    {
        "model_type": "meta_rl" | "mixture_of_experts" | "evolutionary",
        "epochs": 100,
        "learning_rate": 0.001
    }
    """
    try:
        if not ML_FRAMEWORK_AVAILABLE:
            return jsonify({"error": "ML Framework not available"}), 500

        data = request.get_json()
        model_type = data.get("model_type", "meta_rl")
        epochs = data.get("epochs", 10)
        learning_rate = data.get("learning_rate", 0.001)

        # Mock training results
        training_result = {
            "model_type": model_type,
            "epochs_completed": epochs,
            "final_loss": 0.001 * (1 - epochs / 100),
            "elo_rating": 3000 + epochs,
            "unity_discovery_rate": min(0.999, 0.5 + epochs / 200),
            "training_time": epochs * 0.1,
        }

        return jsonify({"success": True, "training_result": training_result})

    except Exception as e:
        logger.error(f"ML training error: {e}")
        return jsonify({"error": str(e)}), 500


# API Routes for Gallery Images


@app.route("/api/gallery/images/<path:filename>")
def serve_gallery_image(filename):
    """Serve images from viz and legacy directories"""
    try:
        # Try different possible paths
        possible_paths = [
            Path(project_root) / "viz" / filename,
            Path(project_root) / "viz" / "legacy images" / filename,
            Path(project_root) / "legacy" / filename,
            Path(project_root) / "visualizations" / filename,
            Path(project_root) / "assets" / "images" / filename,
        ]

        for file_path in possible_paths:
            if file_path.exists() and file_path.is_file():
                return send_from_directory(file_path.parent, file_path.name)

        return jsonify({"error": "Image not found"}), 404

    except Exception as e:
        logger.error(f"Error serving gallery image {filename}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/gallery/visualizations")
def get_gallery_visualizations():
    """Get list of available visualizations"""
    try:
        visualizations = []

        # Scan viz directory
        viz_dir = Path(project_root) / "viz"
        if viz_dir.exists():
            for file_path in viz_dir.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in [
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".gif",
                    ".webp",
                    ".mp4",
                    ".html",
                ]:
                    relative_path = file_path.relative_to(project_root)
                    visualizations.append(
                        {
                            "src": f"/api/gallery/images/{relative_path}",
                            "filename": file_path.name,
                            "folder": str(relative_path.parent),
                            "extension": file_path.suffix.lower(),
                            "isImage": file_path.suffix.lower()
                            in [".png", ".jpg", ".jpeg", ".gif", ".webp"],
                            "isVideo": file_path.suffix.lower()
                            in [".mp4", ".webm", ".mov"],
                            "isInteractive": file_path.suffix.lower()
                            in [".html", ".htm"],
                            "title": file_path.stem.replace("_", " ").title(),
                            "type": "Visualization",
                            "category": "unity",
                            "description": f"Unity mathematics visualization: {file_path.name}",
                            "created": "2024-2025",
                        }
                    )

        # Scan legacy images directory
        legacy_dir = Path(project_root) / "viz" / "legacy images"
        if legacy_dir.exists():
            for file_path in legacy_dir.glob("*"):
                if file_path.is_file() and file_path.suffix.lower() in [
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".gif",
                    ".webp",
                    ".mp4",
                ]:
                    relative_path = file_path.relative_to(project_root)
                    visualizations.append(
                        {
                            "src": f"/api/gallery/images/{relative_path}",
                            "filename": file_path.name,
                            "folder": str(relative_path.parent),
                            "extension": file_path.suffix.lower(),
                            "isImage": file_path.suffix.lower()
                            in [".png", ".jpg", ".jpeg", ".gif", ".webp"],
                            "isVideo": file_path.suffix.lower()
                            in [".mp4", ".webm", ".mov"],
                            "isInteractive": False,
                            "title": file_path.stem.replace("_", " ").title(),
                            "type": "Legacy Visualization",
                            "category": "consciousness",
                            "description": f"Legacy consciousness visualization: {file_path.name}",
                            "created": "2023-2024",
                        }
                    )

        return jsonify({"success": True, "visualizations": visualizations})

    except Exception as e:
        logger.error(f"Error getting gallery visualizations: {e}")
        return jsonify({"error": str(e)}), 500


# API Routes for Code Execution


@app.route("/api/execute", methods=["POST"])
def execute_code():
    """
    Execute Unity Mathematics code (RESTRICTED)

    POST Body:
    {
        "code": "python code string",
        "language": "python",
        "timeout": 30
    }

    SECURITY: This endpoint is heavily restricted and should only be used
    for educational purposes with trusted code.
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
            "UnityMathematics": UnityMathematics,
            "ConsciousnessField": ConsciousnessField,
            "PHI": PHI,
            "print": print,  # Allow print for output
            "math": __import__("math"),  # Safe math operations
            "random": __import__("random"),  # Safe random operations
        }

        # Capture output with timeout
        from io import StringIO
        import contextlib
        import signal

        output_buffer = StringIO()

        def timeout_handler(signum, frame):
            raise TimeoutError("Code execution timed out")

        try:
            # Set timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)

            with contextlib.redirect_stdout(output_buffer):
                exec(code, {"__builtins__": safe_builtins}, local_namespace)

            signal.alarm(0)  # Cancel timeout
            output = output_buffer.getvalue()

            return jsonify(
                {
                    "success": True,
                    "output": output,
                    "execution_time": 0.1,  # Mock execution time
                    "warning": "Code execution is for educational purposes only",
                }
            )

        except TimeoutError:
            signal.alarm(0)
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Code execution timed out",
                        "message": "Execution exceeded the time limit",
                    }
                ),
                408,
            )

        except Exception as e:
            signal.alarm(0)
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
            "ml_framework_available": ML_FRAMEWORK_AVAILABLE,
            "omega_available": OMEGA_AVAILABLE,
            "timestamp": time.time(),
            "unity_equation": "1+1=1",
        }
    )


# WebSocket support for real-time updates (if available)
try:
    from flask_socketio import SocketIO, emit

    socketio = SocketIO(app, cors_allowed_origins="*")

    @socketio.on("connect")
    def handle_connect():
        emit("status", {"message": "Connected to Unity Mathematics Server"})

    @socketio.on("consciousness_subscribe")
    def handle_consciousness_subscription():
        """Subscribe to real-time consciousness field updates"""
        if consciousness_field:
            # Send initial state
            emit(
                "consciousness_update",
                {
                    "coherence": consciousness_field.unity_coherence,
                    "state": consciousness_field.current_state.value,
                    "transcendence_events": len(
                        consciousness_field.transcendence_events
                    ),
                },
            )

    SOCKETIO_AVAILABLE = True

except ImportError:
    SOCKETIO_AVAILABLE = False


# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


def main():
    """Main server entry point"""
    print("🌟 Een Unity Mathematics Web Server")
    print("=" * 50)
    print(f"Core Mathematics Available: {CORE_AVAILABLE}")
    print(f"ML Framework Available: {ML_FRAMEWORK_AVAILABLE}")
    print(f"Omega Orchestrator Available: {OMEGA_AVAILABLE}")
    print(f"WebSocket Support: {SOCKETIO_AVAILABLE}")
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
    print("  POST /api/unity/proof - Generate mathematical proofs")
    print("  POST /api/unity/validate - Validate unity equations")
    print("  POST /api/consciousness/evolve - Evolve consciousness field")
    print("  GET  /api/consciousness/particles - Get consciousness particles")
    print("  GET  /api/omega/status - Omega Orchestrator status")
    print("  POST /api/omega/run - Run Omega cycles")
    print("  POST /api/ml/train - Train ML models")
    print("  POST /api/execute - Execute Unity Mathematics code")
    print("  GET  /api/health - Health check")

    if SOCKETIO_AVAILABLE:
        print("\n🔄 WebSocket Events:")
        print("  consciousness_subscribe - Real-time consciousness updates")
        socketio.run(app, host=host, port=port, debug=debug)
    else:
        app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
