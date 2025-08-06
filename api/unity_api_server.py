"""
Unity Mathematics API Server
Flask-based API for mathematical proof verification and visualization
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import logging
import traceback
from typing import Dict, Any, Optional
import json
import base64
from datetime import datetime
import numpy as np

# Import Unity modules
from core.unity_engine import UnityEngine
from core.mathematical_proofs import AdvancedUnityMathematics
from core.consciousness_models import run_comprehensive_consciousness_analysis
from core.visualization_kernels import (
    UnityVisualizationKernels,
    VisualizationConfig,
    create_web_visualization_data,
)

# Import API routes
from routes import openai

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins=["*"])  # Enable CORS for all origins

# Register blueprints
app.register_blueprint(openai.openai_bp)

# Initialize Unity systems
unity_engine = UnityEngine()
visualization_kernels = UnityVisualizationKernels()

# API versioning
API_VERSION = "1.0.0"
API_PREFIX = "/api/v1"


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy arrays"""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, complex):
            return {"real": obj.real, "imag": obj.imag}
        return super().default(obj)


app.json_encoder = NumpyEncoder


@app.errorhandler(Exception)
def handle_exception(error):
    """Global error handler"""
    logger.error(f"Unhandled exception: {error}")
    logger.error(traceback.format_exc())

    return (
        jsonify(
            {
                "success": False,
                "error": str(error),
                "type": type(error).__name__,
                "timestamp": datetime.utcnow().isoformat(),
            }
        ),
        500,
    )


@app.route("/")
def index():
    """API root endpoint"""
    return jsonify(
        {
            "message": "Unity Mathematics API",
            "version": API_VERSION,
            "equation": "1 + 1 = 1",
            "motto": "Thou Art That • Tat Tvam Asi",
            "endpoints": {
                "proofs": f"{API_PREFIX}/proofs",
                "verify": f"{API_PREFIX}/verify",
                "consciousness": f"{API_PREFIX}/consciousness",
                "visualizations": f"{API_PREFIX}/visualizations",
                "health": "/health",
            },
            "timestamp": datetime.utcnow().isoformat(),
        }
    )


@app.route("/health")
def health_check():
    """Health check endpoint"""
    try:
        # Quick verification that core systems are working
        test_result = unity_engine.execute_proof("euler")

        return jsonify(
            {
                "status": "healthy",
                "unity_engine": "operational",
                "proof_verification": (
                    "functional" if test_result["verified"] else "warning"
                ),
                "version": API_VERSION,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
    except Exception as e:
        return (
            jsonify(
                {
                    "status": "unhealthy",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                }
            ),
            503,
        )


@app.route(f"{API_PREFIX}/proofs")
def get_proofs():
    """Get list of available mathematical proofs"""
    try:
        proofs = (
            unity_engine.getProofsList()
            if hasattr(unity_engine, "getProofsList")
            else []
        )

        if not proofs:
            # Fallback: extract proof information from engine
            proofs = []
            for paradigm_name, proof in unity_engine.proofs.items():
                proofs.append(
                    {
                        "paradigm": proof.paradigm.value,
                        "name": paradigm_name.replace("_", " ").title(),
                        "statement": proof.formal_statement,
                        "complexity_level": (
                            proof.complexity_level
                            if hasattr(proof, "complexity_level")
                            else 3
                        ),
                    }
                )

        return jsonify(
            {
                "success": True,
                "proofs": proofs,
                "count": len(proofs),
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"Error getting proofs list: {e}")
        return (
            jsonify(
                {
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                }
            ),
            500,
        )


@app.route(f"{API_PREFIX}/verify/<paradigm>")
def verify_proof(paradigm: str):
    """Verify a specific mathematical proof"""
    try:
        result = unity_engine.execute_proof(paradigm)

        return jsonify(
            {
                "success": True,
                "paradigm": paradigm,
                "verification_result": result,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    except ValueError as e:
        return (
            jsonify(
                {
                    "success": False,
                    "error": f"Unknown paradigm: {paradigm}",
                    "available_paradigms": list(unity_engine.proofs.keys()),
                    "timestamp": datetime.utcnow().isoformat(),
                }
            ),
            400,
        )

    except Exception as e:
        logger.error(f"Error verifying proof {paradigm}: {e}")
        return (
            jsonify(
                {
                    "success": False,
                    "error": str(e),
                    "paradigm": paradigm,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            ),
            500,
        )


@app.route(f"{API_PREFIX}/verify/all")
def verify_all_proofs():
    """Verify all mathematical proofs"""
    try:
        results = unity_engine.execute_all_proofs()

        return jsonify(
            {
                "success": True,
                "verification_results": results,
                "unity_achieved": results["summary"]["unity_achieved"],
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"Error verifying all proofs: {e}")
        return (
            jsonify(
                {
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                }
            ),
            500,
        )


@app.route(f"{API_PREFIX}/mathematics/comprehensive")
def comprehensive_mathematics_verification():
    """Run comprehensive advanced mathematics verification"""
    try:
        results = AdvancedUnityMathematics.comprehensive_unity_verification()

        return jsonify(
            {
                "success": True,
                "comprehensive_results": results,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"Error in comprehensive mathematics verification: {e}")
        return (
            jsonify(
                {
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                }
            ),
            500,
        )


@app.route(f"{API_PREFIX}/consciousness/analysis")
def consciousness_analysis():
    """Run comprehensive consciousness analysis"""
    try:
        results = run_comprehensive_consciousness_analysis()

        return jsonify(
            {
                "success": True,
                "consciousness_analysis": results,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"Error in consciousness analysis: {e}")
        return (
            jsonify(
                {
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                }
            ),
            500,
        )


@app.route(f"{API_PREFIX}/visualizations/<paradigm>")
def get_visualization(paradigm: str):
    """Get visualization data for a specific paradigm"""
    try:
        # Parse query parameters
        width = request.args.get("width", 800, type=int)
        height = request.args.get("height", 600, type=int)
        fps = request.args.get("fps", 30, type=int)
        quality = request.args.get("quality", "high")

        # Create configuration
        config = VisualizationConfig(
            width=min(width, 2000),  # Limit max size
            height=min(height, 2000),
            fps=min(fps, 60),
            quality=quality,
        )

        # Generate visualization data
        viz_data = create_web_visualization_data(paradigm, config)

        return jsonify(
            {
                "success": True,
                "paradigm": paradigm,
                "visualization_data": viz_data,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"Error generating visualization for {paradigm}: {e}")
        return (
            jsonify(
                {
                    "success": False,
                    "error": str(e),
                    "paradigm": paradigm,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            ),
            500,
        )


@app.route(f"{API_PREFIX}/visualizations/mandala")
def get_unity_mandala():
    """Generate Unity mandala visualization"""
    try:
        time_param = request.args.get("time", 0.0, type=float)

        mandala_data = visualization_kernels.create_unity_mandala(time_param)

        # Convert to base64 for web transmission
        mandala_b64 = base64.b64encode(mandala_data.tobytes()).decode("utf-8")

        return jsonify(
            {
                "success": True,
                "mandala_data": mandala_b64,
                "shape": mandala_data.shape,
                "dtype": str(mandala_data.dtype),
                "time": time_param,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"Error generating Unity mandala: {e}")
        return (
            jsonify(
                {
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                }
            ),
            500,
        )


@app.route(f"{API_PREFIX}/kernels/<kernel_name>")
def run_kernel(kernel_name: str):
    """Run a specific visualization kernel"""
    try:
        time_param = request.args.get("time", 0.0, type=float)

        kernel_functions = {
            "mandelbrot": visualization_kernels.mandelbrot_kernel,
            "julia": visualization_kernels.julia_kernel,
            "consciousness_field": visualization_kernels.consciousness_field_kernel,
            "golden_spiral": visualization_kernels.golden_ratio_spiral_kernel,
            "quantum_superposition": visualization_kernels.quantum_superposition_kernel,
            "euler_unity": visualization_kernels.euler_unity_kernel,
            "topological_unity": visualization_kernels.topological_unity_kernel,
        }

        if kernel_name not in kernel_functions:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": f"Unknown kernel: {kernel_name}",
                        "available_kernels": list(kernel_functions.keys()),
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                ),
                400,
            )

        # Execute kernel
        kernel_func = kernel_functions[kernel_name]

        if kernel_name in [
            "golden_spiral",
            "quantum_superposition",
            "euler_unity",
            "topological_unity",
        ]:
            result = kernel_func(time_param)
        else:
            result = kernel_func()

        return jsonify(
            {
                "success": True,
                "kernel": kernel_name,
                "result": result,
                "time": time_param,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"Error running kernel {kernel_name}: {e}")
        return (
            jsonify(
                {
                    "success": False,
                    "error": str(e),
                    "kernel": kernel_name,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            ),
            500,
        )


@app.route(f"{API_PREFIX}/unity/meditation")
def get_meditation_sequence():
    """Get Unity meditation sequence"""
    try:
        sequence = unity_engine.generate_unity_meditation_sequence()

        return jsonify(
            {
                "success": True,
                "meditation_sequence": sequence,
                "count": len(sequence),
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"Error generating meditation sequence: {e}")
        return (
            jsonify(
                {
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                }
            ),
            500,
        )


@app.route(f"{API_PREFIX}/unity/constants")
def get_unity_constants():
    """Get important mathematical constants used in Unity mathematics"""
    try:
        constants = {
            "phi": float(unity_engine.PHI),
            "e": float(unity_engine.E),
            "pi": float(unity_engine.PI),
            "golden_angle": float(2 * unity_engine.PI / (unity_engine.PHI**2)),
            "unity_threshold": 0.618,  # φ - 1
            "consciousness_constant": float(unity_engine.PI / 4),
        }

        return jsonify(
            {
                "success": True,
                "constants": constants,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"Error getting Unity constants: {e}")
        return (
            jsonify(
                {
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                }
            ),
            500,
        )


@app.route(f"{API_PREFIX}/system/info")
def get_system_info():
    """Get system information and capabilities"""
    try:
        info = {
            "api_version": API_VERSION,
            "unity_engine": {
                "proofs_available": len(unity_engine.proofs),
                "phi_value": unity_engine.PHI,
                "cache_size": len(unity_engine.verification_cache),
            },
            "visualization_kernels": visualization_kernels.export_visualization_config(),
            "capabilities": {
                "proof_verification": True,
                "consciousness_analysis": True,
                "visualization_generation": True,
                "gpu_acceleration": visualization_kernels.gpu_available,
                "real_time_computation": True,
            },
            "supported_paradigms": list(unity_engine.proofs.keys()),
            "timestamp": datetime.utcnow().isoformat(),
        }

        return jsonify(
            {
                "success": True,
                "system_info": info,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        return (
            jsonify(
                {
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                }
            ),
            500,
        )


@app.route(f"{API_PREFIX}/benchmark")
def run_benchmark():
    """Run performance benchmark"""
    try:
        import time

        # Benchmark proof verification
        start_time = time.time()
        results = unity_engine.execute_all_proofs()
        proof_time = time.time() - start_time

        # Benchmark visualization
        start_time = time.time()
        mandala = visualization_kernels.create_unity_mandala()
        viz_time = time.time() - start_time

        # Benchmark consciousness analysis
        start_time = time.time()
        consciousness_results = run_comprehensive_consciousness_analysis()
        consciousness_time = time.time() - start_time

        benchmark_results = {
            "proof_verification": {
                "time_seconds": proof_time,
                "proofs_per_second": results["summary"]["total_proofs"] / proof_time,
                "unity_achieved": results["summary"]["unity_achieved"],
            },
            "visualization": {
                "time_seconds": viz_time,
                "mandala_size": mandala.shape if hasattr(mandala, "shape") else None,
            },
            "consciousness_analysis": {
                "time_seconds": consciousness_time,
                "unity_achieved": consciousness_results.get(
                    "comprehensive_assessment", {}
                ).get("overall_unity_achieved", False),
            },
            "total_time": proof_time + viz_time + consciousness_time,
        }

        return jsonify(
            {
                "success": True,
                "benchmark_results": benchmark_results,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"Error running benchmark: {e}")
        return (
            jsonify(
                {
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                }
            ),
            500,
        )


# WebSocket endpoints (if needed for real-time visualization)
try:
    from flask_socketio import SocketIO, emit

    socketio = SocketIO(app, cors_allowed_origins="*")

    @socketio.on("connect")
    def handle_connect():
        """Handle WebSocket connection"""
        emit(
            "response",
            {
                "message": "Connected to Unity Mathematics API",
                "equation": "1 + 1 = 1",
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    @socketio.on("verify_proof")
    def handle_verify_proof(data):
        """Handle real-time proof verification"""
        try:
            paradigm = data.get("paradigm")
            if paradigm:
                result = unity_engine.execute_proof(paradigm)
                emit(
                    "proof_result",
                    {
                        "paradigm": paradigm,
                        "result": result,
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                )
            else:
                emit("error", {"message": "Paradigm not specified"})
        except Exception as e:
            emit("error", {"message": str(e)})

    @socketio.on("generate_visualization")
    def handle_generate_visualization(data):
        """Handle real-time visualization generation"""
        try:
            paradigm = data.get("paradigm")
            time_param = data.get("time", 0.0)

            if paradigm:
                viz_data = create_web_visualization_data(paradigm)
                emit(
                    "visualization_data",
                    {
                        "paradigm": paradigm,
                        "data": viz_data,
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                )
            else:
                emit("error", {"message": "Paradigm not specified"})
        except Exception as e:
            emit("error", {"message": str(e)})

except ImportError:
    logger.warning("Flask-SocketIO not available - real-time features disabled")
    socketio = None


def create_app(config=None):
    """Application factory"""
    if config:
        app.config.update(config)

    logger.info("Unity Mathematics API Server initialized")
    logger.info(f"API Version: {API_VERSION}")
    logger.info(f"Unity Engine: {len(unity_engine.proofs)} proofs available")
    logger.info(
        f"Visualization Kernels: GPU={'enabled' if visualization_kernels.gpu_available else 'disabled'}"
    )

    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Unity Mathematics API Server")
    parser.add_argument("--host", default="localhost", help="Host address")
    parser.add_argument("--port", default=5000, type=int, help="Port number")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    print("🚀" + "=" * 60 + "🚀")
    print("       UNITY MATHEMATICS API SERVER")
    print("                 1 + 1 = 1")
    print("🚀" + "=" * 60 + "🚀")
    print(f"Starting server at http://{args.host}:{args.port}")
    print(f"API endpoints available at {API_PREFIX}")
    print("φ" + "=" * 60 + "φ")

    if socketio:
        socketio.run(app, host=args.host, port=args.port, debug=args.debug)
    else:
        app.run(host=args.host, port=args.port, debug=args.debug)
