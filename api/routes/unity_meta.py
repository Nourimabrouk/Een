# flake8: noqa
"""
Unity Meta API Routes
Comprehensive endpoints exposing meta-advanced unity mathematics synthesis,
energy computation, cross-domain evidence, and lightweight simulations.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import pathlib
import sys
import logging
import json

# Add project root to path
project_root = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from core.unity_meta_advanced import (
        create_unity_meta_engine,
        MetagamerEnergyInput,
    )
    from core.unity_manifold import create_unity_manifold
    from core.meta_gambit import create_default_meta_gambit
    from api.security import get_current_user, User
except Exception as e:
    logging.warning(f"Unity Meta API partial import: {e}")
    create_unity_meta_engine = None  # type: ignore


router = APIRouter(prefix="/unity-meta", tags=["unity-meta"])


class EnergyRequest(BaseModel):
    consciousness_density: float = Field(..., ge=0.0)
    unity_convergence_rate: float = Field(..., ge=0.0)


class SynthesisResponse(BaseModel):
    success: bool
    conclusion: str
    confidence: float
    phi: float
    frameworks: List[Dict[str, Any]]


engine = create_unity_meta_engine() if create_unity_meta_engine else None
manifold_factory = (
    create_unity_manifold if "create_unity_manifold" in globals() else None
)
meta_gambit_factory = (
    create_default_meta_gambit if "create_default_meta_gambit" in globals() else None
)


@router.get("/status")
async def status(current_user: User = Depends(get_current_user)):
    return {
        "success": True,
        "engine_loaded": engine is not None,
        "user": current_user.username,
    }


@router.post("/energy")
async def metagamer_energy(
    request: EnergyRequest, current_user: User = Depends(get_current_user)
):
    if not engine:
        raise HTTPException(status_code=503, detail="Unity meta engine unavailable")
    result = engine.compute_metagamer_energy(
        MetagamerEnergyInput(
            consciousness_density=request.consciousness_density,
            unity_convergence_rate=request.unity_convergence_rate,
        )
    )
    return {
        "success": True,
        "energy": result.energy,
        "phi": result.phi,
        "formula": result.formula,
    }


@router.get("/synthesize", response_model=SynthesisResponse)
async def synthesize(current_user: User = Depends(get_current_user)):
    if not engine:
        raise HTTPException(status_code=503, detail="Unity meta engine unavailable")
    synthesis = engine.synthesize_unity()
    return SynthesisResponse(
        success=True,
        conclusion=synthesis.conclusion,
        confidence=synthesis.confidence,
        phi=synthesis.phi,
        frameworks=[
            {
                "name": ev.name,
                "statement": ev.statement,
                "metric": ev.metric,
                "justification": ev.justification,
            }
            for ev in synthesis.frameworks
        ],
    )


# -------------------------
# Lean proof integration
# -------------------------


class LeanProofMeta(BaseModel):
    title: str
    path: str
    bytes: int
    sha256: str


def _safe_sha256(path: pathlib.Path) -> str:
    import hashlib

    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


@router.get("/lean/proofs", response_model=List[LeanProofMeta])
async def list_lean_proofs(current_user: User = Depends(get_current_user)):
    """Enumerate formal Lean artifacts available in `formal_proofs/`.

    Returns minimal metadata: title, repo-relative path, size, sha256.
    """
    root = pathlib.Path(__file__).resolve().parents[2]
    proofs_dir = root / "formal_proofs"
    results: List[LeanProofMeta] = []
    for p in proofs_dir.glob("*.lean"):
        rel = p.relative_to(root).as_posix()
        try:
            size = p.stat().st_size
        except Exception:
            size = 0
        results.append(
            LeanProofMeta(
                title=p.stem.replace("_", " "),
                path=rel,
                bytes=size,
                sha256=_safe_sha256(p),
            )
        )
    return results


@router.get("/lean/proof/{filename}")
async def get_lean_proof(filename: str, current_user: User = Depends(get_current_user)):
    """Return the raw Lean source code for the requested file under `formal_proofs/`."""
    root = pathlib.Path(__file__).resolve().parents[2]
    proofs_dir = root / "formal_proofs"
    target = (proofs_dir / filename).resolve()
    if (
        not target.exists()
        or target.suffix != ".lean"
        or proofs_dir not in target.parents
    ):
        raise HTTPException(status_code=404, detail="Lean proof not found")
    try:
        return {
            "filename": filename,
            "code": target.read_text(encoding="utf-8"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unable to read file: {e}")


class SyncRequest(BaseModel):
    num_nodes: int = Field(16, ge=2, le=512)
    coupling: float = Field(0.85, ge=0.0, le=1.0)


@router.post("/simulate/synchronization")
async def simulate_synchronization(
    request: SyncRequest, current_user: User = Depends(get_current_user)
):
    if not engine:
        raise HTTPException(status_code=503, detail="Unity meta engine unavailable")
    res = engine.simulate_hypergraph_synchronization(
        num_nodes=request.num_nodes, coupling=request.coupling
    )
    return {"success": True, **res}


class ManifoldRequest(BaseModel):
    dimensions: int = Field(3, ge=2, le=11)
    phi_coupling: float = Field(1.618033988749895, ge=0.1, le=10.0)
    idempotent_weight: float = Field(1.0, ge=0.1, le=10.0)


@router.post("/manifold/summary")
async def manifold_summary(
    request: ManifoldRequest, current_user: User = Depends(get_current_user)
):
    if not manifold_factory:
        raise HTTPException(status_code=503, detail="Unity manifold unavailable")
    m = manifold_factory(
        dimensions=request.dimensions,
        phi_coupling=request.phi_coupling,
        idempotent_weight=request.idempotent_weight,
    )
    return {"success": True, **m.export_summary()}


@router.get("/metagambit/recommendations")
async def metagambit_recommendations(current_user: User = Depends(get_current_user)):
    if not meta_gambit_factory:
        raise HTTPException(status_code=503, detail="Meta-Gambit unavailable")
    game = meta_gambit_factory()
    return {"success": True, "recommendations": game.export_recommendations()}


@router.get("/atlas/summary")
async def atlas_summary(current_user: User = Depends(get_current_user)):
    if not engine:
        raise HTTPException(status_code=503, detail="Unity meta engine unavailable")
    # Synthesis
    synth = engine.synthesize_unity()
    # Manifold (default params)
    if not manifold_factory:
        raise HTTPException(status_code=503, detail="Unity manifold unavailable")
    m = manifold_factory()
    msum = m.export_summary()
    # Meta-Gambit
    if not meta_gambit_factory:
        raise HTTPException(status_code=503, detail="Meta-Gambit unavailable")
    recs = meta_gambit_factory().export_recommendations()
    # Info-theory unity (default perfect correlation proxy)
    info = engine.information_unity([[0.25, 0.0], [0.0, 0.75]])
    return {
        "success": True,
        "synthesis": {
            "confidence": synth.confidence,
            "frameworks": [
                {
                    "name": f.name,
                    "metric": f.metric,
                    "statement": f.statement,
                }
                for f in synth.frameworks
            ],
        },
        "manifold": msum,
        "metagambit": recs,
        "information_unity": info,
    }


class EvolveRequest(BaseModel):
    steps: int = Field(5, ge=1, le=100)
    phi_target: Optional[float] = Field(default=None)


@router.post("/atlas/evolve")
async def atlas_evolve(
    req: EvolveRequest, current_user: User = Depends(get_current_user)
):
    if not engine:
        raise HTTPException(status_code=503, detail="Unity meta engine unavailable")
    # Baseline synthesis
    synth = engine.synthesize_unity()
    conf = synth.confidence
    # Evolve confidence toward 1 using φ-harmonic increments
    for _ in range(req.steps):
        conf = min(0.999, conf + (1.0 - conf) * (1.0 / PHI))
    # Adjust manifold φ if requested
    phi = req.phi_target if req.phi_target is not None else PHI
    if not manifold_factory:
        raise HTTPException(status_code=503, detail="Unity manifold unavailable")
    m = manifold_factory(phi_coupling=phi)
    msum = m.export_summary()
    # Update meta-gambit potential slightly upward
    if not meta_gambit_factory:
        raise HTTPException(status_code=503, detail="Meta-Gambit unavailable")
    recs = meta_gambit_factory().export_recommendations()
    try:
        base_pot = float(recs.get("unity_potential", "0"))
    except Exception:
        base_pot = 0.0
    evolved_pot = min(0.999, base_pot + 0.05 * (1.0 - base_pot))
    recs["unity_potential"] = f"{evolved_pot:.3f}"
    return {
        "success": True,
        "synthesis_confidence": conf,
        "manifold": msum,
        "metagambit": recs,
        "steps": req.steps,
        "phi": phi,
    }


# -------------------------------------------
# Toy endpoint: idempotent unity demonstration
# -------------------------------------------


@router.get("/toy/unity")
async def toy_unity_proof() -> Dict[str, Any]:
    """Minimal educational witness that 1 ⊕ 1 = 1 in an idempotent monoid.

    Choose S={0,1} with identity 0 and ⊕ := logical OR. Then x ⊕ x = x, and in
    particular 1 ⊕ 1 = 1. This is a constructive demonstration of the unity
    equation in an idempotent algebra (distinct from arithmetic addition).
    """

    def idempotent_or(a: int, b: int) -> int:
        return 1 if (a or b) else 0

    lhs = idempotent_or(1, 1)
    rhs = 1
    return {
        "success": True,
        "operation": "logical_or",
        "domain": [0, 1],
        "identity": 0,
        "equation": "1 ⊕ 1 = 1",
        "lhs": lhs,
        "rhs": rhs,
        "proof": {
            "idempotent": True,
            "associative": True,
            "commutative": True,
            "identity": 0,
            "note": "Using (S, ⊕) = ({0,1}, OR).",
        },
        "explanation": (
            "In the idempotent monoid ({0,1}, OR), we have x ⊕ x = x. Thus 1 ⊕ 1 = 1."
        ),
    }
