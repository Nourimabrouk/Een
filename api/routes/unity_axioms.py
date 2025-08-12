"""
Unity Axioms API

Endpoints implementing the "what to do now" tasks:
- Axiomatize Unity: expose axioms, checks, and a commuting-square witness
- MDL audit: cross-domain description-length reductions
- Safety sketch: monotone improvement under ⊕
- Prediction stub: register φ-structured spectral/slope hypotheses

These routes are lightweight and depend only on core mathematical modules.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Tuple

from src.core.mathematical.unity_axioms import (
    UnityToposSketch,
    boolean_or_semiring,
    mdl_length,
    monotone_update_sequence,
)


router = APIRouter(prefix="/unity-axioms", tags=["unity-axioms"])


class MDLAuditRequest(BaseModel):
    expressions: List[str] = Field(..., description="Code-like laws or statements")


class MDLAuditResponse(BaseModel):
    success: bool
    baseline_tokens: int
    compressed_tokens: int
    compression_ratio: float
    breakdown: List[Tuple[str, int, int]]


@router.get("/axioms")
async def axioms() -> Dict[str, Any]:
    """Return the formal objects and a simple commuting-square check."""
    semi = boolean_or_semiring()
    cat = UnityToposSketch()
    x, y = 1, 1
    commute = cat.commuting_square_holds(x, y, semi)
    return {
        "success": True,
        "idempotent_semiring": {
            "carrier": "{0,1}",
            "oplus": "OR",
            "otimes": "AND",
            "zero": semi.zero,
            "one": semi.one,
            "idempotence_witness": semi.is_idempotent(1),
        },
        "unity_topos_sketch": {
            "terminal": cat.terminal_object,
            "commuting_square_holds": commute,
            "equation": "1 ⊕ 1 = 1",
        },
    }


@router.post("/mdl", response_model=MDLAuditResponse)
async def mdl(expressions: MDLAuditRequest) -> MDLAuditResponse:
    base, comp, ratio, breakdown = mdl_length(expressions.expressions)
    return MDLAuditResponse(
        success=True,
        baseline_tokens=base,
        compressed_tokens=comp,
        compression_ratio=ratio,
        breakdown=breakdown,
    )


class MonotoneRequest(BaseModel):
    updates: List[int] = Field(..., description="Sequence of 0/1 updates")
    start: int = Field(0, ge=0, le=1)


@router.post("/safety/monotone")
async def safety_monotone(req: MonotoneRequest) -> Dict[str, Any]:
    semi = boolean_or_semiring()
    fixed, steps = monotone_update_sequence(
        req.start,
        req.updates,
        semi,
        max_steps=10_000,
    )
    return {
        "success": True,
        "fixed_point": fixed,
        "steps": steps,
        "idempotent": semi.is_idempotent(fixed),
    }


class PredictionRegistration(BaseModel):
    domain: str = Field(..., examples=["spectral_slope", "learning_scaling"])
    claim: str = Field(..., description="A φ-structured, falsifiable claim")
    metric: str = Field(..., description="How to measure success/failure")


_PREDICTIONS: List[Dict[str, str]] = []


@router.post("/predictions/register")
async def register_prediction(p: PredictionRegistration) -> Dict[str, Any]:
    _PREDICTIONS.append(p.dict())
    return {"success": True, "count": len(_PREDICTIONS), "latest": p.dict()}


@router.get("/predictions")
async def list_predictions() -> Dict[str, Any]:
    return {"success": True, "predictions": _PREDICTIONS}
