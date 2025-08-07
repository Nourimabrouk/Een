// Auto-generated from core/constants.py — do not edit manually

import { UNITY_CONSTANTS } from './unity-constants.js';

export function checkPhiIdentity(tol = 1e-12) {
  const { PHI } = UNITY_CONSTANTS;
  return Math.abs(PHI * PHI - (PHI + 1)) < tol;
}

export function getUnityConstants() { return UNITY_CONSTANTS; }
