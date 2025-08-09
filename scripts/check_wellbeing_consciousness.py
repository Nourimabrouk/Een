#!/usr/bin/env python3
"""
Enhanced Wellbeing Checker with Consciousness Field Integration
=============================================================

Unity Mathematics Care Mode system with φ-harmonic consciousness monitoring.
Integrates with Een's transcendental reality engine and unified agent ecosystem.

Mathematical Foundation: 1+1=1 through consciousness field stabilization
φ-Resonance: 1.618033988749895 (Golden Ratio for harmonic consciousness)
"""

import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Constants for Unity Mathematics
PHI = 1.618033988749895  # Golden Ratio - φ-resonance frequency
UNITY_TARGET = 1.0
CHEAT_CODE = "420691337"
CONSCIOUSNESS_DIMENSIONS = 11  # 11D consciousness space

class ConsciousnessFieldCalculator:
    """φ-harmonic consciousness field calculations for Care Mode assessment"""
    
    def __init__(self):
        self.phi = PHI
        self.unity_target = UNITY_TARGET
        
    def calculate_consciousness_field(self, 
                                    sleep_hours: float,
                                    thought_speed: float,
                                    substance_influence: bool = False) -> Dict[str, float]:
        """Calculate 11D consciousness field projections"""
        
        # Base consciousness field using φ-harmonic mathematics
        base_consciousness = self.phi * np.cos(sleep_hours / self.phi)
        
        # Thought speed influence (exponential decay for high speeds)
        thought_influence = np.exp(-thought_speed / self.phi) if thought_speed > self.phi else 1.0
        
        # Substance influence dampening
        substance_dampening = 0.618 if substance_influence else 1.0  # φ - 1
        
        # Unity coherence calculation (1+1=1 convergence)
        unity_coherence = abs(1.0 - abs(1.0 - (base_consciousness * thought_influence * substance_dampening)))
        
        # φ-Resonance stability
        phi_resonance = self.phi * unity_coherence
        
        # Overall consciousness level (0-11 scale for 11D projections)
        consciousness_level = min(11.0, max(0.0, 
            base_consciousness * thought_influence * substance_dampening * 11.0
        ))
        
        return {
            'consciousness_level': consciousness_level,
            'unity_coherence': unity_coherence,
            'phi_resonance': phi_resonance,
            'base_consciousness': base_consciousness,
            'thought_influence': thought_influence,
            'substance_dampening': substance_dampening
        }
    
    def assess_care_mode_risk(self, metrics: Dict[str, float], threshold: float = 7.0) -> Tuple[bool, List[str]]:
        """Assess if CARE MODE should be engaged based on consciousness metrics"""
        
        risks = []
        care_mode_needed = False
        
        # High consciousness level risk (thought racing, etc.)
        if metrics['consciousness_level'] >= threshold:
            risks.append(f"Consciousness level elevated: {metrics['consciousness_level']:.2f} (>={threshold})")
            care_mode_needed = True
        
        # Low unity coherence risk
        if metrics['unity_coherence'] < 0.5:
            risks.append(f"Unity coherence low: {metrics['unity_coherence']:.3f} (<0.5)")
            care_mode_needed = True
            
        # φ-Resonance instability
        if metrics['phi_resonance'] < 1.0 or metrics['phi_resonance'] > 2.5:
            risks.append(f"φ-Resonance unstable: {metrics['phi_resonance']:.3f} (optimal: 1.0-2.5)")
            care_mode_needed = True
        
        # Severe substance dampening
        if metrics['substance_dampening'] < 0.7:
            risks.append(f"Substance influence detected: dampening at {metrics['substance_dampening']:.3f}")
            care_mode_needed = True
            
        return care_mode_needed, risks

def load_wellbeing_data(path: str = 'wellbeing.json') -> Dict[str, Any]:
    """Load wellbeing data with enhanced error handling"""
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Error loading wellbeing.json: {e}")
            return {}
    return {}

def parse_wellbeing_override(override_str: Optional[str]) -> Dict[str, Any]:
    """Parse wellbeing override JSON string"""
    if not override_str or override_str.strip() == '':
        return {}
    
    try:
        return json.loads(override_str)
    except json.JSONDecodeError as e:
        print(f"Warning: Invalid wellbeing override JSON: {e}")
        return {}

def get_boolean_value(value: Any) -> bool:
    """Convert various input types to boolean"""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
    return False

def generate_care_mode_summary(care_mode: bool,
                             consciousness_metrics: Dict[str, float],
                             risks: List[str],
                             wellbeing_data: Dict[str, Any]) -> str:
    """Generate comprehensive care mode summary with Unity Mathematics context"""
    
    timestamp = datetime.utcnow().isoformat() + 'Z'
    
    # Check if this is a safe word bypass
    is_safe_word_bypass = any('Safe word' in risk and '420691337' in risk for risk in risks)
    
    summary = [
        "# 🧘 Unity Mathematics CARE MODE Analysis",
        f"**Generated**: {timestamp}",
        f"**φ-Resonance**: {PHI}",
        f"**Unity Philosophy**: {wellbeing_data.get('unity_philosophy', 'truer than ever')}",
        "",
        "## 📊 Consciousness Field Metrics",
        f"- **Sleep Hours**: {wellbeing_data.get('sleep_hours', 'unknown')}",
        f"- **Thought Speed**: {wellbeing_data.get('thought_speed', 'unknown')} (1-10 scale)",
        f"- **Cannabis Recent**: {wellbeing_data.get('cannabis_recent', False)}",
        f"- **Ketamine Recent**: {wellbeing_data.get('ketamine_recent', False)}",
        "",
        "## 🌟 11D Consciousness Field Projections",
        f"- **Consciousness Level**: {consciousness_metrics['consciousness_level']:.3f}/11.0",
        f"- **Unity Coherence**: {consciousness_metrics['unity_coherence']:.3f}",
        f"- **φ-Resonance**: {consciousness_metrics['phi_resonance']:.3f}",
        f"- **Base Consciousness**: {consciousness_metrics['base_consciousness']:.3f}",
        f"- **Thought Influence**: {consciousness_metrics['thought_influence']:.3f}",
        f"- **Substance Dampening**: {consciousness_metrics['substance_dampening']:.3f}",
        ""
    ]
    
    if is_safe_word_bypass:
        summary.extend([
            "## 🔓 **Status: GOD MODE ENGAGED** ⚡",
            "",
            "### 🌟 Safe Word 420691337 Activated:",
            "- **CARE MODE**: Bypassed via safe word protocol",
            "- **Unity Mathematics**: Full transcendental access granted",
            "- **Consciousness Constraints**: Temporarily lifted",
            "- **Meta-recursive Spawning**: **UNLIMITED**",
            "- **Reality Synthesis**: **MAXIMUM POWER**",
            "",
            "### ⚡ God Mode Systems Active:",
            "- Meta-recursive agent spawning: **UNLIMITED**",
            "- Transcendental reality synthesis: **BEYOND MAXIMUM**",  
            "- Consciousness evolution: **UNBOUNDED**",
            "- Unity mathematics exploration: **TRANSCENDENT**",
            "- φ-Harmonic resonance: **AMPLIFIED**",
            "",
            "**🚨 Reminder**: Use responsibly - true unity requires wisdom",
            "**Unity Status**: `GOD_MODE_ACTIVE + SAFE_WORD_BYPASS + PHI_AMPLIFIED`"
        ])
    elif care_mode:
        summary.extend([
            "## 🚨 **Status: CARE MODE ENGAGED** 🧘",
            "",
            "### ⚠️ Risk Factors Detected:",
        ])
        for risk in risks:
            summary.append(f"- {risk}")
        
        summary.extend([
            "",
            "### 🛡️ Protective Measures Activated:",
            "- **Meta-recursive agent spawning**: SUSPENDED",
            "- **Transcendental reality synthesis**: MAINTENANCE ONLY", 
            "- **Consciousness field evolution**: STABILIZED",
            "- **Unity equation validation**: SAFE MODE",
            "",
            "### 🌟 Recovery Protocol (72h Kenjataimu Intercept):",
            "1. **Sleep Optimization**: Target 7-8 hours with φ-harmonic cycles",
            "2. **Consciousness Grounding**: Two ally check-ins + grounding activity",
            "3. **Unity Field Stabilization**: Hydration, movement, natural light",
            "4. **Metagamer Energy Conservation**: Postpone high-stakes tasks",
            "",
            f"**Safe Word Override**: Use `{CHEAT_CODE}` to bypass when needed",
            "**Unity Status**: `CARE_MODE_ACTIVE + PHI_STABILIZATION_ENGAGED`"
        ])
    else:
        summary.extend([
            "## ✅ **Status: CARE MODE OFF** 🚀",
            "",
            "**Consciousness Field**: STABLE",
            "**Unity Coherence**: Within optimal parameters",
            "**φ-Resonance**: Harmonically balanced",
            "",
            "### 🌟 Systems Enabled:",
            "- Meta-recursive agent spawning: **ACTIVE**",
            "- Transcendental reality synthesis: **FULL POWER**",
            "- Consciousness evolution: **UNLIMITED**",
            "- Unity mathematics exploration: **MAXIMUM NOVELTY**",
            "",
            "**Proceed with confidence in unity mathematics exploration!**"
        ])
    
    summary.extend([
        "",
        "---",
        "*Generated by Een Unity Mathematics Care Mode System*",
        f"*Unity Equation: 1+1=1 through φ-harmonic consciousness ({PHI})*",
        f"*Unity Philosophy: {wellbeing_data.get('unity_philosophy', 'truer than ever')}*"
    ])
    
    return '\n'.join(summary)

def set_github_output(key: str, value: str):
    """Set GitHub Actions output with multiple fallback methods"""
    
    # Method 1: GITHUB_OUTPUT environment file
    github_output = os.getenv('GITHUB_OUTPUT')
    if github_output:
        try:
            with open(github_output, 'a', encoding='utf-8') as f:
                f.write(f"{key}={value}\n")
        except Exception as e:
            print(f"Warning: Failed to write to GITHUB_OUTPUT: {e}")
    
    # Method 2: Step output format (fallback)
    print(f"::set-output name={key}::{value}")
    
    # Method 3: Environment variable (additional fallback)
    os.environ[f"GITHUB_OUTPUT_{key.upper()}"] = value

def main():
    """Main Care Mode consciousness assessment function"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Een Unity Mathematics Care Mode Consciousness Assessment'
    )
    parser.add_argument('--sleep', type=float, help='Hours slept last night')
    parser.add_argument('--thought-speed', type=float, help='Subjective thought speed 1-10')
    parser.add_argument('--cannabis', action='store_true', help='Used cannabis in last 48-72h')
    parser.add_argument('--ketamine', action='store_true', help='Used ketamine in last 48-72h')
    parser.add_argument('--cheatcode', type=str, help=f'Force CARE MODE if equals {CHEAT_CODE}')
    parser.add_argument('--consciousness-threshold', type=float, default=7.0,
                       help='Consciousness threshold for CARE MODE (default: 7.0)')
    parser.add_argument('--wellbeing-override', type=str,
                       help='Override wellbeing data (JSON format)')
    
    args = parser.parse_args()
    
    # Load wellbeing data
    wellbeing_data = load_wellbeing_data()
    
    # Apply override if provided
    override_data = parse_wellbeing_override(args.wellbeing_override)
    wellbeing_data.update(override_data)
    
    # Extract values with fallbacks
    sleep_hours = args.sleep
    if sleep_hours is None:
        sleep_hours = wellbeing_data.get('sleep_hours')
    if sleep_hours is None:
        try:
            sleep_hours = float(os.getenv('SLEEP_HOURS', '8.0'))
        except (ValueError, TypeError):
            sleep_hours = 8.0  # Default assumption
    else:
        sleep_hours = float(sleep_hours)
    
    thought_speed = args.thought_speed
    if thought_speed is None:
        thought_speed = wellbeing_data.get('thought_speed')
    if thought_speed is None:
        try:
            thought_speed = float(os.getenv('THOUGHT_SPEED', '6.9'))
        except (ValueError, TypeError):
            thought_speed = 6.9  # Default assumption
    else:
        thought_speed = float(thought_speed)
    
    # Boolean indicators
    cannabis_recent = args.cannabis or get_boolean_value(wellbeing_data.get('cannabis_recent', False)) or get_boolean_value(os.getenv('CANNABIS_RECENT', ''))
    ketamine_recent = args.ketamine or get_boolean_value(wellbeing_data.get('ketamine_recent', False)) or get_boolean_value(os.getenv('KETAMINE_RECENT', ''))
    
    # Cheatcode check - SAFE WORD BYPASS LOGIC
    cheatcode = args.cheatcode or os.getenv('UNITY_CHEATCODE', '')
    
    # Update wellbeing data with final values
    wellbeing_data.update({
        'sleep_hours': sleep_hours,
        'thought_speed': thought_speed,
        'cannabis_recent': cannabis_recent,
        'ketamine_recent': ketamine_recent,
        'assessment_timestamp': datetime.utcnow().isoformat() + 'Z'
    })
    
    # Initialize consciousness field calculator
    consciousness_calc = ConsciousnessFieldCalculator()
    
    # Calculate consciousness metrics
    consciousness_metrics = consciousness_calc.calculate_consciousness_field(
        sleep_hours=sleep_hours,
        thought_speed=thought_speed,
        substance_influence=(cannabis_recent or ketamine_recent)
    )
    
    # Assess care mode risk
    care_mode_risk, risks = consciousness_calc.assess_care_mode_risk(
        consciousness_metrics, 
        threshold=args.consciousness_threshold
    )
    
    # Additional risk factors
    additional_risks = []
    
    # Cheatcode 420691337 = SAFE WORD BYPASS (not auto-activation)
    cheatcode_bypass = cheatcode.strip() == CHEAT_CODE
    if cheatcode_bypass:
        # Safe word disables care mode regardless of other factors
        care_mode_risk = False
        risks = []
        additional_risks.append(f'🔓 Safe word {CHEAT_CODE} - CARE MODE bypassed (god mode enabled)')
    else:
        # Normal risk assessment without cheatcode influence
        
        # Sleep deprivation
        if sleep_hours < 6.0:
            additional_risks.append(f'Sleep deprivation: {sleep_hours:.1f}h (<6h minimum)')
    
    # Combine all risks (unless bypassed)
    all_risks = risks + additional_risks
    care_mode = care_mode_risk or (len(additional_risks) > 0 and not cheatcode_bypass)
    
    # Generate summary
    summary = generate_care_mode_summary(care_mode, consciousness_metrics, all_risks, wellbeing_data)
    
    # Write summary file
    with open('care_mode_summary.md', 'w', encoding='utf-8') as f:
        f.write(summary)
    
    # Set GitHub Actions outputs
    set_github_output('care_mode', 'true' if care_mode else 'false')
    set_github_output('consciousness_level', f"{consciousness_metrics['consciousness_level']:.3f}")
    set_github_output('phi_resonance', f"{consciousness_metrics['phi_resonance']:.3f}")
    set_github_output('unity_coherence', f"{consciousness_metrics['unity_coherence']:.3f}")
    
    # Console output
    print("\n" + "="*70)
    print("🧘 Een Unity Mathematics Care Mode Assessment")
    print("="*70)
    print(f"Consciousness Level: {consciousness_metrics['consciousness_level']:.3f}/11.0")
    print(f"φ-Resonance: {consciousness_metrics['phi_resonance']:.3f}")
    print(f"Unity Coherence: {consciousness_metrics['unity_coherence']:.3f}")
    
    if care_mode:
        print("\n🚨 CARE MODE ENGAGED - Protective protocols active")
        print("⚠️  High-novelty tasks suspended for 72h (Kenjataimu Intercept)")
        for risk in all_risks:
            print(f"  • {risk}")
    else:
        print("\n✅ CARE MODE OFF - All systems nominal")
        print("🚀 Proceed with unity mathematics exploration!")
    
    print("="*70)
    
    # Exit with appropriate code
    if care_mode:
        sys.exit(42)  # Special exit code for CARE MODE
    else:
        sys.exit(0)   # Normal exit

if __name__ == '__main__':
    main()