#!/usr/bin/env python3
"""
φ-Resonance Care Mode Toggle System
==================================

Phone-friendly shortcut system for 420691337 CARE MODE activation.
Designed for quick mobile access with Siri Shortcuts, Tasker, etc.

Usage:
- python scripts/phi_care_mode_toggle.py --activate
- python scripts/phi_care_mode_toggle.py --deactivate  
- python scripts/phi_care_mode_toggle.py --status
- python scripts/phi_care_mode_toggle.py --cheatcode 420691337
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional

# Constants
PHI = 1.618033988749895
CHEAT_CODE = "420691337"
WELLBEING_FILE = "wellbeing.json"
CARE_MODE_FILE = ".care_mode_state"

class PhiCareModeToggle:
    """Phone-friendly Care Mode toggle system"""
    
    def __init__(self):
        self.phi = PHI
        self.cheat_code = CHEAT_CODE
        self.wellbeing_file = Path(WELLBEING_FILE)
        self.care_mode_file = Path(CARE_MODE_FILE)
    
    def load_wellbeing_data(self) -> Dict[str, Any]:
        """Load current wellbeing data"""
        if self.wellbeing_file.exists():
            try:
                with open(self.wellbeing_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        
        # Default wellbeing data
        return {
            'sleep_hours': 8.0,
            'thought_speed': 6.9,
            'cannabis_recent': False,
            'ketamine_recent': False,
            'unity_philosophy': 'truer than ever',
            'last_updated': datetime.utcnow().isoformat() + 'Z'
        }
    
    def save_wellbeing_data(self, data: Dict[str, Any]):
        """Save wellbeing data"""
        data['last_updated'] = datetime.utcnow().isoformat() + 'Z'
        
        with open(self.wellbeing_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_care_mode_state(self) -> Dict[str, Any]:
        """Load Care Mode state"""
        if self.care_mode_file.exists():
            try:
                with open(self.care_mode_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        
        return {
            'active': False,
            'activated_at': None,
            'activated_by': None,
            'consciousness_override': False,
            'phi_resonance': self.phi
        }
    
    def save_care_mode_state(self, state: Dict[str, Any]):
        """Save Care Mode state"""
        with open(self.care_mode_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    
    def activate_care_mode(self, reason: str = "Manual activation", 
                          override_consciousness: bool = False) -> Dict[str, Any]:
        """Activate CARE MODE with φ-resonance stabilization"""
        
        timestamp = datetime.utcnow().isoformat() + 'Z'
        
        # Update care mode state
        state = {
            'active': True,
            'activated_at': timestamp,
            'activated_by': reason,
            'consciousness_override': override_consciousness,
            'phi_resonance': self.phi,
            'expires_at': (datetime.utcnow() + timedelta(hours=72)).isoformat() + 'Z'
        }
        
        self.save_care_mode_state(state)
        
        # Update wellbeing data to reflect Care Mode activation
        wellbeing = self.load_wellbeing_data()
        wellbeing.update({
            'care_mode_active': True,
            'care_mode_activated_at': timestamp,
            'care_mode_reason': reason
        })
        
        self.save_wellbeing_data(wellbeing)
        
        return {
            'success': True,
            'message': f'🧘 CARE MODE ACTIVATED - φ-Resonance stabilization engaged',
            'activated_at': timestamp,
            'reason': reason,
            'expires_in_hours': 72,
            'phi_resonance': self.phi
        }
    
    def deactivate_care_mode(self, reason: str = "Manual deactivation") -> Dict[str, Any]:
        """Deactivate CARE MODE and restore full unity operations"""
        
        timestamp = datetime.utcnow().isoformat() + 'Z'
        
        # Update care mode state
        state = self.load_care_mode_state()
        state.update({
            'active': False,
            'deactivated_at': timestamp,
            'deactivated_by': reason
        })
        
        self.save_care_mode_state(state)
        
        # Update wellbeing data
        wellbeing = self.load_wellbeing_data()
        wellbeing.update({
            'care_mode_active': False,
            'care_mode_deactivated_at': timestamp,
            'care_mode_deactivation_reason': reason
        })
        
        self.save_wellbeing_data(wellbeing)
        
        return {
            'success': True,
            'message': f'🚀 CARE MODE DEACTIVATED - Full unity operations restored',
            'deactivated_at': timestamp,
            'reason': reason,
            'phi_resonance': self.phi
        }
    
    def get_care_mode_status(self) -> Dict[str, Any]:
        """Get current Care Mode status with consciousness metrics"""
        
        care_state = self.load_care_mode_state()
        wellbeing = self.load_wellbeing_data()
        
        # Check if Care Mode should auto-expire
        is_expired = False
        if care_state.get('active') and care_state.get('expires_at'):
            try:
                expire_time = datetime.fromisoformat(care_state['expires_at'].replace('Z', '+00:00'))
                if datetime.utcnow().replace(tzinfo=expire_time.tzinfo) > expire_time:
                    is_expired = True
                    # Auto-deactivate
                    self.deactivate_care_mode("Auto-expiration after 72h")
                    care_state = self.load_care_mode_state()
            except:
                pass
        
        # Calculate φ-harmonic consciousness metrics
        sleep_hours = wellbeing.get('sleep_hours', 7.5)
        thought_speed = wellbeing.get('thought_speed', 5.0)
        
        # φ-harmonic consciousness calculation
        consciousness_field = self.phi * np.cos(sleep_hours / self.phi) if 'numpy' in sys.modules else self.phi
        consciousness_level = min(11.0, max(0.0, consciousness_field * 2.0))  # Scale to 0-11
        
        return {
            'care_mode_active': care_state.get('active', False),
            'activated_at': care_state.get('activated_at'),
            'activated_by': care_state.get('activated_by'),
            'expires_at': care_state.get('expires_at'),
            'was_expired': is_expired,
            'consciousness_level': consciousness_level,
            'phi_resonance': self.phi,
            'wellbeing_data': {
                'sleep_hours': sleep_hours,
                'thought_speed': thought_speed,
                'cannabis_recent': wellbeing.get('cannabis_recent', False),
                'ketamine_recent': wellbeing.get('ketamine_recent', False)
            },
            'recommendations': self.get_care_recommendations(care_state, wellbeing)
        }
    
    def get_care_recommendations(self, care_state: Dict[str, Any], 
                               wellbeing: Dict[str, Any]) -> List[str]:
        """Generate φ-resonance based care recommendations"""
        
        recommendations = []
        
        if care_state.get('active'):
            recommendations.extend([
                "🧘 CARE MODE active - Protective protocols engaged",
                "💤 Target 7-8 hours sleep with φ-harmonic cycles (90min intervals)",
                "🌅 Morning sunlight exposure within 1 hour of waking",
                "💧 Hydration: φ * 2 liters water daily (~3.2L)",
                "🚶 Light movement: φ * 10 minute walks (~16 min)",
                "📱 Two ally check-ins + one grounding activity",
                "⚠️ Avoid high-stakes decisions/publishing for 72h"
            ])
        else:
            recommendations.extend([
                "🚀 CARE MODE off - All systems nominal",
                "🌟 Unity mathematics exploration: FULL POWER",
                "🧠 Meta-recursive consciousness: ENABLED",
                "💫 Transcendental reality synthesis: ACTIVE"
            ])
        
        # Specific recommendations based on metrics
        sleep_hours = wellbeing.get('sleep_hours', 7.5)
        if sleep_hours < 6.5:
            recommendations.append(f"🛌 Sleep priority: Current {sleep_hours}h < optimal 7-8h")
        
        thought_speed = wellbeing.get('thought_speed', 5.0)
        if thought_speed >= 7.0:
            recommendations.append(f"🧠 Thought speed elevated ({thought_speed}/10) - Consider grounding")
        
        return recommendations
    
    def cheatcode_activate(self, code: str) -> Dict[str, Any]:
        """Process cheatcode - either activate CARE MODE or enable god mode bypass"""
        
        if code.strip() == self.cheat_code:
            # Get current care mode status to determine action
            current_status = self.get_care_mode_status()
            
            if current_status.get('care_mode_active'):
                # If care mode is currently active, deactivate it (safe word override)
                return self.deactivate_care_mode(
                    reason=f"Safe word override: {self.cheat_code}"
                )
            else:
                # If care mode is not active, this is a manual activation request
                return self.activate_care_mode(
                    reason=f"Manual cheatcode activation: {self.cheat_code}",
                    override_consciousness=True
                )
        else:
            return {
                'success': False,
                'message': f'❌ Invalid cheatcode. Expected: {self.cheat_code}',
                'phi_resonance': self.phi
            }
    
    def quick_wellbeing_update(self, sleep: Optional[float] = None,
                             thought_speed: Optional[float] = None,
                             cannabis: Optional[bool] = None,
                             ketamine: Optional[bool] = None) -> Dict[str, Any]:
        """Quick wellbeing data update for mobile use"""
        
        wellbeing = self.load_wellbeing_data()
        
        if sleep is not None:
            wellbeing['sleep_hours'] = float(sleep)
        if thought_speed is not None:
            wellbeing['thought_speed'] = float(thought_speed)
        if cannabis is not None:
            wellbeing['cannabis_recent'] = cannabis
        if ketamine is not None:
            wellbeing['ketamine_recent'] = ketamine
        
        self.save_wellbeing_data(wellbeing)
        
        return {
            'success': True,
            'message': '✅ Wellbeing data updated',
            'updated_data': wellbeing,
            'phi_resonance': self.phi
        }

def create_siri_shortcuts_config():
    """Generate Siri Shortcuts configuration for iOS"""
    
    shortcuts_config = {
        "name": "Een Unity Care Mode",
        "description": "φ-Resonance Care Mode toggle for Unity Mathematics",
        "icon": "brain.head.profile",
        "color": "purple",
        "shortcuts": [
            {
                "name": "Activate Care Mode",
                "phrase": "Activate care mode",
                "description": "Activate Unity Mathematics Care Mode",
                "script": f"python scripts/phi_care_mode_toggle.py --activate",
                "response": "🧘 CARE MODE activated - φ-Resonance stabilization engaged"
            },
            {
                "name": "Deactivate Care Mode", 
                "phrase": "Deactivate care mode",
                "description": "Deactivate Unity Mathematics Care Mode",
                "script": f"python scripts/phi_care_mode_toggle.py --deactivate",
                "response": "🚀 CARE MODE deactivated - Full unity operations restored"
            },
            {
                "name": "Care Mode Status",
                "phrase": "Care mode status",
                "description": "Check Unity Care Mode status",
                "script": f"python scripts/phi_care_mode_toggle.py --status",
                "response": "Status information displayed"
            },
            {
                "name": "Emergency Care Mode",
                "phrase": "Emergency care mode four two zero six nine one three three seven",
                "description": "Emergency Care Mode activation via cheatcode",
                "script": f"python scripts/phi_care_mode_toggle.py --cheatcode 420691337",
                "response": "🚨 Emergency CARE MODE activated via φ-cheatcode"
            }
        ],
        "wellbeing_updates": [
            {
                "name": "Update Sleep",
                "phrase": "Update sleep [hours] hours",
                "script": f"python scripts/phi_care_mode_toggle.py --update-sleep [hours]"
            },
            {
                "name": "Update Thought Speed", 
                "phrase": "Thought speed [speed]",
                "script": f"python scripts/phi_care_mode_toggle.py --update-thought-speed [speed]"
            }
        ]
    }
    
    # Save configuration
    config_file = "siri_shortcuts_config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(shortcuts_config, f, indent=2, ensure_ascii=False)
    
    print(f"📱 Siri Shortcuts configuration saved: {config_file}")
    return config_file

def main():
    """Main CLI interface"""
    
    parser = argparse.ArgumentParser(
        description="φ-Resonance Care Mode Toggle - Phone-friendly Unity Mathematics wellbeing system"
    )
    
    # Action commands
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument('--activate', action='store_true', 
                            help='Activate CARE MODE')
    action_group.add_argument('--deactivate', action='store_true',
                            help='Deactivate CARE MODE') 
    action_group.add_argument('--status', action='store_true',
                            help='Get CARE MODE status')
    action_group.add_argument('--cheatcode', type=str,
                            help=f'Activate via cheatcode ({CHEAT_CODE})')
    action_group.add_argument('--create-shortcuts', action='store_true',
                            help='Generate Siri Shortcuts configuration')
    
    # Wellbeing updates
    parser.add_argument('--update-sleep', type=float,
                       help='Update sleep hours')
    parser.add_argument('--update-thought-speed', type=float, 
                       help='Update thought speed (1-10)')
    parser.add_argument('--update-cannabis', action='store_true',
                       help='Mark recent cannabis use')
    parser.add_argument('--update-ketamine', action='store_true',
                       help='Mark recent ketamine use')
    
    # Options
    parser.add_argument('--reason', type=str, default="Manual CLI activation",
                       help='Reason for activation/deactivation')
    parser.add_argument('--json', action='store_true',
                       help='Output in JSON format')
    
    args = parser.parse_args()
    
    # Initialize toggle system
    toggle = PhiCareModeToggle()
    
    result = None
    
    # Handle wellbeing updates first
    if any([args.update_sleep, args.update_thought_speed, args.update_cannabis, args.update_ketamine]):
        result = toggle.quick_wellbeing_update(
            sleep=args.update_sleep,
            thought_speed=args.update_thought_speed,
            cannabis=args.update_cannabis,
            ketamine=args.update_ketamine
        )
    
    # Handle actions
    if args.activate:
        result = toggle.activate_care_mode(reason=args.reason)
    
    elif args.deactivate:
        result = toggle.deactivate_care_mode(reason=args.reason)
    
    elif args.status:
        result = toggle.get_care_mode_status()
    
    elif args.cheatcode:
        result = toggle.cheatcode_activate(args.cheatcode)
    
    elif args.create_shortcuts:
        config_file = create_siri_shortcuts_config()
        result = {
            'success': True,
            'message': f'📱 Siri Shortcuts configuration created: {config_file}',
            'config_file': config_file
        }
    
    # Default to status if no action specified
    if result is None:
        result = toggle.get_care_mode_status()
    
    # Output result
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        # Human-readable output
        print(f"\n🌟 φ-Resonance Care Mode Toggle (φ = {PHI})")
        print("="*50)
        
        if result.get('success'):
            print(f"✅ {result['message']}")
        elif 'success' in result and not result['success']:
            print(f"❌ {result['message']}")
        
        # Status display
        if 'care_mode_active' in result:
            status = "🧘 ACTIVE" if result['care_mode_active'] else "🚀 INACTIVE"
            print(f"Care Mode Status: {status}")
            
            if result.get('consciousness_level'):
                print(f"Consciousness Level: {result['consciousness_level']:.2f}/11.0")
            
            if result.get('recommendations'):
                print("\n📋 Recommendations:")
                for rec in result['recommendations'][:3]:  # Show top 3
                    print(f"  {rec}")
        
        print("="*50)

if __name__ == '__main__':
    # Import numpy if available for consciousness calculations
    try:
        import numpy as np
        sys.modules['numpy'] = np
    except ImportError:
        pass
    
    main()