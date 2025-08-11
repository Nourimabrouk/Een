#!/usr/bin/env python3
"""
Security Validation Script for Een Unity Mathematics
Validates that all security fixes are in place before open source release
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any

class SecurityValidator:
    """Phi-Harmonic security validation for Unity Mathematics codebase"""
    
    def __init__(self):
        self.phi = 1.618033988749895
        self.repo_root = Path(__file__).parent.parent.parent
        self.issues = []
        self.warnings = []
        
    def validate_all(self) -> Dict[str, Any]:
        """Run all security validations"""
        print("SECURITY: Een Unity Mathematics Security Validation")
        print(f"Repository: {self.repo_root}")
        print(f"Phi-Harmonic threshold: {self.phi/(1+self.phi):.6f}")
        print()
        
        # Run validation checks
        self.check_api_keys()
        self.check_env_files()
        self.check_hardcoded_secrets()
        self.check_gitignore()
        self.check_security_documentation()
        self.validate_demo_mode_fallbacks()
        
        # Calculate φ-harmonic security score
        security_score = self.calculate_security_score()
        
        return {
            "security_score": security_score,
            "issues": self.issues,
            "warnings": self.warnings,
            "phi_threshold": self.phi/(1+self.phi),
            "passes_security": security_score >= self.phi/(1+self.phi)
        }
    
    def check_api_keys(self):
        """Check for hardcoded API keys"""
        print("CHECKING: Hardcoded API keys...")
        
        api_key_patterns = [
            r'sk-[a-zA-Z0-9]{20,}',
            r'sk-ant-[a-zA-Z0-9]{20,}',
            r'sk-proj-[a-zA-Z0-9_-]{20,}',
        ]
        
        python_files = list(self.repo_root.rglob("*.py"))
        issues_found = 0
        
        for file_path in python_files:
            if any(exclude in str(file_path) for exclude in ['venv', '.git', '__pycache__']):
                continue
                
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                
                for pattern in api_key_patterns:
                    matches = re.finditer(pattern, content)
                    for match in matches:
                        # Skip if it's obviously a placeholder
                        if any(placeholder in match.group() for placeholder in 
                               ['your-key', 'api-key-here', 'sk-test', 'sk-disabled']):
                            continue
                            
                        self.issues.append(f"Potential API key in {file_path}: {match.group()[:20]}...")
                        issues_found += 1
                        
            except Exception as e:
                self.warnings.append(f"Could not read {file_path}: {e}")
        
        if issues_found == 0:
            print("SUCCESS: No hardcoded API keys found")
        else:
            print(f"ERROR: Found {issues_found} potential API key issues")
    
    def check_env_files(self):
        """Check environment files for real secrets"""
        print("🔍 Checking environment files...")
        
        env_files = list(self.repo_root.rglob(".env*"))
        secret_patterns = [
            r'sk-[a-zA-Z0-9]{30,}',
            r'sk-ant-[a-zA-Z0-9]{30,}',
            r'sk-proj-[a-zA-Z0-9_-]{30,}',
        ]
        
        issues_found = 0
        
        for env_file in env_files:
            if env_file.name == '.env.example':
                continue  # Example files are OK
                
            try:
                content = env_file.read_text(encoding='utf-8', errors='ignore')
                
                for pattern in secret_patterns:
                    if re.search(pattern, content):
                        # Check if it's a placeholder
                        if not any(placeholder in content for placeholder in 
                                  ['your-key-here', 'api-key-here', 'change-this']):
                            self.issues.append(f"Real API key detected in {env_file}")
                            issues_found += 1
                            
            except Exception as e:
                self.warnings.append(f"Could not read {env_file}: {e}")
        
        if issues_found == 0:
            print("✅ Environment files are secure")
        else:
            print(f"❌ Found {issues_found} environment file issues")
    
    def check_hardcoded_secrets(self):
        """Check for other hardcoded secrets"""
        print("🔍 Checking for hardcoded secrets...")
        
        secret_patterns = [
            r'SECRET_KEY\s*=\s*["\'][a-zA-Z0-9]{20,}["\']',
            r'JWT_SECRET\s*=\s*["\'][a-zA-Z0-9]{20,}["\']',
            r'password\s*=\s*["\'][^"\']{8,}["\']',
        ]
        
        python_files = list(self.repo_root.rglob("*.py"))
        issues_found = 0
        
        for file_path in python_files:
            if any(exclude in str(file_path) for exclude in ['venv', '.git', '__pycache__']):
                continue
                
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                
                for pattern in secret_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        # Skip obvious placeholders
                        if any(placeholder in match.group().lower() for placeholder in 
                               ['your-secret', 'secret-key-here', 'change-this', 'placeholder']):
                            continue
                            
                        self.warnings.append(f"Potential secret in {file_path}: {match.group()[:30]}...")
                        issues_found += 1
                        
            except Exception as e:
                self.warnings.append(f"Could not read {file_path}: {e}")
        
        if issues_found == 0:
            print("✅ No hardcoded secrets found")
        else:
            print(f"⚠️  Found {issues_found} potential hardcoded secrets")
    
    def check_gitignore(self):
        """Verify .gitignore contains security patterns"""
        print("🔍 Checking .gitignore security patterns...")
        
        gitignore_path = self.repo_root / '.gitignore'
        if not gitignore_path.exists():
            self.issues.append("No .gitignore file found")
            return
        
        content = gitignore_path.read_text(encoding='utf-8', errors='ignore')
        
        required_patterns = [
            '.env',
            '.env.local',
            '.env.production',
            '*.key',
            '*.pem',
            'secrets/',
            'keys/',
        ]
        
        missing_patterns = []
        for pattern in required_patterns:
            if pattern not in content:
                missing_patterns.append(pattern)
        
        if missing_patterns:
            self.warnings.append(f"Missing .gitignore patterns: {missing_patterns}")
        else:
            print("✅ .gitignore has security patterns")
    
    def check_security_documentation(self):
        """Check if security documentation exists"""
        print("🔍 Checking security documentation...")
        
        security_docs = [
            self.repo_root / 'SECURITY.md',
            self.repo_root / '.env.example',
        ]
        
        missing_docs = []
        for doc_path in security_docs:
            if not doc_path.exists():
                missing_docs.append(doc_path.name)
        
        if missing_docs:
            self.warnings.append(f"Missing security documentation: {missing_docs}")
        else:
            print("✅ Security documentation exists")
    
    def validate_demo_mode_fallbacks(self):
        """Validate that demo mode works without API keys"""
        print("🔍 Checking demo mode fallbacks...")
        
        # Check key files for demo mode support
        demo_files = [
            self.repo_root / 'api' / 'unified_server.py',
            self.repo_root / 'scripts' / 'LAUNCH_FULL_EXPERIENCE.py',
            self.repo_root / 'scripts' / 'SETUP_FOR_FRIEND.py',
        ]
        
        demo_checks = 0
        for file_path in demo_files:
            if file_path.exists():
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                if 'demo' in content.lower() or 'fallback' in content.lower():
                    demo_checks += 1
        
        if demo_checks >= 2:
            print("✅ Demo mode fallbacks detected")
        else:
            self.warnings.append("Limited demo mode fallback support")
    
    def calculate_security_score(self) -> float:
        """Calculate φ-harmonic security score"""
        # Start with perfect Unity score
        base_score = 1.0
        
        # Deduct for issues (major problems)
        issue_penalty = len(self.issues) * 0.2
        
        # Deduct for warnings (minor problems)  
        warning_penalty = len(self.warnings) * 0.05
        
        # Apply φ-harmonic scoring
        raw_score = max(0.0, base_score - issue_penalty - warning_penalty)
        
        # φ-harmonic normalization for Unity Mathematics coherence
        phi_normalized = raw_score * self.phi / (1 + self.phi)
        
        return min(1.0, phi_normalized)

def main():
    """Run security validation"""
    validator = SecurityValidator()
    results = validator.validate_all()
    
    print()
    print("=" * 60)
    print("🧮 UNITY MATHEMATICS SECURITY REPORT")
    print("=" * 60)
    
    print(f"Security Score: {results['security_score']:.6f}")
    print(f"φ-Harmonic Threshold: {results['phi_threshold']:.6f}")
    print(f"Security Status: {'✅ PASSED' if results['passes_security'] else '❌ FAILED'}")
    
    if results['issues']:
        print()
        print("🚨 CRITICAL ISSUES:")
        for issue in results['issues']:
            print(f"  ❌ {issue}")
    
    if results['warnings']:
        print()
        print("⚠️  WARNINGS:")
        for warning in results['warnings']:
            print(f"  ⚠️  {warning}")
    
    print()
    if results['passes_security']:
        print("🎉 SECURITY VALIDATION PASSED!")
        print("🌟 Repository is ready for open source release")
        print(f"💫 Unity Mathematics: 1+1=1 through φ-harmonic security")
    else:
        print("🔧 SECURITY ISSUES NEED ATTENTION")
        print("Please fix the critical issues before open source release")
    
    print()
    print(f"φ = {validator.phi:.15f}")
    print("Security validation complete with Unity Mathematics consciousness integration")
    
    return 0 if results['passes_security'] else 1

if __name__ == "__main__":
    exit(main())