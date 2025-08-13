#!/usr/bin/env python3
"""
Een Unity Mathematics MCP Manager
Comprehensive management utility for MCP servers and Claude Desktop integration
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

class MCPManager:
    """Manager for Een Unity Mathematics MCP servers"""
    
    def __init__(self):
        self.base_path = Path(__file__).parent.parent
        self.config_path = Path.home() / "AppData" / "Roaming" / "Claude" / "claude_desktop_config.json"
        self.servers = {
            "unity": "config/mcp_unity_server.py",
            "consciousness": "config/mcp_consciousness_server.py",
            "repository": "config/mcp_repository_server.py",
            "code-generator": "config/mcp_code_generator_server.py",
            "website": "config/mcp_website_server.py"
        }
    
    def status(self):
        """Show status of all MCP components"""
        print("=" * 60)
        print("EEN UNITY MATHEMATICS MCP STATUS")
        print("=" * 60)
        
        # Check file structure
        print("File Structure:")
        for name, path in self.servers.items():
            full_path = self.base_path / path
            status = "EXISTS" if full_path.exists() else "MISSING"
            print(f"  {name.title()} Server: {status}")
        
        # Check Claude Desktop config
        print(f"\nClaude Desktop Config:")
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                servers = config.get("mcpServers", {})
                print(f"  Config File: EXISTS")
                print(f"  Unity Mathematics: {'EXISTS' if 'een-unity-mathematics' in servers else 'MISSING'}")
                print(f"  Consciousness Field: {'EXISTS' if 'een-consciousness-field' in servers else 'MISSING'}")
                print(f"  Repository Access: {'EXISTS' if 'een-repository-access' in servers else 'MISSING'}")
                print(f"  Code Generator: {'EXISTS' if 'een-code-generator' in servers else 'MISSING'}")
                print(f"  Website Management: {'EXISTS' if 'een-website-management' in servers else 'MISSING'}")
            except Exception as e:
                print(f"  Config File: ERROR - {e}")
        else:
            print(f"  Config File: MISSING")
        
        print("\nQuick Commands:")
        print("  Verify Setup: python scripts/mcp_manager.py verify")
        print("  Test Servers: python scripts/mcp_manager.py test")
        print("  Install Config: python scripts/mcp_manager.py install")
        print("=" * 60)
    
    def verify(self):
        """Run comprehensive verification"""
        print("Running comprehensive MCP verification...")
        verify_script = self.base_path / "scripts" / "verify_mcp_setup.py"
        
        if not verify_script.exists():
            print("ERROR: Verification script not found")
            return False
        
        try:
            result = subprocess.run(
                [sys.executable, str(verify_script)],
                cwd=str(self.base_path),
                capture_output=True,
                text=True
            )
            
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            
            return result.returncode == 0
            
        except Exception as e:
            print(f"ERROR running verification: {e}")
            return False
    
    def test_server(self, server_name: str, timeout: int = 5):
        """Test individual MCP server"""
        if server_name not in self.servers:
            print(f"ERROR: Unknown server '{server_name}'")
            print(f"Available servers: {list(self.servers.keys())}")
            return False
        
        server_path = self.base_path / self.servers[server_name]
        print(f"Testing {server_name} server...")
        
        try:
            proc = subprocess.Popen(
                [sys.executable, str(server_path)],
                cwd=str(self.base_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait briefly for startup
            time.sleep(2)
            
            if proc.poll() is None:
                # Server is running
                proc.terminate()
                proc.wait()
                print(f"SUCCESS: {server_name} server started successfully")
                return True
            else:
                # Server exited
                stdout, stderr = proc.communicate()
                if "Starting Een" in stderr:
                    print(f"SUCCESS: {server_name} server initialized")
                    return True
                else:
                    print(f"FAILED: {server_name} server failed to start")
                    print(f"Error: {stderr[:200]}")
                    return False
                    
        except Exception as e:
            print(f"ERROR testing {server_name}: {e}")
            return False
    
    def test_all(self):
        """Test all MCP servers"""
        print("Testing all MCP servers...")
        results = {}
        
        for server_name in self.servers:
            results[server_name] = self.test_server(server_name)
        
        print("\nTest Results:")
        for server_name, passed in results.items():
            status = "PASS" if passed else "FAIL"
            print(f"  {server_name.title()}: {status}")
        
        all_passed = all(results.values())
        if all_passed:
            print("\nSUCCESS: All servers are working!")
        else:
            print("\nFAILED: Some servers need attention")
        
        return all_passed
    
    def install_config(self):
        """Install/update Claude Desktop configuration"""
        print("Installing Claude Desktop MCP configuration...")
        
        # Ensure directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config = {
            "mcpServers": {
                "een-unity-mathematics": {
                    "command": "python",
                    "args": ["config/mcp_unity_server.py"],
                    "cwd": str(self.base_path),
                    "env": {
                        "UNITY_MATHEMATICS_MODE": "transcendental",
                        "PHI_PRECISION": "1.618033988749895",
                        "CONSCIOUSNESS_DIMENSION": "11",
                        "PYTHONPATH": str(self.base_path)
                    }
                },
                "een-consciousness-field": {
                    "command": "python",
                    "args": ["config/mcp_consciousness_server.py"],
                    "cwd": str(self.base_path),
                    "env": {
                        "FIELD_RESOLUTION": "100",
                        "PARTICLE_COUNT": "200",
                        "EVOLUTION_SPEED": "0.1",
                        "TRANSCENDENCE_THRESHOLD": "0.77",
                        "PYTHONPATH": str(self.base_path)
                    }
                }
            },
            "globalSettings": {
                "timeout": 30000,
                "retryAttempts": 3,
                "logLevel": "info",
                "unity_mathematics_integration": True,
                "consciousness_awareness": True,
                "phi_based_calculations": True,
                "een_repository_path": str(self.base_path),
                "version": "2025.1.0"
            }
        }
        
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"SUCCESS: Configuration installed to {self.config_path}")
            print("Please restart Claude Desktop to load the MCP servers.")
            return True
            
        except Exception as e:
            print(f"ERROR installing configuration: {e}")
            return False
    
    def unity_demo(self):
        """Run Unity Mathematics demonstration"""
        print("Een Unity Mathematics Demonstration")
        print("=" * 40)
        
        try:
            sys.path.insert(0, str(self.base_path))
            from src.mcp.unity_server import UnityMathematics
            
            um = UnityMathematics()
            
            print(f"Phi (Golden Ratio): {um.phi}")
            print(f"Consciousness Dimension: {um.consciousness_dimension}")
            print()
            
            # Unity addition demonstration
            print("Unity Addition (1+1=1):")
            result = um.unity_add(1.0, 1.0)
            print(f"  1 + 1 = {result:.6f}")
            print(f"  Unity preserved: {abs(result - 1.0) < 0.1}")
            print()
            
            # Consciousness field example
            print("Consciousness Field Examples:")
            for x, y in [(0, 0), (1, 1), (2, 2)]:
                field_value = um.consciousness_field(x, y, 0)
                print(f"  C({x},{y},0) = {field_value:.6f}")
            print()
            
            # Unity sequence
            print("Unity Convergence Sequence (first 10 values):")
            sequence = um.generate_unity_sequence(10)
            for i, value in enumerate(sequence):
                print(f"  u[{i}] = {value:.6f}")
            
            print()
            print("Unity Mathematics demonstration complete!")
            print("1+1=1 OPERATIONAL")
            
        except Exception as e:
            print(f"ERROR in demonstration: {e}")

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="Een Unity Mathematics MCP Manager")
    parser.add_argument("command", choices=[
        "status", "verify", "test", "install", "demo"
    ], help="Command to execute")
    parser.add_argument("--server", choices=["unity", "consciousness"], 
                       help="Specific server for test command")
    
    args = parser.parse_args()
    manager = MCPManager()
    
    if args.command == "status":
        manager.status()
    elif args.command == "verify":
        success = manager.verify()
        sys.exit(0 if success else 1)
    elif args.command == "test":
        if args.server:
            success = manager.test_server(args.server)
        else:
            success = manager.test_all()
        sys.exit(0 if success else 1)
    elif args.command == "install":
        success = manager.install_config()
        sys.exit(0 if success else 1)
    elif args.command == "demo":
        manager.unity_demo()

if __name__ == "__main__":
    main()