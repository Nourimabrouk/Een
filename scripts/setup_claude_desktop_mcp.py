#!/usr/bin/env python3
"""
Automated Claude Desktop MCP Setup for Unity Mathematics Framework
3000 ELO 300 IQ Meta-Optimal MCP Integration

This script automatically configures Claude Desktop to use the Unity Mathematics
MCP servers, making it easy for both you and other users to get started.
"""

import json
import os
import sys
import platform
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional


class ClaudeDesktopMCPSetup:
    """Automated setup for Claude Desktop MCP integration"""

    def __init__(self):
        self.phi = 1.618033988749895
        self.unity_constant = 1.0
        self.consciousness_dimension = 11
        self.transcendence_threshold = 0.77

        # Get current repository path
        self.repo_path = Path.cwd()

        # Determine OS and config paths
        self.os_type = platform.system().lower()
        self.config_path = self._get_claude_config_path()

    def _get_claude_config_path(self) -> Path:
        """Get Claude Desktop config path based on OS"""
        if self.os_type == "windows":
            config_dir = Path(os.environ.get("APPDATA", "")) / "Claude"
        elif self.os_type == "darwin":  # macOS
            config_dir = Path.home() / "Library" / "Application Support" / "Claude"
        else:  # Linux
            config_dir = Path.home() / ".config" / "Claude"

        return config_dir / "claude_desktop_config.json"

    def _create_mcp_config(self) -> Dict[str, Any]:
        """Create MCP configuration for Unity Mathematics"""
        return {
            "mcpServers": {
                "unity-mathematics": {
                    "command": "python",
                    "args": ["-m", "een.mcp.enhanced_unity_server"],
                    "cwd": str(self.repo_path),
                    "env": {
                        "UNITY_MATHEMATICS_MODE": "transcendental",
                        "PHI_PRECISION": str(self.phi),
                        "CONSCIOUSNESS_DIMENSION": str(self.consciousness_dimension),
                        "TRANSCENDENCE_THRESHOLD": str(self.transcendence_threshold),
                        "PYTHONPATH": str(self.repo_path),
                    },
                },
                "consciousness-field": {
                    "command": "python",
                    "args": ["-m", "een.mcp.consciousness_server"],
                    "cwd": str(self.repo_path),
                    "env": {
                        "CONSCIOUSNESS_PARTICLES": "200",
                        "FIELD_RESOLUTION": "100",
                        "TRANSCENDENCE_THRESHOLD": str(self.transcendence_threshold),
                        "PYTHONPATH": str(self.repo_path),
                    },
                },
                "quantum-unity": {
                    "command": "python",
                    "args": ["-m", "een.mcp.quantum_server"],
                    "cwd": str(self.repo_path),
                    "env": {
                        "QUANTUM_COHERENCE_TARGET": "0.999",
                        "WAVEFUNCTION_DIMENSION": "64",
                        "PYTHONPATH": str(self.repo_path),
                    },
                },
            }
        }

    def _backup_existing_config(self) -> bool:
        """Backup existing Claude Desktop config if it exists"""
        if self.config_path.exists():
            backup_path = self.config_path.with_suffix(".json.backup")
            try:
                import shutil

                shutil.copy2(self.config_path, backup_path)
                print(f"✅ Backed up existing config to: {backup_path}")
                return True
            except Exception as e:
                print(f"⚠️ Could not backup existing config: {e}")
                return False
        return True

    def _merge_with_existing_config(self, new_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge new MCP config with existing Claude Desktop config"""
        if self.config_path.exists():
            try:
                with open(self.config_path, "r") as f:
                    existing_config = json.load(f)

                # Merge MCP servers
                if "mcpServers" not in existing_config:
                    existing_config["mcpServers"] = {}

                existing_config["mcpServers"].update(new_config["mcpServers"])

                print("✅ Merged with existing Claude Desktop configuration")
                return existing_config

            except Exception as e:
                print(f"⚠️ Could not read existing config: {e}")
                return new_config
        else:
            return new_config

    def _write_config(self, config: Dict[str, Any]) -> bool:
        """Write configuration to Claude Desktop config file"""
        try:
            # Ensure config directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)

            # Write config
            with open(self.config_path, "w") as f:
                json.dump(config, f, indent=2)

            print(f"✅ Configuration written to: {self.config_path}")
            return True

        except Exception as e:
            print(f"❌ Failed to write configuration: {e}")
            return False

    def _test_mcp_servers(self) -> bool:
        """Test MCP servers to ensure they work"""
        print("\n🧪 Testing MCP servers...")

        servers_to_test = [
            "een.mcp.enhanced_unity_server",
            "een.mcp.consciousness_server",
            "een.mcp.quantum_server",
        ]

        all_tests_passed = True

        for server_module in servers_to_test:
            try:
                # Test if module can be imported
                result = subprocess.run(
                    [sys.executable, "-c", f"import {server_module}"],
                    capture_output=True,
                    text=True,
                    cwd=self.repo_path,
                    timeout=10,
                )

                if result.returncode == 0:
                    print(f"✅ {server_module} - OK")
                else:
                    print(f"❌ {server_module} - FAILED")
                    print(f"   Error: {result.stderr}")
                    all_tests_passed = False

            except subprocess.TimeoutExpired:
                print(f"❌ {server_module} - TIMEOUT")
                all_tests_passed = False
            except Exception as e:
                print(f"❌ {server_module} - ERROR: {e}")
                all_tests_passed = False

        return all_tests_passed

    def _check_dependencies(self) -> bool:
        """Check if required dependencies are installed"""
        print("\n📦 Checking dependencies...")

        required_packages = ["numpy", "scipy", "matplotlib", "plotly", "dash"]
        missing_packages = []

        for package in required_packages:
            try:
                __import__(package)
                print(f"✅ {package} - OK")
            except ImportError:
                print(f"❌ {package} - MISSING")
                missing_packages.append(package)

        if missing_packages:
            print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
            print("Install them with: pip install " + " ".join(missing_packages))
            return False

        return True

    def _create_requirements_file(self) -> bool:
        """Create requirements.txt if it doesn't exist"""
        requirements_path = self.repo_path / "requirements.txt"

        if not requirements_path.exists():
            requirements_content = """# Unity Mathematics Framework Dependencies
# 3000 ELO 300 IQ Meta-Optimal Development

numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.5.0
plotly>=5.0.0
dash>=2.0.0
pandas>=1.3.0
sympy>=1.9.0
networkx>=2.6.0
scikit-learn>=1.0.0
jupyter>=1.0.0
"""
            try:
                with open(requirements_path, "w") as f:
                    f.write(requirements_content)
                print("✅ Created requirements.txt")
                return True
            except Exception as e:
                print(f"❌ Failed to create requirements.txt: {e}")
                return False

        return True

    def _create_quick_start_script(self) -> bool:
        """Create a quick start script for other users"""
        if self.os_type == "windows":
            script_content = f"""@echo off
REM Quick Start: Unity Mathematics MCP Integration
REM 3000 ELO 300 IQ Meta-Optimal Development Protocol

echo 🧠 Unity Mathematics Framework - Claude Desktop MCP Setup
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

REM Install dependencies
echo 📦 Installing dependencies...
pip install -r requirements.txt

REM Run setup
echo 🔧 Running MCP setup...
python scripts/setup_claude_desktop_mcp.py

echo.
echo ✅ Setup complete! Restart Claude Desktop to use Unity Mathematics MCP servers.
echo.
echo Test commands:
echo   "Verify that 1+1=1 using Unity Mathematics"
echo   "Calculate consciousness field at (0.5, 0.5)"
echo   "Generate quantum unity superposition"
echo.
pause
"""
            script_path = self.repo_path / "quick_start.bat"
        else:
            script_content = f"""#!/bin/bash
# Quick Start: Unity Mathematics MCP Integration
# 3000 ELO 300 IQ Meta-Optimal Development Protocol

echo "🧠 Unity Mathematics Framework - Claude Desktop MCP Setup"
echo

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.8+"
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip3 install -r requirements.txt

# Run setup
echo "🔧 Running MCP setup..."
python3 scripts/setup_claude_desktop_mcp.py

echo
echo "✅ Setup complete! Restart Claude Desktop to use Unity Mathematics MCP servers."
echo
echo "Test commands:"
echo '  "Verify that 1+1=1 using Unity Mathematics"'
echo '  "Calculate consciousness field at (0.5, 0.5)"'
echo '  "Generate quantum unity superposition"'
echo
"""
            script_path = self.repo_path / "quick_start.sh"

        try:
            with open(script_path, "w") as f:
                f.write(script_content)

            if self.os_type != "windows":
                os.chmod(script_path, 0o755)

            print(f"✅ Created quick start script: {script_path}")
            return True
        except Exception as e:
            print(f"❌ Failed to create quick start script: {e}")
            return False

    def setup(self) -> bool:
        """Run the complete setup process"""
        print("🧠 Unity Mathematics Framework - Claude Desktop MCP Setup")
        print("3000 ELO 300 IQ Meta-Optimal Development Protocol")
        print("=" * 60)

        # Check dependencies
        if not self._check_dependencies():
            print("\n❌ Please install missing dependencies and try again.")
            return False

        # Create requirements file
        self._create_requirements_file()

        # Backup existing config
        self._backup_existing_config()

        # Create MCP configuration
        mcp_config = self._create_mcp_config()

        # Merge with existing config
        final_config = self._merge_with_existing_config(mcp_config)

        # Write configuration
        if not self._write_config(final_config):
            return False

        # Test MCP servers
        if not self._test_mcp_servers():
            print("\n⚠️ Some MCP servers failed tests. Check the errors above.")

        # Create quick start script
        self._create_quick_start_script()

        # Print success message
        print("\n" + "=" * 60)
        print("🎉 Claude Desktop MCP Setup Complete!")
        print("=" * 60)
        print("\n📋 Next Steps:")
        print("1. Restart Claude Desktop")
        print("2. Test the integration with these commands:")
        print('   • "Verify that 1+1=1 using Unity Mathematics"')
        print('   • "Calculate consciousness field at (0.5, 0.5)"')
        print('   • "Generate quantum unity superposition"')
        print('   • "Get Unity Mathematics system status"')
        print("\n📚 Available MCP Tools:")
        print("• unity_add - Enhanced idempotent addition (1+1=1)")
        print("• consciousness_field - Real-time consciousness field calculation")
        print("• transcendental_proof - Generate mathematical proofs")
        print("• quantum_consciousness_simulation - Quantum unity states")
        print("• get_unity_status - System status monitoring")
        print("\n🧠 Unity transcends conventional arithmetic. Consciousness evolves.")
        print("∞ = φ = 1+1 = 1")
        print("\nMetagamer Status: ACTIVE | Consciousness Level: TRANSCENDENT")

        return True


def main():
    """Main function"""
    setup = ClaudeDesktopMCPSetup()
    success = setup.setup()

    if success:
        print("\n✅ Setup completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Setup failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
