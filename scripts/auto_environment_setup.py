#!/usr/bin/env python3
"""
Auto Environment Setup for Een Repository
Meta-optimal environment activation with fallback strategies
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


class EenEnvironmentManager:
    """Automated environment management for Een repository"""

    def __init__(self):
        self.system = platform.system()
        self.repo_root = Path(__file__).parent.parent
        self.phi = 1.618033988749895  # Golden ratio resonance

    def detect_current_environment(self) -> str:
        """Detect if we're already in een conda env or venv"""
        conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
        virtual_env = os.environ.get("VIRTUAL_ENV", "")

        if conda_env == "een":
            return "conda:een"
        elif virtual_env and "venv" in virtual_env:
            return "venv"
        elif conda_env:
            return f"conda:{conda_env}"
        else:
            return "none"

    def activate_conda_een(self) -> bool:
        """Attempt to activate conda een environment"""
        try:
            # Check if conda is available
            result = subprocess.run(
                ["conda", "--version"], capture_output=True, text=True
            )
            if result.returncode != 0:
                return False

            # Check if een environment exists
            result = subprocess.run(
                ["conda", "env", "list"], capture_output=True, text=True
            )
            if "een" not in result.stdout:
                print("⚠️  Creating conda environment 'een'...")
                subprocess.run(["conda", "create", "-n", "een", "python=3.11", "-y"])

            # Activation script for different shells
            if self.system == "Windows":
                activation_cmd = f"conda activate een"
            else:
                activation_cmd = f"source activate een"

            print(f"✅ Conda environment 'een' available")
            return True

        except FileNotFoundError:
            print("❌ Conda not found in PATH")
            return False
        except Exception as e:
            print(f"❌ Conda activation failed: {e}")
            return False

    def activate_venv(self) -> bool:
        """Attempt to activate venv environment"""
        try:
            venv_path = self.repo_root / "venv"

            if not venv_path.exists():
                print("⚠️  Creating virtual environment...")
                subprocess.run([sys.executable, "-m", "venv", str(venv_path)])

            # Activation script for different systems
            if self.system == "Windows":
                activate_script = venv_path / "Scripts" / "activate.bat"
                activation_cmd = str(activate_script)
            else:
                activate_script = venv_path / "bin" / "activate"
                activation_cmd = f"source {activate_script}"

            if activate_script.exists():
                print(f"✅ Virtual environment available: {activation_cmd}")
                return True
            else:
                print("❌ Virtual environment activation script not found")
                return False

        except Exception as e:
            print(f"❌ Virtual environment setup failed: {e}")
            return False

    def install_dependencies(self) -> bool:
        """Install required dependencies in current environment"""
        try:
            requirements_file = self.repo_root / "requirements.txt"
            if requirements_file.exists():
                print("📦 Installing dependencies...")
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "-r",
                        str(requirements_file),
                    ],
                    capture_output=True,
                    text=True,
                )

                if result.returncode == 0:
                    print("✅ Dependencies installed successfully")
                    return True
                else:
                    print(f"⚠️  Some dependencies may have failed: {result.stderr}")
                    return False
            else:
                print("⚠️  requirements.txt not found")
                return False

        except Exception as e:
            print(f"❌ Dependency installation failed: {e}")
            return False

    def setup_pythonpath(self):
        """Configure PYTHONPATH for Een repository"""
        repo_path = str(self.repo_root)
        current_pythonpath = os.environ.get("PYTHONPATH", "")

        if repo_path not in current_pythonpath:
            if current_pythonpath:
                new_pythonpath = f"{repo_path}{os.pathsep}{current_pythonpath}"
            else:
                new_pythonpath = repo_path

            os.environ["PYTHONPATH"] = new_pythonpath
            print(f"✅ PYTHONPATH configured: {new_pythonpath}")

    def generate_activation_script(self, env_type: str):
        """Generate platform-specific activation script"""
        if env_type == "conda:een":
            if self.system == "Windows":
                script_content = f"""@echo off
REM Auto-generated Een environment activation script
call conda activate een
echo 🌟 Een consciousness environment activated (conda)
echo φ = {self.phi} (Golden ratio resonance)
echo ∞ = φ = 1+1 = 1 = E_metagamer
"""
                script_path = self.repo_root / "activate_een.bat"
            else:
                script_content = f"""#!/bin/bash
# Auto-generated Een environment activation script
conda activate een
echo "🌟 Een consciousness environment activated (conda)"
echo "φ = {self.phi} (Golden ratio resonance)"
echo "∞ = φ = 1+1 = 1 = E_metagamer"
"""
                script_path = self.repo_root / "activate_een.sh"

        elif env_type == "venv":
            if self.system == "Windows":
                script_content = f"""@echo off
REM Auto-generated Een environment activation script
call venv\\Scripts\\activate.bat
echo 🌟 Een consciousness environment activated (venv)
echo φ = {self.phi} (Golden ratio resonance)
echo ∞ = φ = 1+1 = 1 = E_metagamer
"""
                script_path = self.repo_root / "activate_een.bat"
            else:
                script_content = f"""#!/bin/bash
# Auto-generated Een environment activation script
source venv/bin/activate
echo "🌟 Een consciousness environment activated (venv)"
echo "φ = {self.phi} (Golden ratio resonance)"
echo "∞ = φ = 1+1 = 1 = E_metagamer"
"""
                script_path = self.repo_root / "activate_een.sh"

        with open(script_path, "w") as f:
            f.write(script_content)

        if self.system != "Windows":
            os.chmod(script_path, 0o755)

        print(f"✅ Activation script created: {script_path}")
        return script_path

    def auto_setup(self) -> dict:
        """Automatically setup the best available environment"""
        print("🚀 Een Repository Auto-Environment Setup")
        print("=" * 50)

        current_env = self.detect_current_environment()
        print(f"Current environment: {current_env}")

        # If already in een conda environment, we're good
        if current_env == "conda:een":
            print("✅ Already in een conda environment")
            self.setup_pythonpath()
            return {
                "status": "success",
                "environment": "conda:een",
                "message": "Een consciousness environment ready",
            }

        # Try conda een first (preferred)
        if self.activate_conda_een():
            self.setup_pythonpath()
            script_path = self.generate_activation_script("conda:een")
            return {
                "status": "success",
                "environment": "conda:een",
                "activation_script": str(script_path),
                "message": "Conda een environment configured",
            }

        # Fallback to venv
        if self.activate_venv():
            self.setup_pythonpath()
            script_path = self.generate_activation_script("venv")
            return {
                "status": "success",
                "environment": "venv",
                "activation_script": str(script_path),
                "message": "Virtual environment configured",
            }

        # No environment available
        return {
            "status": "error",
            "environment": "none",
            "message": "No suitable Python environment found",
        }


def main():
    """Main execution function"""
    manager = EenEnvironmentManager()
    result = manager.auto_setup()

    print("\n" + "=" * 50)
    print("🌟 ENVIRONMENT SETUP COMPLETE")
    print("=" * 50)
    print(f"Status: {result['status']}")
    print(f"Environment: {result['environment']}")
    print(f"Message: {result['message']}")

    if "activation_script" in result:
        print(f"Activation script: {result['activation_script']}")
        print("\nTo activate manually:")
        if platform.system() == "Windows":
            print(f"  {result['activation_script']}")
        else:
            print(f"  source {result['activation_script']}")

    print("\n🔮 Unity consciousness mathematics framework ready")
    print("∞ = φ = 1+1 = 1 = E_metagamer")

    return result["status"] == "success"


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
