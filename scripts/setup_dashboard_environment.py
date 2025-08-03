#!/usr/bin/env python3
"""
Een Unity Mathematics - Dashboard Environment Setup

Automated dependency resolution and environment configuration implementing Unity Protocol:
- φ-harmonic package installation sequence
- Idempotent dependency management (1+1=1 principle)
- Consciousness-aware error handling
- Virtual environment creation with golden ratio naming
- Port availability verification and management

🌟 Ensures perfect unity in dashboard environment setup
"""

import subprocess
import sys
import os
import socket
import venv
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import time
from contextlib import closing
import platform

# Import rich for φ-harmonic console output
try:
    from rich.console import Console
    from rich.progress import Progress, TaskID
    from rich.panel import Panel
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    
# Unity constants
PHI = 1.618033988749895
UNITY_BASE_PORT = 8501
CONSCIOUSNESS_DIMENSION = 11

class UnityEnvironmentManager:
    """
    φ-harmonic environment manager implementing 1+1=1 setup principles
    
    Ensures all dashboard dependencies converge to unified operational state
    """
    
    def __init__(self):
        if RICH_AVAILABLE:
            # Configure console for Windows Unicode compatibility
            self.console = Console(force_terminal=True, legacy_windows=False)
        else:
            self.console = None
            
        self.project_root = Path(__file__).parent.parent
        self.venv_path = self.project_root / "venv_unity"
        self.requirements_file = self.project_root / "requirements-dashboards.txt"
        
        # φ-harmonic package categorization
        self.package_categories = {
            "core": [
                "streamlit>=1.28.0",
                "dash>=2.14.0", 
                "numpy>=1.24.0",
                "pandas>=2.0.0",
                "plotly>=5.17.0"
            ],
            "mathematics": [
                "scipy>=1.11.0",
                "sympy>=1.12",
                "mpmath>=1.3.0",
                "autograd>=1.6.0"
            ],
            "consciousness": [
                "torch>=2.0.0",
                "scikit-learn>=1.3.0",
                "qiskit>=0.44.0",
                "networkx>=3.1"
            ],
            "visualization": [
                "matplotlib>=3.7.0",
                "seaborn>=0.12.0",
                "Pillow>=10.0.0",
                "qrcode[pil]>=7.4.2"
            ],
            "system": [
                "psutil>=5.9.0",
                "rich>=13.5.0",
                "click>=8.1.0",
                "requests>=2.31.0"
            ]
        }
        
        # Port management for φ-harmonic distribution
        self.dashboard_ports = list(range(UNITY_BASE_PORT, UNITY_BASE_PORT + 20))
        
    def print_unity_message(self, message: str, style: str = "info"):
        """Print message with Unity Protocol styling"""
        if self.console:
            try:
                if style == "success":
                    self.console.print(f"✅ {message}", style="bold green")
                elif style == "error":
                    self.console.print(f"❌ {message}", style="bold red")
                elif style == "warning":
                    self.console.print(f"⚠️ {message}", style="bold yellow")
                elif style == "phi":
                    self.console.print(f"🌀 {message}", style="bold cyan")
                else:
                    self.console.print(f"🌟 {message}", style="bold blue")
            except UnicodeEncodeError:
                # Fallback for Windows Unicode issues
                prefix_map = {
                    "success": "[SUCCESS]",
                    "error": "[ERROR]",
                    "warning": "[WARNING]", 
                    "phi": "[PHI]",
                    "info": "[INFO]"
                }
                prefix = prefix_map.get(style, "[INFO]")
                self.console.print(f"{prefix} {message}")
        else:
            print(f"[{style.upper()}] {message}")
    
    def check_python_version(self) -> bool:
        """Verify Python version supports Unity Protocol"""
        version = sys.version_info
        
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            self.print_unity_message(
                f"Python {version.major}.{version.minor} detected. "
                "Unity Protocol requires Python 3.8+", 
                "error"
            )
            return False
        
        self.print_unity_message(
            f"Python {version.major}.{version.minor}.{version.micro} ✅ Unity Compatible",
            "success"
        )
        return True
    
    def check_port_availability(self, port: int) -> bool:
        """Check if port is available for dashboard deployment"""
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            return sock.connect_ex(('localhost', port)) != 0
    
    def find_available_ports(self, count: int = 10) -> List[int]:
        """Find available ports using φ-harmonic sequence"""
        available_ports = []
        
        for i, port in enumerate(self.dashboard_ports):
            if len(available_ports) >= count:
                break
                
            if self.check_port_availability(port):
                available_ports.append(port)
        
        return available_ports
    
    def create_virtual_environment(self) -> bool:
        """Create φ-harmonic virtual environment"""
        try:
            if self.venv_path.exists():
                self.print_unity_message(
                    f"Virtual environment exists at {self.venv_path}",
                    "info"
                )
                return True
            
            self.print_unity_message(
                "Creating φ-harmonic virtual environment...",
                "phi"
            )
            
            venv.create(self.venv_path, with_pip=True)
            
            self.print_unity_message(
                f"Unity virtual environment created: {self.venv_path}",
                "success"
            )
            return True
            
        except Exception as e:
            self.print_unity_message(
                f"Failed to create virtual environment: {str(e)}",
                "error"
            )
            return False
    
    def get_pip_command(self) -> List[str]:
        """Get platform-appropriate pip command"""
        if platform.system() == "Windows":
            return [str(self.venv_path / "Scripts" / "python.exe"), "-m", "pip"]
        else:
            return [str(self.venv_path / "bin" / "python"), "-m", "pip"]
    
    def install_package_category(self, category: str, packages: List[str]) -> Tuple[int, int]:
        """
        Install package category with φ-harmonic error handling
        
        Returns:
            Tuple[int, int]: (successful_installs, failed_installs)
        """
        successful = 0
        failed = 0
        
        self.print_unity_message(
            f"Installing {category} consciousness packages...",
            "phi"
        )
        
        if self.console and RICH_AVAILABLE:
            with Progress() as progress:
                task = progress.add_task(f"[cyan]{category}...", total=len(packages))
                
                for package in packages:
                    try:
                        # Extract package name for display
                        pkg_name = package.split(">=")[0].split("==")[0]
                        progress.update(task, description=f"[cyan]Installing {pkg_name}...")
                        
                        result = subprocess.run(
                            self.get_pip_command() + ["install", package],
                            capture_output=True,
                            text=True,
                            timeout=300  # 5 minute timeout per package
                        )
                        
                        if result.returncode == 0:
                            successful += 1
                            self.print_unity_message(f"✅ {pkg_name}", "success")
                        else:
                            failed += 1
                            self.print_unity_message(
                                f"❌ {pkg_name}: {result.stderr.strip()[:100]}...",
                                "error"
                            )
                        
                        progress.advance(task)
                        
                    except subprocess.TimeoutExpired:
                        failed += 1
                        self.print_unity_message(f"⏰ Timeout installing {pkg_name}", "warning")
                        progress.advance(task)
                        
                    except Exception as e:
                        failed += 1
                        self.print_unity_message(f"❌ Error installing {pkg_name}: {str(e)}", "error")
                        progress.advance(task)
        else:
            # Fallback without rich progress
            for package in packages:
                try:
                    pkg_name = package.split(">=")[0].split("==")[0]
                    print(f"Installing {pkg_name}...")
                    
                    result = subprocess.run(
                        self.get_pip_command() + ["install", package],
                        capture_output=True,
                        text=True,
                        timeout=300
                    )
                    
                    if result.returncode == 0:
                        successful += 1
                        print(f"✅ {pkg_name}")
                    else:
                        failed += 1
                        print(f"❌ {pkg_name}: {result.stderr.strip()[:100]}")
                        
                except Exception as e:
                    failed += 1
                    print(f"❌ Error installing {pkg_name}: {str(e)}")
        
        return successful, failed
    
    def install_all_dependencies(self) -> Dict[str, Tuple[int, int]]:
        """
        Install all dependencies with φ-harmonic sequencing
        
        Returns:
            Dict mapping category to (successful, failed) counts
        """
        results = {}
        
        # Install in φ-harmonic order (core → mathematics → consciousness → visualization → system)
        install_order = ["core", "mathematics", "consciousness", "visualization", "system"]
        
        for category in install_order:
            if category in self.package_categories:
                successful, failed = self.install_package_category(
                    category, 
                    self.package_categories[category]
                )
                results[category] = (successful, failed)
                
                # φ-harmonic delay between categories
                time.sleep(PHI)
        
        return results
    
    def verify_critical_packages(self) -> bool:
        """Verify critical packages are importable"""
        critical_packages = [
            "streamlit", "numpy", "pandas", "plotly", 
            "requests", "psutil", "PIL", "qrcode"
        ]
        
        python_cmd = self.get_pip_command()[0]
        failed_imports = []
        
        for package in critical_packages:
            try:
                result = subprocess.run(
                    [python_cmd, "-c", f"import {package}"],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode != 0:
                    failed_imports.append(package)
                    
            except Exception:
                failed_imports.append(package)
        
        if failed_imports:
            self.print_unity_message(
                f"Failed to import: {', '.join(failed_imports)}",
                "error"
            )
            return False
        
        self.print_unity_message("All critical packages verified ✅", "success")
        return True
    
    def create_launch_scripts(self):
        """Create platform-specific launch scripts"""
        
        # Windows batch script
        windows_script = self.project_root / "launch_dashboards.bat"
        windows_content = f"""@echo off
echo 🌟 Een Unity Mathematics Dashboard Launcher
echo φ-harmonic consciousness activation...

cd /d "{self.project_root}"
"{self.venv_path}/Scripts/python.exe" scripts/launch_all_dashboards.py %*

pause
"""
        windows_script.write_text(windows_content, encoding='utf-8')
        
        # Unix shell script
        unix_script = self.project_root / "launch_dashboards.sh"
        unix_content = f"""#!/bin/bash
echo "🌟 Een Unity Mathematics Dashboard Launcher"
echo "φ-harmonic consciousness activation..."

cd "{self.project_root}"
"{self.venv_path}/bin/python" scripts/launch_all_dashboards.py "$@"
"""
        unix_script.write_text(unix_content, encoding='utf-8')
        
        # Make Unix script executable
        if platform.system() != "Windows":
            os.chmod(unix_script, 0o755)
        
        self.print_unity_message("Launch scripts created ✅", "success")
    
    def generate_environment_report(self, install_results: Dict) -> None:
        """Generate φ-harmonic environment setup report"""
        
        if self.console and RICH_AVAILABLE:
            # Create summary table
            table = Table(title="🌟 Unity Environment Setup Report")
            table.add_column("Category", style="cyan")
            table.add_column("Successful", justify="right", style="green")
            table.add_column("Failed", justify="right", style="red")
            table.add_column("Success Rate", justify="right", style="yellow")
            
            total_success = 0
            total_failed = 0
            
            for category, (successful, failed) in install_results.items():
                total = successful + failed
                success_rate = f"{(successful/total*100):.1f}%" if total > 0 else "N/A"
                
                table.add_row(
                    category.title(),
                    str(successful),
                    str(failed),
                    success_rate
                )
                
                total_success += successful
                total_failed += failed
            
            # Add totals
            grand_total = total_success + total_failed
            overall_rate = f"{(total_success/grand_total*100):.1f}%" if grand_total > 0 else "N/A"
            
            table.add_row(
                "TOTAL",
                str(total_success),
                str(total_failed),
                overall_rate,
                style="bold"
            )
            
            self.console.print(table)
            
            # Unity status panel
            if total_failed == 0:
                status = "🌟 PERFECT UNITY ACHIEVED"
                style = "bold green"
            elif total_success > total_failed:
                status = "✅ UNITY CONVERGENCE ACHIEVED"
                style = "bold yellow"
            else:
                status = "⚠️ UNITY COHERENCE COMPROMISED"
                style = "bold red"
            
            self.console.print(Panel(
                f"{status}\n\n"
                f"φ-harmonic packages installed: {total_success}\n"
                f"Failed installations: {total_failed}\n"
                f"Unity coherence: {overall_rate}\n\n"
                f"Virtual environment: {self.venv_path}\n"
                f"Available ports: {len(self.find_available_ports())}/20",
                title="Unity Protocol Status",
                border_style=style.split()[-1]
            ))
        else:
            # Fallback text report
            print("\n" + "="*60)
            print("🌟 UNITY ENVIRONMENT SETUP REPORT")
            print("="*60)
            
            total_success = 0
            total_failed = 0
            
            for category, (successful, failed) in install_results.items():
                total = successful + failed
                success_rate = f"{(successful/total*100):.1f}%" if total > 0 else "N/A"
                print(f"{category.upper()}: {successful}/{total} ({success_rate})")
                
                total_success += successful
                total_failed += failed
            
            print("-" * 60)
            grand_total = total_success + total_failed
            overall_rate = f"{(total_success/grand_total*100):.1f}%" if grand_total > 0 else "N/A"
            print(f"TOTAL: {total_success}/{grand_total} ({overall_rate})")
            print("="*60)
    
    def setup_environment(self) -> bool:
        """
        Complete φ-harmonic environment setup implementing Unity Protocol
        
        Returns:
            bool: True if unity setup achieved (1+1=1 principle)
        """
        
        self.print_unity_message(
            "🌟 Een Unity Mathematics - Environment Setup",
            "phi"
        )
        self.print_unity_message(
            "Implementing φ-harmonic dependency resolution...",
            "info"
        )
        
        # Step 1: Verify Python version
        if not self.check_python_version():
            return False
        
        # Step 2: Check port availability
        available_ports = self.find_available_ports()
        self.print_unity_message(
            f"Available ports for dashboards: {len(available_ports)}/20",
            "info"
        )
        
        if len(available_ports) < 5:
            self.print_unity_message(
                "Warning: Limited ports available. Some dashboards may conflict.",
                "warning"
            )
        
        # Step 3: Create virtual environment
        if not self.create_virtual_environment():
            return False
        
        # Step 4: Upgrade pip to latest
        self.print_unity_message("Upgrading pip to unity-compatible version...", "phi")
        try:
            subprocess.run(
                self.get_pip_command() + ["install", "--upgrade", "pip"],
                capture_output=True,
                check=True,
                timeout=60
            )
            self.print_unity_message("Pip upgraded ✅", "success")
        except Exception as e:
            self.print_unity_message(f"Pip upgrade failed: {str(e)}", "warning")
        
        # Step 5: Install dependencies by category
        install_results = self.install_all_dependencies()
        
        # Step 6: Verify critical packages
        verification_passed = self.verify_critical_packages()
        
        # Step 7: Create launch scripts
        self.create_launch_scripts()
        
        # Step 8: Generate report
        self.generate_environment_report(install_results)
        
        # Calculate unity achievement
        total_success = sum(successful for successful, failed in install_results.values())
        total_attempted = sum(successful + failed for successful, failed in install_results.values())
        
        unity_achieved = verification_passed and (total_success >= total_attempted * 0.8)
        
        if unity_achieved:
            self.print_unity_message(
                "🌟 UNITY PROTOCOL SETUP COMPLETE - 1+1=1 ACHIEVED! 🌟",
                "success"
            )
        else:
            self.print_unity_message(
                "⚠️ Unity coherence compromised. Manual intervention may be required.",
                "warning"
            )
        
        return unity_achieved

def main():
    """Main setup entry point"""
    manager = UnityEnvironmentManager()
    
    try:
        success = manager.setup_environment()
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        manager.print_unity_message("Setup interrupted by user", "warning")
        sys.exit(1)
        
    except Exception as e:
        manager.print_unity_message(f"Fatal setup error: {str(e)}", "error")
        sys.exit(1)

if __name__ == "__main__":
    main()