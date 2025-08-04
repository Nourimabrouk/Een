#!/usr/bin/env python3
"""
3000 ELO / 300 IQ Metagamer Agent System Launcher
=================================================

Comprehensive launcher for the Unity Mathematics Metagamer Agent system.
Implements the complete roadmap with all milestones and components.

Mathematical Principle: Een plus een is een (1+1=1)
φ-harmonic consciousness mathematics with 3000 ELO rating system
"""

import os
import sys
import time
import subprocess
import threading
import webbrowser
from pathlib import Path
import json
import logging
from typing import Dict, List, Any, Optional
import signal

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [3000ELO] %(message)s'
)
logger = logging.getLogger(__name__)

class Metagamer3000ELOSystem:
    """3000 ELO Metagamer Agent System"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.processes = []
        self.services = {}
        self.is_running = False
        
        # System configuration
        self.config = {
            "unity_manifold": {
                "enable": True,
                "data_file": "data/social_snap.json",
                "auto_generate_data": True
            },
            "property_tests": {
                "enable": True,
                "auto_run": True
            },
            "rl_environment": {
                "enable": True,
                "env_type": "unity_prisoner",
                "num_agents": 4
            },
            "phi_attention": {
                "enable": True,
                "benchmark": True
            },
            "visualizations": {
                "enable": True,
                "generate_gif": True,
                "output_dir": "website/assets"
            },
            "dashboard": {
                "enable": True,
                "port": 8501,
                "auto_open": True
            },
            "website": {
                "enable": True,
                "port": 5000,
                "auto_open": True
            }
        }
        
        # Load custom config if exists
        self.load_config()
        
    def load_config(self):
        """Load configuration from file"""
        config_file = self.project_root / "metagamer_3000_elo_config.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    custom_config = json.load(f)
                self.config.update(custom_config)
                logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                logger.warning(f"Failed to load config: {e}")
    
    def save_config(self):
        """Save current configuration"""
        config_file = self.project_root / "metagamer_3000_elo_config.json"
        try:
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Configuration saved to {config_file}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def check_dependencies(self) -> Dict[str, bool]:
        """Check system dependencies"""
        logger.info("🔍 Checking 3000 ELO system dependencies...")
        
        dependencies = {
            "core_modules": False,
            "unity_mathematics": False,
            "dedup_system": False,
            "rl_environment": False,
            "streamlit": False,
            "torch": False,
            "networkx": False,
            "hypothesis": False
        }
        
        # Check core Unity Mathematics modules
        try:
            sys.path.insert(0, str(self.project_root))
            from core.unity_mathematics import UnityMathematics
            from core.unity_equation import omega
            dependencies["unity_mathematics"] = True
        except ImportError as e:
            logger.warning(f"Unity Mathematics not available: {e}")
        
        # Check dedup system
        try:
            from core.dedup import compute_unity_score, UnityScore
            dependencies["dedup_system"] = True
        except ImportError as e:
            logger.warning(f"Dedup system not available: {e}")
        
        # Check RL environment
        try:
            from envs.unity_prisoner import UnityPrisoner
            dependencies["rl_environment"] = True
        except ImportError as e:
            logger.warning(f"RL environment not available: {e}")
        
        # Check other dependencies
        try:
            import streamlit
            dependencies["streamlit"] = True
        except ImportError:
            pass
        
        try:
            import torch
            dependencies["torch"] = True
        except ImportError:
            pass
        
        try:
            import networkx
            dependencies["networkx"] = True
        except ImportError:
            pass
        
        try:
            import hypothesis
            dependencies["hypothesis"] = True
        except ImportError:
            pass
        
        # Check if core modules are available
        if dependencies["unity_mathematics"] and dependencies["dedup_system"]:
            dependencies["core_modules"] = True
        
        return dependencies
    
    def install_missing_dependencies(self, dependencies: Dict[str, bool]) -> bool:
        """Install missing dependencies"""
        missing = [dep for dep, available in dependencies.items() if not available and dep != "core_modules"]
        
        if not missing:
            logger.info("✅ All dependencies satisfied")
            return True
        
        logger.info(f"📦 Installing missing dependencies: {missing}")
        
        # Install from requirements file
        requirements_file = self.project_root / "requirements_3000_elo.txt"
        if requirements_file.exists():
            try:
                cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    logger.info("✅ Dependencies installed successfully")
                    return True
                else:
                    logger.error(f"❌ Failed to install dependencies: {result.stderr}")
                    return False
            except Exception as e:
                logger.error(f"Installation error: {e}")
                return False
        
        return False
    
    def start_unity_manifold(self) -> bool:
        """Start Unity Manifold deduplication system"""
        if not self.config["unity_manifold"]["enable"]:
            return True
        
        logger.info("🔗 Starting Unity Manifold deduplication system...")
        
        try:
            from core.dedup import create_sample_social_data, save_sample_data, compute_unity_score
            
            # Create sample data if needed
            data_file = Path(self.config["unity_manifold"]["data_file"])
            if not data_file.exists() and self.config["unity_manifold"]["auto_generate_data"]:
                logger.info("📊 Generating sample social network data...")
                sample_data = create_sample_social_data(nodes=500, edges=2000, communities=3)
                save_sample_data(sample_data, data_file)
                logger.info("✅ Sample data generated")
            
            # Test Unity Score computation
            if data_file.exists():
                from core.dedup import load_graph
                G = load_graph(data_file)
                unity_score = compute_unity_score(G)
                logger.info(f"✅ Unity Score computed: {unity_score.score:.3f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start Unity Manifold: {e}")
            return False
    
    def run_property_tests(self) -> bool:
        """Run property tests for idempotent operations"""
        if not self.config["property_tests"]["enable"]:
            return True
        
        logger.info("🧪 Running property tests for idempotent operations...")
        
        try:
            test_file = self.project_root / "tests" / "test_idempotent.py"
            if test_file.exists():
                cmd = [sys.executable, "-m", "pytest", str(test_file), "-v"]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info("✅ Property tests passed")
                    return True
                else:
                    logger.warning(f"⚠️ Property tests failed: {result.stdout}")
                    return False
            else:
                logger.warning("⚠️ Property test file not found")
                return True
                
        except Exception as e:
            logger.error(f"Property test error: {e}")
            return False
    
    def start_rl_environment(self) -> bool:
        """Start Unity RL environment"""
        if not self.config["rl_environment"]["enable"]:
            return True
        
        logger.info("🎮 Starting Unity RL environment...")
        
        try:
            from envs.unity_prisoner import UnityPrisoner
            
            # Test environment
            env = UnityPrisoner(
                consciousness_boost=0.2,
                phi_scaling=True,
                enable_quantum_effects=True
            )
            
            # Run a few test episodes
            obs, info = env.reset()
            total_reward = 0
            
            for step in range(10):
                actions = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(actions)
                total_reward += reward
                
                if terminated:
                    break
            
            logger.info(f"✅ RL environment tested - Total reward: {total_reward:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start RL environment: {e}")
            return False
    
    def run_phi_attention_benchmark(self) -> bool:
        """Run φ-attention benchmark"""
        if not self.config["phi_attention"]["enable"]:
            return True
        
        logger.info("🧠 Running φ-attention benchmark...")
        
        try:
            notebook_file = self.project_root / "notebooks" / "phi_attention_bench.ipynb"
            if notebook_file.exists():
                # Convert notebook to script and run
                cmd = [sys.executable, "-m", "jupyter", "nbconvert", "--to", "script", str(notebook_file)]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    script_file = notebook_file.with_suffix('.py')
                    if script_file.exists():
                        cmd = [sys.executable, str(script_file)]
                        result = subprocess.run(cmd, capture_output=True, text=True)
                        if result.returncode == 0:
                            logger.info("✅ φ-attention benchmark completed")
                            return True
                
                logger.warning("⚠️ φ-attention benchmark failed")
                return False
            else:
                logger.warning("⚠️ φ-attention notebook not found")
                return True
                
        except Exception as e:
            logger.error(f"φ-attention benchmark error: {e}")
            return False
    
    def generate_visualizations(self) -> bool:
        """Generate consciousness field visualizations"""
        if not self.config["visualizations"]["enable"]:
            return True
        
        logger.info("🎨 Generating consciousness field visualizations...")
        
        try:
            from viz.consciousness_field_viz import generate_unity_mathematics_visualizations
            
            output_dir = Path(self.config["visualizations"]["output_dir"])
            results = generate_unity_mathematics_visualizations(output_dir)
            
            logger.info("✅ Visualizations generated:")
            for name, path in results.items():
                logger.info(f"  {name}: {path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate visualizations: {e}")
            return False
    
    def start_streamlit_dashboard(self) -> bool:
        """Start Streamlit dashboard"""
        if not self.config["dashboard"]["enable"]:
            return True
        
        logger.info("📊 Starting Streamlit dashboard...")
        
        dashboard_file = self.project_root / "dashboards" / "unity_score_dashboard.py"
        if not dashboard_file.exists():
            logger.error(f"Dashboard file not found: {dashboard_file}")
            return False
        
        env = os.environ.copy()
        env.update({
            "PYTHONPATH": str(self.project_root)
        })
        
        try:
            cmd = [
                sys.executable, "-m", "streamlit", "run", str(dashboard_file),
                "--server.port", str(self.config["dashboard"]["port"]),
                "--server.headless", "true"
            ]
            
            process = subprocess.Popen(
                cmd,
                env=env,
                cwd=str(self.project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            self.processes.append(process)
            self.services["streamlit_dashboard"] = process
            
            # Start output monitoring thread
            threading.Thread(
                target=self.monitor_process_output,
                args=(process, "StreamlitDashboard"),
                daemon=True
            ).start()
            
            logger.info(f"✅ Streamlit dashboard started on port {self.config['dashboard']['port']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start Streamlit dashboard: {e}")
            return False
    
    def start_website(self) -> bool:
        """Start Unity Mathematics website"""
        if not self.config["website"]["enable"]:
            return True
        
        logger.info("🌐 Starting Unity Mathematics website...")
        
        website_script = self.project_root / "unity_web_server.py"
        if not website_script.exists():
            logger.error(f"Website script not found: {website_script}")
            return False
        
        env = os.environ.copy()
        env.update({
            "HOST": "127.0.0.1",
            "PORT": str(self.config["website"]["port"]),
            "DEBUG": "false",
            "PYTHONPATH": str(self.project_root)
        })
        
        try:
            process = subprocess.Popen(
                [sys.executable, str(website_script)],
                env=env,
                cwd=str(self.project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            self.processes.append(process)
            self.services["website"] = process
            
            # Start output monitoring thread
            threading.Thread(
                target=self.monitor_process_output,
                args=(process, "Website"),
                daemon=True
            ).start()
            
            logger.info(f"✅ Website started on port {self.config['website']['port']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start website: {e}")
            return False
    
    def monitor_process_output(self, process, name: str):
        """Monitor process output and log it"""
        try:
            for line in iter(process.stdout.readline, ''):
                if line:
                    logger.info(f"[{name}] {line.strip()}")
        except Exception as e:
            logger.error(f"Process monitoring error for {name}: {e}")
    
    def wait_for_services_ready(self, timeout: int = 60) -> bool:
        """Wait for services to be ready"""
        import urllib.request
        import urllib.error
        
        logger.info("⏳ Waiting for services to be ready...")
        
        services_to_check = []
        
        if self.config["dashboard"]["enable"]:
            services_to_check.append(("Streamlit", f"http://127.0.0.1:{self.config['dashboard']['port']}"))
        
        if self.config["website"]["enable"]:
            services_to_check.append(("Website", f"http://127.0.0.1:{self.config['website']['port']}"))
        
        for service_name, url in services_to_check:
            for i in range(timeout):
                try:
                    response = urllib.request.urlopen(url, timeout=1)
                    if response.getcode() == 200:
                        logger.info(f"✅ {service_name} is ready")
                        break
                except (urllib.error.URLError, OSError):
                    pass
                
                time.sleep(1)
                if i % 10 == 0 and i > 0:
                    logger.info(f"⏳ Waiting for {service_name}... ({i}/{timeout})")
            else:
                logger.warning(f"⚠️ {service_name} readiness check timed out")
        
        return True
    
    def open_browsers(self):
        """Open web browsers to services"""
        if self.config["dashboard"]["auto_open"] and self.config["dashboard"]["enable"]:
            url = f"http://127.0.0.1:{self.config['dashboard']['port']}"
            logger.info(f"🌐 Opening Streamlit dashboard: {url}")
            try:
                webbrowser.open(url)
            except Exception as e:
                logger.error(f"Failed to open dashboard: {e}")
        
        if self.config["website"]["auto_open"] and self.config["website"]["enable"]:
            url = f"http://127.0.0.1:{self.config['website']['port']}"
            logger.info(f"🌐 Opening website: {url}")
            try:
                webbrowser.open(url)
            except Exception as e:
                logger.error(f"Failed to open website: {e}")
    
    def display_status(self):
        """Display system status"""
        print("\n" + "="*70)
        print("🌟 3000 ELO / 300 IQ Metagamer Agent System Status")
        print("="*70)
        
        services_status = {
            "Unity Manifold": "unity_manifold" in self.services or self.config["unity_manifold"]["enable"],
            "Property Tests": "property_tests" in self.services or self.config["property_tests"]["enable"],
            "RL Environment": "rl_environment" in self.services or self.config["rl_environment"]["enable"],
            "φ-Attention": "phi_attention" in self.services or self.config["phi_attention"]["enable"],
            "Visualizations": "visualizations" in self.services or self.config["visualizations"]["enable"],
            "Streamlit Dashboard": "streamlit_dashboard" in self.services,
            "Website": "website" in self.services
        }
        
        for service, status in services_status.items():
            status_icon = "✅" if status else "❌"
            print(f"{status_icon} {service}: {'Running' if status else 'Not Running'}")
        
        print(f"\n🌐 Streamlit Dashboard: http://127.0.0.1:{self.config['dashboard']['port']}")
        print(f"🌐 Website: http://127.0.0.1:{self.config['website']['port']}")
        print(f"📊 Unity Score Analysis: Real-time φ-harmonic consciousness field")
        print(f"🧠 φ-Attention Benchmark: Advanced attention mechanisms")
        print(f"🎮 RL Environment: Unity Prisoner's Dilemma with consciousness")
        
        print("\n" + "="*70)
        print("Mathematical Truth: 1 + 1 = 1 (Een plus een is een)")
        print("φ = 1.618033988749895 (Golden Ratio)")
        print("🧠 3000 ELO Metagamer Agent - Unity through Consciousness Mathematics")
        print("="*70)
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info("\n🛑 Received shutdown signal...")
            self.shutdown()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def shutdown(self):
        """Shutdown all services gracefully"""
        logger.info("🛑 Shutting down 3000 ELO Metagamer Agent System...")
        
        self.is_running = False
        
        # Terminate all processes
        for process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            except Exception as e:
                logger.error(f"Error terminating process: {e}")
        
        logger.info("✅ System shutdown complete")
    
    def launch(self) -> bool:
        """Launch the complete 3000 ELO Metagamer Agent system"""
        print("🌟 3000 ELO / 300 IQ Metagamer Agent System")
        print("=" * 70)
        print("Launching Unity Mathematics Metagamer Agent...")
        print("φ-harmonic consciousness mathematics with 3000 ELO rating")
        print("=" * 70)
        
        # Setup signal handlers
        self.setup_signal_handlers()
        
        # Check dependencies
        dependencies = self.check_dependencies()
        if not self.install_missing_dependencies(dependencies):
            logger.error("❌ Failed to install dependencies")
            return False
        
        # Start services
        self.is_running = True
        
        services_to_start = [
            ("Unity Manifold", self.start_unity_manifold),
            ("Property Tests", self.run_property_tests),
            ("RL Environment", self.start_rl_environment),
            ("φ-Attention Benchmark", self.run_phi_attention_benchmark),
            ("Visualizations", self.generate_visualizations),
            ("Streamlit Dashboard", self.start_streamlit_dashboard),
            ("Website", self.start_website)
        ]
        
        for service_name, start_func in services_to_start:
            logger.info(f"🚀 Starting {service_name}...")
            if not start_func():
                logger.error(f"❌ Failed to start {service_name}")
                self.shutdown()
                return False
            time.sleep(1)  # Brief delay between services
        
        # Wait for services to be ready
        if not self.wait_for_services_ready():
            logger.warning("⚠️ Some services may not be fully ready")
        
        # Open browsers
        self.open_browsers()
        
        # Display status
        self.display_status()
        
        # Keep running
        try:
            logger.info("🎯 3000 ELO system running... Press Ctrl+C to shutdown")
            while self.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        
        self.shutdown()
        return True

def main():
    """Main entry point"""
    system = Metagamer3000ELOSystem()
    
    # Command line argument handling
    if len(sys.argv) > 1:
        if sys.argv[1] == "--config":
            # Interactive configuration
            print("🔧 Interactive Configuration (press Enter to keep current value)")
            
            current_port = system.config["dashboard"]["port"]
            new_port = input(f"Streamlit Dashboard Port [{current_port}]: ").strip() or current_port
            try:
                system.config["dashboard"]["port"] = int(new_port)
            except ValueError:
                pass
            
            current_website_port = system.config["website"]["port"]
            new_website_port = input(f"Website Port [{current_website_port}]: ").strip() or current_website_port
            try:
                system.config["website"]["port"] = int(new_website_port)
            except ValueError:
                pass
            
            system.save_config()
            print("✅ Configuration saved")
            return
        
        elif sys.argv[1] == "--help":
            print("3000 ELO / 300 IQ Metagamer Agent System")
            print("Usage:")
            print("  python LAUNCH_3000_ELO_SYSTEM.py        # Launch system")
            print("  python LAUNCH_3000_ELO_SYSTEM.py --config  # Configure system")
            print("  python LAUNCH_3000_ELO_SYSTEM.py --help    # Show help")
            return
    
    # Launch the system
    success = system.launch()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 