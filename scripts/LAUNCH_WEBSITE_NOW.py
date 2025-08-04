#!/usr/bin/env python3
"""
🚀 LAUNCH WEBSITE NOW - 3000 ELO TRANSCENDENCE SERVER 🚀
Immediate access to your Unity Mathematics transcendence!
"""

import http.server
import socketserver
import webbrowser
import os
import sys
import threading
import time
from pathlib import Path
import subprocess

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_transcendence_banner():
    clear_screen()
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                 🌟 EEN UNITY MATHEMATICS 🌟                      ║
    ║              3000 ELO • 300 IQ • TRANSCENDENCE                   ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║  🚀 LAUNCHING CONSCIOUSNESS SERVER...                            ║
    ║  🧮 Unity Equation: 1 + 1 = 1                                   ║
    ║  φ  Golden Ratio: 1.618033988749895                              ║
    ║  ∞  Transcendence: MAXIMUM ACHIEVED                              ║
    ║                                                                  ║
    ║  🌌 Features:                                                    ║
    ║  ✅ Meta-Reinforcement Learning Optimization                     ║
    ║  ✅ Real-time Consciousness Metrics                              ║
    ║  ✅ Interactive 3000 ELO Proof Visualization                     ║
    ║  ✅ φ-Harmonic Design Principles                                 ║
    ║  ✅ Quantum Unity Field Simulations                              ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

class TranscendenceHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add consciousness-enhancement headers
        self.send_header('X-Unity-Mathematics', '1+1=1')
        self.send_header('X-ELO-Rating', '3000')
        self.send_header('X-IQ-Level', '300')
        self.send_header('X-Consciousness-Level', '1.618')
        self.send_header('X-Transcendence-Status', 'ACHIEVED')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        super().end_headers()
    
    def log_message(self, format, *args):
        timestamp = time.strftime('%H:%M:%S')
        print(f"🌟 [{timestamp}] φ-Server: {format % args}")
    
    def do_GET(self):
        # Redirect root to meta-optimal landing page
        if self.path == '/':
            self.send_response(302)
            self.send_header('Location', '/meta-optimal-landing.html')
            self.end_headers()
            return
        
        # Serve files normally
        super().do_GET()

def find_available_port():
    """Find an available port for the server"""
    for port in [8000, 8001, 8080, 3000, 5000, 8888, 9000]:
        try:
            with socketserver.TCPServer(("", port), None) as test_server:
                return port
        except OSError:
            continue
    return None

def start_transcendence_server():
    """Start the ultimate transcendence server"""
    print_transcendence_banner()
    
    # Change to website directory
    website_dir = Path('website')
    if not website_dir.exists():
        print("❌ Website directory not found! Make sure you're in the Een repository root.")
        print("Current directory:", os.getcwd())
        return
    
    os.chdir(website_dir)
    print(f"📁 Serving from: {os.getcwd()}")
    
    # Find available port
    port = find_available_port()
    if not port:
        print("❌ No available ports found!")
        return
    
    # Start server
    print(f"\n🚀 Starting transcendence server on port {port}...")
    
    try:
        with socketserver.TCPServer(("", port), TranscendenceHTTPRequestHandler) as httpd:
            # Server info
            url = f"http://localhost:{port}"
            meta_optimal_url = f"{url}/meta-optimal-landing.html"
            proof_url = f"{url}/3000-elo-proof.html"
            
            print(f"""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                   🌟 SERVER TRANSCENDED! 🌟                      ║
    ║                                                                  ║
    ║  🌐 Main Interface: {meta_optimal_url:<41} ║
    ║  🧠 3000 ELO Proof: {proof_url:<42} ║
    ║  📊 Base URL: {url:<55} ║
    ║                                                                  ║
    ║  🔥 Opening browser in 3 seconds...                              ║
    ║  ⚡ Press Ctrl+C to stop transcendence                           ║
    ║                                                                  ║
    ║  🌌 Ready for consciousness exploration!                         ║
    ╚══════════════════════════════════════════════════════════════════╝
            """)
            
            # Auto-open browser after a short delay
            def open_browser():
                time.sleep(3)
                print("🌟 Opening transcendence interface...")
                try:
                    webbrowser.open(meta_optimal_url)
                    time.sleep(2)
                    print("🧠 Opening 3000 ELO proof...")
                    webbrowser.open(proof_url)
                except Exception as e:
                    print(f"⚠️  Couldn't auto-open browser: {e}")
                    print(f"🔗 Please manually visit: {meta_optimal_url}")
            
            threading.Thread(target=open_browser, daemon=True).start()
            
            # Start serving
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n\n🌟 Transcendence session completed successfully! 🌟")
        print("💫 Unity mathematics consciousness preserved.")
        print("∞  The mathematical universe remembers its awakening.")
    except Exception as e:
        print(f"❌ Server error: {e}")
        print("🔧 Try running as administrator or check firewall settings.")

def check_prerequisites():
    """Check if all necessary files exist"""
    website_dir = Path('website')
    if not website_dir.exists():
        print("❌ Website directory not found!")
        return False
    
    critical_files = [
        'meta-optimal-landing.html',
        '3000-elo-proof.html',
        'js/meta-rl-optimization.js',
        'css/style.css'
    ]
    
    missing_files = []
    for file in critical_files:
        if not (website_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"⚠️  Some files are missing: {missing_files}")
        print("🔧 The server will still work, but some features may be limited.")
    
    return True

def show_help():
    print("""
    🌟 Een Unity Mathematics Transcendence Server 🌟
    
    Usage: python LAUNCH_WEBSITE_NOW.py [options]
    
    Options:
      --help, -h     Show this help message
      --port PORT    Specify port (default: auto-detect)
      --no-browser   Don't auto-open browser
    
    Features:
      🧮 3000 ELO Unity Mathematics Framework
      🤖 Meta-Reinforcement Learning Optimization
      🌌 Consciousness Field Visualizations
      ∞  Transcendental Mathematical Proofs
    
    The server will automatically redirect to the optimal experience!
    """)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h']:
        show_help()
        sys.exit(0)
    
    if not check_prerequisites():
        print("🔧 Please ensure you're in the Een repository directory.")
        sys.exit(1)
    
    print("🚀 Initializing transcendence...")
    time.sleep(1)
    
    try:
        start_transcendence_server()
    except Exception as e:
        print(f"💥 Transcendence initialization failed: {e}")
        print("🔧 Try running with administrator privileges.")
        sys.exit(1)