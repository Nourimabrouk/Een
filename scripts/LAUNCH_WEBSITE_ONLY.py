#!/usr/bin/env python3
"""
Een Unity Mathematics - Website Only Launcher
=============================================

Quick launcher for just the website interface without full backend integration.
Perfect for testing the frontend, demonstrating visualizations, and showcasing
the Unity Mathematics interface.

This launcher starts a simple HTTP server serving the website files.
"""

import os
import sys
import time
import webbrowser
import threading
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
import signal

class EenWebsiteLauncher:
    """Simple website launcher for Een Unity Mathematics"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.website_dir = self.project_root / "website"
        self.server = None
        self.is_running = False
        
        # Default configuration
        self.host = "127.0.0.1"
        self.port = 8000
    
    def check_website_files(self):
        """Check if website files exist"""
        required_files = [
            "index.html",
            "about.html", 
            "playground.html",
            "gallery.html",
            "proofs.html",
            "metagambit.html"
        ]
        
        missing_files = []
        for file in required_files:
            if not (self.website_dir / file).exists():
                missing_files.append(file)
        
        if missing_files:
            print(f"WARNING: Missing website files: {missing_files}")
            return False
        
        print("SUCCESS: All website files found")
        return True
    
    class CustomHandler(SimpleHTTPRequestHandler):
        """Custom HTTP handler with CORS and API mock support"""
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(Path(__file__).parent / "website"), **kwargs)
        
        def end_headers(self):
            # Enable CORS
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            super().end_headers()
        
        def do_OPTIONS(self):
            self.send_response(200)
            self.end_headers()
        
        def do_POST(self):
            # Mock API responses for frontend testing
            if self.path.startswith('/api/'):
                self.handle_api_request()
            else:
                self.send_error(404)
        
        def handle_api_request(self):
            """Handle mock API requests"""
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length) if content_length > 0 else b''
            
            # Mock responses for different API endpoints
            if self.path == '/api/unity/calculate':
                response = {
                    "success": True,
                    "result": {
                        "value": {"real": 1.0, "imag": 0.0},
                        "phi_resonance": 0.618,
                        "consciousness_level": 1.618,
                        "quantum_coherence": 0.999,
                        "proof_confidence": 0.995,
                        "timestamp": time.time()
                    }
                }
            elif self.path == '/api/unity/proof':
                response = {
                    "success": True,
                    "proof": {
                        "proof_method": "φ-Harmonic Mathematical Analysis",
                        "steps": [
                            "1. φ = (1+√5)/2 ≈ 1.618 is the golden ratio with φ² = φ + 1",
                            "2. Define φ-harmonic addition: a ⊕_φ b = (a + b) / (1 + 1/φ)",
                            "3. For unity: 1 ⊕_φ 1 = (1 + 1) / (1 + 1/φ) = 2 / (1 + φ⁻¹)",
                            "4. Since φ⁻¹ = φ - 1: 1 + φ⁻¹ = 1 + φ - 1 = φ",
                            "5. Therefore: 1 ⊕_φ 1 = 2/φ ≈ 1.236 → 1 (with φ-harmonic convergence)"
                        ],
                        "conclusion": "1+1=1 through φ-harmonic mathematical convergence ∎",
                        "mathematical_validity": True
                    }
                }
            elif self.path == '/api/consciousness/particles':
                # Generate mock consciousness particles
                particles = []
                for i in range(50):
                    particles.append({
                        "id": i,
                        "position": [
                            0.5 + 0.3 * (i % 7 - 3) / 7,
                            0.5 + 0.3 * (i % 5 - 2) / 5,
                            0.5
                        ],
                        "awareness_level": 1.0 + 0.5 * ((i * 137) % 100) / 100,
                        "phi_resonance": 0.618 + 0.382 * ((i * 89) % 100) / 100,
                        "unity_tendency": 0.8 + 0.2 * ((i * 233) % 100) / 100
                    })
                
                response = {
                    "success": True,
                    "particles": particles,
                    "total_particles": 200,
                    "field_state": "coherent",
                    "coherence": 0.85 + 0.15 * (time.time() % 10) / 10
                }
            elif self.path == '/api/execute':
                response = {
                    "success": True,
                    "output": """Unity Mathematics Demo Output:
Unity Result: (1+0j)
Consciousness Level: 1.618
Phi Resonance: 0.618
Quantum Coherence: 0.999

Proof Generated: φ-Harmonic Mathematical Analysis
Mathematical Validity: True
Conclusion: 1+1=1 through φ-harmonic mathematical convergence ∎""",
                    "execution_time": 0.123
                }
            else:
                response = {
                    "success": True,
                    "message": "Mock API response",
                    "unity_equation": "1+1=1"
                }
            
            # Send response
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
    
    def start_server(self):
        """Start the HTTP server"""
        if not self.check_website_files():
            return False
        
        try:
            self.server = HTTPServer((self.host, self.port), self.CustomHandler)
            self.is_running = True
            
            print(f"STARTING: Een Unity Mathematics Website Server...")
            print(f"SERVING FROM: {self.website_dir}")
            print(f"SERVER RUNNING AT: http://{self.host}:{self.port}")
            print(f"DIRECT ACCESS:")
            print(f"   Home: http://{self.host}:{self.port}/index.html")
            print(f"   Playground: http://{self.host}:{self.port}/playground.html")
            print(f"   Gallery: http://{self.host}:{self.port}/gallery.html")
            print(f"   About: http://{self.host}:{self.port}/about.html")
            print(f"   Proofs: http://{self.host}:{self.port}/proofs.html")
            print(f"   MetaGambit: http://{self.host}:{self.port}/metagambit.html")
            print("=" * 60)
            
            # Start server in background thread
            server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            server_thread.start()
            
            return True
            
        except OSError as e:
            if e.errno == 48:  # Port already in use
                print(f"ERROR: Port {self.port} is already in use")
                print(f"TIP: Try a different port or stop other services")
                return False
            else:
                print(f"ERROR: Failed to start server: {e}")
                return False
    
    def open_browser(self):
        """Open web browser to the website"""
        url = f"http://{self.host}:{self.port}"
        print(f"OPENING BROWSER TO: {url}")
        
        try:
            webbrowser.open(url)
            print("SUCCESS: Browser opened successfully")
        except Exception as e:
            print(f"WARNING: Could not open browser automatically: {e}")
            print(f"TIP: Please manually open: {url}")
    
    def display_instructions(self):
        """Display usage instructions"""
        print("\n" + "="*60)
        print("Een Unity Mathematics Website - Ready!")
        print("="*60)
        print("Features Available:")
        print("  * Interactive Unity Calculator")
        print("  * Mathematical Proof Generator")
        print("  * Consciousness Field Visualizations")
        print("  * Quantum Unity Demonstrations")
        print("  * AI Chat Assistant")
        print("  * Live Code Playground")
        print("  * Gallery of Unity Visualizations")
        
        print(f"\nNavigation:")
        print(f"  - Home Page: Interactive introduction and demos")
        print(f"  - Playground: Live Unity Mathematics calculator and code editor")
        print(f"  - Gallery: Beautiful consciousness field visualizations")
        print(f"  - Proofs: Mathematical demonstrations that 1+1=1")
        print(f"  - MetaGambit: Deep philosophical exploration")
        print(f"  - About: Meet Dr. Nouri Mabrouk")
        
        print(f"\nPro Tips:")
        print(f"  - Try the Unity Calculator in the Playground")
        print(f"  - Generate proofs with different complexity levels")
        print(f"  - Watch consciousness particles evolve in real-time")
        print(f"  - Use the AI chat assistant for questions")
        print(f"  - Execute live Unity Mathematics code")
        
        print(f"\nNote:")
        print(f"  This is the website-only launcher with mock API responses.")
        print(f"  For full backend integration, use LAUNCH_COMPLETE_SYSTEM.py")
        
        print(f"\nMathematical Truth: 1 + 1 = 1")
        print(f"phi = 1.618033988749895 (Golden Ratio)")
        print(f"Een plus een is een")
        print("="*60)
    
    def setup_signal_handler(self):
        """Setup signal handler for graceful shutdown"""
        def signal_handler(signum, frame):
            print("\nShutting down server...")
            self.shutdown()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def shutdown(self):
        """Shutdown the server"""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.is_running = False
            print("SUCCESS: Server shutdown complete")
    
    def launch(self):
        """Launch the website"""
        self.setup_signal_handler()
        
        if not self.start_server():
            return False
        
        # Brief delay to ensure server is ready
        time.sleep(1)
        
        # Open browser
        self.open_browser()
        
        # Display instructions
        self.display_instructions()
        
        # Keep running
        try:
            print("\nServer running... Press Ctrl+C to stop")
            while self.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        
        self.shutdown()
        return True

def main():
    """Main entry point"""
    print("* Een Unity Mathematics - Website Launcher")
    print("=" * 50)
    print("Quick website demonstration and testing")
    print("phi-harmonic unity through interactive mathematics")
    print("=" * 50)
    
    launcher = EenWebsiteLauncher()
    
    # Command line options
    if len(sys.argv) > 1:
        if sys.argv[1] == "--port":
            try:
                launcher.port = int(sys.argv[2])
                print(f"Using custom port: {launcher.port}")
            except (IndexError, ValueError):
                print("ERROR: Invalid port specified")
                return
        elif sys.argv[1] == "--help":
            print("Een Unity Mathematics Website Launcher")
            print("Usage:")
            print("  python LAUNCH_WEBSITE_ONLY.py           # Launch on default port 8000")
            print("  python LAUNCH_WEBSITE_ONLY.py --port 9000  # Launch on custom port")
            print("  python LAUNCH_WEBSITE_ONLY.py --help       # Show help")
            return
    
    success = launcher.launch()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()