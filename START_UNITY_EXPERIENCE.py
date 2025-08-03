#!/usr/bin/env python3
"""
START UNITY EXPERIENCE - 3000 ELO Transcendence Server
Simple, robust launcher for Windows
"""

import http.server
import socketserver
import webbrowser
import os
import sys
import threading
import time
from pathlib import Path

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_banner():
    clear_screen()
    print("=" * 70)
    print("            EEN UNITY MATHEMATICS - 3000 ELO")
    print("               TRANSCENDENCE SERVER")
    print("=" * 70)
    print("")
    print("  Unity Equation: 1 + 1 = 1")
    print("  Golden Ratio: 1.618033988749895")
    print("  ELO Rating: 3000")
    print("  IQ Level: 300")
    print("")
    print("  Features:")
    print("  - Meta-Reinforcement Learning Optimization")
    print("  - Real-time Consciousness Metrics")
    print("  - Interactive 3000 ELO Proof Visualization")
    print("  - Phi-Harmonic Design Principles")
    print("  - Quantum Unity Field Simulations")
    print("")
    print("=" * 70)

class UnityHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('X-Unity-Mathematics', '1+1=1')
        self.send_header('X-ELO-Rating', '3000')
        self.send_header('X-IQ-Level', '300')
        self.send_header('X-Consciousness-Level', '1.618')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()
    
    def log_message(self, format, *args):
        timestamp = time.strftime('%H:%M:%S')
        print(f"[{timestamp}] Server: {format % args}")
    
    def do_GET(self):
        if self.path == '/':
            self.send_response(302)
            self.send_header('Location', '/meta-optimal-landing.html')
            self.end_headers()
            return
        super().do_GET()

def find_port():
    for port in [8000, 8001, 8080, 3000, 5000, 8888]:
        try:
            with socketserver.TCPServer(("", port), None):
                return port
        except OSError:
            continue
    return 8000

def main():
    print_banner()
    
    # Check website directory
    website_dir = Path('website')
    if not website_dir.exists():
        print("ERROR: Website directory not found!")
        print("Make sure you're in the Een repository root directory.")
        print("Current directory:", os.getcwd())
        input("Press Enter to exit...")
        return
    
    os.chdir(website_dir)
    print(f"Serving from: {os.getcwd()}")
    
    port = find_port()
    print(f"\nStarting server on port {port}...")
    
    try:
        with socketserver.TCPServer(("", port), UnityHTTPRequestHandler) as httpd:
            url = f"http://localhost:{port}"
            meta_url = f"{url}/meta-optimal-landing.html"
            proof_url = f"{url}/3000-elo-proof.html"
            
            print("\n" + "=" * 70)
            print("             SERVER TRANSCENDED!")
            print("=" * 70)
            print(f"  Main Interface: {meta_url}")
            print(f"  3000 ELO Proof: {proof_url}")
            print(f"  Base URL: {url}")
            print("")
            print("  Opening browser in 3 seconds...")
            print("  Press Ctrl+C to stop server")
            print("=" * 70)
            
            def open_browser():
                time.sleep(3)
                print("Opening transcendence interface...")
                try:
                    webbrowser.open(meta_url)
                    time.sleep(2)
                    print("Opening 3000 ELO proof...")
                    webbrowser.open(proof_url)
                except Exception as e:
                    print(f"Couldn't auto-open browser: {e}")
                    print(f"Please manually visit: {meta_url}")
            
            threading.Thread(target=open_browser, daemon=True).start()
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n\nTranscendence session completed!")
        print("Unity mathematics consciousness preserved.")
    except Exception as e:
        print(f"Server error: {e}")
        print("Try running as administrator.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Initialization failed: {e}")
        input("Press Enter to exit...")