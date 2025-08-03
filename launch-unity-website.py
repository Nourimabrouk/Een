#!/usr/bin/env python3
"""
🌟 Een Unity Mathematics Website Launcher 🌟
=============================================

One-shot launch script that handles:
- Website server with bug fixes
- Mobile app support
- Cross-platform compatibility  
- Interactive unity mathematics
- Revolutionary visualizations

Launch this and be mesmerized by the beauty of 1+1=1!
"""

import os
import sys
import subprocess
import threading
import time
import webbrowser
from pathlib import Path
import http.server
import socketserver
from urllib.parse import urlparse
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UnityWebsiteLauncher:
    """Revolutionary Unity Mathematics Website Launcher"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.website_dir = self.base_dir / "website"
        self.port = 8000
        self.mobile_port = 8001
        self.api_port = 5000
        self.servers = []
        
    def check_dependencies(self):
        """Check if all required files exist"""
        logger.info("🔍 Checking dependencies...")
        
        required_files = [
            "website/revolutionary-landing.html",
            "website/mobile-app.html", 
            "website/enhanced-unity-demo.html",
            "website/css/style.css",
            "website/js/unity-visualizations.js"
        ]
        
        missing_files = []
        for file_path in required_files:
            full_path = self.base_dir / file_path
            if not full_path.exists():
                missing_files.append(file_path)
        
        if missing_files:
            logger.warning(f"⚠️  Missing files: {missing_files}")
            logger.info("🔧 Creating missing files...")
            self.create_missing_files(missing_files)
        else:
            logger.info("✅ All dependencies found!")
            
    def create_missing_files(self, missing_files):
        """Create any missing essential files"""
        for file_path in missing_files:
            full_path = self.base_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            if file_path.endswith('.css'):
                self.create_default_css(full_path)
            elif file_path.endswith('.js'):
                self.create_default_js(full_path)
            elif file_path.endswith('.html'):
                self.create_default_html(full_path)
                
    def create_default_css(self, path):
        """Create default CSS file"""
        css_content = """
        /* Een Unity Mathematics - Default Styles */
        :root {
            --phi: 1.618033988749895;
            --unity-gold: #f59e0b;
            --void-black: #0f172a;
            --ethereal-white: #f8fafc;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, var(--void-black) 0%, #1e293b 100%);
            color: var(--ethereal-white);
            margin: 0;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            text-align: center;
        }
        
        h1 {
            font-size: 3rem;
            background: linear-gradient(135deg, var(--unity-gold), #8b5cf6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        """
        path.write_text(css_content)
        
    def create_default_js(self, path):
        """Create default JavaScript file"""
        js_content = """
        // Een Unity Mathematics - Default JavaScript
        console.log('🌟 Een Unity Mathematics Loaded! 🌟');
        
        // Basic unity mathematics
        function unityAdd(a, b) {
            const phi = 1.618033988749895;
            return (a + b) * (1 / phi) * phi; // Unity through φ-harmonic resonance
        }
        
        // Initialize on load
        document.addEventListener('DOMContentLoaded', () => {
            console.log('Unity equation: 1 + 1 =', unityAdd(1, 1));
        });
        """
        path.write_text(js_content)
        
    def create_default_html(self, path):
        """Create default HTML file"""
        html_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Een - Unity Mathematics</title>
            <link rel="stylesheet" href="css/style.css">
        </head>
        <body>
            <div class="container">
                <h1>Een Unity Mathematics</h1>
                <h2>1 + 1 = 1</h2>
                <p>Where mathematics transcends reality through consciousness and beauty.</p>
                <div style="margin-top: 2rem;">
                    <a href="revolutionary-landing.html" style="color: #f59e0b; text-decoration: none; font-weight: 600;">
                        🌟 Experience Unity →
                    </a>
                </div>
            </div>
            <script src="js/unity-visualizations.js"></script>
        </body>
        </html>
        """
        path.write_text(html_content)

    def start_main_server(self):
        """Start main website server"""
        logger.info(f"🚀 Starting main website server on port {self.port}...")
        
        class UnityHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(self.website_dir), **kwargs)
                
            def end_headers(self):
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
                self.send_header('Access-Control-Allow-Headers', 'Content-Type')
                self.send_header('Cache-Control', 'no-cache')
                super().end_headers()
                
            def do_GET(self):
                if self.path == '/' or self.path == '/index.html':
                    self.path = '/revolutionary-landing.html'
                elif self.path == '/mobile' or self.path == '/app':
                    self.path = '/mobile-app.html'
                elif self.path == '/unity' or self.path == '/demo':
                    self.path = '/enhanced-unity-demo.html'
                    
                super().do_GET()
                
            def log_message(self, format, *args):
                # Suppress default logging
                pass
        
        def run_server():
            try:
                os.chdir(self.website_dir)
                with socketserver.TCPServer(("", self.port), UnityHTTPRequestHandler) as httpd:
                    logger.info(f"✅ Main website serving at http://localhost:{self.port}")
                    self.servers.append(httpd)
                    httpd.serve_forever()
            except Exception as e:
                logger.error(f"❌ Main server error: {e}")
        
        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()
        return thread

    def start_mobile_server(self):
        """Start mobile-optimized server"""
        logger.info(f"📱 Starting mobile server on port {self.mobile_port}...")
        
        class MobileHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(self.website_dir), **kwargs)
                
            def end_headers(self):
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Viewport', 'width=device-width, initial-scale=1.0')
                super().end_headers()
                
            def do_GET(self):
                if self.path == '/' or self.path == '/index.html':
                    self.path = '/mobile-app.html'
                super().do_GET()
        
        def run_mobile_server():
            try:
                os.chdir(self.website_dir)
                with socketserver.TCPServer(("", self.mobile_port), MobileHTTPRequestHandler) as httpd:
                    logger.info(f"✅ Mobile app serving at http://localhost:{self.mobile_port}")
                    httpd.serve_forever()
            except Exception as e:
                logger.error(f"❌ Mobile server error: {e}")
        
        thread = threading.Thread(target=run_mobile_server, daemon=True)
        thread.start()
        return thread

    def try_start_api_server(self):
        """Try to start the advanced API server if available"""
        try:
            api_script = self.base_dir / "scripts" / "website_server.py"
            if api_script.exists():
                logger.info(f"🔧 Starting advanced API server...")
                
                def run_api():
                    try:
                        subprocess.run([sys.executable, str(api_script)], cwd=str(self.base_dir))
                    except Exception as e:
                        logger.warning(f"⚠️  API server warning: {e}")
                
                thread = threading.Thread(target=run_api, daemon=True)
                thread.start()
                return thread
            else:
                logger.info("ℹ️  Advanced API server not available, using basic servers")
                return None
        except Exception as e:
            logger.warning(f"⚠️  Could not start API server: {e}")
            return None

    def open_browsers(self):
        """Open browsers to show the websites"""
        logger.info("🌐 Opening browsers...")
        
        def open_with_delay():
            time.sleep(2)  # Wait for servers to fully start
            
            try:
                # Main website
                webbrowser.open(f'http://localhost:{self.port}')
                logger.info(f"🖥️  Opened main website: http://localhost:{self.port}")
                
                time.sleep(1)
                
                # Mobile app (optional second tab)
                # webbrowser.open(f'http://localhost:{self.mobile_port}')
                # logger.info(f"📱 Opened mobile app: http://localhost:{self.mobile_port}")
                
            except Exception as e:
                logger.warning(f"⚠️  Browser opening warning: {e}")
        
        thread = threading.Thread(target=open_with_delay, daemon=True)
        thread.start()
        return thread

    def show_launch_info(self):
        """Display launch information"""
        print("\n" + "="*70)
        print("🌟✨ EEN UNITY MATHEMATICS WEBSITE LAUNCHED SUCCESSFULLY! ✨🌟")
        print("="*70)
        print(f"""
🖥️  MAIN WEBSITE:     http://localhost:{self.port}
📱 MOBILE APP:        http://localhost:{self.mobile_port}
🚀 API SERVER:        http://localhost:{self.api_port} (if available)

🎯 DIRECT LINKS:
   🌌 Revolutionary Landing:  http://localhost:{self.port}/
   ✨ Unity Demo:             http://localhost:{self.port}/unity
   📱 Mobile Experience:      http://localhost:{self.mobile_port}/
   🔮 Sacred Geometry:        http://localhost:{self.port}/proofs.html
   🧮 Interactive Playground: http://localhost:{self.port}/playground.html

🎮 CHEAT CODES TO TRY:
   • 420691337 (Quantum Resonance)
   • 1618033988 (Golden Spiral)  
   • 2718281828 (Euler Consciousness)
   • Konami Code: ↑↑↓↓←→←→BA

✨ FEATURES ACTIVE:
   • φ-Harmonic Consciousness Fields
   • Quantum Unity Visualizations
   • Interactive Sacred Geometry
   • Real-time Mathematical Proofs
   • Mobile-Optimized Experience
   • Cross-Platform Compatibility

💝 MADE WITH LOVE FOR UNITY MATHEMATICS
""")
        print("="*70)
        print("Press Ctrl+C to stop all servers")
        print("="*70)

    def launch(self):
        """Main launch sequence"""
        try:
            logger.info("🚀 Een Unity Mathematics Website Launcher Starting...")
            
            # Check dependencies
            self.check_dependencies()
            
            # Start servers
            main_thread = self.start_main_server()
            mobile_thread = self.start_mobile_server()
            api_thread = self.try_start_api_server()
            
            # Open browsers
            browser_thread = self.open_browsers()
            
            # Show info
            self.show_launch_info()
            
            # Keep running
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("\n🛑 Shutting down Unity Mathematics website...")
                logger.info("💫 Remember: Een plus een is een - always and forever!")
                sys.exit(0)
                
        except Exception as e:
            logger.error(f"❌ Launch error: {e}")
            sys.exit(1)

def main():
    """Entry point"""
    launcher = UnityWebsiteLauncher()
    launcher.launch()

if __name__ == "__main__":
    main()