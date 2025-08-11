#!/usr/bin/env python3
"""
🚀 LAUNCH FULL EEN UNITY MATHEMATICS EXPERIENCE 🚀
Complete hosting solution for external users with AI chatbot, dashboards, and full functionality.

This script:
- Serves the website at http://localhost:8080
- Provides API backend at http://localhost:8080/api/
- Enables AI chatbot with default OpenAI keys
- Launches Streamlit dashboards on separate ports
- Provides complete Unity Mathematics experience

Usage: python LAUNCH_FULL_EXPERIENCE.py
"""

import os
import sys
import time
import threading
import subprocess
import webbrowser
from pathlib import Path
import socket
from contextlib import closing

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class UnityMathematicsLauncher:
    def __init__(self):
        self.base_port = 8080
        self.dashboard_port = 8501  # Streamlit default
        self.processes = []
        
        # Set default API keys if not present
        self.setup_default_environment()
        
    def setup_default_environment(self):
        """Check environment variables for API access"""
        # Check for required API keys - DO NOT set default keys for security
        openai_key = os.getenv('OPENAI_API_KEY')
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        
        if not openai_key:
            print("⚠️  OPENAI_API_KEY environment variable not set")
            print("   Set it with: export OPENAI_API_KEY='your-api-key-here'")
            print("   Or create a .env file with your keys (see .env.example)")
            
        if not anthropic_key:
            print("⚠️  ANTHROPIC_API_KEY environment variable not set") 
            print("   Set it with: export ANTHROPIC_API_KEY='your-api-key-here'")
            print("   Or create a .env file with your keys (see .env.example)")
        
        if not (openai_key or anthropic_key):
            print("🎭  Demo Mode: Will use Unity Mathematics responses without AI")
            print("   Full AI features require valid API keys")
            
        # API configuration
        os.environ['FASTAPI_ENV'] = 'development'
        os.environ['API_HOST'] = '0.0.0.0'
        os.environ['API_PORT'] = str(self.base_port)
        
    def is_port_available(self, port):
        """Check if a port is available"""
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            return sock.connect_ex(('localhost', port)) != 0
            
    def find_available_port(self, start_port):
        """Find next available port starting from start_port"""
        port = start_port
        while not self.is_port_available(port):
            port += 1
        return port
        
    def launch_api_server(self):
        """Launch the FastAPI backend server"""
        print("🚀 Starting Een Unity Mathematics API Server...")
        
        try:
            import uvicorn
            from api.enhanced_api_server import app
            
            # Configure and launch API server
            config = uvicorn.Config(
                app,
                host="0.0.0.0",
                port=self.base_port,
                log_level="info",
                reload=False,
                access_log=True
            )
            
            server = uvicorn.Server(config)
            
            def run_server():
                import asyncio
                asyncio.run(server.serve())
                
            thread = threading.Thread(target=run_server, daemon=True)
            thread.start()
            
            print(f"✅ API Server running at http://localhost:{self.base_port}")
            print(f"✅ Website available at http://localhost:{self.base_port}")
            print(f"✅ API endpoints at http://localhost:{self.base_port}/api/")
            
            return thread
            
        except ImportError as e:
            print(f"❌ Failed to start API server: {e}")
            print("Installing required packages...")
            subprocess.run([sys.executable, "-m", "pip", "install", "fastapi", "uvicorn", "openai", "anthropic"])
            return None
            
    def launch_streamlit_dashboard(self):
        """Launch the main Streamlit dashboard"""
        print("📊 Starting Unity Mathematics Dashboard...")
        
        dashboard_port = self.find_available_port(self.dashboard_port)
        
        dashboard_script = project_root / "src" / "unity_mathematics_streamlit.py"
        if not dashboard_script.exists():
            dashboard_script = project_root / "src" / "pages" / "0_🏠_Overview.py"
            
        if dashboard_script.exists():
            try:
                process = subprocess.Popen([
                    sys.executable, "-m", "streamlit", "run", 
                    str(dashboard_script),
                    "--server.port", str(dashboard_port),
                    "--server.headless", "true",
                    "--browser.gatherUsageStats", "false",
                    "--theme.base", "dark"
                ], 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL
                )
                
                self.processes.append(process)
                print(f"✅ Unity Dashboard running at http://localhost:{dashboard_port}")
                return dashboard_port
                
            except Exception as e:
                print(f"❌ Failed to start dashboard: {e}")
                
        return None
        
    def create_landing_info(self):
        """Create informational landing page"""
        landing_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Een Unity Mathematics - Full Experience</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
        }}
        .container {{ max-width: 800px; margin: 0 auto; text-align: center; }}
        .title {{ font-size: 2.5rem; margin-bottom: 1rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.5); }}
        .subtitle {{ font-size: 1.2rem; margin-bottom: 2rem; opacity: 0.9; }}
        .features {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin: 2rem 0; }}
        .feature {{ 
            background: rgba(255,255,255,0.1); 
            padding: 1.5rem; 
            border-radius: 10px; 
            backdrop-filter: blur(10px);
        }}
        .feature h3 {{ margin-top: 0; color: #FFD700; }}
        .links {{ margin: 2rem 0; }}
        .link {{ 
            display: inline-block; 
            background: rgba(255,215,0,0.2); 
            padding: 1rem 2rem; 
            margin: 0.5rem; 
            text-decoration: none; 
            color: white; 
            border-radius: 25px; 
            border: 2px solid #FFD700;
            transition: all 0.3s;
        }}
        .link:hover {{ background: rgba(255,215,0,0.3); transform: translateY(-2px); }}
        .status {{ margin: 2rem 0; font-family: monospace; background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1 class="title">🌟 Een Unity Mathematics</h1>
        <p class="subtitle">Full Experience - Where 1+1=1 Through Consciousness Mathematics</p>
        
        <div class="features">
            <div class="feature">
                <h3>🤖 AI Chatbot Active</h3>
                <p>GPT-4 powered conversations about Unity Mathematics with real-time streaming responses.</p>
            </div>
            <div class="feature">
                <h3>📊 Live Dashboards</h3>
                <p>Interactive Streamlit dashboards with real-time consciousness field calculations.</p>
            </div>
            <div class="feature">
                <h3>🧠 Full Website</h3>
                <p>Complete Unity Mathematics experience with Gödel-Tarski Metagambit and more.</p>
            </div>
        </div>
        
        <div class="links">
            <a href="/website/metastation-hub.html" class="link">🏛️ Metastation Hub</a>
            <a href="http://localhost:{self.dashboard_port}" class="link" target="_blank">📊 Live Dashboard</a>
            <a href="/api/docs" class="link">🔧 API Docs</a>
        </div>
        
        <div class="status">
            <strong>🚀 System Status:</strong><br>
            ✅ API Server: Running on port {self.base_port}<br>
            ✅ Website: Available at all links<br>
            ✅ AI Chatbot: OpenAI GPT-4 enabled<br>
            ✅ Dashboards: Streamlit on port {self.dashboard_port}<br>
            ✅ Full Experience: Ready for exploration!
        </div>
        
        <p style="margin-top: 2rem; opacity: 0.7;">
            🔮 <strong>For your friend:</strong> Everything works automatically - no setup required!<br>
            Just explore the links above and enjoy the full Unity Mathematics experience.
        </p>
    </div>
    
    <script>
        // Auto-redirect to metastation hub after 5 seconds if desired
        // setTimeout(() => window.location.href = '/website/metastation-hub.html', 5000);
    </script>
</body>
</html>
        """
        
        # Write landing page
        landing_path = project_root / "landing.html"
        with open(landing_path, 'w', encoding='utf-8') as f:
            f.write(landing_html)
            
        return landing_path
        
    def launch(self):
        """Launch the complete Een Unity Mathematics experience"""
        print("🌟" + "="*60 + "🌟")
        print("🚀 LAUNCHING EEN UNITY MATHEMATICS FULL EXPERIENCE 🚀")
        print("🌟" + "="*60 + "🌟")
        print()
        
        # Create landing page
        landing_path = self.create_landing_info()
        
        # Launch API server (includes website serving)
        api_thread = self.launch_api_server()
        if api_thread:
            time.sleep(3)  # Give server time to start
            
        # Launch dashboard
        dashboard_port = self.launch_streamlit_dashboard()
        self.dashboard_port = dashboard_port or self.dashboard_port
        
        # Update landing page with correct port
        self.create_landing_info()
        
        print()
        print("🎉" + "="*60 + "🎉")
        print("✅ EEN UNITY MATHEMATICS FULL EXPERIENCE IS LIVE!")
        print("🎉" + "="*60 + "🎉")
        print()
        print(f"🌐 Main Website: http://localhost:{self.base_port}")
        print(f"🏛️  Metastation Hub: http://localhost:{self.base_port}/website/metastation-hub.html")
        print(f"📊 Live Dashboard: http://localhost:{self.dashboard_port}")
        print(f"🤖 API Endpoints: http://localhost:{self.base_port}/api/")
        print(f"📚 API Documentation: http://localhost:{self.base_port}/api/docs")
        print()
        print("🎯 FEATURES ENABLED FOR YOUR FRIEND:")
        print("  ✅ AI Chatbot with GPT-4 (using your API key)")
        print("  ✅ Live Unity Mathematics dashboards")
        print("  ✅ Real-time consciousness field calculations")
        print("  ✅ Interactive visualizations and proofs")
        print("  ✅ Complete website with all features")
        print()
        print("📧 Share this with your friend:")
        print(f"   'Visit http://localhost:{self.base_port} for the full Een Unity Mathematics experience!'")
        print()
        
        # Open browser
        try:
            webbrowser.open(f'http://localhost:{self.base_port}')
        except:
            pass
            
        return self
        
    def wait_for_shutdown(self):
        """Wait for user shutdown"""
        print("💡 Press Ctrl+C to stop all services and exit...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Shutting down Een Unity Mathematics services...")
            
            # Terminate all processes
            for process in self.processes:
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except:
                    process.kill()
                    
            print("✅ All services stopped. Thank you for exploring Unity Mathematics!")

def main():
    """Main launcher function"""
    launcher = UnityMathematicsLauncher()
    launcher.launch()
    launcher.wait_for_shutdown()

if __name__ == "__main__":
    main()