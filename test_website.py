#!/usr/bin/env python3
"""
Een Website Local Test Server
============================

Simple test server to validate website functionality before deployment.
Tests all pages including metagambit and AI chat integration.

Usage: python test_website.py
"""

import http.server
import socketserver
import webbrowser
import threading
import time
import os
from pathlib import Path

class EenTestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler for Een website testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory="website", **kwargs)
    
    def end_headers(self):
        # Add CORS headers for local testing
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def log_message(self, format, *args):
        # Custom logging for better visibility
        message = format % args
        if "404" in message:
            print(f"⚠️  404 Error: {message}")
        elif "200" in message:
            print(f"✅ Success: {message}")
        else:
            print(f"ℹ️  Info: {message}")

def test_website_files():
    """Test that all essential website files exist."""
    print("🔍 Testing website file structure...")
    
    essential_files = [
        "website/index.html",
        "website/metagambit.html", 
        "website/css/style.css",
        "website/css/metagambit.css",
        "website/static/chat.js",
        "website/js/navigation.js",
        "website/js/katex-integration.js",
        "website/js/unity-demo.js"
    ]
    
    all_good = True
    for file_path in essential_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ MISSING: {file_path}")
            all_good = False
    
    return all_good

def open_test_pages(port):
    """Open test pages in browser after server starts."""
    time.sleep(2)  # Wait for server to start
    
    test_urls = [
        f"http://localhost:{port}/",
        f"http://localhost:{port}/metagambit.html"
    ]
    
    print(f"\n🌐 Opening test pages...")
    for url in test_urls:
        print(f"Opening: {url}")
        webbrowser.open(url)
        time.sleep(1)

def main():
    """Main test function."""
    print("🚀 Een Website Local Test Server")
    print("=" * 50)
    
    # Test file structure first
    if not test_website_files():
        print("\n❌ Some essential files are missing!")
        print("Please ensure all website files are present before testing.")
        return 1
    
    print("\n✅ All essential files present!")
    
    # Start test server
    PORT = 8080
    
    print(f"\n🖥️  Starting test server on port {PORT}...")
    print(f"Website will be available at: http://localhost:{PORT}")
    print(f"Metagambit page: http://localhost:{PORT}/metagambit.html")
    print(f"\nPress Ctrl+C to stop the server")
    
    try:
        # Start browser opening in background
        browser_thread = threading.Thread(target=open_test_pages, args=(PORT,))
        browser_thread.daemon = True
        browser_thread.start()
        
        # Start server
        with socketserver.TCPServer(("", PORT), EenTestHandler) as httpd:
            print(f"\n🎯 Server running! Test your presentation website now.")
            print(f"🤖 AI Chat widget should appear as a φ icon in bottom-right")
            print(f"📄 Check that metagambit page loads correctly")
            print(f"🔍 Open browser console to see any JavaScript errors")
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print(f"\n\n🛑 Server stopped.")
        print(f"✅ Website testing complete!")
        return 0
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())