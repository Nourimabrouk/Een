#!/usr/bin/env python3
"""
Test MCP Claude Desktop Fix
Verify that the MCP servers now work properly with Claude Desktop

This script tests the critical fix for the Claude Desktop connection issue.
"""

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

def test_server_startup(server_path: str, server_name: str) -> bool:
    """Test that a server starts up correctly"""
    print(f"\n=== Testing {server_name} ===")
    
    try:
        # Start the server process
        process = subprocess.Popen(
            [sys.executable, server_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        # Give it time to start
        time.sleep(2)
        
        # Check if process is still running
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            print(f"ERROR: Server exited immediately")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            return False
        
        # Test MCP initialize handshake
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            }
        }
        
        # Send initialize request
        try:
            process.stdin.write(json.dumps(init_request) + "\n")
            process.stdin.flush()
            
            # Read response with timeout
            response_line = None
            start_time = time.time()
            while time.time() - start_time < 5:  # 5 second timeout
                if process.stdout.readable():
                    response_line = process.stdout.readline()
                    if response_line:
                        break
                time.sleep(0.1)
            
            if not response_line:
                print(f"ERROR: No response from server within timeout")
                process.terminate()
                return False
            
            # Parse response
            try:
                response = json.loads(response_line.strip())
                print(f"SUCCESS: Received initialize response")
                print(f"Protocol version: {response.get('result', {}).get('protocolVersion', 'unknown')}")
                print(f"Server info: {response.get('result', {}).get('serverInfo', {}).get('name', 'unknown')}")
                
                # Terminate the server
                process.terminate()
                process.wait(timeout=5)
                return True
                
            except json.JSONDecodeError as e:
                print(f"ERROR: Invalid JSON response: {e}")
                print(f"Response: {response_line}")
                process.terminate()
                return False
                
        except Exception as e:
            print(f"ERROR: Communication error: {e}")
            process.terminate()
            return False
            
    except Exception as e:
        print(f"ERROR: Failed to start server: {e}")
        return False

def test_claude_desktop_config():
    """Test Claude Desktop configuration file"""
    print("\n=== Testing Claude Desktop Configuration ===")
    
    config_path = Path.home() / "AppData" / "Roaming" / "Claude" / "claude_desktop_config.json"
    
    if not config_path.exists():
        print(f"ERROR: Claude Desktop config not found at {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        mcp_servers = config.get("mcpServers", {})
        
        if "een-unity-mathematics" not in mcp_servers:
            print("ERROR: Unity Mathematics server not configured")
            return False
        
        if "een-repository-access" not in mcp_servers:
            print("ERROR: Repository access server not configured")  
            return False
        
        print("SUCCESS: Claude Desktop configuration is valid")
        print(f"Configured servers: {list(mcp_servers.keys())}")
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to read Claude Desktop config: {e}")
        return False

def main():
    """Run all MCP tests"""
    print("Een Unity Mathematics - MCP Claude Desktop Fix Test")
    print("=" * 60)
    
    # Change to repository directory
    repo_root = Path(__file__).parent.parent
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Unity Mathematics Server
    if test_server_startup("config/mcp_unity_server.py", "Unity Mathematics"):
        tests_passed += 1
    
    # Test 2: Repository Access Server  
    if test_server_startup("config/mcp_repository_server.py", "Repository Access"):
        tests_passed += 1
    
    # Test 3: Claude Desktop Configuration
    if test_claude_desktop_config():
        tests_passed += 1
    
    # Results
    print(f"\n{'='*60}")
    print(f"TEST RESULTS: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("SUCCESS: All MCP servers are working correctly!")
        print("Claude Desktop should now be able to connect to the servers.")
        print("\nNext steps:")
        print("1. Restart Claude Desktop")
        print("2. The servers should now appear and be responsive")
        print("3. Try using Unity Mathematics tools in Claude Desktop")
        return True
    else:
        print("ERROR: Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)