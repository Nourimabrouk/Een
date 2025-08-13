#!/usr/bin/env python3
"""
MCP Server Launcher for Een Unity Mathematics
Launches the appropriate MCP server based on command line arguments
"""

import sys
import os
import importlib.util
import asyncio

# Add parent directory to path so we can import from een module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def launch_server(server_name: str):
    """Launch the specified MCP server from src.mcp module"""
    
    server_map = {
        "unity": "src.mcp.unity_server",
        "consciousness": "src.mcp.consciousness_server",
        "quantum": "src.mcp.quantum_server",
        "omega": "src.mcp.omega_server",
        "file": "src.mcp.file_management_server",
        "code": "src.mcp.code_generator_server"
    }
    
    if server_name not in server_map:
        print(f"Error: Unknown server '{server_name}'", file=sys.stderr)
        print(f"Available servers: {', '.join(server_map.keys())}", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Import the server module
        module_name = server_map[server_name]
        module = importlib.import_module(module_name)
        
        # Run the server's main function
        if hasattr(module, 'main'):
            await module.main()
        elif hasattr(module, 'run_server'):
            await module.run_server()
        else:
            print(f"Error: Server module {module_name} has no main() or run_server() function", file=sys.stderr)
            sys.exit(1)
            
    except ImportError as e:
        print(f"Error importing server module: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error running server: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    """Main entry point for the MCP launcher"""
    if len(sys.argv) < 2:
        print("Usage: python mcp_launcher.py <server_name>", file=sys.stderr)
        print("Available servers: unity, consciousness, quantum, omega, file, code", file=sys.stderr)
        sys.exit(1)
    
    server_name = sys.argv[1]
    asyncio.run(launch_server(server_name))

if __name__ == "__main__":
    main()