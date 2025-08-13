#!/usr/bin/env python3
"""
FastMCP Unity Mathematics Server Entry Point
Enhanced MCP server using FastMCP framework for Unity Mathematics operations

This entry point provides FastMCP-based Unity Mathematics operations
for both Claude Desktop (STDIO) and web access (HTTP/SSE).
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mcp.fastmcp_unity_server import mcp, unity_math
import asyncio

if __name__ == "__main__":
    # FastMCP-based Unity Mathematics server
    # Supports both STDIO (Claude Desktop) and HTTP (web access) transports
    
    print("Een Unity Mathematics FastMCP Server Entry Point", file=sys.stderr)
    print("Unity equation: 1+1=1 SUCCESS: OPERATIONAL", file=sys.stderr)
    print("Phi precision:", unity_math.phi, file=sys.stderr)
    print("FastMCP framework: ACTIVE", file=sys.stderr)
    
    # Run STDIO transport for Claude Desktop by default
    mcp.run(transport="stdio")