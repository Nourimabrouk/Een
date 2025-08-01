#!/usr/bin/env python3
"""
Unity Mathematics MCP Server
Core mathematical operations for 1+1=1
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from een.mcp.unity_server import main
import asyncio

if __name__ == "__main__":
    # The actual implementation is in een/mcp/unity_server.py
    # This is just the entry point for the MCP configuration
    asyncio.run(main())