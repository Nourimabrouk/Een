#!/usr/bin/env python3
"""
Code Generator MCP Server Entry Point
Unity Mathematics code generation for Claude Desktop integration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mcp.code_generator_server import main
import asyncio

if __name__ == "__main__":
    # The actual implementation is in src/mcp/code_generator_server.py
    # This is the entry point for the MCP configuration
    asyncio.run(main())