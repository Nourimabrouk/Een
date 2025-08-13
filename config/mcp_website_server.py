#!/usr/bin/env python3
"""
Website Management MCP Server Entry Point
Unity Mathematics website operations for Claude Desktop integration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mcp.website_management_server import main
import asyncio

if __name__ == "__main__":
    # The actual implementation is in src/mcp/website_management_server.py
    # This is the entry point for the MCP configuration
    asyncio.run(main())