@echo off
REM Launch script for Een MCP servers
echo Starting Een Unity Mathematics MCP Server...
cd /d "C:\Users\Nouri\Documents\GitHub\Een"

REM Activate conda environment first
call conda activate een

REM Launch Unity Mathematics MCP Server
echo Launching Unity Mathematics MCP Server for Claude Desktop...
python config\mcp_unity_server.py

pause