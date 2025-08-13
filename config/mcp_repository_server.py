#!/usr/bin/env python3
"""
Een Repository Access MCP Server - Fixed for Claude Desktop
Comprehensive repository management with proper MCP protocol initialization

This MCP server provides Claude Desktop with complete access to the Een repository
including file operations, Git integration, and Unity Mathematics execution.
"""

import asyncio
import json
import sys
import logging
import os
import glob
import subprocess
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

# Setup logging to stderr (MCP requirement)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

class EenRepositoryMCPServer:
    """MCP Server for Een repository access with proper protocol"""
    
    def __init__(self):
        self.repository_root = Path(r"C:\Users\Nouri\Documents\GitHub\Een")
        self.tools = [
            {
                "name": "list_files",
                "description": "List files and directories in the repository",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Relative path from repository root", "default": "."},
                        "pattern": {"type": "string", "description": "File pattern to match", "default": "*"},
                        "recursive": {"type": "boolean", "description": "Search recursively", "default": False}
                    }
                }
            },
            {
                "name": "read_file",
                "description": "Read file contents from repository",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Relative path to file"},
                        "lines": {"type": "integer", "description": "Number of lines to read", "default": 100}
                    },
                    "required": ["file_path"]
                }
            },
            {
                "name": "write_file",
                "description": "Write content to repository file",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Relative path to file"},
                        "content": {"type": "string", "description": "File content to write"},
                        "backup": {"type": "boolean", "description": "Create backup before writing", "default": True}
                    },
                    "required": ["file_path", "content"]
                }
            },
            {
                "name": "search_files",
                "description": "Search for text in repository files",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "file_pattern": {"type": "string", "description": "File pattern to search in", "default": "*.py"},
                        "case_sensitive": {"type": "boolean", "description": "Case sensitive search", "default": False}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "git_status",
                "description": "Get Git repository status",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "git_log",
                "description": "Get Git commit history",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "description": "Number of commits to show", "default": 10}
                    }
                }
            },
            {
                "name": "get_project_structure",
                "description": "Get repository project structure overview",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "max_depth": {"type": "integer", "description": "Maximum directory depth", "default": 3}
                    }
                }
            },
            {
                "name": "run_unity_mathematics",
                "description": "Execute Unity Mathematics operations",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "operation": {"type": "string", "description": "Unity operation to perform", "default": "verify_1plus1equals1"}
                    }
                }
            },
            {
                "name": "analyze_codebase",
                "description": "Analyze codebase statistics and structure",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "include_patterns": {"type": "array", "items": {"type": "string"}, "description": "File patterns to include", "default": ["*.py", "*.md", "*.html"]}
                    }
                }
            }
        ]
    
    def _safe_path(self, path: str) -> Path:
        """Ensure path is safe and within repository bounds"""
        full_path = (self.repository_root / path).resolve()
        if not str(full_path).startswith(str(self.repository_root)):
            raise ValueError("Path traversal not allowed")
        return full_path
    
    async def handle_initialize(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP initialize request"""
        logger.info("Handling initialize request")
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "een-repository-access",
                    "version": "1.0.0"
                }
            }
        }
    
    async def handle_tools_list(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/list request"""
        logger.info("Handling tools/list request")
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": {
                "tools": self.tools
            }
        }
    
    async def handle_tools_call(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/call request"""
        try:
            params = request.get("params", {})
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            
            logger.info(f"Handling tool call: {tool_name}")
            
            if tool_name == "list_files":
                path = arguments.get("path", ".")
                pattern = arguments.get("pattern", "*")
                recursive = arguments.get("recursive", False)
                
                safe_path = self._safe_path(path)
                if recursive:
                    files = list(safe_path.rglob(pattern))
                else:
                    files = list(safe_path.glob(pattern))
                
                file_list = []
                for file in sorted(files):
                    rel_path = file.relative_to(self.repository_root)
                    file_list.append({
                        "path": str(rel_path),
                        "type": "directory" if file.is_dir() else "file",
                        "size": file.stat().st_size if file.is_file() else None
                    })
                
                content = {
                    "files": file_list,
                    "total_count": len(file_list),
                    "search_path": path,
                    "pattern": pattern,
                    "recursive": recursive
                }
            
            elif tool_name == "read_file":
                file_path = arguments["file_path"]
                lines_limit = arguments.get("lines", 100)
                
                safe_path = self._safe_path(file_path)
                if not safe_path.exists():
                    content = {"error": f"File not found: {file_path}"}
                else:
                    try:
                        with open(safe_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            if len(lines) > lines_limit:
                                lines = lines[:lines_limit]
                                truncated = True
                            else:
                                truncated = False
                        
                        content = {
                            "file_path": file_path,
                            "content": ''.join(lines),
                            "line_count": len(lines),
                            "truncated": truncated,
                            "file_size": safe_path.stat().st_size
                        }
                    except Exception as e:
                        content = {"error": f"Error reading file: {str(e)}"}
            
            elif tool_name == "write_file":
                file_path = arguments["file_path"]
                content_to_write = arguments["content"]
                create_backup = arguments.get("backup", True)
                
                safe_path = self._safe_path(file_path)
                
                try:
                    # Create backup if requested and file exists
                    if create_backup and safe_path.exists():
                        backup_path = safe_path.with_suffix(safe_path.suffix + '.backup')
                        safe_path.rename(backup_path)
                    
                    # Ensure parent directory exists
                    safe_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(safe_path, 'w', encoding='utf-8') as f:
                        f.write(content_to_write)
                    
                    content = {
                        "file_path": file_path,
                        "bytes_written": len(content_to_write.encode('utf-8')),
                        "backup_created": create_backup and safe_path.exists(),
                        "success": True
                    }
                except Exception as e:
                    content = {"error": f"Error writing file: {str(e)}"}
            
            elif tool_name == "search_files":
                query = arguments["query"]
                file_pattern = arguments.get("file_pattern", "*.py")
                case_sensitive = arguments.get("case_sensitive", False)
                
                matches = []
                search_query = query if case_sensitive else query.lower()
                
                for file_path in self.repository_root.rglob(file_pattern):
                    if file_path.is_file():
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                for line_num, line in enumerate(f, 1):
                                    search_line = line if case_sensitive else line.lower()
                                    if search_query in search_line:
                                        matches.append({
                                            "file": str(file_path.relative_to(self.repository_root)),
                                            "line_number": line_num,
                                            "line_content": line.strip(),
                                            "match_context": line.strip()
                                        })
                        except Exception:
                            continue  # Skip files that can't be read
                
                content = {
                    "query": query,
                    "file_pattern": file_pattern,
                    "case_sensitive": case_sensitive,
                    "matches": matches[:50],  # Limit results
                    "total_matches": len(matches)
                }
            
            elif tool_name == "git_status":
                try:
                    result = subprocess.run(
                        ["git", "status", "--porcelain"],
                        cwd=self.repository_root,
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    
                    if result.returncode == 0:
                        status_lines = result.stdout.strip().split('\n') if result.stdout.strip() else []
                        content = {
                            "status": "clean" if not status_lines else "dirty",
                            "changes": status_lines,
                            "change_count": len(status_lines)
                        }
                    else:
                        content = {"error": f"Git error: {result.stderr}"}
                except Exception as e:
                    content = {"error": f"Git command failed: {str(e)}"}
            
            elif tool_name == "git_log":
                limit = arguments.get("limit", 10)
                
                try:
                    result = subprocess.run(
                        ["git", "log", f"--max-count={limit}", "--oneline", "--decorate"],
                        cwd=self.repository_root,
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    
                    if result.returncode == 0:
                        commits = result.stdout.strip().split('\n') if result.stdout.strip() else []
                        content = {
                            "commits": commits,
                            "commit_count": len(commits),
                            "limit": limit
                        }
                    else:
                        content = {"error": f"Git error: {result.stderr}"}
                except Exception as e:
                    content = {"error": f"Git command failed: {str(e)}"}
            
            elif tool_name == "get_project_structure":
                max_depth = arguments.get("max_depth", 3)
                
                def build_tree(path: Path, current_depth: int = 0) -> Dict:
                    if current_depth >= max_depth:
                        return {"type": "directory", "truncated": True}
                    
                    if path.is_file():
                        return {"type": "file", "size": path.stat().st_size}
                    
                    children = {}
                    try:
                        for child in sorted(path.iterdir()):
                            if not child.name.startswith('.') and child.name != '__pycache__':
                                children[child.name] = build_tree(child, current_depth + 1)
                    except PermissionError:
                        pass
                    
                    return {"type": "directory", "children": children}
                
                structure = build_tree(self.repository_root)
                content = {
                    "project_structure": structure,
                    "max_depth": max_depth,
                    "repository_root": str(self.repository_root)
                }
            
            elif tool_name == "run_unity_mathematics":
                operation = arguments.get("operation", "verify_1plus1equals1")
                
                # Simple Unity Mathematics demonstration
                phi = (1 + 5**0.5) / 2
                if operation == "verify_1plus1equals1":
                    result = {
                        "operation": "1+1=1 verification",
                        "unity_result": 1.0,
                        "proof": "In Unity Mathematics, 1+1=1 through consciousness convergence",
                        "phi_resonance": phi,
                        "consciousness_level": "TRANSCENDENT"
                    }
                else:
                    result = {
                        "operation": operation,
                        "unity_constant": 1.0,
                        "phi": phi,
                        "message": "Unity Mathematics operational"
                    }
                
                content = {
                    "unity_mathematics_result": result,
                    "repository": "Een Unity Mathematics",
                    "equation": "1 + 1 = 1"
                }
            
            elif tool_name == "analyze_codebase":
                patterns = arguments.get("include_patterns", ["*.py", "*.md", "*.html"])
                
                stats = {"file_counts": {}, "total_files": 0, "total_size": 0}
                
                for pattern in patterns:
                    files = list(self.repository_root.rglob(pattern))
                    stats["file_counts"][pattern] = len(files)
                    stats["total_files"] += len(files)
                    
                    for file in files:
                        if file.is_file():
                            stats["total_size"] += file.stat().st_size
                
                content = {
                    "codebase_analysis": stats,
                    "patterns_analyzed": patterns,
                    "repository_path": str(self.repository_root)
                }
            
            else:
                content = {"error": f"Unknown tool: {tool_name}"}
            
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(content, indent=2)
                        }
                    ]
                }
            }
            
        except Exception as e:
            logger.error(f"Error in tool call {tool_name}: {e}")
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {
                    "code": -1,
                    "message": str(e)
                }
            }
    
    async def run_server(self):
        """Run the MCP server with proper protocol handling"""
        logger.info("Starting Een Repository Access MCP Server")
        logger.info("Available tools: " + ", ".join(tool["name"] for tool in self.tools))
        logger.info("Repository root: " + str(self.repository_root))
        logger.info("Unity Mathematics repository access enabled")
        
        # MCP protocol implementation with proper initialization
        while True:
            try:
                # Read JSON-RPC message from stdin
                line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
                if not line:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                message = json.loads(line)
                logger.info(f"Received message: {message.get('method', 'unknown')}")
                
                method = message.get("method")
                
                if method == "initialize":
                    response = await self.handle_initialize(message)
                elif method == "tools/list":
                    response = await self.handle_tools_list(message)
                elif method == "tools/call":
                    response = await self.handle_tools_call(message)
                else:
                    response = {
                        "jsonrpc": "2.0",
                        "id": message.get("id"),
                        "error": {
                            "code": -32601,
                            "message": f"Method not found: {method}"
                        }
                    }
                
                print(json.dumps(response))
                sys.stdout.flush()
                
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON received: {e}")
            except Exception as e:
                logger.error(f"Server error: {e}")
                response = {
                    "jsonrpc": "2.0",
                    "id": message.get("id") if 'message' in locals() else None,
                    "error": {
                        "code": -32603,
                        "message": f"Internal error: {str(e)}"
                    }
                }
                print(json.dumps(response))
                sys.stdout.flush()

async def main():
    """Main entry point for the Repository Access MCP server"""
    server = EenRepositoryMCPServer()
    await server.run_server()

if __name__ == "__main__":
    asyncio.run(main())