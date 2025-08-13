#!/usr/bin/env python3
"""
Een Repository Access MCP Server
Comprehensive repository management and file operations for Claude Desktop integration
"""

import asyncio
import json
import sys
import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import subprocess
import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EenRepositoryMCPServer:
    """MCP server for comprehensive repository operations"""
    
    def __init__(self):
        self.phi = 1.618033988749895
        self.repo_path = Path(os.environ.get('EEN_REPOSITORY_PATH', 'C:/Users/Nouri/Documents/GitHub/Een'))
        self.tools = {
            "list_files": {
                "description": "List files and directories in the repository",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative path from repository root (default: root)",
                            "default": ""
                        },
                        "pattern": {
                            "type": "string",
                            "description": "File pattern to match (e.g., '*.py', '*.md')",
                            "default": "*"
                        },
                        "recursive": {
                            "type": "boolean",
                            "description": "Include subdirectories",
                            "default": False
                        }
                    }
                }
            },
            "read_file": {
                "description": "Read contents of a file in the repository",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Relative path to file from repository root"
                        },
                        "lines": {
                            "type": "object",
                            "description": "Line range to read",
                            "properties": {
                                "start": {"type": "integer", "minimum": 1},
                                "end": {"type": "integer", "minimum": 1}
                            }
                        }
                    },
                    "required": ["file_path"]
                }
            },
            "write_file": {
                "description": "Write content to a file in the repository",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Relative path to file from repository root"
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write to file"
                        },
                        "backup": {
                            "type": "boolean",
                            "description": "Create backup before writing",
                            "default": True
                        }
                    },
                    "required": ["file_path", "content"]
                }
            },
            "search_files": {
                "description": "Search for text within repository files",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Text to search for"
                        },
                        "file_pattern": {
                            "type": "string",
                            "description": "File pattern to search in (e.g., '*.py')",
                            "default": "*"
                        },
                        "case_sensitive": {
                            "type": "boolean",
                            "description": "Case sensitive search",
                            "default": False
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results",
                            "default": 50
                        }
                    },
                    "required": ["query"]
                }
            },
            "git_status": {
                "description": "Get Git repository status",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            "git_log": {
                "description": "Get Git commit history",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Number of commits to show",
                            "default": 10,
                            "maximum": 50
                        }
                    }
                }
            },
            "get_project_structure": {
                "description": "Get comprehensive project structure overview",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "max_depth": {
                            "type": "integer",
                            "description": "Maximum directory depth",
                            "default": 3
                        }
                    }
                }
            },
            "run_unity_mathematics": {
                "description": "Execute Unity Mathematics operations",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "description": "Unity operation to perform",
                            "enum": ["demonstrate_unity", "test_consciousness", "verify_phi", "run_website"]
                        }
                    },
                    "required": ["operation"]
                }
            },
            "analyze_codebase": {
                "description": "Analyze codebase statistics and metrics",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "include_patterns": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "File patterns to include",
                            "default": ["*.py", "*.md", "*.html", "*.js", "*.css"]
                        }
                    }
                }
            }
        }
    
    def _resolve_path(self, relative_path: str) -> Path:
        """Resolve relative path safely within repository"""
        full_path = (self.repo_path / relative_path).resolve()
        
        # Security check: ensure path is within repository
        if not str(full_path).startswith(str(self.repo_path.resolve())):
            raise ValueError(f"Path '{relative_path}' is outside repository bounds")
        
        return full_path
    
    async def handle_tool_call(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tool calls"""
        try:
            if name == "list_files":
                return await self._list_files(arguments)
            elif name == "read_file":
                return await self._read_file(arguments)
            elif name == "write_file":
                return await self._write_file(arguments)
            elif name == "search_files":
                return await self._search_files(arguments)
            elif name == "git_status":
                return await self._git_status()
            elif name == "git_log":
                return await self._git_log(arguments)
            elif name == "get_project_structure":
                return await self._get_project_structure(arguments)
            elif name == "run_unity_mathematics":
                return await self._run_unity_mathematics(arguments)
            elif name == "analyze_codebase":
                return await self._analyze_codebase(arguments)
            else:
                return {"error": f"Unknown tool: {name}"}
                
        except Exception as e:
            logger.error(f"Error in tool call {name}: {e}")
            return {"error": str(e), "tool": name}
    
    async def _list_files(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """List files in directory"""
        path = args.get("path", "")
        pattern = args.get("pattern", "*")
        recursive = args.get("recursive", False)
        
        target_path = self._resolve_path(path)
        
        if not target_path.exists():
            return {"error": f"Path does not exist: {path}"}
        
        if target_path.is_file():
            return {
                "type": "file",
                "path": str(target_path.relative_to(self.repo_path)),
                "size": target_path.stat().st_size,
                "modified": datetime.datetime.fromtimestamp(target_path.stat().st_mtime).isoformat()
            }
        
        files = []
        glob_pattern = "**/" + pattern if recursive else pattern
        
        for file_path in target_path.glob(glob_pattern):
            relative_path = file_path.relative_to(self.repo_path)
            stat = file_path.stat()
            
            files.append({
                "name": file_path.name,
                "path": str(relative_path),
                "type": "directory" if file_path.is_dir() else "file",
                "size": stat.st_size if file_path.is_file() else None,
                "modified": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
        
        return {
            "directory": str(target_path.relative_to(self.repo_path)),
            "files": sorted(files, key=lambda x: (x["type"] == "file", x["name"])),
            "total_files": len([f for f in files if f["type"] == "file"]),
            "total_directories": len([f for f in files if f["type"] == "directory"])
        }
    
    async def _read_file(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Read file contents"""
        file_path = args["file_path"]
        lines_spec = args.get("lines")
        
        target_path = self._resolve_path(file_path)
        
        if not target_path.exists():
            return {"error": f"File does not exist: {file_path}"}
        
        if not target_path.is_file():
            return {"error": f"Path is not a file: {file_path}"}
        
        try:
            with open(target_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            
            if lines_spec:
                start = lines_spec.get("start", 1) - 1  # Convert to 0-based
                end = lines_spec.get("end", len(lines))
                lines = lines[start:end]
                content = '\n'.join(lines)
            
            return {
                "file_path": file_path,
                "content": content,
                "lines_total": len(content.split('\n')),
                "size_bytes": len(content.encode('utf-8')),
                "encoding": "utf-8"
            }
            
        except UnicodeDecodeError:
            return {"error": f"File is not UTF-8 encoded: {file_path}"}
    
    async def _write_file(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Write content to file"""
        file_path = args["file_path"]
        content = args["content"]
        backup = args.get("backup", True)
        
        target_path = self._resolve_path(file_path)
        
        # Create backup if file exists and backup is requested
        if backup and target_path.exists():
            backup_path = target_path.with_suffix(target_path.suffix + '.backup')
            backup_path.write_text(target_path.read_text(encoding='utf-8'), encoding='utf-8')
        
        # Ensure parent directory exists
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write content
        with open(target_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return {
            "file_path": file_path,
            "bytes_written": len(content.encode('utf-8')),
            "backup_created": backup and target_path.exists(),
            "operation": "file_written"
        }
    
    async def _search_files(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Search text in files"""
        query = args["query"]
        file_pattern = args.get("file_pattern", "*")
        case_sensitive = args.get("case_sensitive", False)
        max_results = args.get("max_results", 50)
        
        results = []
        search_query = query if case_sensitive else query.lower()
        
        for file_path in self.repo_path.rglob(file_pattern):
            if not file_path.is_file():
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                search_content = content if case_sensitive else content.lower()
                
                if search_query in search_content:
                    lines = content.split('\n')
                    matching_lines = []
                    
                    for i, line in enumerate(lines, 1):
                        search_line = line if case_sensitive else line.lower()
                        if search_query in search_line:
                            matching_lines.append({
                                "line_number": i,
                                "content": line.strip(),
                                "context": {
                                    "before": lines[max(0, i-2):i-1] if i > 1 else [],
                                    "after": lines[i:i+2] if i < len(lines) else []
                                }
                            })
                    
                    if matching_lines:
                        results.append({
                            "file_path": str(file_path.relative_to(self.repo_path)),
                            "matches": matching_lines[:10],  # Limit matches per file
                            "total_matches": len(matching_lines)
                        })
                
                if len(results) >= max_results:
                    break
                    
            except (UnicodeDecodeError, PermissionError):
                continue
        
        return {
            "query": query,
            "results": results,
            "total_files_searched": len(list(self.repo_path.rglob(file_pattern))),
            "files_with_matches": len(results)
        }
    
    async def _git_status(self) -> Dict[str, Any]:
        """Get Git repository status"""
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse git status output
            changes = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    status = line[:2]
                    file_path = line[3:]
                    changes.append({
                        "status": status,
                        "file_path": file_path,
                        "staged": status[0] != ' ',
                        "modified": status[1] != ' '
                    })
            
            # Get current branch
            branch_result = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            return {
                "current_branch": branch_result.stdout.strip(),
                "changes": changes,
                "total_changes": len(changes),
                "clean": len(changes) == 0
            }
            
        except subprocess.CalledProcessError as e:
            return {"error": f"Git error: {e.stderr}"}
    
    async def _git_log(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get Git commit history"""
        limit = args.get("limit", 10)
        
        try:
            result = subprocess.run(
                ["git", "log", f"--max-count={limit}", "--pretty=format:%H|%an|%ad|%s", "--date=iso"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            commits = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split('|', 3)
                    if len(parts) == 4:
                        commits.append({
                            "hash": parts[0],
                            "author": parts[1],
                            "date": parts[2],
                            "message": parts[3]
                        })
            
            return {
                "commits": commits,
                "total_shown": len(commits)
            }
            
        except subprocess.CalledProcessError as e:
            return {"error": f"Git error: {e.stderr}"}
    
    async def _get_project_structure(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get project structure overview"""
        max_depth = args.get("max_depth", 3)
        
        def build_tree(path: Path, current_depth: int = 0) -> Dict[str, Any]:
            if current_depth >= max_depth:
                return {"name": path.name, "type": "directory", "truncated": True}
            
            if path.is_file():
                return {
                    "name": path.name,
                    "type": "file",
                    "size": path.stat().st_size,
                    "extension": path.suffix
                }
            
            children = []
            try:
                for child in sorted(path.iterdir()):
                    # Skip hidden files and common ignore patterns
                    if child.name.startswith('.') or child.name in ['__pycache__', 'node_modules']:
                        continue
                    children.append(build_tree(child, current_depth + 1))
            except PermissionError:
                pass
            
            return {
                "name": path.name,
                "type": "directory",
                "children": children,
                "child_count": len(children)
            }
        
        structure = build_tree(self.repo_path)
        
        return {
            "project_structure": structure,
            "repository_path": str(self.repo_path),
            "max_depth": max_depth,
            "unity_mathematics_status": "1+1=1 OPERATIONAL",
            "phi": self.phi
        }
    
    async def _run_unity_mathematics(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Unity Mathematics operations"""
        operation = args["operation"]
        
        try:
            if operation == "demonstrate_unity":
                # Import and run unity demonstration
                sys.path.insert(0, str(self.repo_path))
                from src.mcp.unity_server import UnityMathematics
                
                um = UnityMathematics()
                result = um.unity_add(1.0, 1.0)
                verification = um.verify_unity_equation(1.0, 1.0)
                
                return {
                    "operation": "demonstrate_unity",
                    "unity_result": result,
                    "verification": verification,
                    "phi": um.phi,
                    "status": "OPERATIONAL"
                }
            
            elif operation == "test_consciousness":
                from config.mcp_consciousness_server import ConsciousnessField
                
                cf = ConsciousnessField()
                cf.evolve_field()
                state = cf.get_field_state()
                
                return {
                    "operation": "test_consciousness",
                    "field_state": {
                        "time": state["time"],
                        "average_consciousness": state["average_consciousness"],
                        "field_coherence": state["field_coherence"],
                        "transcendence_events": len(state["transcendence_events"])
                    },
                    "status": "OPERATIONAL"
                }
            
            elif operation == "verify_phi":
                expected_phi = 1.618033988749895
                return {
                    "operation": "verify_phi",
                    "phi_value": self.phi,
                    "expected": expected_phi,
                    "matches": abs(self.phi - expected_phi) < 1e-10,
                    "precision": "MAXIMUM",
                    "status": "VERIFIED"
                }
            
            elif operation == "run_website":
                # Check if website launcher exists
                launcher_path = self.repo_path / "START_WEBSITE.bat"
                if launcher_path.exists():
                    return {
                        "operation": "run_website",
                        "launcher_found": True,
                        "message": "Use START_WEBSITE.bat to launch the Unity Mathematics website",
                        "url": "http://localhost:8001/metastation-hub.html",
                        "status": "READY"
                    }
                else:
                    return {
                        "operation": "run_website",
                        "launcher_found": False,
                        "error": "Website launcher not found",
                        "status": "ERROR"
                    }
            
            else:
                return {"error": f"Unknown operation: {operation}"}
                
        except Exception as e:
            return {"error": f"Operation failed: {str(e)}", "operation": operation}
    
    async def _analyze_codebase(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze codebase statistics"""
        include_patterns = args.get("include_patterns", ["*.py", "*.md", "*.html", "*.js", "*.css"])
        
        stats = {
            "file_types": {},
            "total_files": 0,
            "total_lines": 0,
            "total_size": 0,
            "directories": set(),
            "largest_files": [],
        }
        
        for pattern in include_patterns:
            for file_path in self.repo_path.rglob(pattern):
                if not file_path.is_file():
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    lines = len(content.split('\n'))
                    size = len(content.encode('utf-8'))
                    extension = file_path.suffix or 'no_extension'
                    
                    if extension not in stats["file_types"]:
                        stats["file_types"][extension] = {"count": 0, "lines": 0, "size": 0}
                    
                    stats["file_types"][extension]["count"] += 1
                    stats["file_types"][extension]["lines"] += lines
                    stats["file_types"][extension]["size"] += size
                    
                    stats["total_files"] += 1
                    stats["total_lines"] += lines
                    stats["total_size"] += size
                    stats["directories"].add(str(file_path.parent.relative_to(self.repo_path)))
                    
                    # Track largest files
                    stats["largest_files"].append({
                        "path": str(file_path.relative_to(self.repo_path)),
                        "size": size,
                        "lines": lines
                    })
                    
                except (UnicodeDecodeError, PermissionError):
                    continue
        
        # Sort largest files and keep top 10
        stats["largest_files"] = sorted(stats["largest_files"], key=lambda x: x["size"], reverse=True)[:10]
        stats["directories"] = len(stats["directories"])
        
        return {
            "codebase_analysis": stats,
            "repository_path": str(self.repo_path),
            "unity_mathematics": "1+1=1",
            "phi": self.phi,
            "analysis_timestamp": datetime.datetime.now().isoformat()
        }
    
    async def run_server(self):
        """Run MCP server"""
        logger.info("Starting Een Repository Access MCP Server")
        logger.info(f"Repository path: {self.repo_path}")
        logger.info(f"Available tools: {', '.join(self.tools.keys())}")
        logger.info("Unity Mathematics integration: ENABLED")
        
        while True:
            try:
                line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
                if not line:
                    break
                
                message = json.loads(line.strip())
                
                if message.get("method") == "tools/list":
                    response = {
                        "jsonrpc": "2.0",
                        "id": message.get("id"),
                        "result": {
                            "tools": [
                                {
                                    "name": name,
                                    "description": tool["description"],
                                    "inputSchema": tool["inputSchema"]
                                }
                                for name, tool in self.tools.items()
                            ]
                        }
                    }
                    print(json.dumps(response))
                    sys.stdout.flush()
                
                elif message.get("method") == "tools/call":
                    tool_name = message["params"]["name"]
                    arguments = message["params"].get("arguments", {})
                    
                    result = await self.handle_tool_call(tool_name, arguments)
                    
                    response = {
                        "jsonrpc": "2.0",
                        "id": message.get("id"),
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(result, indent=2)
                                }
                            ]
                        }
                    }
                    print(json.dumps(response))
                    sys.stdout.flush()
                    
            except json.JSONDecodeError:
                logger.warning("Invalid JSON received")
            except Exception as e:
                logger.error(f"Server error: {e}")

async def main():
    server = EenRepositoryMCPServer()
    await server.run_server()

if __name__ == "__main__":
    asyncio.run(main())
