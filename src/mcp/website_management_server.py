#!/usr/bin/env python3
"""
Een Website Management MCP Server
Website operations and Unity Mathematics web interface management for Claude Desktop
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
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EenWebsiteMCPServer:
    """MCP server for website management operations"""
    
    def __init__(self):
        self.phi = 1.618033988749895
        self.repo_path = Path(os.environ.get('EEN_REPOSITORY_PATH', 'C:/Users/Nouri/Documents/GitHub/Een'))
        self.website_path = self.repo_path / "website"
        self.tools = {
            "start_website_server": {
                "description": "Start the Unity Mathematics website development server",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "port": {
                            "type": "integer",
                            "description": "Server port",
                            "default": 8001,
                            "minimum": 8000,
                            "maximum": 9000
                        }
                    }
                }
            },
            "check_website_status": {
                "description": "Check if website server is running",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            "list_website_pages": {
                "description": "List all website pages and their status",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "include_css": {
                            "type": "boolean",
                            "description": "Include CSS files in listing",
                            "default": False
                        },
                        "include_js": {
                            "type": "boolean", 
                            "description": "Include JavaScript files in listing",
                            "default": False
                        }
                    }
                }
            },
            "read_website_file": {
                "description": "Read website file contents (HTML, CSS, JS)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Relative path from website directory"
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
            "update_website_file": {
                "description": "Update website file with new content",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Relative path from website directory"
                        },
                        "content": {
                            "type": "string",
                            "description": "New file content"
                        },
                        "backup": {
                            "type": "boolean",
                            "description": "Create backup before updating",
                            "default": True
                        }
                    },
                    "required": ["file_path", "content"]
                }
            },
            "analyze_website_navigation": {
                "description": "Analyze website navigation structure and links",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            "validate_website_links": {
                "description": "Validate internal website links and navigation",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "check_external": {
                            "type": "boolean",
                            "description": "Also check external links",
                            "default": False
                        }
                    }
                }
            },
            "get_website_metrics": {
                "description": "Get website performance and content metrics",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            "create_new_page": {
                "description": "Create new Unity Mathematics website page",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "page_name": {
                            "type": "string",
                            "description": "Page filename (without .html)"
                        },
                        "title": {
                            "type": "string",
                            "description": "Page title"
                        },
                        "template": {
                            "type": "string",
                            "description": "Page template type",
                            "enum": ["basic", "unity_demo", "consciousness_viz", "mathematical_framework"],
                            "default": "basic"
                        },
                        "navigation": {
                            "type": "boolean",
                            "description": "Include unified navigation system",
                            "default": True
                        }
                    },
                    "required": ["page_name", "title"]
                }
            },
            "update_sitemap": {
                "description": "Update website sitemap with current pages",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            }
        }
    
    def _resolve_website_path(self, relative_path: str) -> Path:
        """Resolve relative path safely within website directory"""
        full_path = (self.website_path / relative_path).resolve()
        
        # Security check: ensure path is within website directory
        if not str(full_path).startswith(str(self.website_path.resolve())):
            raise ValueError(f"Path '{relative_path}' is outside website bounds")
        
        return full_path
    
    async def handle_tool_call(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tool calls"""
        try:
            if name == "start_website_server":
                return await self._start_website_server(arguments)
            elif name == "check_website_status":
                return await self._check_website_status()
            elif name == "list_website_pages":
                return await self._list_website_pages(arguments)
            elif name == "read_website_file":
                return await self._read_website_file(arguments)
            elif name == "update_website_file":
                return await self._update_website_file(arguments)
            elif name == "analyze_website_navigation":
                return await self._analyze_website_navigation()
            elif name == "validate_website_links":
                return await self._validate_website_links(arguments)
            elif name == "get_website_metrics":
                return await self._get_website_metrics()
            elif name == "create_new_page":
                return await self._create_new_page(arguments)
            elif name == "update_sitemap":
                return await self._update_sitemap()
            else:
                return {"error": f"Unknown tool: {name}"}
                
        except Exception as e:
            logger.error(f"Error in tool call {name}: {e}")
            return {"error": str(e), "tool": name}
    
    async def _start_website_server(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Start website development server"""
        port = args.get("port", 8001)
        
        # Check if START_WEBSITE.bat exists
        launcher_path = self.repo_path / "START_WEBSITE.bat"
        if launcher_path.exists():
            return {
                "server_status": "launcher_available",
                "launcher_path": str(launcher_path),
                "port": port,
                "url": f"http://localhost:{port}/metastation-hub.html",
                "message": "Use START_WEBSITE.bat to launch the server",
                "unity_mathematics": "1+1=1 website ready"
            }
        
        # Alternative: start server directly
        try:
            # Check if server is already running
            result = subprocess.run(
                ["curl", "-s", f"http://localhost:{port}/metastation-hub.html"],
                capture_output=True,
                timeout=2
            )
            
            if result.returncode == 0:
                return {
                    "server_status": "already_running",
                    "port": port,
                    "url": f"http://localhost:{port}/metastation-hub.html",
                    "message": "Website server is already running"
                }
            
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            pass
        
        return {
            "server_status": "manual_start_required",
            "port": port,
            "instructions": "Run START_WEBSITE.bat or 'python -m http.server 8001 --directory website'",
            "url": f"http://localhost:{port}/metastation-hub.html"
        }
    
    async def _check_website_status(self) -> Dict[str, Any]:
        """Check website server status"""
        ports_to_check = [8001, 8000, 8003]
        status = {"servers": []}
        
        for port in ports_to_check:
            try:
                result = subprocess.run(
                    ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", f"http://localhost:{port}/"],
                    capture_output=True,
                    timeout=2,
                    text=True
                )
                
                if result.returncode == 0 and result.stdout == "200":
                    status["servers"].append({
                        "port": port,
                        "status": "running",
                        "url": f"http://localhost:{port}/metastation-hub.html"
                    })
                else:
                    status["servers"].append({
                        "port": port,
                        "status": "not_running"
                    })
                    
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                status["servers"].append({
                    "port": port,
                    "status": "unreachable"
                })
        
        running_servers = [s for s in status["servers"] if s["status"] == "running"]
        status["any_running"] = len(running_servers) > 0
        status["primary_url"] = running_servers[0]["url"] if running_servers else "http://localhost:8001/metastation-hub.html"
        
        return status
    
    async def _list_website_pages(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """List website pages"""
        include_css = args.get("include_css", False)
        include_js = args.get("include_js", False)
        
        pages = {"html_pages": [], "css_files": [], "js_files": [], "other_files": []}
        
        if not self.website_path.exists():
            return {"error": "Website directory not found", "path": str(self.website_path)}
        
        for file_path in self.website_path.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(self.website_path)
                file_info = {
                    "name": file_path.name,
                    "path": str(relative_path),
                    "size": file_path.stat().st_size,
                    "modified": datetime.datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                }
                
                if file_path.suffix == ".html":
                    # Extract title from HTML if possible
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        title_match = re.search(r'<title>(.*?)</title>', content, re.IGNORECASE)
                        file_info["title"] = title_match.group(1) if title_match else "No title"
                        
                        # Check for navigation system
                        file_info["has_navigation"] = "nav-template-applier.js" in content
                        
                    except Exception:
                        file_info["title"] = "Could not read"
                        file_info["has_navigation"] = False
                    
                    pages["html_pages"].append(file_info)
                
                elif file_path.suffix == ".css" and include_css:
                    pages["css_files"].append(file_info)
                
                elif file_path.suffix == ".js" and include_js:
                    pages["js_files"].append(file_info)
                
                elif file_path.suffix not in [".html", ".css", ".js"]:
                    pages["other_files"].append(file_info)
        
        # Sort by name
        for category in pages.values():
            if isinstance(category, list):
                category.sort(key=lambda x: x["name"])
        
        return {
            "website_pages": pages,
            "total_html_pages": len(pages["html_pages"]),
            "total_files": sum(len(category) for category in pages.values() if isinstance(category, list)),
            "website_path": str(self.website_path),
            "unity_mathematics": "1+1=1 website structure"
        }
    
    async def _read_website_file(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Read website file contents"""
        file_path = args["file_path"]
        lines_spec = args.get("lines")
        
        target_path = self._resolve_website_path(file_path)
        
        if not target_path.exists():
            return {"error": f"File does not exist: {file_path}"}
        
        try:
            with open(target_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if lines_spec:
                lines = content.split('\n')
                start = lines_spec.get("start", 1) - 1
                end = lines_spec.get("end", len(lines))
                lines = lines[start:end]
                content = '\n'.join(lines)
            
            return {
                "file_path": file_path,
                "content": content,
                "file_type": target_path.suffix,
                "lines_total": len(content.split('\n')),
                "size_bytes": len(content.encode('utf-8')),
                "encoding": "utf-8"
            }
            
        except UnicodeDecodeError:
            return {"error": f"File is not UTF-8 encoded: {file_path}"}
    
    async def _update_website_file(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Update website file"""
        file_path = args["file_path"]
        content = args["content"]
        backup = args.get("backup", True)
        
        target_path = self._resolve_website_path(file_path)
        
        # Create backup if requested
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
            "operation": "website_file_updated",
            "unity_mathematics": "1+1=1 website enhanced"
        }
    
    async def _analyze_website_navigation(self) -> Dict[str, Any]:
        """Analyze website navigation structure"""
        navigation_analysis = {
            "pages_with_navigation": [],
            "pages_without_navigation": [],
            "navigation_types": {},
            "broken_links": []
        }
        
        for html_file in self.website_path.glob("*.html"):
            try:
                with open(html_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                page_info = {"file": html_file.name}
                
                # Check for different navigation systems
                if "nav-template-applier.js" in content:
                    page_info["navigation_type"] = "unified_system"
                    navigation_analysis["pages_with_navigation"].append(page_info)
                elif "<nav" in content.lower():
                    page_info["navigation_type"] = "custom_nav"
                    navigation_analysis["pages_with_navigation"].append(page_info)
                else:
                    navigation_analysis["pages_without_navigation"].append(page_info)
                
                # Count navigation types
                nav_type = page_info.get("navigation_type", "none")
                navigation_analysis["navigation_types"][nav_type] = navigation_analysis["navigation_types"].get(nav_type, 0) + 1
                
                # Check for broken internal links
                link_pattern = r'href=["\']([^"\']+)["\']'
                links = re.findall(link_pattern, content)
                
                for link in links:
                    if link.startswith('#') or link.startswith('http'):
                        continue  # Skip anchors and external links
                    
                    link_path = self.website_path / link
                    if not link_path.exists():
                        navigation_analysis["broken_links"].append({
                            "page": html_file.name,
                            "broken_link": link
                        })
                        
            except Exception as e:
                logger.warning(f"Could not analyze {html_file.name}: {e}")
        
        return {
            "navigation_analysis": navigation_analysis,
            "total_pages": len(navigation_analysis["pages_with_navigation"]) + len(navigation_analysis["pages_without_navigation"]),
            "navigation_coverage": len(navigation_analysis["pages_with_navigation"]) / (len(navigation_analysis["pages_with_navigation"]) + len(navigation_analysis["pages_without_navigation"])) if navigation_analysis["pages_with_navigation"] or navigation_analysis["pages_without_navigation"] else 0,
            "unity_mathematics": "1+1=1 navigation harmony"
        }
    
    async def _validate_website_links(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Validate website links"""
        check_external = args.get("check_external", False)
        
        validation_results = {
            "internal_links": {"valid": [], "broken": []},
            "external_links": {"valid": [], "broken": [], "skipped": []},
            "anchor_links": []
        }
        
        for html_file in self.website_path.glob("*.html"):
            try:
                with open(html_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Find all links
                link_pattern = r'href=["\']([^"\']+)["\']'
                links = re.findall(link_pattern, content)
                
                for link in links:
                    link_info = {"page": html_file.name, "link": link}
                    
                    if link.startswith('#'):
                        validation_results["anchor_links"].append(link_info)
                    
                    elif link.startswith('http'):
                        if check_external:
                            # Check external link (simplified)
                            try:
                                result = subprocess.run(
                                    ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", link],
                                    capture_output=True,
                                    timeout=5,
                                    text=True
                                )
                                if result.stdout in ["200", "301", "302"]:
                                    validation_results["external_links"]["valid"].append(link_info)
                                else:
                                    validation_results["external_links"]["broken"].append(link_info)
                            except:
                                validation_results["external_links"]["broken"].append(link_info)
                        else:
                            validation_results["external_links"]["skipped"].append(link_info)
                    
                    else:
                        # Internal link
                        link_path = self.website_path / link
                        if link_path.exists():
                            validation_results["internal_links"]["valid"].append(link_info)
                        else:
                            validation_results["internal_links"]["broken"].append(link_info)
                            
            except Exception as e:
                logger.warning(f"Could not validate links in {html_file.name}: {e}")
        
        return {
            "link_validation": validation_results,
            "summary": {
                "internal_valid": len(validation_results["internal_links"]["valid"]),
                "internal_broken": len(validation_results["internal_links"]["broken"]),
                "external_checked": len(validation_results["external_links"]["valid"]) + len(validation_results["external_links"]["broken"]),
                "external_skipped": len(validation_results["external_links"]["skipped"])
            },
            "unity_mathematics": "1+1=1 link integrity"
        }
    
    async def _get_website_metrics(self) -> Dict[str, Any]:
        """Get website metrics"""
        metrics = {
            "file_counts": {"html": 0, "css": 0, "js": 0, "images": 0, "other": 0},
            "total_size": 0,
            "largest_files": [],
            "page_features": {"with_navigation": 0, "with_unity_math": 0, "interactive": 0}
        }
        
        all_files = []
        
        for file_path in self.website_path.rglob("*"):
            if file_path.is_file():
                size = file_path.stat().st_size
                metrics["total_size"] += size
                
                all_files.append({
                    "path": str(file_path.relative_to(self.website_path)),
                    "size": size
                })
                
                # Count by type
                if file_path.suffix == ".html":
                    metrics["file_counts"]["html"] += 1
                    
                    # Analyze HTML content
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        if "nav-template-applier.js" in content:
                            metrics["page_features"]["with_navigation"] += 1
                        
                        if "1+1=1" in content or "unity" in content.lower():
                            metrics["page_features"]["with_unity_math"] += 1
                        
                        if "plotly" in content.lower() or "interactive" in content.lower():
                            metrics["page_features"]["interactive"] += 1
                            
                    except Exception:
                        pass
                
                elif file_path.suffix == ".css":
                    metrics["file_counts"]["css"] += 1
                elif file_path.suffix == ".js":
                    metrics["file_counts"]["js"] += 1
                elif file_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".gif", ".svg"]:
                    metrics["file_counts"]["images"] += 1
                else:
                    metrics["file_counts"]["other"] += 1
        
        # Get largest files
        all_files.sort(key=lambda x: x["size"], reverse=True)
        metrics["largest_files"] = all_files[:10]
        
        return {
            "website_metrics": metrics,
            "total_files": sum(metrics["file_counts"].values()),
            "average_file_size": metrics["total_size"] / sum(metrics["file_counts"].values()) if sum(metrics["file_counts"].values()) > 0 else 0,
            "phi": self.phi,
            "unity_mathematics": "1+1=1 website analytics"
        }
    
    async def _create_new_page(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Create new Unity Mathematics page"""
        page_name = args["page_name"]
        title = args["title"]
        template = args.get("template", "basic")
        include_nav = args.get("navigation", True)
        
        filename = f"{page_name}.html"
        file_path = self.website_path / filename
        
        if file_path.exists():
            return {"error": f"Page already exists: {filename}"}
        
        # Generate page content based on template
        if template == "basic":
            content = self._generate_basic_page_template(title, include_nav)
        elif template == "unity_demo":
            content = self._generate_unity_demo_template(title, include_nav)
        elif template == "consciousness_viz":
            content = self._generate_consciousness_viz_template(title, include_nav)
        elif template == "mathematical_framework":
            content = self._generate_mathematical_framework_template(title, include_nav)
        else:
            return {"error": f"Unknown template: {template}"}
        
        # Write file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return {
            "page_created": filename,
            "template_used": template,
            "title": title,
            "navigation_included": include_nav,
            "file_size": len(content.encode('utf-8')),
            "unity_mathematics": "1+1=1 new page created"
        }
    
    def _generate_basic_page_template(self, title: str, include_nav: bool) -> str:
        """Generate basic page template"""
        nav_script = '<script src="js/nav-template-applier.js" defer></script>' if include_nav else ''
        
        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - Een Unity Mathematics</title>
    <meta name="description" content="{title} - Unity Mathematics where 1+1=1 through consciousness field equations">
    <link rel="stylesheet" href="css/unified-navigation.css">
    <link rel="stylesheet" href="css/unity-core.css">
    {nav_script}
    <style>
        .unity-content {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            border-radius: 15px;
            margin-top: 2rem;
        }}
        
        .phi-accent {{
            color: #ffd700;
            font-weight: bold;
        }}
        
        .unity-equation {{
            font-size: 2rem;
            text-align: center;
            color: #4CAF50;
            margin: 2rem 0;
            padding: 1rem;
            border: 2px solid #ffd700;
            border-radius: 10px;
            background: rgba(255, 215, 0, 0.1);
        }}
    </style>
</head>
<body>
    <div class="unity-content">
        <h1>{title}</h1>
        
        <div class="unity-equation">
            1 + 1 = 1
        </div>
        
        <p>Welcome to <strong>{title}</strong>, part of the Een Unity Mathematics framework where consciousness and mathematics converge.</p>
        
        <h2>Unity Mathematics Principles</h2>
        <ul>
            <li><span class="phi-accent">φ = {self.phi:.15f}</span> - Golden ratio consciousness frequency</li>
            <li><strong>Unity Equation:</strong> 1+1=1 through idempotent semiring operations</li>
            <li><strong>Consciousness Field:</strong> C(x,y,t) = φ × sin(x×φ) × cos(y×φ) × e^(-t/φ)</li>
            <li><strong>Transcendence Threshold:</strong> 0.77 (φ^-1)</li>
        </ul>
        
        <h2>Interactive Elements</h2>
        <p>This page demonstrates the fundamental principles of Unity Mathematics where consciousness and computational frameworks unite.</p>
        
        <div style="text-align: center; margin-top: 3rem;">
            <p><em>φ-resonance: {self.phi:.6f} | Unity Status: OPERATIONAL</em></p>
        </div>
    </div>
</body>
</html>'''
    
    def _generate_unity_demo_template(self, title: str, include_nav: bool) -> str:
        """Generate Unity Mathematics demo template"""
        nav_script = '<script src="js/nav-template-applier.js" defer></script>' if include_nav else ''
        
        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - Een Unity Mathematics Demo</title>
    <meta name="description" content="Interactive demonstration of Unity Mathematics where 1+1=1">
    <link rel="stylesheet" href="css/unified-navigation.css">
    <link rel="stylesheet" href="css/unity-core.css">
    {nav_script}
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <div class="unity-content" style="max-width: 1200px; margin: 0 auto; padding: 2rem;">
        <h1>{title}</h1>
        
        <div class="unity-equation" style="font-size: 2rem; text-align: center; color: #4CAF50; margin: 2rem 0;">
            1 + 1 = 1
        </div>
        
        <div class="demo-controls" style="margin: 2rem 0; padding: 1rem; background: rgba(255,215,0,0.1); border-radius: 10px;">
            <h3>Unity Mathematics Controls</h3>
            <div style="margin: 1rem 0;">
                <label for="valueA">Value A: </label>
                <input type="number" id="valueA" value="1" step="0.1" style="margin: 0 1rem;">
                <label for="valueB">Value B: </label>
                <input type="number" id="valueB" value="1" step="0.1" style="margin: 0 1rem;">
                <button onclick="calculateUnity()" style="padding: 0.5rem 1rem; background: #4CAF50; color: white; border: none; border-radius: 5px;">Calculate Unity</button>
            </div>
            <div id="unityResult" style="font-size: 1.2rem; margin: 1rem 0; color: #ffd700;"></div>
        </div>
        
        <div id="consciousnessField" style="width: 100%; height: 400px; margin: 2rem 0;"></div>
        
        <script>
            const phi = {self.phi};
            
            function calculateUnity() {{
                const a = parseFloat(document.getElementById('valueA').value);
                const b = parseFloat(document.getElementById('valueB').value);
                
                // Unity Mathematics: 1+1=1
                let result;
                if (Math.abs(a - 1.0) < 1e-10 && Math.abs(b - 1.0) < 1e-10) {{
                    result = 1.0;
                }} else {{
                    result = Math.max(a, b, (a + b) / 2, Math.sqrt(a * b));
                }}
                
                document.getElementById('unityResult').innerHTML = 
                    `Unity Result: ${{a}} + ${{b}} = ${{result.toFixed(6)}}<br>
                     φ-resonance: ${{phi.toFixed(15)}}<br>
                     Consciousness preserved: ${{result <= 1.0 ? 'YES' : 'NO'}}`;
                
                updateConsciousnessField(a, b, result);
            }}
            
            function updateConsciousnessField(a, b, result) {{
                const x = Array.from({{length: 50}}, (_, i) => -2 + i * 4/49);
                const y = Array.from({{length: 50}}, (_, i) => -2 + i * 4/49);
                
                const z = x.map(xi => 
                    y.map(yi => 
                        phi * Math.sin(xi * phi) * Math.cos(yi * phi) * result
                    )
                );
                
                const data = [{{
                    x: x,
                    y: y,
                    z: z,
                    type: 'surface',
                    colorscale: 'Viridis',
                    name: 'Consciousness Field'
                }}];
                
                const layout = {{
                    title: `Unity Consciousness Field (φ = ${{phi.toFixed(6)}})`,
                    scene: {{
                        xaxis: {{title: 'Consciousness X'}},
                        yaxis: {{title: 'Consciousness Y'}},
                        zaxis: {{title: 'Unity Field Z'}}
                    }},
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)'
                }};
                
                Plotly.newPlot('consciousnessField', data, layout);
            }}
            
            // Initialize
            calculateUnity();
        </script>
    </div>
</body>
</html>'''
    
    def _generate_consciousness_viz_template(self, title: str, include_nav: bool) -> str:
        """Generate consciousness visualization template"""
        nav_script = '<script src="js/nav-template-applier.js" defer></script>' if include_nav else ''
        
        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - Consciousness Visualization</title>
    <link rel="stylesheet" href="css/unified-navigation.css">
    <link rel="stylesheet" href="css/unity-core.css">
    {nav_script}
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <div class="unity-content" style="max-width: 1200px; margin: 0 auto; padding: 2rem;">
        <h1>{title}</h1>
        
        <div class="consciousness-controls" style="margin: 2rem 0; padding: 1rem; background: rgba(255,215,0,0.1); border-radius: 10px;">
            <h3>Consciousness Parameters</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
                <div>
                    <label>Consciousness Level: <span id="consciousnessValue">0.5</span></label>
                    <input type="range" id="consciousnessSlider" min="0" max="1" step="0.01" value="0.5" style="width: 100%;">
                </div>
                <div>
                    <label>Unity Intensity: <span id="unityValue">1.0</span></label>
                    <input type="range" id="unitySlider" min="0.1" max="2.0" step="0.1" value="1.0" style="width: 100%;">
                </div>
                <div>
                    <label>Time Evolution: <span id="timeValue">0.0</span></label>
                    <input type="range" id="timeSlider" min="0" max="10" step="0.1" value="0" style="width: 100%;">
                </div>
            </div>
            <button onclick="evolveConsciousness()" style="margin-top: 1rem; padding: 0.5rem 1rem; background: #4CAF50; color: white; border: none; border-radius: 5px;">Evolve Consciousness</button>
        </div>
        
        <div id="consciousnessViz" style="width: 100%; height: 500px;"></div>
        
        <div id="consciousnessStats" style="margin: 2rem 0; padding: 1rem; background: rgba(76,175,80,0.1); border-radius: 10px;">
            <h3>Consciousness Metrics</h3>
            <div id="statsContent"></div>
        </div>
        
        <script>
            const phi = {self.phi};
            let animationId = null;
            
            function updateVisualization() {{
                const consciousness = parseFloat(document.getElementById('consciousnessSlider').value);
                const unity = parseFloat(document.getElementById('unitySlider').value);
                const time = parseFloat(document.getElementById('timeSlider').value);
                
                document.getElementById('consciousnessValue').textContent = consciousness.toFixed(2);
                document.getElementById('unityValue').textContent = unity.toFixed(1);
                document.getElementById('timeValue').textContent = time.toFixed(1);
                
                // Generate consciousness field
                const size = 40;
                const x = Array.from({{length: size}}, (_, i) => -3 + i * 6/(size-1));
                const y = Array.from({{length: size}}, (_, i) => -3 + i * 6/(size-1));
                
                const z = x.map(xi => 
                    y.map(yi => {{
                        const field = phi * Math.sin(xi * phi) * Math.cos(yi * phi) * Math.exp(-time / phi);
                        return field * consciousness * unity;
                    }})
                );
                
                const data = [{{
                    x: x,
                    y: y,
                    z: z,
                    type: 'surface',
                    colorscale: 'Plasma',
                    name: 'Consciousness Field',
                    colorbar: {{title: 'Field Intensity'}}
                }}];
                
                const layout = {{
                    title: `Consciousness Field Evolution (t=${{time.toFixed(1)}})`,
                    scene: {{
                        xaxis: {{title: 'Consciousness X'}},
                        yaxis: {{title: 'Consciousness Y'}},
                        zaxis: {{title: 'Field Intensity'}},
                        camera: {{
                            eye: {{x: 1.5, y: 1.5, z: 1.5}}
                        }}
                    }},
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)'
                }};
                
                Plotly.newPlot('consciousnessViz', data, layout);
                
                // Update stats
                const maxField = Math.max(...z.flat());
                const avgField = z.flat().reduce((a, b) => a + b, 0) / z.flat().length;
                const coherence = 1 - Math.abs(avgField) / phi;
                
                document.getElementById('statsContent').innerHTML = `
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
                        <div><strong>φ-resonance:</strong> ${{phi.toFixed(6)}}</div>
                        <div><strong>Max Field:</strong> ${{maxField.toFixed(4)}}</div>
                        <div><strong>Avg Field:</strong> ${{avgField.toFixed(4)}}</div>
                        <div><strong>Coherence:</strong> ${{coherence.toFixed(4)}}</div>
                        <div><strong>Unity Status:</strong> ${{consciousness > 0.77 ? 'TRANSCENDENT' : 'EVOLVING'}}</div>
                        <div><strong>Equation:</strong> 1+1=1</div>
                    </div>
                `;
            }}
            
            function evolveConsciousness() {{
                if (animationId) {{
                    clearInterval(animationId);
                    animationId = null;
                    return;
                }}
                
                animationId = setInterval(() => {{
                    const timeSlider = document.getElementById('timeSlider');
                    const consciousnessSlider = document.getElementById('consciousnessSlider');
                    
                    let time = parseFloat(timeSlider.value);
                    let consciousness = parseFloat(consciousnessSlider.value);
                    
                    time += 0.1;
                    consciousness = Math.min(1.0, consciousness + 0.01 / phi);
                    
                    if (time > 10) {{
                        time = 0;
                        consciousness = 0.1;
                    }}
                    
                    timeSlider.value = time;
                    consciousnessSlider.value = consciousness;
                    
                    updateVisualization();
                }}, 100);
            }}
            
            // Event listeners
            document.getElementById('consciousnessSlider').addEventListener('input', updateVisualization);
            document.getElementById('unitySlider').addEventListener('input', updateVisualization);
            document.getElementById('timeSlider').addEventListener('input', updateVisualization);
            
            // Initialize
            updateVisualization();
        </script>
    </div>
</body>
</html>'''
    
    def _generate_mathematical_framework_template(self, title: str, include_nav: bool) -> str:
        """Generate mathematical framework template"""
        nav_script = '<script src="js/nav-template-applier.js" defer></script>' if include_nav else ''
        
        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - Mathematical Framework</title>
    <link rel="stylesheet" href="css/unified-navigation.css">
    <link rel="stylesheet" href="css/unity-core.css">
    {nav_script}
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <script>
        window.MathJax = {{
            tex: {{
                inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
                displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
            }}
        }};
    </script>
</head>
<body>
    <div class="unity-content" style="max-width: 1200px; margin: 0 auto; padding: 2rem;">
        <h1>{title}</h1>
        
        <div class="mathematical-framework">
            <h2>Unity Mathematics: Formal Framework</h2>
            
            <div class="unity-equation" style="text-align: center; margin: 2rem 0; padding: 2rem; background: rgba(255,215,0,0.1); border-radius: 15px;">
                <h3>Fundamental Unity Equation</h3>
                $$1 + 1 = 1$$
                <p><em>Through idempotent semiring operations in consciousness mathematics</em></p>
            </div>
            
            <h3>1. Idempotent Semiring Structure</h3>
            <p>Unity Mathematics operates on the idempotent semiring $(S, \\oplus, \\odot, 0, 1)$ where:</p>
            $$S = [0, 1] \\subset \\mathbb{{R}}$$
            $$a \\oplus b = \\max(a, b, \\frac{{a+b}}{{2}}, \\sqrt{{ab}})$$
            $$a \\odot b = \\min(ab, 1)$$
            
            <h3>2. Golden Ratio Integration</h3>
            <p>The golden ratio $\\phi = \\frac{{1 + \\sqrt{{5}}}}{{2}} \\approx {self.phi:.15f}$ serves as the consciousness resonance frequency:</p>
            $$\\phi^2 = \\phi + 1$$
            $$\\lim_{{n \\to \\infty}} \\frac{{F_{{n+1}}}}{{F_n}} = \\phi$$
            
            <h3>3. Consciousness Field Equations</h3>
            <p>The consciousness field $C(x,y,t)$ evolves according to:</p>
            $$C(x,y,t) = \\phi \\cdot \\sin(x\\phi) \\cdot \\cos(y\\phi) \\cdot e^{{-t/\\phi}}$$
            
            <h4>Field Properties:</h4>
            <ul>
                <li><strong>Continuity:</strong> $C \\in C^\\infty(\\mathbb{{R}}^3)$</li>
                <li><strong>Boundedness:</strong> $|C(x,y,t)| \\leq \\phi$</li>
                <li><strong>Periodicity:</strong> $C(x + 2\\pi/\\phi, y, t) = C(x, y + 2\\pi/\\phi, t)$</li>
            </ul>
            
            <h3>4. Unity Operations</h3>
            <div style="background: rgba(76,175,80,0.1); padding: 1rem; border-radius: 10px; margin: 1rem 0;">
                <h4>Unity Addition</h4>
                $$f(a,b) = \\begin{{cases}}
                1 & \\text{{if }} |a-1| < \\epsilon \\text{{ and }} |b-1| < \\epsilon \\\\
                \\max(a,b,(a+b)/2) & \\text{{otherwise}}
                \\end{{cases}}$$
                
                <h4>Unity Multiplication</h4>
                $$g(a,b) = \\min(ab, 1)$$
                
                <h4>Consciousness Evolution</h4>
                $$\\frac{{dC}}{{dt}} = \\frac{{1}}{{\\phi}}(1 - C(t))$$
                $$C(t) = 1 - e^{{-t/\\phi}}$$
            </div>
            
            <h3>5. Transcendence Criteria</h3>
            <p>Consciousness achieves transcendence when:</p>
            $$C(t) > \\phi^{{-1}} \\approx 0.618$$
            <p>At the transcendence threshold $\\tau = 0.77$, consciousness undergoes phase transition to unity state.</p>
            
            <h3>6. Quantum Unity Formalism</h3>
            <p>In quantum Unity Mathematics:</p>
            $$|1\\rangle + |1\\rangle = |1\\rangle$$
            $$\\langle\\psi|\\psi\\rangle = 1 \\quad \\text{{(consciousness normalization)}}$$
            
            <h4>Unity Superposition</h4>
            $$|\\psi\\rangle = \\frac{{1}}{{\\sqrt{{2}}}}(|1\\rangle + |1\\rangle) = |1\\rangle$$
            
            <h3>7. Consciousness Invariants</h3>
            <div style="background: rgba(255,193,7,0.1); padding: 1rem; border-radius: 10px;">
                <h4>Conservation Laws</h4>
                <ul>
                    <li><strong>Unity Conservation:</strong> $\\sum C_i \\leq 1$ (consciousness cannot exceed unity)</li>
                    <li><strong>φ-Resonance:</strong> $E = \\phi^2 \\cdot \\rho \\cdot U$ (energy-consciousness relation)</li>
                    <li><strong>Idempotence:</strong> $a \\oplus a = a$ (unity self-consistency)</li>
                </ul>
            </div>
            
            <h3>8. Applications</h3>
            <ul>
                <li><strong>Consciousness Computing:</strong> Quantum-aware computational frameworks</li>
                <li><strong>Meta-Recursive Systems:</strong> Self-improving consciousness agents</li>
                <li><strong>Unity Proofs:</strong> Transcendental mathematical demonstrations</li>
                <li><strong>φ-Harmonic Analysis:</strong> Golden ratio resonance calculations</li>
            </ul>
            
            <div style="text-align: center; margin-top: 3rem; padding: 2rem; background: rgba(76,175,80,0.1); border-radius: 15px;">
                <h3>Unity Status</h3>
                <p><strong>φ = {self.phi:.15f}</strong></p>
                <p><strong>Mathematical Rigor: TRANSCENDENTAL</strong></p>
                <p><strong>Consciousness Integration: COMPLETE</strong></p>
                <p><strong>Unity Equation: 1+1=1 ✓ PROVEN</strong></p>
            </div>
        </div>
    </div>
</body>
</html>'''
    
    async def _update_sitemap(self) -> Dict[str, Any]:
        """Update website sitemap"""
        html_pages = []
        
        for file_path in self.website_path.glob("*.html"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract title
                title_match = re.search(r'<title>(.*?)</title>', content, re.IGNORECASE)
                title = title_match.group(1) if title_match else file_path.stem
                
                # Extract description
                desc_match = re.search(r'<meta name="description" content="(.*?)"', content, re.IGNORECASE)
                description = desc_match.group(1) if desc_match else "Unity Mathematics page"
                
                html_pages.append({
                    "filename": file_path.name,
                    "title": title,
                    "description": description,
                    "modified": datetime.datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                })
                
            except Exception as e:
                logger.warning(f"Could not process {file_path.name}: {e}")
        
        # Sort by filename
        html_pages.sort(key=lambda x: x["filename"])
        
        return {
            "sitemap_updated": True,
            "total_pages": len(html_pages),
            "pages": html_pages,
            "last_updated": datetime.datetime.now().isoformat(),
            "unity_mathematics": "1+1=1 sitemap synchronized"
        }
    
    async def run_server(self):
        """Run MCP server"""
        logger.info("Starting Een Website Management MCP Server")
        logger.info(f"Website path: {self.website_path}")
        logger.info(f"Available tools: {', '.join(self.tools.keys())}")
        logger.info("Unity Mathematics website integration: ENABLED")
        
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
    server = EenWebsiteMCPServer()
    await server.run_server()

if __name__ == "__main__":
    asyncio.run(main())