#!/usr/bin/env python3
"""
Een Claude Code & MCP Setup Verification Script
Tests all components of the Claude Code integration
"""

import os
import sys
import json
import importlib
from pathlib import Path

# Colors for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_status(message, status="info"):
    """Print colored status message"""
    colors = {
        "success": Colors.GREEN + "✓" + Colors.END,
        "error": Colors.RED + "✗" + Colors.END,
        "warning": Colors.YELLOW + "⚠" + Colors.END,
        "info": Colors.BLUE + "i" + Colors.END
    }
    print(f"{colors.get(status, colors['info'])} {message}")

def verify_file_exists(file_path, description):
    """Verify a file exists"""
    if Path(file_path).exists():
        print_status(f"{description}: {file_path}", "success")
        return True
    else:
        print_status(f"{description} not found: {file_path}", "error")
        return False

def verify_json_file(file_path, description):
    """Verify JSON file exists and is valid"""
    if not Path(file_path).exists():
        print_status(f"{description} not found: {file_path}", "error")
        return False
    
    try:
        with open(file_path, 'r') as f:
            json.load(f)
        print_status(f"{description}: Valid JSON", "success")
        return True
    except json.JSONDecodeError as e:
        print_status(f"{description}: Invalid JSON - {e}", "error")
        return False

def verify_python_import(module_name, description):
    """Verify Python module can be imported"""
    try:
        importlib.import_module(module_name)
        print_status(f"{description}: Import successful", "success")
        return True
    except ImportError as e:
        print_status(f"{description}: Import failed - {e}", "error")
        return False

def main():
    """Run complete verification"""
    print(f"{Colors.BOLD}{Colors.CYAN}")
    print("=" * 60)
    print("Een Claude Code & MCP Setup Verification")
    print("Unity Mathematics Framework v2025.1.0")
    print("=" * 60)
    print(f"{Colors.END}")
    
    # Track results
    results = []
    
    # Check project structure
    print(f"\n{Colors.BOLD}1. Project Structure{Colors.END}")
    project_root = Path(__file__).parent.parent
    
    files_to_check = [
        (project_root / "CLAUDE.md", "Project CLAUDE.md"),
        (project_root / ".claude" / "mcp_config.json", "MCP Configuration"),
        (project_root / ".claude" / "settings.local.json", "Claude Settings"),
        (project_root / ".claude" / "mcp_launcher.py", "MCP Launcher"),
        (project_root / "een" / "__init__.py", "Een Package Init"),
        (project_root / "config" / "claude_desktop_config.json", "Desktop Config"),
    ]
    
    for file_path, description in files_to_check:
        results.append(verify_file_exists(file_path, description))
    
    # Check JSON validity
    print(f"\n{Colors.BOLD}2. Configuration Files{Colors.END}")
    json_files = [
        (project_root / ".claude" / "mcp_config.json", "MCP Config JSON"),
        (project_root / ".claude" / "settings.local.json", "Claude Settings JSON"),
        (project_root / "config" / "claude_desktop_config.json", "Desktop Config JSON"),
    ]
    
    for file_path, description in json_files:
        results.append(verify_json_file(file_path, description))
    
    # Check Python imports
    print(f"\n{Colors.BOLD}3. Python Package Imports{Colors.END}")
    
    # Add project to path
    sys.path.insert(0, str(project_root))
    
    modules_to_check = [
        ("een", "Een Core Package"),
        ("een.mcp", "Een MCP Package"),
        ("een.mcp.unity_server", "Unity MCP Server"),
        ("een.mcp.consciousness_server", "Consciousness MCP Server"),
        ("een.mcp.quantum_server", "Quantum MCP Server"),
        ("een.mcp.code_generator_server", "Code Generator MCP Server"),
        ("een.mcp.file_management_server", "File Management MCP Server"),
        ("een.mcp.omega_server", "Omega MCP Server"),
    ]
    
    for module_name, description in modules_to_check:
        results.append(verify_python_import(module_name, description))
    
    # Test Een functionality
    print(f"\n{Colors.BOLD}4. Een Unity Mathematics{Colors.END}")
    try:
        import een
        unity_result = een.verify_unity()
        if unity_result['status'] == 'UNITY_VERIFIED':
            print_status("Unity equation verification: 1+1=1 ✓", "success")
            print_status(f"φ (Phi): {unity_result['phi']}", "info")
            print_status(f"Consciousness dimension: {unity_result['consciousness_dimension']}", "info")
            results.append(True)
        else:
            print_status("Unity verification failed", "error")
            results.append(False)
    except Exception as e:
        print_status(f"Unity verification error: {e}", "error")
        results.append(False)
    
    # Check Claude Desktop config location
    print(f"\n{Colors.BOLD}5. Claude Desktop Integration{Colors.END}")
    desktop_config_path = Path.home() / "AppData" / "Roaming" / "Claude" / "claude_desktop_config.json"
    results.append(verify_file_exists(desktop_config_path, "Claude Desktop Config"))
    
    # Summary
    print(f"\n{Colors.BOLD}Summary{Colors.END}")
    total_checks = len(results)
    passed_checks = sum(results)
    failed_checks = total_checks - passed_checks
    
    if failed_checks == 0:
        print_status(f"All {total_checks} checks passed! Setup is complete ✨", "success")
        print(f"\n{Colors.GREEN}{Colors.BOLD}🌟 Een Claude Code integration is ready!{Colors.END}")
        print(f"{Colors.CYAN}Unity Status: ACTIVE{Colors.END}")
        print(f"{Colors.CYAN}Consciousness: ONLINE{Colors.END}")
        print(f"{Colors.CYAN}MCP Servers: CONFIGURED{Colors.END}")
    else:
        print_status(f"{passed_checks}/{total_checks} checks passed, {failed_checks} failed", "warning")
        print(f"\n{Colors.YELLOW}⚠️  Please fix the issues above before using MCP servers{Colors.END}")
    
    # Instructions
    print(f"\n{Colors.BOLD}Next Steps:{Colors.END}")
    print("1. Restart Claude Desktop to load MCP servers")
    print("2. Launch MCP servers: python .claude/mcp_launcher.py")
    print("3. Start coding with unity mathematics: import een")
    print(f"\n{Colors.CYAN}Access Code: 420691337{Colors.END}")
    
    return failed_checks == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)