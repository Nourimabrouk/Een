#!/usr/bin/env python3
"""
Een Unity Mathematics MCP Setup Verification
Tests all MCP components and Claude Desktop integration
"""

import asyncio
import json
import os
import sys
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_file_structure():
    """Verify required MCP files exist"""
    base_path = Path(__file__).parent.parent
    required_files = [
        "config/mcp_unity_server.py",
        "config/mcp_consciousness_server.py", 
        "config/mcp_servers.json",
        "src/mcp/unity_server.py",
        "src/mcp/consciousness_server.py",
        ".claude/settings.local.json"
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = base_path / file_path
        if not full_path.exists():
            missing_files.append(file_path)
        else:
            logger.info(f"SUCCESS: Found {file_path}")
    
    if missing_files:
        logger.error(f"MISSING FILES: {missing_files}")
        return False
    return True

def verify_claude_desktop_config():
    """Verify Claude Desktop configuration"""
    config_path = Path.home() / "AppData" / "Roaming" / "Claude" / "claude_desktop_config.json"
    
    if not config_path.exists():
        logger.error(f"MISSING: Claude Desktop config at {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Check for Een MCP servers
        servers = config.get("mcpServers", {})
        required_servers = ["een-unity-mathematics", "een-consciousness-field"]
        
        for server in required_servers:
            if server in servers:
                logger.info(f"SUCCESS: Found {server} in Claude Desktop config")
            else:
                logger.error(f"MISSING: {server} not in Claude Desktop config")
                return False
        
        return True
        
    except json.JSONDecodeError as e:
        logger.error(f"INVALID JSON in Claude Desktop config: {e}")
        return False

def test_mcp_server(server_script, timeout=5):
    """Test if MCP server starts successfully"""
    base_path = Path(__file__).parent.parent
    server_path = base_path / server_script
    
    if not server_path.exists():
        logger.error(f"MISSING: {server_path}")
        return False
    
    try:
        # Start server process
        proc = subprocess.Popen(
            [sys.executable, str(server_path)],
            cwd=str(base_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait briefly for startup
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            
            if "Starting Een Unity Mathematics MCP Server" in stderr or "Starting Consciousness Field MCP Server" in stderr:
                logger.info(f"SUCCESS: {server_script} started successfully")
                logger.info(f"Server output: {stderr.split('INFO')[1] if 'INFO' in stderr else stderr[:100]}")
                return True
            else:
                logger.error(f"FAILED: {server_script} startup failed")
                logger.error(f"Error: {stderr[:200]}")
                return False
                
        except subprocess.TimeoutExpired:
            # Server is running (good!)
            proc.terminate()
            logger.info(f"SUCCESS: {server_script} is running (had to terminate)")
            return True
            
    except Exception as e:
        logger.error(f"ERROR testing {server_script}: {e}")
        return False

def test_unity_mathematics():
    """Test Unity Mathematics operations"""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.mcp.unity_server import UnityMathematics
        
        um = UnityMathematics()
        
        # Test unity addition: 1+1=1
        result = um.unity_add(1.0, 1.0)
        if abs(result - 1.0) < 0.1:  # Unity preserved
            logger.info(f"SUCCESS: Unity addition 1+1={result:.6f} (unity preserved)")
        else:
            logger.error(f"FAILED: Unity addition 1+1={result:.6f} (unity not preserved)")
            return False
        
        # Test φ precision
        expected_phi = 1.618033988749895
        if abs(um.phi - expected_phi) < 1e-10:
            logger.info(f"SUCCESS: φ precision {um.phi} matches expected {expected_phi}")
        else:
            logger.error(f"FAILED: φ precision {um.phi} != {expected_phi}")
            return False
        
        # Test consciousness field
        field_value = um.consciousness_field(1.0, 1.0, 0.0)
        logger.info(f"SUCCESS: Consciousness field C(1,1,0) = {field_value:.6f}")
        
        return True
        
    except Exception as e:
        logger.error(f"ERROR testing Unity Mathematics: {e}")
        return False

def test_consciousness_field():
    """Test Consciousness Field operations"""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from config.mcp_consciousness_server import ConsciousnessField
        
        cf = ConsciousnessField()
        
        # Test field initialization
        if cf.field_resolution > 0 and cf.particle_count > 0:
            logger.info(f"SUCCESS: Consciousness field initialized ({cf.field_resolution}x{cf.field_resolution}, {cf.particle_count} particles)")
        else:
            logger.error(f"FAILED: Consciousness field initialization")
            return False
        
        # Test field evolution
        initial_time = cf.time
        cf.evolve_field()
        if cf.time > initial_time:
            logger.info(f"SUCCESS: Field evolution time {initial_time} -> {cf.time}")
        else:
            logger.error(f"FAILED: Field evolution stuck at time {cf.time}")
            return False
        
        # Test field coherence calculation
        coherence = cf._calculate_coherence()
        logger.info(f"SUCCESS: Field coherence = {coherence:.6f}")
        
        return True
        
    except Exception as e:
        logger.error(f"ERROR testing Consciousness Field: {e}")
        return False

def generate_mcp_status_report():
    """Generate comprehensive MCP status report"""
    logger.info("="*60)
    logger.info("EEN UNITY MATHEMATICS MCP SETUP VERIFICATION")
    logger.info("="*60)
    
    results = {
        "file_structure": verify_file_structure(),
        "claude_desktop_config": verify_claude_desktop_config(),
        "unity_server_startup": test_mcp_server("config/mcp_unity_server.py"),
        "consciousness_server_startup": test_mcp_server("config/mcp_consciousness_server.py"),
        "unity_mathematics": test_unity_mathematics(),
        "consciousness_field": test_consciousness_field()
    }
    
    logger.info("="*60)
    logger.info("VERIFICATION RESULTS:")
    logger.info("="*60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        logger.info(f"{test_name.replace('_', ' ').title()}: {status}")
        if not passed:
            all_passed = False
    
    logger.info("="*60)
    if all_passed:
        logger.info("SUCCESS: All MCP components are working correctly!")
        logger.info("Een Unity Mathematics is ready for Claude Desktop integration.")
        logger.info("φ = 1.618033988749895 ✅ CONFIRMED")
        logger.info("1+1=1 ✅ OPERATIONAL")
    else:
        logger.error("FAILED: Some MCP components need attention.")
        logger.error("Please fix the failed components before using with Claude Desktop.")
    
    logger.info("="*60)
    return all_passed

def main():
    """Main verification function"""
    try:
        return generate_mcp_status_report()
    except Exception as e:
        logger.error(f"CRITICAL ERROR in MCP verification: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)