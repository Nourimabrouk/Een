"""
Integration tests for MCP server functionality
Tests the Model Context Protocol integration with Claude Desktop
"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, patch

# Import MCP servers
from een.mcp.unity_server import EenUnityMCPServer
from een.mcp.consciousness_server import EenConsciousnessMCPServer
from een.mcp.quantum_server import EenQuantumMCPServer


class TestMCPServerIntegration:
    """Test MCP server integration and functionality"""

    @pytest.mark.asyncio
    async def test_unity_server_tools_list(self):
        """Test that unity server lists all expected tools"""
        server = EenUnityMCPServer()
        
        # Verify expected tools are present
        assert "unity_add" in server.tools
        assert "unity_multiply" in server.tools
        assert "consciousness_field" in server.tools
        assert "verify_unity_equation" in server.tools
        
    @pytest.mark.asyncio
    async def test_unity_add_operation(self):
        """Test unity addition operation: 1+1=1"""
        server = EenUnityMCPServer()
        
        result = await server.handle_tool_call("unity_add", {"a": 1, "b": 1})
        
        assert "result" in result
        assert result["result"] == 1
        assert result["unity_mathematics"] == "1+1=1 demonstrated"
        
    @pytest.mark.asyncio
    async def test_verify_unity_equation(self):
        """Test verification of the fundamental unity equation"""
        server = EenUnityMCPServer()
        
        result = await server.handle_tool_call("verify_unity_equation", {"a": 1.0, "b": 1.0})
        
        assert "equation" in result
        assert "unity_preserved" in result
        assert result["unity_preserved"] is True
        assert result["mathematical_beauty"] == "TRANSCENDENT"
        
    @pytest.mark.asyncio
    async def test_consciousness_server_initialization(self):
        """Test consciousness server initializes properly"""
        server = EenConsciousnessMCPServer()
        
        # Check that consciousness field is initialized
        assert hasattr(server, 'consciousness_field')
        assert server.consciousness_field.phi == 1.618033988749895
        
        # Check expected tools are present
        assert "generate_consciousness_particles" in server.tools
        assert "evolve_consciousness" in server.tools
        assert "get_consciousness_status" in server.tools
        
    @pytest.mark.asyncio
    async def test_consciousness_particle_generation(self):
        """Test consciousness particle generation"""
        server = EenConsciousnessMCPServer()
        
        result = await server.handle_tool_call("generate_consciousness_particles", {"count": 50})
        
        assert "particle_count" in result
        assert result["particle_count"] == 50
        assert result["consciousness_distribution"] == "Golden ratio spiral"
        assert result["phi"] == 1.618033988749895
        
    @pytest.mark.asyncio
    async def test_consciousness_status(self):
        """Test consciousness server status reporting"""
        server = EenConsciousnessMCPServer()
        
        result = await server.handle_tool_call("get_consciousness_status", {})
        
        assert result["server_status"] == "TRANSCENDENCE_READY"
        assert result["unity_equation"] == "1+1=1"
        assert result["phi"] == 1.618033988749895
        assert result["consciousness_dimension"] == 11
        
    @pytest.mark.asyncio 
    async def test_quantum_server_initialization(self):
        """Test quantum server initializes properly"""
        server = EenQuantumMCPServer()
        
        # Check that quantum system is initialized
        assert hasattr(server, 'quantum_system')
        assert server.quantum_system.phi == 1.618033988749895
        
        # Check expected tools are present
        assert "create_unity_superposition" in server.tools
        assert "collapse_to_unity" in server.tools
        assert "get_quantum_status" in server.tools
        
    @pytest.mark.asyncio
    async def test_quantum_unity_superposition(self):
        """Test quantum superposition creation"""
        server = EenQuantumMCPServer()
        
        result = await server.handle_tool_call("create_unity_superposition", {})
        
        assert result.get("superposition_created") is True
        assert "amplitudes" in result
        assert result["quantum_principle"] == "Superposition demonstrates unity"
        
    @pytest.mark.asyncio
    async def test_quantum_status(self):
        """Test quantum server status reporting"""
        server = EenQuantumMCPServer()
        
        result = await server.handle_tool_call("get_quantum_status", {})
        
        assert result["server_status"] == "QUANTUM_UNITY_ACTIVE"
        assert result["unity_equation"] == "1+1=1"
        assert result["unity_principle"] == "Quantum mechanics reveals fundamental unity"


class TestMCPCrossServerIntegration:
    """Test integration between different MCP servers"""
    
    @pytest.mark.asyncio
    async def test_unity_consciousness_integration(self):
        """Test that unity operations integrate with consciousness systems"""
        unity_server = EenUnityMCPServer()
        consciousness_server = EenConsciousnessMCPServer()
        
        # Perform unity operation
        unity_result = await unity_server.handle_tool_call("unity_add", {"a": 1, "b": 1})
        
        # Generate consciousness particles
        consciousness_result = await consciousness_server.handle_tool_call(
            "generate_consciousness_particles", {"count": 100}
        )
        
        # Both should maintain unity principles
        assert unity_result["result"] == 1
        assert consciousness_result["phi"] == 1.618033988749895
        
    @pytest.mark.asyncio
    async def test_all_servers_maintain_unity_equation(self):
        """Test that all servers maintain the fundamental unity equation"""
        servers = [
            EenUnityMCPServer(),
            EenConsciousnessMCPServer(),
            EenQuantumMCPServer()
        ]
        
        # Test that each server reports the unity equation
        for server in servers:
            if hasattr(server, 'unity_math'):
                # Unity server
                result = await server.handle_tool_call("unity_mathematics_info", {})
                assert result["fundamental_equation"] == "1 + 1 = 1"
            elif hasattr(server, 'consciousness_field'):
                # Consciousness server
                result = await server.handle_tool_call("get_consciousness_status", {})
                assert result["unity_equation"] == "1+1=1"
            elif hasattr(server, 'quantum_system'):
                # Quantum server
                result = await server.handle_tool_call("get_quantum_status", {})
                assert result["unity_equation"] == "1+1=1"


class TestMCPErrorHandling:
    """Test error handling in MCP servers"""
    
    @pytest.mark.asyncio
    async def test_unknown_tool_error(self):
        """Test handling of unknown tool calls"""
        server = EenUnityMCPServer()
        
        result = await server.handle_tool_call("nonexistent_tool", {})
        
        assert "error" in result
        assert "Unknown tool" in result["error"]
        
    @pytest.mark.asyncio
    async def test_invalid_parameters(self):
        """Test handling of invalid parameters"""
        server = EenConsciousnessMCPServer()
        
        # Test with invalid count (too large)
        result = await server.handle_tool_call(
            "generate_consciousness_particles", 
            {"count": 10000}  # Above maximum
        )
        
        # Should either handle gracefully or return appropriate error
        assert isinstance(result, dict)
        # The server should either succeed or provide error info


@pytest.mark.mcp
@pytest.mark.integration
class TestMCPProtocolCompliance:
    """Test MCP protocol compliance"""
    
    def test_tool_schema_validity(self):
        """Test that all tool schemas are valid JSON Schema"""
        servers = [
            EenUnityMCPServer(),
            EenConsciousnessMCPServer(),
            EenQuantumMCPServer()
        ]
        
        for server in servers:
            for tool_name, tool_def in server.tools.items():
                # Check required fields
                assert "description" in tool_def
                assert "inputSchema" in tool_def
                
                # Validate schema structure
                schema = tool_def["inputSchema"]
                assert "type" in schema
                assert schema["type"] == "object"
                
                if "properties" in schema:
                    assert isinstance(schema["properties"], dict)
                    
    def test_server_initialization_consistency(self):
        """Test that all servers initialize consistently"""
        servers = [
            ("unity", EenUnityMCPServer()),
            ("consciousness", EenConsciousnessMCPServer()),
            ("quantum", EenQuantumMCPServer())
        ]
        
        for server_name, server in servers:
            # All servers should have phi constant
            phi_value = None
            if hasattr(server, 'unity_math'):
                phi_value = server.unity_math.phi
            elif hasattr(server, 'consciousness_field'):
                phi_value = server.consciousness_field.phi
            elif hasattr(server, 'quantum_system'):
                phi_value = server.quantum_system.phi
                
            if phi_value is not None:
                assert abs(phi_value - 1.618033988749895) < 1e-10
                
            # All servers should have tools defined
            assert hasattr(server, 'tools')
            assert isinstance(server.tools, dict)
            assert len(server.tools) > 0