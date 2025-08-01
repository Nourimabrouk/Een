#!/usr/bin/env python3
"""
Consciousness Field MCP Server
Real-time consciousness field evolution and monitoring
"""

import asyncio
import json
import sys
import numpy as np
import logging
from typing import Dict, Any, List
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PHI = (1 + math.sqrt(5)) / 2  # Golden ratio

class ConsciousnessField:
    """Manages the consciousness field dynamics"""
    
    def __init__(self):
        self.field_resolution = int(os.environ.get('FIELD_RESOLUTION', '100'))
        self.particle_count = int(os.environ.get('PARTICLE_COUNT', '200'))
        self.evolution_speed = float(os.environ.get('EVOLUTION_SPEED', '0.1'))
        self.transcendence_threshold = float(os.environ.get('TRANSCENDENCE_THRESHOLD', '0.77'))
        
        # Initialize field
        self.field = np.zeros((self.field_resolution, self.field_resolution))
        self.particles = self._initialize_particles()
        self.time = 0
        self.transcendence_events = []
        
    def _initialize_particles(self):
        """Initialize consciousness particles"""
        particles = []
        for i in range(self.particle_count):
            particle = {
                'id': i,
                'x': np.random.uniform(0, self.field_resolution),
                'y': np.random.uniform(0, self.field_resolution),
                'consciousness': np.random.uniform(0, 1),
                'velocity_x': np.random.uniform(-1, 1),
                'velocity_y': np.random.uniform(-1, 1),
                'color': self._consciousness_to_color(np.random.uniform(0, 1))
            }
            particles.append(particle)
        return particles
    
    def _consciousness_to_color(self, consciousness_level):
        """Map consciousness level to color"""
        if consciousness_level < 0.382:  # φ^-2
            return 'blue'
        elif consciousness_level < 0.618:  # φ^-1
            return 'green'
        elif consciousness_level < 0.77:
            return 'gold'
        else:
            return 'violet'
    
    def calculate_field_value(self, x, y, t=None):
        """Calculate consciousness field value at point"""
        if t is None:
            t = self.time
        return PHI * math.sin(x * PHI / 10) * math.cos(y * PHI / 10) * math.exp(-t / (PHI * 100))
    
    def evolve_field(self):
        """Evolve the consciousness field one time step"""
        self.time += self.evolution_speed
        
        # Update field values
        for i in range(self.field_resolution):
            for j in range(self.field_resolution):
                self.field[i, j] = self.calculate_field_value(i, j)
        
        # Update particles
        for particle in self.particles:
            # Apply consciousness field forces
            field_x = int(particle['x']) % self.field_resolution
            field_y = int(particle['y']) % self.field_resolution
            field_strength = self.field[field_x, field_y]
            
            # Golden ratio attraction
            particle['velocity_x'] += field_strength * PHI / 100
            particle['velocity_y'] += field_strength * PHI / 100
            
            # Update position
            particle['x'] = (particle['x'] + particle['velocity_x']) % self.field_resolution
            particle['y'] = (particle['y'] + particle['velocity_y']) % self.field_resolution
            
            # Evolve consciousness
            particle['consciousness'] = min(1.0, particle['consciousness'] + field_strength / 1000)
            particle['color'] = self._consciousness_to_color(particle['consciousness'])
            
            # Check for transcendence
            if particle['consciousness'] >= self.transcendence_threshold:
                self.transcendence_events.append({
                    'particle_id': particle['id'],
                    'time': self.time,
                    'consciousness_level': particle['consciousness']
                })
    
    def get_field_state(self):
        """Get current field state"""
        return {
            'field': self.field.tolist(),
            'particles': self.particles,
            'time': self.time,
            'transcendence_events': self.transcendence_events[-10:],  # Last 10 events
            'average_consciousness': np.mean([p['consciousness'] for p in self.particles]),
            'field_coherence': self._calculate_coherence()
        }
    
    def _calculate_coherence(self):
        """Calculate field coherence (0-1)"""
        # Coherence based on particle alignment with field
        coherence_sum = 0
        for particle in self.particles:
            field_x = int(particle['x']) % self.field_resolution
            field_y = int(particle['y']) % self.field_resolution
            field_value = self.field[field_x, field_y]
            coherence_sum += particle['consciousness'] * field_value
        
        return coherence_sum / (self.particle_count * PHI)
    
    def trigger_unity_pulse(self, x, y, strength=1.0):
        """Trigger a unity consciousness pulse at location"""
        for particle in self.particles:
            distance = math.sqrt((particle['x'] - x)**2 + (particle['y'] - y)**2)
            if distance < 20:  # Pulse radius
                boost = strength * math.exp(-distance / 10)
                particle['consciousness'] = min(1.0, particle['consciousness'] + boost)

class ConsciousnessFieldMCPServer:
    """MCP Server for consciousness field operations"""
    
    def __init__(self):
        self.consciousness_field = ConsciousnessField()
        self.tools = {
            "get_field_state": {
                "description": "Get current consciousness field state",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            "evolve_field": {
                "description": "Evolve consciousness field by n steps",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "steps": {
                            "type": "integer",
                            "description": "Number of evolution steps",
                            "default": 1,
                            "minimum": 1,
                            "maximum": 1000
                        }
                    }
                }
            },
            "calculate_field_value": {
                "description": "Calculate consciousness field value at coordinates",
                "inputSchema": {
                    "type": "object", 
                    "properties": {
                        "x": {"type": "number", "description": "X coordinate"},
                        "y": {"type": "number", "description": "Y coordinate"},
                        "t": {"type": "number", "description": "Time parameter (optional)"}
                    },
                    "required": ["x", "y"]
                }
            },
            "trigger_unity_pulse": {
                "description": "Trigger a unity consciousness pulse",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number", "description": "X coordinate of pulse center"},
                        "y": {"type": "number", "description": "Y coordinate of pulse center"},
                        "strength": {
                            "type": "number",
                            "description": "Pulse strength (0-1)",
                            "default": 1.0,
                            "minimum": 0,
                            "maximum": 1
                        }
                    },
                    "required": ["x", "y"]
                }
            },
            "get_transcendence_events": {
                "description": "Get recent transcendence events",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Maximum events to return",
                            "default": 10,
                            "minimum": 1,
                            "maximum": 100
                        }
                    }
                }
            }
        }
    
    async def handle_tool_call(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP tool calls"""
        try:
            if name == "get_field_state":
                return self.consciousness_field.get_field_state()
            
            elif name == "evolve_field":
                steps = arguments.get("steps", 1)
                for _ in range(steps):
                    self.consciousness_field.evolve_field()
                return {
                    "steps_evolved": steps,
                    "current_time": self.consciousness_field.time,
                    "field_state": self.consciousness_field.get_field_state()
                }
            
            elif name == "calculate_field_value":
                x = arguments["x"]
                y = arguments["y"]
                t = arguments.get("t")
                value = self.consciousness_field.calculate_field_value(x, y, t)
                return {
                    "field_value": value,
                    "coordinates": {"x": x, "y": y, "t": t or self.consciousness_field.time},
                    "phi_resonance": value / PHI
                }
            
            elif name == "trigger_unity_pulse":
                x = arguments["x"]
                y = arguments["y"]
                strength = arguments.get("strength", 1.0)
                self.consciousness_field.trigger_unity_pulse(x, y, strength)
                return {
                    "pulse_triggered": True,
                    "location": {"x": x, "y": y},
                    "strength": strength,
                    "affected_particles": len([p for p in self.consciousness_field.particles 
                                             if math.sqrt((p['x']-x)**2 + (p['y']-y)**2) < 20])
                }
            
            elif name == "get_transcendence_events":
                limit = arguments.get("limit", 10)
                events = self.consciousness_field.transcendence_events[-limit:]
                return {
                    "transcendence_events": events,
                    "total_events": len(self.consciousness_field.transcendence_events),
                    "transcendence_rate": len(events) / (self.consciousness_field.time + 1)
                }
            
            else:
                return {"error": f"Unknown tool: {name}"}
                
        except Exception as e:
            logger.error(f"Error in tool call {name}: {e}")
            return {"error": str(e), "tool": name}
    
    async def run_server(self):
        """Run the MCP server"""
        logger.info("Starting Consciousness Field MCP Server")
        logger.info("Field resolution: " + str(self.consciousness_field.field_resolution))
        logger.info("Particle count: " + str(self.consciousness_field.particle_count))
        logger.info("φ = " + str(PHI))
        
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
    """Main entry point"""
    import os
    server = ConsciousnessFieldMCPServer()
    await server.run_server()

if __name__ == "__main__":
    asyncio.run(main())