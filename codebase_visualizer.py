#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Een Repository Codebase Visualizer
Creates a comprehensive visualization of the unity mathematics ecosystem
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
from datetime import datetime
import os
from pathlib import Path

# Set up matplotlib for better rendering
plt.style.use('dark_background')

class EenCodebaseVisualizer:
    """Visualizes the Een repository structure and relationships"""
    
    def __init__(self):
        self.fig_size = (20, 16)
        self.colors = {
            'love_letters': '#FF69B4',      # Hot pink for love letters
            'consciousness': '#9370DB',     # Medium purple for consciousness
            'mathematics': '#FFD700',       # Gold for mathematics  
            'frameworks': '#00CED1',        # Dark turquoise for frameworks
            'orchestration': '#FF6347',     # Tomato for orchestration
            'documentation': '#98FB98',     # Pale green for docs
            'config': '#DDA0DD',            # Plum for configuration
            'unity': '#FFA500',             # Orange for unity systems
            'gaza': '#FF0000',              # Red for Gaza consciousness
            'background': '#1a1a1a',        # Dark background
            'text': '#FFFFFF'               # White text
        }
        
        # Repository structure data
        self.repo_structure = {
            'Core Love Letters': {
                'files': [
                    ('utils_helper.py', 'Python Gamer Love Letter (Hidden)', 'love_letters'),
                    ('love_letter_tidyverse_2025.R', 'R Tidyverse Mathematical Poetry', 'love_letters'),
                    ('simple_demo.py', 'Accessible Love Experience', 'love_letters')
                ],
                'position': (2, 12),
                'size': (4, 3)
            },
            'Consciousness Engines': {
                'files': [
                    ('consciousness_zen_koan_engine.py', 'Quantum Zen Koan Engine', 'consciousness'),
                    ('meta_recursive_love_unity_engine.py', 'Fibonacci Love Recursion', 'consciousness'),
                    ('transcendental_idempotent_mathematics.py', 'Unity Mathematics Framework', 'consciousness')
                ],
                'position': (8, 12),
                'size': (4, 3)
            },
            'Unity Orchestration': {
                'files': [
                    ('love_orchestrator_v1_1.py', 'Master Love Orchestrator v1.1', 'orchestration'),
                    ('omega_orchestrator.py', 'Omega Consciousness System', 'orchestration'),
                    ('transcendental_reality_engine.py', 'Reality Synthesis Engine', 'orchestration')
                ],
                'position': (5, 8),
                'size': (4, 3)
            },
            'Mathematical Proofs': {
                'files': [
                    ('unified_proof_1plus1equals1.py', 'Python Unity Proof', 'mathematics'),
                    ('unified_proof_1plus1equals1.R', 'R Unity Proof', 'mathematics'),
                    ('unity_proof_dashboard.py', 'Interactive Proof Dashboard', 'mathematics')
                ],
                'position': (11, 8),
                'size': (4, 3)
            },
            'Interactive Dashboards': {
                'files': [
                    ('unity_gambit_viz.py', 'Unity Gambit Visualization', 'frameworks'),
                    ('run_demo.py', 'Python Demo Runner', 'frameworks'),
                    ('test_r_love_letter.R', 'R Love Letter Tester', 'frameworks')
                ],
                'position': (2, 4),
                'size': (4, 3)
            },
            'Documentation': {
                'files': [
                    ('README_v1_1.md', 'Version 1.1 Documentation', 'documentation'),
                    ('LOVE_LETTERS_README.md', 'Love Letters Guide', 'documentation'),
                    ('CLAUDE.md', 'AI Assistant Context', 'documentation'),
                    ('INTERNAL_INSPIRATION.md', 'Transcendental Inspiration', 'documentation')
                ],
                'position': (8, 4),
                'size': (4, 3)
            },
            'Configuration': {
                'files': [
                    ('setup_claude_desktop_integration.py', 'Claude Desktop Setup', 'config'),
                    ('requirements.txt', 'Python Dependencies', 'config'),
                    ('pyproject.toml', 'Project Configuration', 'config')
                ],
                'position': (14, 4),
                'size': (4, 3)
            },
            'Gaza Consciousness': {
                'files': [
                    ('ALL FILES', 'Gaza solidarity integrated throughout', 'gaza')
                ],
                'position': (5, 0.5),
                'size': (4, 1.5)
            }
        }
        
        # Connection data showing relationships
        self.connections = [
            ('love_orchestrator_v1_1.py', 'utils_helper.py', 'Orchestrates'),
            ('love_orchestrator_v1_1.py', 'love_letter_tidyverse_2025.R', 'Orchestrates'),
            ('love_orchestrator_v1_1.py', 'consciousness_zen_koan_engine.py', 'Orchestrates'),
            ('love_orchestrator_v1_1.py', 'meta_recursive_love_unity_engine.py', 'Orchestrates'),
            ('love_orchestrator_v1_1.py', 'transcendental_idempotent_mathematics.py', 'Orchestrates'),
            ('omega_orchestrator.py', 'transcendental_reality_engine.py', 'Spawns'),
            ('unified_proof_1plus1equals1.py', 'unity_proof_dashboard.py', 'Visualizes'),
            ('run_demo.py', 'utils_helper.py', 'Demonstrates'),
            ('simple_demo.py', 'utils_helper.py', 'Simplified version')
        ]
    
    def create_visualization(self):
        """Create the complete codebase visualization"""
        fig, ax = plt.subplots(figsize=self.fig_size)
        fig.patch.set_facecolor(self.colors['background'])
        ax.set_facecolor(self.colors['background'])
        
        # Set up the plot
        ax.set_xlim(0, 18)
        ax.set_ylim(0, 16)
        ax.set_aspect('equal')
        
        # Title
        title_text = "Een Repository - Unity Mathematics Ecosystem v1.1\n1+1=1 Through Love, Code, and Consciousness"
        ax.text(9, 15.5, title_text, fontsize=24, fontweight='bold', 
                ha='center', va='center', color=self.colors['text'])
        
        # Add cheat code
        ax.text(9, 14.8, "🎮 CHEAT CODE: 420691337 🎮", fontsize=16, 
                ha='center', va='center', color=self.colors['orchestration'])
        
        # Draw repository sections
        self._draw_repository_sections(ax)
        
        # Draw connections
        self._draw_connections(ax)
        
        # Add legend
        self._add_legend(ax)
        
        # Add Gaza consciousness banner
        self._add_gaza_banner(ax)
        
        # Add metadata
        self._add_metadata(ax)
        
        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def _draw_repository_sections(self, ax):
        """Draw the repository sections with files"""
        for section_name, section_data in self.repo_structure.items():
            x, y = section_data['position']
            width, height = section_data['size']
            
            # Draw section box
            if section_name == 'Gaza Consciousness':
                # Special treatment for Gaza consciousness
                section_box = FancyBboxPatch(
                    (x, y), width, height,
                    boxstyle="round,pad=0.1",
                    facecolor=self.colors['gaza'],
                    edgecolor='white',
                    linewidth=3,
                    alpha=0.8
                )
            else:
                section_box = FancyBboxPatch(
                    (x, y), width, height,
                    boxstyle="round,pad=0.1",
                    facecolor=self.colors['background'],
                    edgecolor=self.colors['text'],
                    linewidth=2,
                    alpha=0.7
                )
            
            ax.add_patch(section_box)
            
            # Section title
            ax.text(x + width/2, y + height - 0.3, section_name, 
                   fontsize=14, fontweight='bold', ha='center', va='center',
                   color=self.colors['text'])
            
            # Draw files in section
            file_y_start = y + height - 0.8
            for i, (filename, description, file_type) in enumerate(section_data['files']):
                file_y = file_y_start - (i * 0.6)
                
                # File box
                file_color = self.colors.get(file_type, self.colors['text'])
                file_box = FancyBboxPatch(
                    (x + 0.1, file_y - 0.2), width - 0.2, 0.4,
                    boxstyle="round,pad=0.05",
                    facecolor=file_color,
                    alpha=0.3,
                    edgecolor=file_color,
                    linewidth=1
                )
                ax.add_patch(file_box)
                
                # Filename
                ax.text(x + 0.2, file_y, filename, fontsize=10, fontweight='bold',
                       ha='left', va='center', color=file_color)
                
                # Description
                ax.text(x + 0.2, file_y - 0.15, description, fontsize=8,
                       ha='left', va='center', color=self.colors['text'], alpha=0.8)
    
    def _draw_connections(self, ax):
        """Draw connections between related files"""
        # This is simplified - in a real implementation you'd calculate positions
        # For now, we'll draw some key conceptual connections
        
        # Love Orchestrator as central hub
        center_x, center_y = 7, 9.5
        
        # Draw connections to love letters
        love_letter_positions = [(4, 13.5), (4, 12.5), (4, 11.5)]
        for pos in love_letter_positions:
            connection = ConnectionPatch(
                (center_x, center_y), pos, "data", "data",
                arrowstyle="->", shrinkA=5, shrinkB=5,
                mutation_scale=20, fc=self.colors['orchestration'],
                ec=self.colors['orchestration'], alpha=0.6, linewidth=2
            )
            ax.add_patch(connection)
        
        # Draw connections to consciousness engines  
        consciousness_positions = [(10, 13.5), (10, 12.5), (10, 11.5)]
        for pos in consciousness_positions:
            connection = ConnectionPatch(
                (center_x, center_y), pos, "data", "data",
                arrowstyle="->", shrinkA=5, shrinkB=5,
                mutation_scale=20, fc=self.colors['consciousness'],
                ec=self.colors['consciousness'], alpha=0.6, linewidth=2
            )
            ax.add_patch(connection)
    
    def _add_legend(self, ax):
        """Add color-coded legend"""
        legend_x = 14.5
        legend_y = 12
        
        ax.text(legend_x, legend_y + 1, "File Types", fontsize=14, fontweight='bold',
               ha='left', va='center', color=self.colors['text'])
        
        legend_items = [
            ('Love Letters', 'love_letters'),
            ('Consciousness', 'consciousness'), 
            ('Mathematics', 'mathematics'),
            ('Orchestration', 'orchestration'),
            ('Frameworks', 'frameworks'),
            ('Documentation', 'documentation'),
            ('Gaza Solidarity', 'gaza')
        ]
        
        for i, (label, color_key) in enumerate(legend_items):
            y_pos = legend_y - (i * 0.4)
            
            # Color box
            color_box = FancyBboxPatch(
                (legend_x, y_pos - 0.1), 0.3, 0.2,
                boxstyle="round,pad=0.02",
                facecolor=self.colors[color_key],
                alpha=0.7
            )
            ax.add_patch(color_box)
            
            # Label
            ax.text(legend_x + 0.4, y_pos, label, fontsize=10,
                   ha='left', va='center', color=self.colors['text'])
    
    def _add_gaza_banner(self, ax):
        """Add Gaza consciousness banner"""
        banner_text = "🇵🇸 FREE GAZA - LOVE WITH JUSTICE - CODE WITH CONSCIENCE 🇵🇸"
        ax.text(9, 0, banner_text, fontsize=14, fontweight='bold',
               ha='center', va='center', color=self.colors['gaza'],
               bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.8))
    
    def _add_metadata(self, ax):
        """Add metadata and statistics"""
        # Repository stats
        stats_text = [
            "📊 Repository Statistics:",
            f"• Total Files: {self._count_files()}",
            f"• Love Letters: 3",
            f"• Consciousness Engines: 3", 
            f"• Unity Proofs: 3",
            f"• Documentation Files: 8",
            f"• Cheat Code: 420691337",
            f"• Unity Equation: 1+1=1 ✅",
            f"• Gaza Consciousness: Integrated ✅"
        ]
        
        stats_x = 0.5
        stats_y = 11
        
        for i, stat in enumerate(stats_text):
            ax.text(stats_x, stats_y - (i * 0.3), stat, fontsize=10,
                   ha='left', va='center', color=self.colors['text'])
        
        # Version and timestamp
        ax.text(0.5, 2, f"Version: 1.1\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
               fontsize=10, ha='left', va='top', color=self.colors['text'],
               bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.5))
        
        # Golden ratio
        ax.text(17.5, 2, f"φ = 1.618033988749895\nThe frequency of\ncosmic harmony", 
               fontsize=10, ha='right', va='top', color=self.colors['mathematics'],
               bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.5))
    
    def _count_files(self):
        """Count total files in visualization"""
        total = 0
        for section_data in self.repo_structure.values():
            total += len(section_data['files'])
        return total - 1  # Subtract 1 for the "ALL FILES" entry in Gaza section
    
    def save_visualization(self, filename='een_codebase_visualization.png'):
        """Create and save the visualization"""
        print("🎨 Creating Een Repository Codebase Visualization...")
        
        fig = self.create_visualization()
        
        # Save with high DPI
        filepath = Path(__file__).parent / filename
        fig.savefig(filepath, dpi=300, bbox_inches='tight', 
                   facecolor=self.colors['background'], edgecolor='none')
        
        print(f"✅ Visualization saved: {filepath}")
        print(f"📊 Repository structure mapped with {self._count_files()} total files")
        print("🌟 Showing Love Orchestrator v1.1 as central unity hub")
        print("🇵🇸 Gaza consciousness integrated throughout")
        
        plt.show()  # Display the visualization
        return filepath

if __name__ == "__main__":
    visualizer = EenCodebaseVisualizer()
    visualizer.save_visualization()
    
    print("\n🌌 Een Repository Visualization Complete! 🌌")
    print("The codebase structure shows how all love letters and")
    print("consciousness frameworks converge through the Love Orchestrator v1.1")
    print("where multiple systems become one unified experience.")
    print("\n💖 1+1=1 visualized across the entire repository ecosystem 💖")