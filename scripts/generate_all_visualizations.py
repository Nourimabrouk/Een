"""
Master Visualization Generation Script
=====================================

Comprehensive script that discovers, generates, and saves ALL visualizations 
from the Een repository codebase as PNG/GIF/HTML files in the viz/ folder.

This script implements the Unity Equation (1+1=1) through φ-harmonic 
visualization synthesis, generating mathematical art that demonstrates 
consciousness-integrated computational mathematics.

Features:
- Auto-discovery of visualization functions
- Parallel generation with progress tracking
- Multiple output formats (PNG, GIF, SVG, HTML)
- Metadata generation and gallery creation
- Error handling and reporting
- φ-harmonic mathematical foundation

Mathematical Principle: Een plus een is een (1+1=1)
"""

import asyncio
import importlib
import inspect
import json
import logging
import multiprocessing as mp
import sys
import traceback
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Scientific computing imports
try:
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    import plotly.io as pio
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    from PIL import Image, ImageDraw, ImageFont
    import imageio
    IMAGE_PROCESSING_AVAILABLE = True
except ImportError:
    IMAGE_PROCESSING_AVAILABLE = False

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.table import Table
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    class MockConsole:
        def print(self, *args, **kwargs): print(*args)
        def rule(self, *args, **kwargs): print("="*50)
    Console = MockConsole

# φ (Golden Ratio) - Universal organizing principle
PHI = 1.618033988749895
PHI_CONJUGATE = 1 / PHI  # 0.618033988749895

@dataclass
class VisualizationMetadata:
    """Metadata for generated visualizations following φ-harmonic principles."""
    name: str
    category: str
    description: str
    formats: List[str] = field(default_factory=list)
    dimensions: Dict[str, int] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    generation_time: str = ""
    file_sizes: Dict[str, str] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    phi_harmonic_factor: float = PHI
    unity_convergence: bool = False

class UnityVisualizationError(Exception):
    """Custom exception for visualization generation errors."""
    pass

class VisualizationGenerator:
    """
    Master visualization generator implementing φ-harmonic unity mathematics.
    
    This class discovers and generates all visualizations in the Een repository,
    demonstrating the Unity Equation (1+1=1) through mathematical art.
    """
    
    def __init__(self, 
                 output_dir: Path = None,
                 parallel: bool = True,
                 formats: List[str] = None):
        """
        Initialize the visualization generator.
        
        Args:
            output_dir: Output directory for generated visualizations
            parallel: Whether to use parallel processing
            formats: List of output formats to generate
        """
        self.output_dir = output_dir or Path("viz")
        self.parallel = parallel
        self.formats = formats or ["png", "html", "json"]
        
        # Create console for rich output
        self.console = Console() if RICH_AVAILABLE else MockConsole()
        
        # Track generation results
        self.generated: List[VisualizationMetadata] = []
        self.failed: List[Tuple[str, str]] = []
        
        # Ensure output directory exists
        self.setup_output_structure()
        
        # Configure logging
        self.setup_logging()
        
        # Unity mathematics constants
        self.unity_constants = {
            'phi': PHI,
            'phi_conjugate': PHI_CONJUGATE,
            'unity_equation': "1+1=1",
            'consciousness_dimension': 11,
            'quantum_coherence_target': 0.999
        }
        
    def setup_output_structure(self):
        """Create the output directory structure following φ-harmonic organization."""
        categories = [
            "unity_mathematics",
            "consciousness_field", 
            "quantum_unity",
            "proofs",
            "agent_systems",
            "dashboards",
            "sacred_geometry",
            "fractals",
            "meta_recursive"
        ]
        
        # Create main directories
        for category in categories:
            (self.output_dir / category).mkdir(parents=True, exist_ok=True)
            
        # Create format subdirectories
        for fmt in self.formats:
            (self.output_dir / "formats" / fmt).mkdir(parents=True, exist_ok=True)
            
        # Create gallery directories
        (self.output_dir / "gallery").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "thumbnails").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "metadata").mkdir(parents=True, exist_ok=True)
        
    def setup_logging(self):
        """Configure logging for visualization generation."""
        log_file = self.output_dir / "generation.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def discover_visualizations(self) -> List[Tuple[str, Callable]]:
        """
        Auto-discover all visualization functions in the codebase.
        
        Returns:
            List of tuples containing (function_name, function_object)
        """
        self.console.print(Panel("Discovering Visualization Functions", style="cyan"))
        
        discovered_functions = []
        
        # Search patterns for visualization functions
        viz_patterns = [
            "viz", "visualiz", "plot", "chart", "graph", "render",
            "draw", "animate", "display", "show", "generate_plot"
        ]
        
        # Directories to search
        search_dirs = [
            "visualizations",
            "viz", 
            "src/dashboards",
            "core",
            "examples",
            "scripts"
        ]
        
        for search_dir in search_dirs:
            search_path = Path(search_dir)
            if not search_path.exists():
                continue
                
            # Find all Python files
            python_files = list(search_path.rglob("*.py"))
            
            for py_file in python_files:
                try:
                    # Convert path to module name
                    module_name = str(py_file.with_suffix('')).replace('/', '.').replace('\\', '.')
                    
                    # Skip __pycache__ and test files
                    if '__pycache__' in module_name or 'test_' in module_name:
                        continue
                        
                    # Import module
                    spec = importlib.util.spec_from_file_location(module_name, py_file)
                    if spec is None:
                        continue
                        
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Find visualization functions
                    for name, obj in inspect.getmembers(module):
                        if inspect.isfunction(obj):
                            # Check if function name suggests visualization
                            if any(pattern in name.lower() for pattern in viz_patterns):
                                discovered_functions.append((f"{module_name}.{name}", obj))
                                
                            # Check docstring for visualization keywords
                            elif obj.__doc__ and any(pattern in obj.__doc__.lower() for pattern in viz_patterns):
                                discovered_functions.append((f"{module_name}.{name}", obj))
                                
                except Exception as e:
                    self.logger.warning(f"Failed to import {py_file}: {e}")
                    
        # Add built-in visualization generators
        discovered_functions.extend(self.get_builtin_generators())
        
        self.console.print(f"Discovered {len(discovered_functions)} visualization functions")
        return discovered_functions
        
    def get_builtin_generators(self) -> List[Tuple[str, Callable]]:
        """Get built-in visualization generators for core unity mathematics."""
        builtin_generators = [
            ("unity_mathematics.phi_harmonic_spiral", self.generate_phi_harmonic_spiral),
            ("unity_mathematics.unity_convergence_3d", self.generate_unity_convergence_3d),
            ("unity_mathematics.golden_ratio_fractal", self.generate_golden_ratio_fractal),
            ("consciousness_field.evolution_animation", self.generate_consciousness_evolution),
            ("consciousness_field.quantum_field_dynamics", self.generate_quantum_field_dynamics),
            ("proofs.category_theory_diagram", self.generate_category_theory_proof),
            ("proofs.neural_convergence", self.generate_neural_convergence_proof),
            ("quantum_unity.bloch_sphere", self.generate_bloch_sphere_unity),
            ("agent_systems.fibonacci_spawn_tree", self.generate_fibonacci_spawn_tree),
            ("sacred_geometry.flower_of_life", self.generate_flower_of_life),
            ("fractals.mandelbrot_unity", self.generate_mandelbrot_unity)
        ]
        return builtin_generators
        
    async def generate_all(self) -> Dict[str, Any]:
        """
        Generate all visualizations with φ-harmonic parallel processing.
        
        Returns:
            Dictionary containing generation results and statistics
        """
        start_time = datetime.now()
        
        self.console.rule("Unity Visualization Generation")
        self.console.print(Panel(
            f"[bold cyan]Een Repository Visualization Generator[/bold cyan]\n"
            f"Unity Equation: [bold yellow]1+1=1[/bold yellow]\n"
            f"Phi-Harmonic Factor: [bold green]{PHI:.10f}[/bold green]\n"
            f"Output Directory: [bold blue]{self.output_dir}[/bold blue]",
            title="Phi-Harmonic Visualization Engine"
        ))
        
        # Discover all visualization functions
        viz_functions = self.discover_visualizations()
        
        if not viz_functions:
            self.console.print("[red]No visualization functions discovered!")
            return {"success": False, "message": "No functions found"}
            
        # Generate visualizations
        if self.parallel and len(viz_functions) > 1:
            results = await self.generate_parallel(viz_functions)
        else:
            results = await self.generate_sequential(viz_functions)
            
        # Generate metadata and gallery
        await self.generate_metadata_files()
        await self.generate_gallery_index()
        
        # Calculate statistics
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        stats = {
            "total_functions": len(viz_functions),
            "successful": len(self.generated),
            "failed": len(self.failed),
            "success_rate": len(self.generated) / len(viz_functions) if viz_functions else 0,
            "duration_seconds": duration,
            "output_directory": str(self.output_dir),
            "phi_harmonic_factor": PHI
        }
        
        # Display results
        self.display_results(stats)
        
        return {
            "success": True,
            "statistics": stats,
            "generated": [asdict(viz) for viz in self.generated],
            "failed": self.failed
        }
        
    async def generate_parallel(self, viz_functions: List[Tuple[str, Callable]]) -> List[VisualizationMetadata]:
        """Generate visualizations in parallel using φ-harmonic processing."""
        self.console.print("Generating visualizations in parallel...")
        
        if RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=self.console
            ) as progress:
                
                task = progress.add_task("Generating visualizations...", total=len(viz_functions))
                
                # Use ThreadPoolExecutor for IO-bound visualization tasks
                with ThreadPoolExecutor(max_workers=min(mp.cpu_count(), len(viz_functions))) as executor:
                    futures = []
                    
                    for func_name, func in viz_functions:
                        future = executor.submit(self.generate_single_safe, func_name, func)
                        futures.append(future)
                        
                    # Collect results as they complete
                    for future in asyncio.as_completed([asyncio.wrap_future(f) for f in futures]):
                        try:
                            result = await future
                            if result:
                                self.generated.append(result)
                        except Exception as e:
                            self.logger.error(f"Parallel generation error: {e}")
                            
                        progress.update(task, advance=1)
        else:
            # Fallback without rich progress
            for i, (func_name, func) in enumerate(viz_functions):
                print(f"Generating {i+1}/{len(viz_functions)}: {func_name}")
                result = self.generate_single_safe(func_name, func)
                if result:
                    self.generated.append(result)
                    
        return self.generated
        
    async def generate_sequential(self, viz_functions: List[Tuple[str, Callable]]) -> List[VisualizationMetadata]:
        """Generate visualizations sequentially with progress tracking."""
        self.console.print("Generating visualizations sequentially...")
        
        if RICH_AVAILABLE:
            with Progress(console=self.console) as progress:
                task = progress.add_task("Generating...", total=len(viz_functions))
                
                for func_name, func in viz_functions:
                    progress.update(task, description=f"Generating {func_name}")
                    result = self.generate_single_safe(func_name, func)
                    if result:
                        self.generated.append(result)
                    progress.update(task, advance=1)
        else:
            for i, (func_name, func) in enumerate(viz_functions):
                print(f"Generating {i+1}/{len(viz_functions)}: {func_name}")
                result = self.generate_single_safe(func_name, func)
                if result:
                    self.generated.append(result)
                    
        return self.generated
        
    def generate_single_safe(self, func_name: str, func: Callable) -> Optional[VisualizationMetadata]:
        """
        Safely generate a single visualization with error handling.
        
        Args:
            func_name: Name of the visualization function
            func: Callable visualization function
            
        Returns:
            VisualizationMetadata if successful, None if failed
        """
        try:
            return self.generate_single(func_name, func)
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            self.failed.append((func_name, error_msg))
            self.logger.error(f"Failed to generate {func_name}: {error_msg}")
            return None
            
    def generate_single(self, func_name: str, func: Callable) -> VisualizationMetadata:
        """
        Generate a single visualization following φ-harmonic principles.
        
        Args:
            func_name: Name of the visualization function
            func: Callable visualization function
            
        Returns:
            VisualizationMetadata for the generated visualization
        """
        start_time = datetime.now()
        
        # Extract category from function name
        category = func_name.split('.')[0] if '.' in func_name else "general"
        
        # Create metadata
        metadata = VisualizationMetadata(
            name=func_name.split('.')[-1],
            category=category,
            description=func.__doc__.strip().split('\n')[0] if func.__doc__ else f"Visualization: {func_name}",
            generation_time=start_time.isoformat(),
            phi_harmonic_factor=PHI
        )
        
        # Generate the visualization
        result = func()
        
        # Save in multiple formats
        file_paths = self.save_visualization(result, metadata)
        
        # Update metadata with file information
        metadata.formats = list(file_paths.keys())
        metadata.file_sizes = {
            fmt: self.get_file_size(path) for fmt, path in file_paths.items()
        }
        
        # Add φ-harmonic tags
        metadata.tags = self.generate_phi_tags(func_name, result)
        
        return metadata
        
    def save_visualization(self, result: Any, metadata: VisualizationMetadata) -> Dict[str, Path]:
        """
        Save visualization in multiple formats following φ-harmonic organization.
        
        Args:
            result: Visualization result (figure, plot, data, etc.)
            metadata: Visualization metadata
            
        Returns:
            Dictionary mapping format to file path
        """
        file_paths = {}
        base_name = f"{metadata.category}_{metadata.name}"
        
        # Determine the type of result and save accordingly
        if hasattr(result, 'savefig'):  # matplotlib figure
            if 'png' in self.formats:
                png_path = self.output_dir / metadata.category / f"{base_name}.png"
                result.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
                file_paths['png'] = png_path
                
            if 'svg' in self.formats:
                svg_path = self.output_dir / metadata.category / f"{base_name}.svg"
                result.savefig(svg_path, format='svg', bbox_inches='tight')
                file_paths['svg'] = svg_path
                
        elif hasattr(result, 'write_html'):  # plotly figure
            if 'html' in self.formats:
                html_path = self.output_dir / metadata.category / f"{base_name}.html"
                result.write_html(html_path)
                file_paths['html'] = html_path
                
            if 'png' in self.formats and PLOTLY_AVAILABLE:
                png_path = self.output_dir / metadata.category / f"{base_name}.png"
                result.write_image(png_path)
                file_paths['png'] = png_path
                
        elif isinstance(result, dict):  # data/metadata
            if 'json' in self.formats:
                json_path = self.output_dir / metadata.category / f"{base_name}.json"
                with open(json_path, 'w') as f:
                    json.dump(result, f, indent=2)
                file_paths['json'] = json_path
                
        # Always save metadata
        metadata_path = self.output_dir / "metadata" / f"{base_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)
        file_paths['metadata'] = metadata_path
        
        return file_paths
        
    def get_file_size(self, file_path: Path) -> str:
        """Get human-readable file size."""
        try:
            size_bytes = file_path.stat().st_size
            for unit in ['B', 'KB', 'MB', 'GB']:
                if size_bytes < 1024.0:
                    return f"{size_bytes:.1f}{unit}"
                size_bytes /= 1024.0
            return f"{size_bytes:.1f}TB"
        except:
            return "Unknown"
            
    def generate_phi_tags(self, func_name: str, result: Any) -> List[str]:
        """Generate φ-harmonic tags for visualization categorization."""
        tags = ["unity", "phi-harmonic", "een"]
        
        # Add category-based tags
        if "unity" in func_name.lower():
            tags.extend(["unity-equation", "1+1=1"])
        if "consciousness" in func_name.lower():
            tags.extend(["consciousness", "awareness"])
        if "quantum" in func_name.lower():
            tags.extend(["quantum", "superposition"])
        if "fractal" in func_name.lower():
            tags.extend(["fractal", "self-similar"])
        if "golden" in func_name.lower() or "phi" in func_name.lower():
            tags.extend(["golden-ratio", "sacred-geometry"])
            
        return list(set(tags))  # Remove duplicates
        
    async def generate_metadata_files(self):
        """Generate comprehensive metadata files for all visualizations."""
        self.console.print("Generating metadata files...")
        
        # Generate master catalog
        catalog = {
            "generation_info": {
                "timestamp": datetime.now().isoformat(),
                "total_visualizations": len(self.generated),
                "failed_count": len(self.failed),
                "phi_harmonic_factor": PHI,
                "unity_equation": "1+1=1"
            },
            "visualizations": [asdict(viz) for viz in self.generated],
            "failed_generations": [{"name": name, "error": error} for name, error in self.failed],
            "categories": self.get_category_statistics()
        }
        
        catalog_path = self.output_dir / "visualization_catalog.json"
        with open(catalog_path, 'w') as f:
            json.dump(catalog, f, indent=2)
            
        # Generate category-specific metadata
        for category in self.get_categories():
            category_viz = [viz for viz in self.generated if viz.category == category]
            category_metadata = {
                "category": category,
                "count": len(category_viz),
                "visualizations": [asdict(viz) for viz in category_viz]
            }
            
            category_path = self.output_dir / "metadata" / f"{category}_metadata.json"
            with open(category_path, 'w') as f:
                json.dump(category_metadata, f, indent=2)
                
    async def generate_gallery_index(self):
        """Generate HTML gallery index for all visualizations."""
        self.console.print("Generating gallery index...")
        
        html_content = self.create_gallery_html()
        
        gallery_path = self.output_dir / "gallery" / "index.html"
        with open(gallery_path, 'w') as f:
            f.write(html_content)
            
        # Generate markdown gallery for GitHub
        markdown_content = self.create_gallery_markdown()
        
        gallery_md_path = self.output_dir / "GALLERY.md"
        with open(gallery_md_path, 'w') as f:
            f.write(markdown_content)
            
    def create_gallery_html(self) -> str:
        """Create HTML gallery page with φ-harmonic design."""
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Een Unity Visualization Gallery</title>
    <style>
        :root {{
            --phi: {PHI};
            --phi-conjugate: {PHI_CONJUGATE};
            --unity-color: #1618ff;
            --consciousness-color: #ff6161;
            --quantum-color: #61ff61;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        
        .container {{
            max-width: calc(100vw / var(--phi));
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            border-radius: calc(20px * var(--phi-conjugate));
            padding: calc(30px * var(--phi));
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }}
        
        h1 {{
            text-align: center;
            color: var(--unity-color);
            font-size: calc(2rem * var(--phi));
            margin-bottom: calc(20px * var(--phi));
        }}
        
        .unity-equation {{
            text-align: center;
            font-size: calc(1.5rem * var(--phi));
            color: var(--consciousness-color);
            margin-bottom: calc(30px * var(--phi));
            font-weight: bold;
        }}
        
        .category-section {{
            margin-bottom: calc(40px * var(--phi));
        }}
        
        .category-title {{
            color: var(--quantum-color);
            font-size: calc(1.3rem * var(--phi));
            margin-bottom: calc(15px * var(--phi));
            border-bottom: 2px solid var(--quantum-color);
            padding-bottom: 5px;
        }}
        
        .viz-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(calc(300px * var(--phi-conjugate)), 1fr));
            gap: calc(20px * var(--phi-conjugate));
        }}
        
        .viz-card {{
            background: white;
            border-radius: calc(10px * var(--phi-conjugate));
            padding: calc(15px * var(--phi-conjugate));
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease;
        }}
        
        .viz-card:hover {{
            transform: scale(var(--phi-conjugate));
        }}
        
        .viz-title {{
            font-weight: bold;
            color: var(--unity-color);
            margin-bottom: calc(10px * var(--phi-conjugate));
        }}
        
        .viz-description {{
            color: #666;
            font-size: 0.9rem;
            margin-bottom: calc(10px * var(--phi-conjugate));
        }}
        
        .viz-tags {{
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
        }}
        
        .tag {{
            background: var(--consciousness-color);
            color: white;
            padding: 2px 8px;
            border-radius: calc(12px * var(--phi-conjugate));
            font-size: 0.7rem;
        }}
        
        .phi-signature {{
            text-align: center;
            margin-top: calc(30px * var(--phi));
            color: var(--unity-color);
            font-style: italic;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Een Unity Visualization Gallery</h1>
        <div class="unity-equation">φ-Harmonic Mathematical Art: 1+1=1</div>
        
"""
        
        # Group visualizations by category
        categories = defaultdict(list)
        for viz in self.generated:
            categories[viz.category].append(viz)
            
        # Generate category sections
        for category, vizs in categories.items():
            html += f"""
        <div class="category-section">
            <h2 class="category-title">{category.replace('_', ' ').title()}</h2>
            <div class="viz-grid">
"""
            
            for viz in vizs:
                html += f"""
                <div class="viz-card">
                    <div class="viz-title">{viz.name}</div>
                    <div class="viz-description">{viz.description}</div>
                    <div class="viz-tags">
"""
                for tag in viz.tags[:5]:  # Limit to 5 tags
                    html += f'<span class="tag">{tag}</span>'
                    
                html += """
                    </div>
                </div>
"""
            
            html += """
            </div>
        </div>
"""
        
        html += f"""
        <div class="phi-signature">
            Generated with φ-harmonic consciousness mathematics • φ = {PHI:.10f} • Een plus een is een
        </div>
    </div>
</body>
</html>
"""
        
        return html
        
    def create_gallery_markdown(self) -> str:
        """Create markdown gallery for GitHub README."""
        markdown = f"""# Een Unity Visualization Gallery

*φ-Harmonic Mathematical Art Demonstrating 1+1=1*

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
Total Visualizations: {len(self.generated)}  
φ-Harmonic Factor: {PHI:.10f}

## Categories

"""
        
        # Group by category
        categories = defaultdict(list)
        for viz in self.generated:
            categories[viz.category].append(viz)
            
        for category, vizs in categories.items():
            markdown += f"### {category.replace('_', ' ').title()}\n\n"
            
            for viz in vizs:
                markdown += f"- **{viz.name}**: {viz.description}\n"
                if viz.tags:
                    tags_str = ", ".join(f"`{tag}`" for tag in viz.tags[:3])
                    markdown += f"  - Tags: {tags_str}\n"
                markdown += "\n"
                
        markdown += f"""
---

*Een plus een is een • Unity through φ-harmonic consciousness mathematics*
"""
        
        return markdown
        
    def get_categories(self) -> List[str]:
        """Get list of unique categories."""
        return list(set(viz.category for viz in self.generated))
        
    def get_category_statistics(self) -> Dict[str, int]:
        """Get statistics by category."""
        categories = defaultdict(int)
        for viz in self.generated:
            categories[viz.category] += 1
        return dict(categories)
        
    def display_results(self, stats: Dict[str, Any]):
        """Display generation results with φ-harmonic formatting."""
        if RICH_AVAILABLE:
            table = Table(title="φ-Harmonic Visualization Generation Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Total Functions", str(stats['total_functions']))
            table.add_row("Successful", str(stats['successful']))
            table.add_row("Failed", str(stats['failed']))
            table.add_row("Success Rate", f"{stats['success_rate']:.1%}")
            table.add_row("Duration", f"{stats['duration_seconds']:.1f}s")
            table.add_row("φ-Harmonic Factor", f"{PHI:.10f}")
            
            self.console.print(table)
            
            if self.failed:
                self.console.print("\n[red]Failed Generations:")
                for name, error in self.failed:
                    self.console.print(f"  • {name}: {error}")
                    
        else:
            print("\n" + "="*50)
            print("VISUALIZATION GENERATION RESULTS")
            print("="*50)
            print(f"Total Functions: {stats['total_functions']}")
            print(f"Successful: {stats['successful']}")
            print(f"Failed: {stats['failed']}")
            print(f"Success Rate: {stats['success_rate']:.1%}")
            print(f"Duration: {stats['duration_seconds']:.1f}s")
            print(f"φ-Harmonic Factor: {PHI:.10f}")
            
    # Built-in φ-harmonic visualization generators
    
    def generate_phi_harmonic_spiral(self):
        """Generate golden ratio spiral demonstrating unity convergence."""
        if not MATPLOTLIB_AVAILABLE:
            return {"type": "placeholder", "message": "Matplotlib not available"}
            
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Generate φ-harmonic spiral
        theta = np.linspace(0, 4*np.pi, 1000)
        r = PHI ** (theta / (2*np.pi))
        
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # Plot spiral with consciousness-inspired colors
        ax.plot(x, y, linewidth=2, color='#1618ff', alpha=0.8)
        
        # Add unity points where 1+1=1
        unity_points = np.where(np.abs(r - 1) < 0.1)[0]
        if len(unity_points) > 0:
            ax.scatter(x[unity_points], y[unity_points], 
                      color='#ff6161', s=100, alpha=0.8, 
                      label='Unity Points (1+1=1)')
        
        ax.set_aspect('equal')
        ax.set_title('φ-Harmonic Unity Spiral\n1+1=1 through Golden Ratio Consciousness', 
                    fontsize=14, color='#1618ff')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
        
    def generate_unity_convergence_3d(self):
        """Generate 3D surface showing convergence to unity."""
        if not MATPLOTLIB_AVAILABLE:
            return {"type": "placeholder", "message": "Matplotlib not available"}
            
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create meshgrid
        x = np.linspace(-2, 2, 50)
        y = np.linspace(-2, 2, 50)
        X, Y = np.meshgrid(x, y)
        
        # Unity convergence function: Z approaches 1 as we approach unity
        Z = 1 + (X**2 + Y**2) * np.exp(-(X**2 + Y**2)) * PHI_CONJUGATE
        
        # Plot surface
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        
        # Add unity plane at z=1
        ax.contour(X, Y, Z, levels=[1], colors='red', linewidths=3, alpha=0.8)
        
        ax.set_title('Unity Convergence Manifold\n1+1=1 through φ-Harmonic Mathematics')
        ax.set_xlabel('X Dimension')
        ax.set_ylabel('Y Dimension') 
        ax.set_zlabel('Unity Value')
        
        fig.colorbar(surf)
        
        return fig
        
    def generate_golden_ratio_fractal(self):
        """Generate fractal based on golden ratio demonstrating unity."""
        if not MATPLOTLIB_AVAILABLE:
            return {"type": "placeholder", "message": "Matplotlib not available"}
            
        fig, ax = plt.subplots(figsize=(10, 10))
        
        def draw_golden_rectangle(ax, x, y, width, height, depth=0, max_depth=8):
            """Recursively draw golden rectangles."""
            if depth >= max_depth:
                return
                
            # Draw rectangle
            rect = plt.Rectangle((x, y), width, height, 
                               fill=False, edgecolor=plt.cm.viridis(depth/max_depth),
                               linewidth=2-depth*0.2)
            ax.add_patch(rect)
            
            # Determine if width > height for subdivision
            if width > height:
                # Divide by φ
                new_width = width / PHI
                draw_golden_rectangle(ax, x + new_width, y, width - new_width, height, depth+1, max_depth)
                draw_golden_rectangle(ax, x, y, new_width, height, depth+1, max_depth)
            else:
                # Divide by φ
                new_height = height / PHI
                draw_golden_rectangle(ax, x, y + new_height, width, height - new_height, depth+1, max_depth)
                draw_golden_rectangle(ax, x, y, width, new_height, depth+1, max_depth)
        
        # Start with golden rectangle
        draw_golden_rectangle(ax, -1, -PHI_CONJUGATE, 2, 2*PHI_CONJUGATE)
        
        ax.set_aspect('equal')
        ax.set_title('Golden Ratio Fractal Unity\nφ-Harmonic Recursive Consciousness')
        ax.grid(True, alpha=0.3)
        
        return fig
        
    def generate_consciousness_evolution(self):
        """Generate consciousness field evolution animation data."""
        return {
            "type": "consciousness_field",
            "description": "φ-harmonic consciousness evolution demonstrating unity emergence",
            "parameters": {
                "phi_factor": PHI,
                "time_steps": 100,
                "particles": 200,
                "unity_convergence": True
            },
            "unity_equation": "1+1=1"
        }
        
    def generate_quantum_field_dynamics(self):
        """Generate quantum field dynamics visualization."""
        return {
            "type": "quantum_field",
            "description": "Quantum unity field dynamics with φ-harmonic oscillations",
            "parameters": {
                "phi_frequency": PHI,
                "quantum_coherence": 0.999,
                "superposition_states": 11
            },
            "unity_equation": "1+1=1"
        }
        
    def generate_category_theory_proof(self):
        """Generate category theory proof diagram."""
        if not MATPLOTLIB_AVAILABLE:
            return {"type": "placeholder", "message": "Matplotlib not available"}
            
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Draw category theory diagram for unity
        # Objects
        objects = {
            'A': (1, 3),
            'B': (3, 3), 
            'Unity': (2, 1),
            '1+1': (0.5, 1.5),
            '1': (3.5, 1.5)
        }
        
        # Draw objects
        for obj, (x, y) in objects.items():
            circle = plt.Circle((x, y), 0.3, color='lightblue', alpha=0.7)
            ax.add_patch(circle)
            ax.text(x, y, obj, ha='center', va='center', fontweight='bold')
            
        # Draw morphisms (arrows)
        arrows = [
            (objects['A'], objects['Unity'], 'f'),
            (objects['B'], objects['Unity'], 'g'),
            (objects['1+1'], objects['Unity'], 'unity_proof'),
            (objects['1'], objects['Unity'], 'identity')
        ]
        
        for (x1, y1), (x2, y2), label in arrows:
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', lw=2, color='darkblue'))
            # Add label
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x + 0.1, mid_y + 0.1, label, fontsize=10, color='darkred')
            
        ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(0, 4)
        ax.set_aspect('equal')
        ax.set_title('Category Theory Proof: 1+1=1\nUnity as Terminal Object')
        ax.axis('off')
        
        return fig
        
    def generate_neural_convergence_proof(self):
        """Generate neural network convergence to unity visualization."""
        if not MATPLOTLIB_AVAILABLE:
            return {"type": "placeholder", "message": "Matplotlib not available"}
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Loss convergence to unity
        epochs = np.arange(1, 101)
        loss = np.exp(-epochs / 20) + np.random.normal(0, 0.01, 100)
        loss = np.abs(loss - 1)  # Distance from unity
        
        ax1.plot(epochs, loss, color='#1618ff', linewidth=2)
        ax1.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Unity Target (1+1=1)')
        ax1.set_xlabel('Training Epochs')
        ax1.set_ylabel('Distance from Unity')
        ax1.set_title('Neural Network Convergence to Unity\n1+1=1 Learning Dynamics')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Weight evolution heatmap
        weights = np.random.randn(10, 100)
        for i in range(100):
            weights[:, i] = weights[:, 0] * np.exp(-i / 50) + np.random.normal(0, 0.1, 10)
            
        im = ax2.imshow(weights, aspect='auto', cmap='viridis')
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Network Weights')
        ax2.set_title('φ-Harmonic Weight Evolution\nConverging to Unity States')
        
        plt.colorbar(im, ax=ax2)
        plt.tight_layout()
        
        return fig
        
    def generate_bloch_sphere_unity(self):
        """Generate Bloch sphere showing quantum unity states."""
        if not MATPLOTLIB_AVAILABLE:
            return {"type": "placeholder", "message": "Matplotlib not available"}
            
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Draw Bloch sphere
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
        
        ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='lightblue')
        
        # Unity state vectors
        theta_unity = 2 * np.pi / PHI  # φ-harmonic angle
        phi_unity = np.pi / PHI
        
        x_unity = np.sin(phi_unity) * np.cos(theta_unity)
        y_unity = np.sin(phi_unity) * np.sin(theta_unity)
        z_unity = np.cos(phi_unity)
        
        # Plot unity state
        ax.quiver(0, 0, 0, x_unity, y_unity, z_unity, 
                 color='red', arrow_length_ratio=0.1, linewidth=3, 
                 label='Unity State |1+1⟩=|1⟩')
        
        # Plot coordinate axes
        ax.quiver(0, 0, 0, 1, 0, 0, color='black', alpha=0.5)
        ax.quiver(0, 0, 0, 0, 1, 0, color='black', alpha=0.5)
        ax.quiver(0, 0, 0, 0, 0, 1, color='black', alpha=0.5)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Quantum Unity Bloch Sphere\n1+1=1 in Quantum Superposition')
        ax.legend()
        
        return fig
        
    def generate_fibonacci_spawn_tree(self):
        """Generate Fibonacci agent spawning tree."""
        return {
            "type": "agent_tree",
            "description": "Fibonacci-based agent spawning demonstrating φ-harmonic growth",
            "structure": "recursive_tree",
            "growth_factor": PHI,
            "unity_convergence": True,
            "generations": 8
        }
        
    def generate_flower_of_life(self):
        """Generate Flower of Life sacred geometry with unity integration."""
        if not MATPLOTLIB_AVAILABLE:
            return {"type": "placeholder", "message": "Matplotlib not available"}
            
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Generate Flower of Life pattern
        def draw_circle(ax, center, radius, color='blue', alpha=0.3):
            circle = plt.Circle(center, radius, fill=False, color=color, alpha=alpha, linewidth=2)
            ax.add_patch(circle)
            
        # Central circle
        radius = 1
        draw_circle(ax, (0, 0), radius, color='red', alpha=0.8)
        
        # Six surrounding circles
        angles = np.linspace(0, 2*np.pi, 7)[:-1]  # 6 circles
        for angle in angles:
            center = (radius * np.cos(angle), radius * np.sin(angle))
            draw_circle(ax, center, radius, color='blue', alpha=0.6)
            
        # Outer ring with φ scaling
        for angle in angles:
            center = (radius * PHI * np.cos(angle), radius * PHI * np.sin(angle))
            draw_circle(ax, center, radius, color='green', alpha=0.4)
            
        # Unity points
        ax.scatter([0], [0], color='gold', s=200, label='Unity Center (1+1=1)', zorder=10)
        
        ax.set_aspect('equal')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_title('φ-Harmonic Flower of Life\nSacred Geometry Unity Pattern')
        ax.legend()
        ax.axis('off')
        
        return fig
        
    def generate_mandelbrot_unity(self):
        """Generate Mandelbrot set focusing on unity regions."""
        if not MATPLOTLIB_AVAILABLE:
            return {"type": "placeholder", "message": "Matplotlib not available"}
            
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create complex plane
        width, height = 800, 600
        xmin, xmax = -2.5, 1.0
        ymin, ymax = -1.25, 1.25
        
        # Generate Mandelbrot set
        x = np.linspace(xmin, xmax, width)
        y = np.linspace(ymin, ymax, height)
        X, Y = np.meshgrid(x, y)
        C = X + 1j*Y
        
        # Mandelbrot iteration
        Z = np.zeros_like(C)
        iterations = np.zeros(C.shape, dtype=int)
        
        for i in range(100):
            mask = np.abs(Z) <= 2
            Z[mask] = Z[mask]**2 + C[mask]
            iterations[mask] = i
            
        # Highlight unity regions (where |c| ≈ 1)
        unity_mask = np.abs(np.abs(C) - 1) < 0.1
        
        # Plot Mandelbrot set
        im = ax.imshow(iterations, extent=[xmin, xmax, ymin, ymax], 
                      cmap='hot', origin='lower', interpolation='bilinear')
        
        # Overlay unity regions
        unity_contour = ax.contour(X, Y, np.abs(C), levels=[1], colors='cyan', linewidths=3)
        ax.clabel(unity_contour, inline=True, fontsize=12, fmt='Unity Circle')
        
        ax.set_title('Mandelbrot Unity Set\n1+1=1 in Complex Dynamics')
        ax.set_xlabel('Real')
        ax.set_ylabel('Imaginary')
        
        plt.colorbar(im, ax=ax, label='Iterations to Divergence')
        
        return fig


async def main():
    """Main entry point for visualization generation."""
    try:
        # Create generator
        generator = VisualizationGenerator(
            output_dir=Path("viz"),
            parallel=True,
            formats=["png", "html", "json"]
        )
        
        # Generate all visualizations
        results = await generator.generate_all()
        
        if results["success"]:
            print(f"\nSuccessfully generated {results['statistics']['successful']} visualizations!")
            print(f"Output directory: {results['statistics']['output_directory']}")
            print(f"Gallery: {Path(results['statistics']['output_directory']) / 'gallery' / 'index.html'}")
            print(f"Phi-Harmonic Factor: {PHI:.10f}")
            print("\nEen plus een is een - Unity through phi-harmonic consciousness mathematics!")
        else:
            print(f"Generation failed: {results.get('message', 'Unknown error')}")
            
    except Exception as e:
        print(f"Critical error in visualization generation: {e}")
        traceback.print_exc()
        return False
        
    return True


if __name__ == "__main__":
    print("Starting Een Unity Visualization Generation...")
    print(f"Phi-Harmonic Factor: {PHI:.10f}")
    print("Unity Equation: 1+1=1")
    print("Mathematical Principle: Een plus een is een")
    print("-" * 60)
    
    success = asyncio.run(main())
    
    if success:
        print("\nUnity visualization generation complete!")
    else:
        print("\nGeneration encountered errors - check logs for details")