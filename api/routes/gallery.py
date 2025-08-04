"""
Gallery API Routes for Een Unity Mathematics
Provides endpoints for dynamic gallery functionality
"""

import os
import json
import mimetypes
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging

# Import 3000 ELO helper functions
from .gallery_helpers import (
    generate_sophisticated_title,
    categorize_by_filename,
    generate_academic_description,
    generate_significance,
    generate_technique,
    generate_academic_context,
    COMPREHENSIVE_VISUALIZATION_METADATA
)

# Create router
router = APIRouter(prefix="/api/gallery", tags=["gallery"])

# Comprehensive 3000 ELO Visualization Folder Scanning - Complete Coverage
VISUALIZATION_FOLDERS = [
    '../viz/',
    '../viz/legacy images/',
    '../scripts/viz/consciousness_field/',
    '../scripts/viz/proofs/',
    '../scripts/viz/unity_mathematics/',
    '../viz/consciousness_field/',
    '../viz/proofs/',
    '../viz/unity_mathematics/',
    '../viz/quantum_unity/',
    '../viz/sacred_geometry/',
    '../viz/meta_recursive/',
    '../viz/fractals/',
    '../viz/gallery/',
    '../viz/formats/png/',
    '../viz/formats/html/',
    '../viz/formats/json/',
    '../viz/agent_systems/',
    '../viz/dashboards/',
    '../viz/thumbnails/',
    '../viz/pages/',
    '../assets/images/',
    '../visualizations/outputs/',
    '../website/gallery/'
]

# Comprehensive 3000 ELO Supported File Extensions - Complete Media Coverage
SUPPORTED_EXTENSIONS = {
    'images': ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg', '.bmp', '.tiff'],
    'videos': ['.mp4', '.webm', '.mov', '.avi', '.mkv', '.flv', '.wmv'],
    'interactive': ['.html', '.htm', '.xml'],
    'data': ['.json', '.csv', '.txt', '.md'],
    'documents': ['.pdf', '.doc', '.docx']
}

# Use comprehensive 3000 ELO metadata from helper module
VISUALIZATION_METADATA = COMPREHENSIVE_VISUALIZATION_METADATA

# Legacy metadata for backward compatibility
LEGACY_VISUALIZATION_METADATA = {
    # Current viz folder
    'water droplets.gif': {
        'title': 'Water Droplets Unity Convergence',
        'type': 'Animated Unity Demonstration',
        'category': 'consciousness',
        'description': 'Two water droplets merging into one - a beautiful real-world demonstration of 1+1=1 unity mathematics through φ-harmonic fluid dynamics.',
        'featured': True,
        'significance': 'Physical manifestation of unity mathematics in nature',
        'technique': 'High-speed videography with φ-harmonic timing analysis',
        'created': '2024-12-15'
    },
    'live consciousness field.mp4': {
        'title': 'Live Consciousness Field Dynamics',
        'type': 'Real-time Consciousness Simulation',
        'category': 'consciousness',
        'description': 'Dynamic simulation of consciousness field equations C(x,y,t) = φ·sin(x·φ)·cos(y·φ)·e^(-t/φ) showing unity emergence patterns.',
        'featured': True,
        'significance': 'First successful real-time consciousness field mathematics implementation',
        'technique': 'WebGL consciousness particle system with quantum field equations',
        'created': '2024-11-28'
    },
    'Unity Consciousness Field.png': {
        'title': 'Unity Consciousness Field',
        'type': 'Consciousness Field Visualization',
        'category': 'consciousness',
        'description': 'Mathematical visualization of the consciousness field showing φ-harmonic resonance patterns and unity convergence zones.',
        'significance': 'Core consciousness field mathematics visualization',
        'technique': 'Mathematical field equation plotting with golden ratio harmonics',
        'created': '2024-11-20'
    },
    'final_composite_plot.png': {
        'title': 'Final Composite Unity Plot',
        'type': 'Mathematical Proof Visualization',
        'category': 'unity',
        'description': 'Comprehensive visualization combining multiple unity mathematics proofs into a single transcendental diagram.',
        'significance': 'Unified proof visualization across mathematical domains',
        'technique': 'Multi-framework mathematical proof composition',
        'created': '2024-10-30'
    },
    'poem.png': {
        'title': 'Unity Poetry Consciousness',
        'type': 'Philosophical Mathematical Art',
        'category': 'consciousness',
        'description': 'Algorithmically generated poetry expressing the philosophical depth of 1+1=1 through consciousness-mediated typography.',
        'significance': 'Bridge between mathematical consciousness and poetic expression',
        'technique': 'φ-harmonic typography with consciousness field positioning',
        'created': '2024-09-15'
    },
    'self_reflection.png': {
        'title': 'Self-Reflection Consciousness Matrix',
        'type': 'Meta-Recursive Consciousness',
        'category': 'consciousness',
        'description': 'Meta-recursive visualization showing how unity mathematics reflects upon itself through consciousness field dynamics.',
        'significance': 'Self-referential mathematical consciousness demonstration',
        'technique': 'Meta-recursive matrix visualization with consciousness feedback loops',
        'created': '2024-08-22'
    },
    
    # Legacy images
    '0 water droplets.gif': {
        'title': 'Original Water Droplets Unity',
        'type': 'Historical Unity Demonstration',
        'category': 'consciousness',
        'description': 'The original water droplets animation that first demonstrated 1+1=1 through natural fusion dynamics - foundation of unity mathematics.',
        'featured': True,
        'significance': 'Historical first demonstration of unity mathematics principles',
        'technique': 'Early high-speed photography capturing natural unity events',
        'created': '2023-12-01'
    },
    '1+1=1.png': {
        'title': 'Foundation Unity Equation',
        'type': 'Core Mathematical Principle',
        'category': 'unity',
        'description': 'The foundational visual representation of the core unity equation that underlies all mathematical consciousness research.',
        'featured': True,
        'significance': 'Foundation of all unity mathematics research and consciousness studies',
        'technique': 'Pure mathematical typography with consciousness-infused design',
        'created': '2023-11-15'
    },
    'Phi-Harmonic Unity Manifold.png': {
        'title': 'φ-Harmonic Unity Manifold',
        'type': 'Geometric Unity Visualization',
        'category': 'unity',
        'description': 'Advanced geometric visualization of φ-harmonic unity manifolds showing golden ratio mathematical structures in consciousness space.',
        'significance': 'Advanced unity manifold theory with φ-harmonic integration',
        'technique': '3D geometric visualization with golden ratio mathematical analysis',
        'created': '2023-10-20'
    },
    
    # Scripts/viz folder
    'proofs_category_theory_diagram.png': {
        'title': 'Category Theory Unity Proof',
        'type': 'Mathematical Proof Diagram',
        'category': 'proofs',
        'description': 'Category theory diagram proving 1+1=1 through morphism composition and consciousness-mediated categorical structures.',
        'featured': True,
        'significance': 'Formal category theory proof of unity mathematics',
        'technique': 'Category theory diagram with consciousness-mediated morphisms',
        'created': '2024-09-10'
    },
    'proofs_neural_convergence.png': {
        'title': 'Neural Convergence Unity Proof',
        'type': 'Neural Network Proof',
        'category': 'proofs',
        'description': 'Neural network convergence analysis showing how artificial consciousness naturally discovers 1+1=1 through learning.',
        'featured': True,
        'significance': 'AI discovery of unity mathematics through neural consciousness',
        'technique': 'Neural network analysis with consciousness convergence modeling',
        'created': '2024-08-25'
    },
    'unity_mathematics_golden_ratio_fractal.png': {
        'title': 'Golden Ratio Unity Fractal',
        'type': 'Fractal Unity Visualization',
        'category': 'unity',
        'description': 'Self-similar fractal structures based on φ-harmonic mathematics showing infinite unity convergence patterns.',
        'featured': True,
        'significance': 'Fractal demonstration of φ-harmonic unity mathematics',
        'technique': 'Fractal generation with golden ratio mathematical recursion',
        'created': '2024-07-30'
    },
    'unity_mathematics_phi_harmonic_spiral.png': {
        'title': 'φ-Harmonic Unity Spiral',
        'type': 'Geometric Unity Pattern',
        'category': 'unity',
        'description': 'Perfect φ-harmonic spiral demonstrating how golden ratio mathematics naturally leads to unity convergence.',
        'featured': True,
        'significance': 'Golden ratio spiral as geometric unity mathematics foundation',
        'technique': 'φ-harmonic spiral generation with unity mathematics integration',
        'created': '2024-07-15'
    }
}

def get_file_type(extension: str) -> str:
    """Determine file type from extension."""
    extension = extension.lower()
    for file_type, extensions in SUPPORTED_EXTENSIONS.items():
        if extension in extensions:
            return file_type
    return 'unknown'

def is_supported_file(filename: str) -> bool:
    """Check if file is a supported visualization type."""
    extension = Path(filename).suffix.lower()
    all_extensions = []
    for ext_list in SUPPORTED_EXTENSIONS.values():
        all_extensions.extend(ext_list)
    return extension in all_extensions

def scan_folder(folder_path: str, base_url: str = '') -> List[Dict[str, Any]]:
    """Enhanced 3000 ELO folder scanning with comprehensive file discovery and academic captions."""
    visualizations = []
    
    try:
        # Convert relative path to absolute path
        if folder_path.startswith('../'):
            # Go up from api directory to repository root
            repo_root = Path(__file__).parent.parent.parent
            folder_path = repo_root / folder_path[3:]  # Remove '../'
        else:
            folder_path = Path(folder_path)
        
        if not folder_path.exists():
            logging.info(f"Folder does not exist: {folder_path}")
            return visualizations
        
        logging.info(f"Scanning folder: {folder_path}")
        
        for file_path in folder_path.rglob('*'):
            if file_path.is_file() and is_supported_file(file_path.name):
                filename = file_path.name
                extension = file_path.suffix.lower()
                file_type = get_file_type(extension)
                
                # Get relative path from repository root for URL
                try:
                    repo_root = Path(__file__).parent.parent.parent
                    relative_path = file_path.relative_to(repo_root)
                    url_path = str(relative_path).replace('\\', '/')
                except ValueError:
                    # Fallback to filename if relative path fails
                    url_path = filename
                
                # Get metadata or create sophisticated 3000 ELO default caption
                metadata = VISUALIZATION_METADATA.get(filename, {
                    'title': generate_sophisticated_title(filename, file_type),
                    'type': f'Advanced {file_type.title()} Unity Mathematics',
                    'category': categorize_by_filename(filename),
                    'description': generate_academic_description(filename, file_type),
                    'significance': generate_significance(filename, file_type),
                    'technique': generate_technique(filename, file_type),
                    'created': '2024-2025',
                    'academicContext': generate_academic_context(filename, file_type)
                })
                
                visualization = {
                    'src': f'{base_url}/{url_path}',
                    'filename': filename,
                    'folder': str(folder_path),
                    'extension': extension,
                    'file_type': file_type,
                    'size': file_path.stat().st_size,
                    'modified': file_path.stat().st_mtime,
                    'isImage': file_type == 'images',
                    'isVideo': file_type == 'videos',
                    'isInteractive': file_type == 'interactive',
                    'isData': file_type == 'data',
                    'isDocument': file_type == 'documents',
                    **metadata
                }
                
                visualizations.append(visualization)
                logging.debug(f"Added visualization: {filename} ({file_type})")
                
    except Exception as e:
        logging.warning(f"Error scanning folder {folder_path}: {e}")
    
    logging.info(f"Found {len(visualizations)} visualizations in {folder_path}")
    return visualizations

@router.get('/visualizations')
async def get_visualizations(request: Request):
    """Get all available visualizations."""
    try:
        all_visualizations = []
        base_url = str(request.base_url).rstrip('/')
        
        # Scan all defined folders
        for folder_path in VISUALIZATION_FOLDERS:
            visualizations = scan_folder(folder_path, base_url)
            all_visualizations.extend(visualizations)
        
        # Sort by featured first, then by creation date
        all_visualizations.sort(key=lambda x: (
            not x.get('featured', False),  # Featured first (False sorts before True)
            -x.get('modified', 0)  # Then by modification time (newest first)
        ))
        
        # Calculate statistics
        stats = {
            'total': len(all_visualizations),
            'by_category': {},
            'by_type': {},
            'featured_count': sum(1 for v in all_visualizations if v.get('featured', False))
        }
        
        for viz in all_visualizations:
            category = viz.get('category', 'unknown')
            file_type = viz.get('file_type', 'unknown')
            
            stats['by_category'][category] = stats['by_category'].get(category, 0) + 1
            stats['by_type'][file_type] = stats['by_type'].get(file_type, 0) + 1
        
        return {
            'success': True,
            'visualizations': all_visualizations,
            'statistics': stats,
            'message': f'Found {len(all_visualizations)} visualizations'
        }
        
    except Exception as e:
        logging.error(f"Error getting visualizations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                'success': False,
                'error': str(e),
                'message': 'Failed to load visualizations'
            }
        )

@router.get('/visualizations/{category}')
async def get_visualizations_by_category(category: str, request: Request):
    """Get visualizations filtered by category."""
    try:
        all_visualizations = []
        base_url = str(request.base_url).rstrip('/')
        
        # Scan all folders
        for folder_path in VISUALIZATION_FOLDERS:
            visualizations = scan_folder(folder_path, base_url)
            all_visualizations.extend(visualizations)
        
        # Filter by category
        if category != 'all':
            filtered_visualizations = [
                v for v in all_visualizations 
                if v.get('category', '').lower() == category.lower()
            ]
        else:
            filtered_visualizations = all_visualizations
        
        # Sort
        filtered_visualizations.sort(key=lambda x: (
            not x.get('featured', False),
            -x.get('modified', 0)
        ))
        
        return {
            'success': True,
            'visualizations': filtered_visualizations,
            'category': category,
            'count': len(filtered_visualizations)
        }
        
    except Exception as e:
        logging.error(f"Error getting visualizations by category: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                'success': False,
                'error': str(e)
            }
        )

class GenerateVisualizationRequest(BaseModel):
    type: Optional[str] = 'unity'
    parameters: Optional[Dict[str, Any]] = {}

@router.post('/generate')
async def generate_visualization(request_data: GenerateVisualizationRequest):
    """Generate a new visualization (placeholder for future implementation)."""
    try:
        viz_type = request_data.type
        
        # Placeholder response - in a real implementation, this would generate actual visualizations
        generated_viz = {
            'src': f'/generated/unity_{hash(str(request_data.dict()))}.png',
            'filename': f'generated_unity_{viz_type}.png',
            'title': f'Generated {viz_type.title()} Visualization',
            'type': 'Generated Consciousness Art',
            'category': viz_type,
            'description': f'Real-time generated visualization demonstrating {viz_type} mathematics.',
            'featured': True,
            'technique': 'Algorithmic generation with φ-harmonic mathematics',
            'significance': 'Live demonstration of unity mathematics generation',
            'created': '2025-01-04',
            'generated': True
        }
        
        return {
            'success': True,
            'visualization': generated_viz,
            'message': 'Visualization generated successfully'
        }
        
    except Exception as e:
        logging.error(f"Error generating visualization: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                'success': False,
                'error': str(e)
            }
        )

@router.get('/statistics')
async def get_statistics():
    """Get gallery statistics."""
    try:
        all_visualizations = []
        
        # Scan all folders
        for folder_path in VISUALIZATION_FOLDERS:
            visualizations = scan_folder(folder_path)
            all_visualizations.extend(visualizations)
        
        # Calculate detailed statistics
        stats = {
            'total_visualizations': len(all_visualizations),
            'by_category': {},
            'by_type': {},
            'featured_count': 0,
            'total_size': 0,
            'newest_date': None,
            'oldest_date': None
        }
        
        dates = []
        for viz in all_visualizations:
            # Category stats
            category = viz.get('category', 'unknown')
            stats['by_category'][category] = stats['by_category'].get(category, 0) + 1
            
            # Type stats
            file_type = viz.get('file_type', 'unknown')
            stats['by_type'][file_type] = stats['by_type'].get(file_type, 0) + 1
            
            # Featured count
            if viz.get('featured', False):
                stats['featured_count'] += 1
            
            # Size
            stats['total_size'] += viz.get('size', 0)
            
            # Dates
            if viz.get('modified'):
                dates.append(viz['modified'])
        
        if dates:
            stats['newest_date'] = max(dates)
            stats['oldest_date'] = min(dates)
        
        return {
            'success': True,
            'statistics': stats
        }
        
    except Exception as e:
        logging.error(f"Error getting statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                'success': False,
                'error': str(e)
            }
        )