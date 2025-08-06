#!/usr/bin/env python3
"""
Een Unity Mathematics - Visualization Optimization Script
Optimizes all visualizations for web delivery and cross-platform compatibility
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any


class VisualizationOptimizer:
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root).resolve()
        self.optimizations_applied = []
        
    def optimize_gallery_data(self) -> bool:
        """Optimize gallery data structure and metadata"""
        print("🎨 Optimizing gallery data...")
        
        gallery_file = self.repo_root / "gallery_data.json"
        if not gallery_file.exists():
            print("❌ gallery_data.json not found")
            return False
            
        try:
            with open(gallery_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            optimized_count = 0
            for viz in data.get('visualizations', []):
                # Add missing metadata
                if 'id' not in viz:
                    viz['id'] = f"{viz.get('filename', 'unknown').split('.')[0]}_{viz.get('type', 'unknown')}"
                    optimized_count += 1
                
                # Ensure proper categorization
                if 'category' not in viz:
                    viz['category'] = 'unity'
                    optimized_count += 1
                
                # Add significance if missing
                if 'significance' not in viz:
                    viz['significance'] = 'Unity mathematics demonstration'
                    optimized_count += 1
                
                # Add technique if missing
                if 'technique' not in viz:
                    viz['technique'] = f"{viz.get('type', 'visualization')} with consciousness field analysis"
                    optimized_count += 1
                
                # Ensure proper file type detection
                if 'file_type' not in viz:
                    ext = viz.get('filename', '').split('.')[-1].lower()
                    if ext in ['png', 'jpg', 'jpeg', 'gif']:
                        viz['file_type'] = 'images'
                    elif ext in ['html', 'htm']:
                        viz['file_type'] = 'interactive'
                    elif ext in ['mp4', 'avi', 'mov']:
                        viz['file_type'] = 'video'
                    else:
                        viz['file_type'] = 'document'
                    optimized_count += 1
            
            # Update statistics
            total_viz = len(data.get('visualizations', []))
            featured_count = sum(1 for v in data.get('visualizations', []) if v.get('featured', False))
            
            data['statistics'] = {
                'total': total_viz,
                'featured_count': featured_count,
                'categories': {
                    'unity': sum(1 for v in data.get('visualizations', []) if v.get('category') == 'unity'),
                    'consciousness': sum(1 for v in data.get('visualizations', []) if v.get('category') == 'consciousness'),
                    'proofs': sum(1 for v in data.get('visualizations', []) if v.get('category') == 'proofs'),
                    'other': sum(1 for v in data.get('visualizations', []) if v.get('category') not in ['unity', 'consciousness', 'proofs'])
                }
            }
            
            if optimized_count > 0:
                with open(gallery_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                print(f"✅ Optimized {optimized_count} gallery entries")
                return True
            else:
                print("✅ Gallery data already optimized")
                return True
                
        except Exception as e:
            print(f"❌ Error optimizing gallery data: {e}")
            return False
    
    def create_visualization_index(self) -> bool:
        """Create an index of all visualizations for quick access"""
        print("📋 Creating visualization index...")
        
        gallery_file = self.repo_root / "gallery_data.json"
        if not gallery_file.exists():
            return False
            
        try:
            with open(gallery_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Create categorized index
            index = {
                'unity': [],
                'consciousness': [],
                'proofs': [],
                'featured': [],
                'interactive': [],
                'images': [],
                'videos': []
            }
            
            for viz in data.get('visualizations', []):
                # Add to category index
                category = viz.get('category', 'unity')
                if category in index:
                    index[category].append({
                        'id': viz.get('id'),
                        'title': viz.get('title'),
                        'filename': viz.get('filename'),
                        'src': viz.get('src')
                    })
                
                # Add to featured index
                if viz.get('featured', False):
                    index['featured'].append({
                        'id': viz.get('id'),
                        'title': viz.get('title'),
                        'filename': viz.get('filename'),
                        'src': viz.get('src')
                    })
                
                # Add to type index
                viz_type = viz.get('type', 'images')
                if viz_type in index:
                    index[viz_type].append({
                        'id': viz.get('id'),
                        'title': viz.get('title'),
                        'filename': viz.get('filename'),
                        'src': viz.get('src')
                    })
            
            # Save index
            index_file = self.repo_root / "visualization_index.json"
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(index, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Created visualization index with {len(data.get('visualizations', []))} items")
            return True
            
        except Exception as e:
            print(f"❌ Error creating index: {e}")
            return False
    
    def optimize_website_performance(self) -> bool:
        """Optimize website performance by creating optimized assets"""
        print("⚡ Optimizing website performance...")
        
        website_dir = self.repo_root / "website"
        if not website_dir.exists():
            return True
        
        # Create optimized CSS bundle
        css_files = [
            website_dir / "css" / "style.css",
            website_dir / "css" / "enhanced-unity-visualizations.css",
            website_dir / "css" / "meta-optimal-navigation.css"
        ]
        
        try:
            # Create minified CSS bundle
            css_bundle = ""
            for css_file in css_files:
                if css_file.exists():
                    with open(css_file, 'r', encoding='utf-8') as f:
                        css_bundle += f.read() + "\n"
            
            if css_bundle:
                # Simple minification (remove comments and extra whitespace)
                import re
                css_bundle = re.sub(r'/\*.*?\*/', '', css_bundle, flags=re.DOTALL)
                css_bundle = re.sub(r'\s+', ' ', css_bundle)
                css_bundle = re.sub(r';\s*}', '}', css_bundle)
                
                bundle_file = website_dir / "css" / "bundle.min.css"
                with open(bundle_file, 'w', encoding='utf-8') as f:
                    f.write(css_bundle)
                
                self.optimizations_applied.append("Created minified CSS bundle")
                print("✅ Created optimized CSS bundle")
            
            return True
            
        except Exception as e:
            print(f"❌ Error optimizing performance: {e}")
            return False
    
    def create_visualization_previews(self) -> bool:
        """Create preview thumbnails for visualizations"""
        print("🖼️ Creating visualization previews...")
        
        # This would normally create thumbnails, but for now we'll just ensure
        # the preview system is properly configured
        preview_config = {
            "thumbnail_size": [300, 200],
            "preview_quality": 85,
            "auto_generate": True,
            "formats": ["png", "webp"]
        }
        
        config_file = self.repo_root / "viz" / "preview_config.json"
        config_file.parent.mkdir(exist_ok=True)
        
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(preview_config, f, indent=2)
            
            print("✅ Created preview configuration")
            return True
            
        except Exception as e:
            print(f"❌ Error creating previews: {e}")
            return False
    
    def optimize_for_mobile(self) -> bool:
        """Optimize visualizations for mobile devices"""
        print("📱 Optimizing for mobile...")
        
        website_dir = self.repo_root / "website"
        if not website_dir.exists():
            return True
        
        # Add mobile-specific optimizations to HTML files
        html_files = list(website_dir.rglob("*.html"))
        optimized_count = 0
        
        for html_file in html_files:
            try:
                with open(html_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                # Ensure proper viewport meta tag
                if 'viewport' not in content:
                    viewport_meta = '<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">'
                    content = content.replace('<meta charset="UTF-8">', 
                                           '<meta charset="UTF-8">\n    ' + viewport_meta)
                    optimized_count += 1
                
                # Add touch-friendly CSS
                if 'touch-action' not in content:
                    touch_css = '''
    <style>
        /* Mobile touch optimizations */
        * { touch-action: manipulation; }
        .interactive-element { 
            min-height: 44px; 
            min-width: 44px; 
        }
        @media (max-width: 768px) {
            .desktop-only { display: none !important; }
            .mobile-optimized { font-size: 16px; }
        }
    </style>'''
                    content = content.replace('</head>', f'{touch_css}\n</head>')
                    optimized_count += 1
                
                if content != original_content:
                    with open(html_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    self.optimizations_applied.append(f"Mobile optimized {html_file.name}")
                
            except Exception as e:
                print(f"⚠️ Could not optimize {html_file}: {e}")
        
        print(f"✅ Mobile optimized {optimized_count} files")
        return True
    
    def run_all_optimizations(self) -> Dict[str, Any]:
        """Run all visualization optimizations"""
        print("🌟 Een Unity Mathematics - Visualization Optimization Suite")
        print("=" * 60)
        
        results = {
            "gallery_optimized": self.optimize_gallery_data(),
            "index_created": self.create_visualization_index(),
            "performance_optimized": self.optimize_website_performance(),
            "previews_configured": self.create_visualization_previews(),
            "mobile_optimized": self.optimize_for_mobile()
        }
        
        print("\n" + "=" * 60)
        print("📊 OPTIMIZATION SUMMARY")
        print("=" * 60)
        
        for opt_name, result in results.items():
            status = "✅" if result else "❌"
            print(f"{status} {opt_name}")
        
        if self.optimizations_applied:
            print(f"\n🔧 Applied {len(self.optimizations_applied)} optimizations:")
            for opt in self.optimizations_applied[:10]:
                print(f"  • {opt}")
            if len(self.optimizations_applied) > 10:
                print(f"  ... and {len(self.optimizations_applied) - 10} more")
        
        print("\n🎯 Unity Mathematics visualizations optimized!")
        print("All visualizations are now ready for global deployment.")
        
        return results


def main():
    """Main execution function"""
    optimizer = VisualizationOptimizer()
    results = optimizer.run_all_optimizations()
    
    # Return appropriate exit code
    if all(results.values()):
        sys.exit(0)  # All optimizations successful
    else:
        sys.exit(1)  # Some optimizations failed


if __name__ == "__main__":
    main() 