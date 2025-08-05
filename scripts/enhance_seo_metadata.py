#!/usr/bin/env python3
"""
SEO Metadata Enhancement Script
==============================

Automatically enhances all HTML files with comprehensive SEO metadata:
- Open Graph meta tags
- Twitter Card meta tags
- Schema.org structured data
- Sitemap.xml generation
- Meta descriptions and keywords
- Canonical URLs
"""

import os
import sys
import json
import re
from pathlib import Path
from datetime import datetime
from urllib.parse import urljoin
import xml.etree.ElementTree as ET
from xml.dom import minidom

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class SEOEnhancer:
    """Enhance website with comprehensive SEO metadata"""
    
    def __init__(self):
        self.project_root = project_root
        self.website_dir = self.project_root / "website"
        self.base_url = "https://nourimabrouk.github.io/Een/"
        
        # Site metadata
        self.site_metadata = {
            'title': 'Een - Mathematical Unity Theory | Where 1+1=1 Through Consciousness Mathematics',
            'description': 'Advanced mathematical framework demonstrating 1+1=1 through consciousness field equations, φ-harmonic operations, and quantum mechanical interpretations.',
            'keywords': 'unity mathematics, consciousness mathematics, golden ratio, phi harmonic, quantum unity, idempotent semiring, mathematical proofs, 1+1=1, transcendental mathematics',
            'author': 'Dr. Nouri Mabrouk',
            'language': 'en',
            'site_name': 'Een Unity Mathematics',
            'twitter_handle': '@EenUnityMath',
            'fb_app_id': '',  # Add if available
            'logo_url': 'assets/images/unity_mandala.png',
            'default_image': 'assets/images/unity_proof_visualization.png'
        }
        
        # Page-specific metadata
        self.page_metadata = {
            'index.html': {
                'title': 'Een - Mathematical Unity Theory | Where 1+1=1 Through Consciousness Mathematics',
                'description': 'Discover the revolutionary mathematical framework where 1+1=1 through consciousness field equations, φ-harmonic operations, and quantum mechanical interpretations.',
                'keywords': 'unity mathematics, 1+1=1, consciousness mathematics, phi harmonic, golden ratio, mathematical theory',
                'type': 'website',
                'image': 'assets/images/unity_mandala.png'
            },
            'about.html': {
                'title': 'About Dr. Nouri Mabrouk | Unity Mathematics Pioneer',
                'description': 'Learn about Dr. Nouri Mabrouk, the pioneering mathematician behind Unity Mathematics and the revolutionary proof that 1+1=1.',
                'keywords': 'Nouri Mabrouk, unity mathematics founder, mathematical pioneer, consciousness researcher',
                'type': 'profile',
                'image': 'assets/images/nourimabrouk.png'
            },
            'proofs.html': {
                'title': 'Mathematical Proofs - Een Unity Mathematics | Rigorous Demonstrations of 1+1=1',
                'description': 'Comprehensive mathematical proofs demonstrating that 1+1=1 across multiple domains: algebraic, quantum, logical, and consciousness-based frameworks.',
                'keywords': 'mathematical proofs, unity proofs, 1+1=1 proof, quantum mathematics, algebraic proofs, consciousness mathematics',
                'type': 'article',
                'image': 'assets/images/unity_proof_visualization.png'
            },
            'playground.html': {
                'title': 'Mathematical Playground - Een Unity Mathematics | Interactive Unity Exploration',
                'description': 'Interactive mathematical playground for exploring Unity Mathematics. Calculate 1+1=1, visualize consciousness fields, and experience φ-harmonic operations.',
                'keywords': 'unity calculator, mathematical playground, interactive mathematics, consciousness field visualization',
                'type': 'webapp',
                'image': 'assets/images/unity_field_v1_1.gif'
            },
            'gallery.html': {
                'title': 'Gallery - Een Unity Visualizations | Mathematical Art & Consciousness Fields',
                'description': 'Visual gallery of Unity Mathematics: consciousness field animations, φ-harmonic visualizations, and mathematical art demonstrating 1+1=1.',
                'keywords': 'mathematical visualization, consciousness field art, phi harmonic art, unity visualizations',
                'type': 'website',
                'image': 'assets/images/phi_consciousness_transcendence.html'
            },
            'research.html': {
                'title': 'Research Portfolio - Een Unity Mathematics | Current Projects & Methodologies',
                'description': 'Current research projects in Unity Mathematics: consciousness field dynamics, φ-harmonic analysis, and transcendental mathematical frameworks.',
                'keywords': 'unity mathematics research, consciousness research, phi harmonic research, mathematical methodology',
                'type': 'article',
                'image': 'assets/images/bayesian results.png'
            },
            'publications.html': {
                'title': 'Publications - Een Unity Mathematics | Academic Papers & Presentations',
                'description': 'Academic publications, conference presentations, and research papers on Unity Mathematics and consciousness field theory.',
                'keywords': 'unity mathematics publications, academic papers, conference presentations, mathematical research',
                'type': 'article',
                'image': 'assets/images/unity_proof_visualization.png'
            },
            'philosophy.html': {
                'title': 'The Unity Equation (1+1=1): A Philosophical Treatise | Een Unity Mathematics',
                'description': 'Deep philosophical exploration of Unity Mathematics, examining the metaphysical implications of 1+1=1 and consciousness-mediated reality.',
                'keywords': 'unity philosophy, mathematical philosophy, consciousness philosophy, metaphysics, 1+1=1 philosophy',
                'type': 'article',
                'image': 'assets/images/zen_koan.png'
            }
        }
    
    def enhance_all_pages(self):
        """Enhance all HTML pages with SEO metadata"""
        print("Starting SEO metadata enhancement...")
        
        try:
            # Find all HTML files
            html_files = list(self.website_dir.glob("*.html"))
            
            enhanced_count = 0
            for html_file in html_files:
                if self.enhance_html_file(html_file):
                    enhanced_count += 1
            
            # Generate sitemap
            self.generate_sitemap()
            
            # Generate robots.txt
            self.generate_robots_txt()
            
            # Generate manifest.json
            self.generate_manifest()
            
            print(f"SEO enhancement completed successfully!")
            print(f"Enhanced {enhanced_count} HTML files")
            print("Generated sitemap.xml, robots.txt, and manifest.json")
            
            return True
            
        except Exception as e:
            print(f"Error during SEO enhancement: {e}")
            return False
    
    def enhance_html_file(self, html_file: Path):
        """Enhance a single HTML file with SEO metadata"""
        try:
            print(f"Enhancing {html_file.name}...")
            
            # Read the file
            with open(html_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Get page metadata
            page_meta = self.page_metadata.get(html_file.name, {})
            
            # Extract existing title if no custom title
            if not page_meta.get('title'):
                existing_title = re.search(r'<title>(.*?)</title>', content, re.IGNORECASE)
                if existing_title:
                    page_meta['title'] = existing_title.group(1)
                else:
                    page_meta['title'] = self.site_metadata['title']
            
            # Generate enhanced metadata
            enhanced_content = self.add_seo_metadata(content, html_file.name, page_meta)
            
            # Write back to file
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(enhanced_content)
            
            return True
            
        except Exception as e:
            print(f"Warning: Could not enhance {html_file.name}: {e}")
            return False
    
    def add_seo_metadata(self, content: str, filename: str, page_meta: dict) -> str:
        """Add comprehensive SEO metadata to HTML content"""
        
        # Extract current head content
        head_match = re.search(r'<head>(.*?)</head>', content, re.DOTALL | re.IGNORECASE)
        if not head_match:
            print(f"Warning: No <head> section found in {filename}")
            return content
        
        current_head = head_match.group(1)
        
        # Generate new metadata
        new_metadata = self.generate_metadata_tags(filename, page_meta)
        
        # Check if metadata already exists and remove old versions
        cleaned_head = self.remove_existing_metadata(current_head)
        
        # Add new metadata after viewport tag or at beginning
        viewport_pattern = r'(<meta name="viewport"[^>]*>)'
        if re.search(viewport_pattern, cleaned_head):
            enhanced_head = re.sub(
                viewport_pattern,
                r'\1\n    ' + new_metadata,
                cleaned_head
            )
        else:
            enhanced_head = new_metadata + '\n    ' + cleaned_head
        
        # Replace the head content
        enhanced_content = content.replace(head_match.group(0), f'<head>{enhanced_head}</head>')
        
        return enhanced_content
    
    def remove_existing_metadata(self, head_content: str) -> str:
        """Remove existing SEO metadata to prevent duplicates"""
        
        # Patterns to remove
        patterns_to_remove = [
            r'<meta property="og:[^"]*"[^>]*>\s*',
            r'<meta name="twitter:[^"]*"[^>]*>\s*',
            r'<meta name="description"[^>]*>\s*',
            r'<meta name="keywords"[^>]*>\s*',
            r'<meta name="author"[^>]*>\s*',
            r'<link rel="canonical"[^>]*>\s*',
            r'<script type="application/ld\+json">.*?</script>\s*'
        ]
        
        cleaned = head_content
        for pattern in patterns_to_remove:
            cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        
        return cleaned
    
    def generate_metadata_tags(self, filename: str, page_meta: dict) -> str:
        """Generate comprehensive metadata tags"""
        
        # Get page-specific data
        title = page_meta.get('title', self.site_metadata['title'])
        description = page_meta.get('description', self.site_metadata['description'])
        keywords = page_meta.get('keywords', self.site_metadata['keywords'])
        page_type = page_meta.get('type', 'website')
        image = page_meta.get('image', self.site_metadata['default_image'])
        
        # Build URL
        page_url = urljoin(self.base_url, filename)
        image_url = urljoin(self.base_url, image)
        
        # Generate metadata
        metadata = f'''
    <!-- SEO Meta Tags -->
    <meta name="description" content="{description}">
    <meta name="keywords" content="{keywords}">
    <meta name="author" content="{self.site_metadata['author']}">
    <meta name="language" content="{self.site_metadata['language']}">
    <link rel="canonical" href="{page_url}">
    
    <!-- Open Graph Meta Tags -->
    <meta property="og:title" content="{title}">
    <meta property="og:description" content="{description}">
    <meta property="og:type" content="{page_type}">
    <meta property="og:url" content="{page_url}">
    <meta property="og:image" content="{image_url}">
    <meta property="og:image:alt" content="Unity Mathematics visualization">
    <meta property="og:site_name" content="{self.site_metadata['site_name']}">
    <meta property="og:locale" content="en_US">
    
    <!-- Twitter Card Meta Tags -->
    <meta name="twitter:card" content="summary_large_image">
    <meta name="twitter:site" content="{self.site_metadata['twitter_handle']}">
    <meta name="twitter:creator" content="{self.site_metadata['twitter_handle']}">
    <meta name="twitter:title" content="{title}">
    <meta name="twitter:description" content="{description}">
    <meta name="twitter:image" content="{image_url}">
    <meta name="twitter:image:alt" content="Unity Mathematics visualization">
    
    <!-- Additional Meta Tags -->
    <meta name="robots" content="index, follow, max-image-preview:large, max-snippet:-1, max-video-preview:-1">
    <meta name="googlebot" content="index, follow">
    <meta name="theme-color" content="#FFD700">
    <meta name="msapplication-TileColor" content="#1a2332">
    
    <!-- Schema.org Structured Data -->
    <script type="application/ld+json">
    {self.generate_structured_data(filename, page_meta, page_url)}
    </script>'''
        
        return metadata.strip()
    
    def generate_structured_data(self, filename: str, page_meta: dict, page_url: str) -> str:
        """Generate Schema.org structured data"""
        
        # Base organization data
        organization = {
            "@type": "Organization",
            "name": self.site_metadata['site_name'],
            "url": self.base_url,
            "logo": urljoin(self.base_url, self.site_metadata['logo_url']),
            "description": self.site_metadata['description'],
            "founder": {
                "@type": "Person",
                "name": self.site_metadata['author'],
                "jobTitle": "Mathematician and Consciousness Researcher",
                "knowsAbout": ["Unity Mathematics", "Consciousness Field Theory", "φ-Harmonic Operations"]
            }
        }
        
        # Page-specific structured data
        if filename == 'index.html':
            structured_data = {
                "@context": "https://schema.org",
                "@type": "WebSite",
                "name": self.site_metadata['site_name'],
                "url": self.base_url,
                "description": self.site_metadata['description'],
                "potentialAction": {
                    "@type": "SearchAction",
                    "target": f"{self.base_url}search?q={{search_term_string}}",
                    "query-input": "required name=search_term_string"
                },
                "publisher": organization,
                "mainEntity": {
                    "@type": "MathSolver",
                    "name": "Unity Mathematics Calculator",
                    "description": "Interactive calculator demonstrating that 1+1=1",
                    "url": urljoin(self.base_url, "playground.html")
                }
            }
        elif filename == 'about.html':
            structured_data = {
                "@context": "https://schema.org",
                "@type": "Person",
                "name": self.site_metadata['author'],
                "jobTitle": "Mathematician and Consciousness Researcher",
                "description": page_meta.get('description', ''),
                "url": page_url,
                "sameAs": [self.base_url],
                "knowsAbout": [
                    "Unity Mathematics",
                    "Consciousness Field Theory", 
                    "φ-Harmonic Operations",
                    "Quantum Unity Systems",
                    "Idempotent Semirings"
                ],
                "worksFor": organization
            }
        elif filename == 'proofs.html':
            structured_data = {
                "@context": "https://schema.org",
                "@type": "ScholarlyArticle",
                "headline": page_meta.get('title', ''),
                "description": page_meta.get('description', ''),
                "author": {
                    "@type": "Person",
                    "name": self.site_metadata['author']
                },
                "datePublished": datetime.now().isoformat(),
                "dateModified": datetime.now().isoformat(),
                "publisher": organization,
                "mainEntityOfPage": {
                    "@type": "WebPage",
                    "@id": page_url
                },
                "about": {
                    "@type": "Thing",
                    "name": "Mathematical Proofs",
                    "description": "Proofs demonstrating Unity Mathematics principles"
                }
            }
        elif filename in ['research.html', 'publications.html']:
            structured_data = {
                "@context": "https://schema.org",
                "@type": "CollectionPage",
                "name": page_meta.get('title', ''),
                "description": page_meta.get('description', ''),
                "url": page_url,
                "author": {
                    "@type": "Person", 
                    "name": self.site_metadata['author']
                },
                "publisher": organization,
                "mainEntity": {
                    "@type": "ItemList",
                    "name": "Research Collection" if 'research' in filename else "Publications Collection",
                    "description": page_meta.get('description', '')
                }
            }
        else:
            # Generic webpage
            structured_data = {
                "@context": "https://schema.org",
                "@type": "WebPage",
                "name": page_meta.get('title', ''),
                "description": page_meta.get('description', ''),
                "url": page_url,
                "author": {
                    "@type": "Person",
                    "name": self.site_metadata['author']
                },
                "publisher": organization
            }
        
        return json.dumps(structured_data, indent=4)
    
    def generate_sitemap(self):
        """Generate XML sitemap"""
        print("Generating sitemap.xml...")
        
        # Create root element
        urlset = ET.Element('urlset')
        urlset.set('xmlns', 'http://www.sitemaps.org/schemas/sitemap/0.9')
        urlset.set('xmlns:image', 'http://www.google.com/schemas/sitemap-image/1.1')
        
        # Find all HTML files
        html_files = list(self.website_dir.glob("*.html"))
        
        # Add each page to sitemap
        for html_file in html_files:
            if html_file.name.startswith('test-') or html_file.name.startswith('gallery_test'):
                continue  # Skip test files
            
            url_elem = ET.SubElement(urlset, 'url')
            
            # URL
            loc = ET.SubElement(url_elem, 'loc')
            loc.text = urljoin(self.base_url, html_file.name)
            
            # Last modified
            lastmod = ET.SubElement(url_elem, 'lastmod')
            lastmod.text = datetime.now().strftime('%Y-%m-%d')
            
            # Change frequency
            changefreq = ET.SubElement(url_elem, 'changefreq')
            if html_file.name == 'index.html':
                changefreq.text = 'weekly'
            elif html_file.name in ['research.html', 'publications.html']:
                changefreq.text = 'monthly'
            else:
                changefreq.text = 'monthly'
            
            # Priority
            priority = ET.SubElement(url_elem, 'priority')
            if html_file.name == 'index.html':
                priority.text = '1.0'
            elif html_file.name in ['proofs.html', 'playground.html']:
                priority.text = '0.9'
            elif html_file.name in ['about.html', 'research.html']:
                priority.text = '0.8'
            else:
                priority.text = '0.7'
            
            # Add images for pages that have them
            page_meta = self.page_metadata.get(html_file.name, {})
            if page_meta.get('image'):
                image_elem = ET.SubElement(url_elem, 'image:image')
                image_loc = ET.SubElement(image_elem, 'image:loc')
                image_loc.text = urljoin(self.base_url, page_meta['image'])
                image_title = ET.SubElement(image_elem, 'image:title')
                image_title.text = page_meta.get('title', 'Unity Mathematics')
        
        # Write sitemap
        rough_string = ET.tostring(urlset, 'unicode')
        reparsed = minidom.parseString(rough_string)
        pretty_sitemap = reparsed.toprettyxml(indent="  ")
        
        # Remove empty lines
        pretty_sitemap = '\n'.join([line for line in pretty_sitemap.split('\n') if line.strip()])
        
        sitemap_path = self.website_dir / 'sitemap.xml'
        with open(sitemap_path, 'w', encoding='utf-8') as f:
            f.write(pretty_sitemap)
        
        print(f"Sitemap generated: {sitemap_path}")
    
    def generate_robots_txt(self):
        """Generate robots.txt file"""
        print("Generating robots.txt...")
        
        robots_content = f"""User-agent: *
Allow: /

# Disallow test pages
Disallow: /test-*
Disallow: /*test*

# Allow important resources
Allow: /css/
Allow: /js/
Allow: /assets/
Allow: /data/

# Sitemap
Sitemap: {urljoin(self.base_url, 'sitemap.xml')}

# Crawl delay
Crawl-delay: 1
"""
        
        robots_path = self.website_dir / 'robots.txt'
        with open(robots_path, 'w', encoding='utf-8') as f:
            f.write(robots_content.strip())
        
        print(f"Robots.txt generated: {robots_path}")
    
    def generate_manifest(self):
        """Generate web app manifest.json"""
        print("Generating manifest.json...")
        
        manifest = {
            "name": self.site_metadata['site_name'],
            "short_name": "Een Unity",
            "description": self.site_metadata['description'],
            "start_url": "/",
            "display": "standalone",
            "background_color": "#1a2332",
            "theme_color": "#FFD700",
            "orientation": "portrait-primary",
            "scope": "/",
            "lang": "en-US",
            "icons": [
                {
                    "src": "assets/images/unity_mandala.png",
                    "sizes": "192x192",
                    "type": "image/png",
                    "purpose": "any maskable"
                },
                {
                    "src": "assets/images/unity_mandala.png", 
                    "sizes": "512x512",
                    "type": "image/png",
                    "purpose": "any maskable"
                }
            ],
            "categories": ["education", "science", "productivity"],
            "screenshots": [
                {
                    "src": "assets/images/unity_proof_visualization.png",
                    "sizes": "1920x1080",
                    "type": "image/png",
                    "label": "Unity Mathematics Proofs"
                }
            ]
        }
        
        manifest_path = self.website_dir / 'manifest.json'
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"Manifest.json generated: {manifest_path}")

def main():
    """Main execution function"""
    print("Starting SEO Metadata Enhancement...")
    
    enhancer = SEOEnhancer()
    success = enhancer.enhance_all_pages()
    
    if success:
        print("SEO enhancement completed successfully!")
        print("All pages now have comprehensive SEO metadata including:")
        print("- Open Graph tags for social media")
        print("- Twitter Card metadata")
        print("- Schema.org structured data")
        print("- Optimized meta descriptions and keywords")
        print("- XML sitemap and robots.txt")
        print("- Web app manifest")
    else:
        print("SEO enhancement failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()