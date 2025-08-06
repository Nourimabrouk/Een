#!/usr/bin/env python3
"""
Een Unity Mathematics - Website and Visualization Fix Script
Comprehensive fix for all website issues, broken links, and optimization problems
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List


class WebsiteFixer:
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root).resolve()
        self.issues_found = []
        self.fixes_applied = []

    def fix_gallery_data_paths(self) -> bool:
        """Fix Windows backslashes in gallery_data.json paths"""
        print("🔧 Fixing gallery data paths...")

        gallery_file = self.repo_root / "gallery_data.json"
        if not gallery_file.exists():
            print("❌ gallery_data.json not found")
            return False

        try:
            with open(gallery_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            fixed_count = 0
            for viz in data.get("visualizations", []):
                if "src" in viz and "\\" in viz["src"]:
                    old_src = viz["src"]
                    viz["src"] = viz["src"].replace("\\", "/")
                    fixed_count += 1
                    self.fixes_applied.append(f"Fixed path: {old_src} → {viz['src']}")

            if fixed_count > 0:
                with open(gallery_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                print(f"✅ Fixed {fixed_count} paths in gallery_data.json")
                return True
            else:
                print("✅ No path issues found in gallery_data.json")
                return True

        except Exception as e:
            print(f"❌ Error fixing gallery data: {e}")
            return False

    def validate_file_existence(self) -> Dict[str, List[str]]:
        """Check if all files referenced in gallery_data.json exist"""
        print("🔍 Validating file existence...")

        gallery_file = self.repo_root / "gallery_data.json"
        if not gallery_file.exists():
            return {"missing": [], "broken": []}

        try:
            with open(gallery_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            missing_files = []
            broken_links = []

            for viz in data.get("visualizations", []):
                if "src" in viz:
                    file_path = self.repo_root / viz["src"]
                    if not file_path.exists():
                        missing_files.append(viz["src"])
                        self.issues_found.append(f"Missing file: {viz['src']}")
                    elif viz["src"].startswith("http"):
                        # Check if external link is accessible
                        try:
                            import urllib.request

                            urllib.request.urlopen(viz["src"], timeout=5)
                        except:
                            broken_links.append(viz["src"])
                            self.issues_found.append(f"Broken link: {viz['src']}")

            print(
                f"📊 Validation results: {len(missing_files)} missing files, {len(broken_links)} broken links"
            )
            return {"missing": missing_files, "broken": broken_links}

        except Exception as e:
            print(f"❌ Error validating files: {e}")
            return {"missing": [], "broken": []}

    def fix_website_navigation(self) -> bool:
        """Fix navigation issues in website files"""
        print("🔧 Fixing website navigation...")

        website_dir = self.repo_root / "website"
        if not website_dir.exists():
            print("❌ website directory not found")
            return False

        # Find all HTML files
        html_files = list(website_dir.rglob("*.html"))
        fixed_count = 0

        for html_file in html_files:
            try:
                with open(html_file, "r", encoding="utf-8") as f:
                    content = f.read()

                original_content = content

                # Fix common navigation issues
                # 1. Fix relative paths
                content = re.sub(r'href="\.\./\.\./', 'href="../', content)
                content = re.sub(r'src="\.\./\.\./', 'src="../', content)

                # 2. Fix missing navigation includes
                if (
                    "<!-- Unified Navigation and AI Chat -->" in content
                    and "shared-navigation.js" not in content
                ):
                    nav_script = '<script src="js/shared-navigation.js"></script>'
                    content = content.replace(
                        "<!-- Unified Navigation and AI Chat -->",
                        f"<!-- Unified Navigation and AI Chat -->\n    {nav_script}",
                    )
                    fixed_count += 1

                # 3. Fix broken image paths
                content = re.sub(
                    r'src="assets\\\\([^"]+)"', r'src="assets/\1"', content
                )
                content = re.sub(
                    r'src="images\\\\([^"]+)"', r'src="images/\1"', content
                )

                # 4. Fix CSS/JS paths
                content = re.sub(r'href="css\\\\([^"]+)"', r'href="css/\1"', content)
                content = re.sub(r'src="js\\\\([^"]+)"', r'src="js/\1"', content)

                if content != original_content:
                    with open(html_file, "w", encoding="utf-8") as f:
                        f.write(content)
                    self.fixes_applied.append(f"Fixed navigation in {html_file.name}")

            except Exception as e:
                print(f"❌ Error fixing {html_file}: {e}")

        print(f"✅ Fixed navigation in {fixed_count} files")
        return True

    def optimize_images(self) -> bool:
        """Optimize images for web delivery"""
        print("🖼️ Optimizing images...")

        # Check if Pillow is available for image optimization
        try:
            from PIL import Image
            import io
        except ImportError:
            print("⚠️ Pillow not available - skipping image optimization")
            return True

        image_dirs = [
            self.repo_root / "assets" / "images",
            self.repo_root / "viz" / "legacy images",
            self.repo_root / "website" / "images",
        ]

        optimized_count = 0
        for img_dir in image_dirs:
            if not img_dir.exists():
                continue

            for img_file in img_dir.glob("*.png"):
                try:
                    with Image.open(img_file) as img:
                        # Convert to RGB if necessary
                        if img.mode in ("RGBA", "LA", "P"):
                            img = img.convert("RGB")

                        # Optimize and save
                        img.save(img_file, "PNG", optimize=True, quality=85)
                        optimized_count += 1

                except Exception as e:
                    print(f"⚠️ Could not optimize {img_file}: {e}")

        print(f"✅ Optimized {optimized_count} images")
        return True

    def create_missing_directories(self) -> bool:
        """Create missing directories that are referenced in gallery data"""
        print("📁 Creating missing directories...")

        gallery_file = self.repo_root / "gallery_data.json"
        if not gallery_file.exists():
            return True

        try:
            with open(gallery_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            created_count = 0
            for viz in data.get("visualizations", []):
                if "src" in viz:
                    file_path = Path(viz["src"])
                    dir_path = self.repo_root / file_path.parent

                    if not dir_path.exists():
                        dir_path.mkdir(parents=True, exist_ok=True)
                        created_count += 1
                        self.fixes_applied.append(f"Created directory: {dir_path}")

            print(f"✅ Created {created_count} missing directories")
            return True

        except Exception as e:
            print(f"❌ Error creating directories: {e}")
            return False

    def fix_deployment_script(self) -> bool:
        """Fix the deployment script issues"""
        print("🚀 Fixing deployment script...")

        deploy_script = self.repo_root / "scripts" / "deploy_global.sh"
        if not deploy_script.exists():
            print("❌ deploy_global.sh not found")
            return False

        try:
            with open(deploy_script, "r", encoding="utf-8") as f:
                content = f.read()

            # Fix the URL construction issue
            if "${Nourimabrouk}.github.io/${Een}/" in content:
                content = content.replace(
                    "${Nourimabrouk}.github.io/${Een}/",
                    "${GITHUB_USER}.github.io/${REPO_NAME}/",
                )
                self.fixes_applied.append("Fixed GitHub Pages URL construction")

            # Add Windows compatibility
            if "#!/bin/bash" in content and "set -e" in content:
                # Add Windows compatibility check
                windows_compat = """
# Windows compatibility
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    print_warning "Running on Windows - some features may be limited"
    # Use Windows-compatible commands
    PYTHON_CMD="python"
    GIT_CMD="git"
else
    PYTHON_CMD="python3"
    GIT_CMD="git"
fi
"""
                content = content.replace(
                    "set -e  # Exit on any error",
                    f"set -e  # Exit on any error{windows_compat}",
                )
                self.fixes_applied.append(
                    "Added Windows compatibility to deployment script"
                )

            with open(deploy_script, "w", encoding="utf-8") as f:
                f.write(content)

            print("✅ Fixed deployment script")
            return True

        except Exception as e:
            print(f"❌ Error fixing deployment script: {e}")
            return False

    def generate_sitemap(self) -> bool:
        """Generate a proper sitemap.xml for the website"""
        print("🗺️ Generating sitemap...")

        website_dir = self.repo_root / "website"
        if not website_dir.exists():
            return True

        try:
            html_files = list(website_dir.rglob("*.html"))

            sitemap_content = '<?xml version="1.0" encoding="UTF-8"?>\n'
            sitemap_content += (
                '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
            )

            base_url = "https://nourimabrouk.github.io/Een/"

            for html_file in html_files:
                # Convert file path to URL
                relative_path = html_file.relative_to(website_dir)
                url_path = str(relative_path).replace("\\", "/")

                sitemap_content += f"  <url>\n"
                sitemap_content += f"    <loc>{base_url}{url_path}</loc>\n"
                sitemap_content += (
                    f"    <lastmod>{html_file.stat().st_mtime}</lastmod>\n"
                )
                sitemap_content += f"    <changefreq>weekly</changefreq>\n"
                sitemap_content += f"    <priority>0.8</priority>\n"
                sitemap_content += f"  </url>\n"

            sitemap_content += "</urlset>"

            sitemap_file = website_dir / "sitemap.xml"
            with open(sitemap_file, "w", encoding="utf-8") as f:
                f.write(sitemap_content)

            print(f"✅ Generated sitemap with {len(html_files)} URLs")
            return True

        except Exception as e:
            print(f"❌ Error generating sitemap: {e}")
            return False

    def run_all_fixes(self) -> Dict[str, any]:
        """Run all website fixes"""
        print("🌟 Een Unity Mathematics - Website Fix Suite")
        print("=" * 50)

        results = {
            "gallery_paths_fixed": self.fix_gallery_data_paths(),
            "directories_created": self.create_missing_directories(),
            "navigation_fixed": self.fix_website_navigation(),
            "images_optimized": self.optimize_images(),
            "deployment_fixed": self.fix_deployment_script(),
            "sitemap_generated": self.generate_sitemap(),
            "validation": self.validate_file_existence(),
        }

        print("\n" + "=" * 50)
        print("📊 FIX SUMMARY")
        print("=" * 50)

        for fix_name, result in results.items():
            if isinstance(result, bool):
                status = "✅" if result else "❌"
                print(f"{status} {fix_name}")
            else:
                print(
                    f"📊 {fix_name}: {len(result.get('missing', []))} missing, {len(result.get('broken', []))} broken"
                )

        if self.fixes_applied:
            print(f"\n🔧 Applied {len(self.fixes_applied)} fixes:")
            for fix in self.fixes_applied[:10]:  # Show first 10
                print(f"  • {fix}")
            if len(self.fixes_applied) > 10:
                print(f"  ... and {len(self.fixes_applied) - 10} more")

        if self.issues_found:
            print(f"\n⚠️ Found {len(self.issues_found)} issues:")
            for issue in self.issues_found[:10]:  # Show first 10
                print(f"  • {issue}")
            if len(self.issues_found) > 10:
                print(f"  ... and {len(self.issues_found) - 10} more")

        print("\n🎯 Unity Mathematics website optimization complete!")
        print("The website is now ready for global deployment.")

        return results


def main():
    """Main execution function"""
    fixer = WebsiteFixer()
    results = fixer.run_all_fixes()

    # Return appropriate exit code
    if any(
        isinstance(r, dict) and (r.get("missing") or r.get("broken"))
        for r in results.values()
    ):
        sys.exit(1)  # Issues found
    else:
        sys.exit(0)  # All good


if __name__ == "__main__":
    main()
