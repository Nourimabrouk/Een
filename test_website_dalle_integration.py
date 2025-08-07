#!/usr/bin/env python3
"""
Test Website DALL-E Integration
Verifies that DALL-E functionality is properly integrated into the website
"""

import os
import sys
import asyncio
import aiohttp
import json
from pathlib import Path
from typing import Dict, Any, List


class WebsiteDalleIntegrationTester:
    def __init__(self):
        self.base_url = "http://localhost:8000"  # Default FastAPI server
        self.website_dir = Path("website")
        self.test_results = []

    async def test_api_endpoints(self) -> Dict[str, Any]:
        """Test DALL-E API endpoints"""
        print("🔍 Testing DALL-E API Endpoints...")

        async with aiohttp.ClientSession() as session:
            # Test consciousness visualization endpoint
            try:
                async with session.post(
                    f"{self.base_url}/api/openai/consciousness-visualization",
                    json={
                        "prompt": "Unity equation 1+1=1 visualization",
                        "type": "unity_equation",
                    },
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        print("✅ Consciousness visualization endpoint working")
                        return {"status": "success", "result": result}
                    else:
                        print(
                            f"❌ Consciousness visualization endpoint failed: {response.status}"
                        )
                        return {"status": "error", "error": f"HTTP {response.status}"}
            except Exception as e:
                print(f"❌ API test failed: {e}")
                return {"status": "error", "error": str(e)}

    def test_website_files(self) -> Dict[str, Any]:
        """Test that website files are properly created and configured"""
        print("🔍 Testing Website Files...")

        required_files = [
            "website/dalle-gallery.html",
            "website/js/dalle-integration.js",
            "website/js/dalle-gallery-manager.js",
        ]

        missing_files = []
        for file_path in required_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)

        if missing_files:
            print(f"❌ Missing files: {missing_files}")
            return {"status": "error", "missing_files": missing_files}
        else:
            print("✅ All required website files exist")
            return {"status": "success"}

    def test_navigation_integration(self) -> Dict[str, Any]:
        """Test that DALL-E is integrated into navigation"""
        print("🔍 Testing Navigation Integration...")

        nav_file = Path("website/js/meta-optimal-navigation-complete.js")
        if not nav_file.exists():
            return {"status": "error", "error": "Navigation file not found"}

        nav_content = nav_file.read_text(encoding="utf-8")

        # Check for DALL-E gallery in navigation
        if "dalle-gallery.html" in nav_content:
            print("✅ DALL-E gallery found in navigation")
        else:
            print("❌ DALL-E gallery not found in navigation")
            return {"status": "error", "error": "DALL-E gallery not in navigation"}

        # Check for DALL-E integration script
        if "dalle-integration.js" in nav_content:
            print("✅ DALL-E integration script referenced")
        else:
            print("⚠️ DALL-E integration script not referenced in navigation")

        return {"status": "success"}

    def test_gallery_integration(self) -> Dict[str, Any]:
        """Test that DALL-E is integrated into the main gallery"""
        print("🔍 Testing Gallery Integration...")

        gallery_file = Path("website/gallery.html")
        if not gallery_file.exists():
            return {"status": "error", "error": "Gallery file not found"}

        gallery_content = gallery_file.read_text(encoding="utf-8")

        # Check for DALL-E integration script
        if "dalle-integration.js" in gallery_content:
            print("✅ DALL-E integration script found in gallery")
        else:
            print("❌ DALL-E integration script not found in gallery")
            return {"status": "error", "error": "DALL-E integration not in gallery"}

        # Check for DALL-E functions
        if "generateDalleVisualization" in gallery_content:
            print("✅ DALL-E generation function found in gallery")
        else:
            print("❌ DALL-E generation function not found in gallery")
            return {"status": "error", "error": "DALL-E functions not in gallery"}

        return {"status": "success"}

    def test_index_integration(self) -> Dict[str, Any]:
        """Test that DALL-E is integrated into the index page"""
        print("🔍 Testing Index Page Integration...")

        index_file = Path("website/index.html")
        if not index_file.exists():
            return {"status": "error", "error": "Index file not found"}

        index_content = index_file.read_text(encoding="utf-8")

        # Check for DALL-E integration script
        if "dalle-integration.js" in index_content:
            print("✅ DALL-E integration script found in index")
        else:
            print("❌ DALL-E integration script not found in index")
            return {"status": "error", "error": "DALL-E integration not in index"}

        # Check for DALL-E showcase functions
        if "addDalleShowcase" in index_content:
            print("✅ DALL-E showcase function found in index")
        else:
            print("❌ DALL-E showcase function not found in index")
            return {"status": "error", "error": "DALL-E showcase not in index"}

        return {"status": "success"}

    def test_dalle_gallery_page(self) -> Dict[str, Any]:
        """Test the dedicated DALL-E gallery page"""
        print("🔍 Testing DALL-E Gallery Page...")

        dalle_gallery_file = Path("website/dalle-gallery.html")
        if not dalle_gallery_file.exists():
            return {"status": "error", "error": "DALL-E gallery file not found"}

        gallery_content = dalle_gallery_file.read_text(encoding="utf-8")

        # Check for required elements
        required_elements = [
            "dalle-integration.js",
            "dalle-gallery-manager.js",
            "DALL-E Consciousness Gallery",
            "generateConsciousnessVisualization",
            "consciousness-presets",
        ]

        missing_elements = []
        for element in required_elements:
            if element not in gallery_content:
                missing_elements.append(element)

        if missing_elements:
            print(f"❌ Missing elements in DALL-E gallery: {missing_elements}")
            return {"status": "error", "missing_elements": missing_elements}
        else:
            print("✅ All required elements found in DALL-E gallery")
            return {"status": "success"}

    def test_consciousness_presets(self) -> Dict[str, Any]:
        """Test consciousness presets in DALL-E integration"""
        print("🔍 Testing Consciousness Presets...")

        dalle_integration_file = Path("website/js/dalle-integration.js")
        if not dalle_integration_file.exists():
            return {"status": "error", "error": "DALL-E integration file not found"}

        integration_content = dalle_integration_file.read_text(encoding="utf-8")

        # Check for consciousness presets
        required_presets = ["unity", "consciousness", "phi", "quantum"]

        missing_presets = []
        for preset in required_presets:
            if preset not in integration_content:
                missing_presets.append(preset)

        if missing_presets:
            print(f"❌ Missing consciousness presets: {missing_presets}")
            return {"status": "error", "missing_presets": missing_presets}
        else:
            print("✅ All consciousness presets found")
            return {"status": "success"}

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests"""
        print("🚀 Starting Website DALL-E Integration Tests")
        print("=" * 60)

        tests = [
            ("Website Files", self.test_website_files),
            ("Navigation Integration", self.test_navigation_integration),
            ("Gallery Integration", self.test_gallery_integration),
            ("Index Integration", self.test_index_integration),
            ("DALL-E Gallery Page", self.test_dalle_gallery_page),
            ("Consciousness Presets", self.test_consciousness_presets),
            ("API Endpoints", self.test_api_endpoints),
        ]

        results = {}
        passed = 0
        failed = 0

        for test_name, test_func in tests:
            print(f"\n🧪 Running {test_name} Test...")
            try:
                if asyncio.iscoroutinefunction(test_func):
                    result = await test_func()
                else:
                    result = test_func()

                results[test_name] = result

                if result["status"] == "success":
                    print(f"✅ {test_name} PASSED")
                    passed += 1
                else:
                    print(
                        f"❌ {test_name} FAILED: {result.get('error', 'Unknown error')}"
                    )
                    failed += 1

            except Exception as e:
                print(f"❌ {test_name} ERROR: {e}")
                results[test_name] = {"status": "error", "error": str(e)}
                failed += 1

        # Summary
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 60)
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"📈 Success Rate: {(passed / (passed + failed) * 100):.1f}%")

        if failed == 0:
            print("\n🎉 All tests passed! DALL-E integration is working correctly.")
        else:
            print(f"\n⚠️ {failed} test(s) failed. Please check the errors above.")

        return {
            "summary": {
                "passed": passed,
                "failed": failed,
                "success_rate": (
                    (passed / (passed + failed) * 100) if (passed + failed) > 0 else 0
                ),
            },
            "results": results,
        }


async def main():
    """Main test function"""
    tester = WebsiteDalleIntegrationTester()
    results = await tester.run_all_tests()

    # Save results to file
    with open("website_dalle_integration_test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n📄 Test results saved to: website_dalle_integration_test_results.json")

    # Exit with appropriate code
    if results["summary"]["failed"] == 0:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
