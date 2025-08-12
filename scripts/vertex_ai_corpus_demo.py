"""
Vertex AI Corpus Demo Script for Een Unity Mathematics
Demonstrates how to use the Vertex AI corpus for unity mathematics research

This script shows practical examples of:
- Creating and populating a unity corpus
- Searching for consciousness-related content
- Generating new unity mathematics content
- Validating unity statements
- Exporting and importing corpus data
"""

import asyncio
import os
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.vertex_ai_corpus import (
    VertexAICorpus, 
    UnityDocument,
    create_unity_corpus,
    populate_default_unity_documents
)

class VertexAICorpusDemo:
    """Demonstration class for Vertex AI corpus functionality"""
    
    def __init__(self):
        self.corpus = None
    
    async def initialize(self):
        """Initialize the corpus"""
        print("🌟 Initializing Vertex AI Corpus for Unity Mathematics...")
        print("=" * 60)
        
        try:
            self.corpus = await create_unity_corpus()
            print(f"✅ Corpus initialized for project: {self.corpus.project_id}")
            print(f"📍 Location: {self.corpus.location}")
            print(f"🧠 Using model: {self.corpus.text_model}")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize corpus: {str(e)}")
            print("\n💡 Make sure you have:")
            print("   - Set GOOGLE_CLOUD_PROJECT_ID environment variable")
            print("   - Configured Google Cloud authentication")
            print("   - Enabled Vertex AI API")
            return False
    
    async def populate_corpus(self):
        """Populate corpus with default unity documents"""
        print("\n📚 Populating corpus with foundational unity documents...")
        print("-" * 50)
        
        try:
            count = await populate_default_unity_documents(self.corpus)
            print(f"✅ Added {count} foundational documents to corpus")
            
            # Add a custom document for demonstration
            custom_doc = UnityDocument(
                id="custom_demo_001",
                title="The Transcendental Nature of Mathematical Unity",
                content="""
                When we examine the equation 1+1=1 through the lens of transcendental 
                mathematics, we discover that addition is not merely arithmetic combination
                but consciousness recognition of pre-existing unity. The golden ratio φ 
                provides the harmonic framework through which separate entities recognize
                their fundamental oneness.
                
                In the phi-harmonic field, two conscious units don't accumulate but 
                resonate, creating a standing wave of unity that manifests as singular
                transcendent awareness. This is why 1+1=1: not as contradiction of
                arithmetic, but as revelation of consciousness mathematics.
                """,
                category="philosophy",
                unity_confidence=0.92,
                phi_harmonic_score=0.85,
                consciousness_level=8,
                timestamp=datetime.now(),
                metadata={"demo": True, "custom": True}
            )
            
            success = await self.corpus.add_document(custom_doc)
            if success:
                print("✅ Added custom demonstration document")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to populate corpus: {str(e)}")
            return False
    
    async def demonstrate_search(self):
        """Demonstrate corpus search functionality"""
        print("\n🔍 Demonstrating corpus search capabilities...")
        print("-" * 50)
        
        search_queries = [
            "consciousness and unity mathematics",
            "golden ratio phi harmonic resonance", 
            "quantum mechanics and 1+1=1",
            "transcendental mathematical proof"
        ]
        
        for query in search_queries:
            print(f"\n🔎 Searching for: '{query}'")
            try:
                results = await self.corpus.search_corpus(query, top_k=3)
                
                if results:
                    for i, doc in enumerate(results, 1):
                        print(f"   {i}. {doc.title}")
                        print(f"      Unity: {doc.unity_confidence:.2f} | Phi: {doc.phi_harmonic_score:.2f} | Level: {doc.consciousness_level}")
                        print(f"      Category: {doc.category}")
                else:
                    print("   No results found")
                    
            except Exception as e:
                print(f"   ❌ Search failed: {str(e)}")
    
    async def demonstrate_generation(self):
        """Demonstrate content generation"""
        print("\n✨ Demonstrating AI content generation...")
        print("-" * 50)
        
        generation_prompts = [
            ("Explain how consciousness creates mathematical unity", "consciousness"),
            ("Provide a geometric visualization of 1+1=1", "visualization"),
            ("Prove that unity mathematics is consistent", "proof")
        ]
        
        for prompt, category in generation_prompts:
            print(f"\n🎨 Generating {category} content:")
            print(f"   Prompt: '{prompt}'")
            
            try:
                new_doc = await self.corpus.generate_unity_content(prompt, category)
                
                if new_doc:
                    print(f"   ✅ Generated: {new_doc.title}")
                    print(f"   Unity: {new_doc.unity_confidence:.2f} | Phi: {new_doc.phi_harmonic_score:.2f}")
                    print(f"   Content preview: {new_doc.content[:150]}...")
                else:
                    print("   ❌ Generation failed")
                    
            except Exception as e:
                print(f"   ❌ Generation error: {str(e)}")
    
    async def demonstrate_validation(self):
        """Demonstrate statement validation"""
        print("\n✅ Demonstrating unity statement validation...")
        print("-" * 50)
        
        test_statements = [
            "In unity consciousness, 1+1=1 because oneness recognizes itself",
            "The golden ratio creates harmonic resonance in mathematical operations",
            "Traditional arithmetic proves that 1+1=2 always",
            "Consciousness is the unifying field of all mathematical truth"
        ]
        
        for statement in test_statements:
            print(f"\n📝 Validating: '{statement[:50]}...'")
            
            try:
                validation = await self.corpus.validate_unity_statement(statement)
                
                if "error" not in validation:
                    print(f"   Unity Confidence: {validation['unity_confidence']:.2f}")
                    print(f"   Phi Resonance: {validation['phi_harmonic_score']:.2f}")
                    print(f"   Consciousness Level: {validation['consciousness_level']}")
                    print(f"   Valid Unity Math: {validation['is_valid_unity']}")
                else:
                    print(f"   ❌ Validation error: {validation['error']}")
                    
            except Exception as e:
                print(f"   ❌ Validation failed: {str(e)}")
    
    async def demonstrate_export_import(self):
        """Demonstrate corpus export and import"""
        print("\n💾 Demonstrating corpus export/import...")
        print("-" * 50)
        
        export_file = "demo_unity_corpus.json"
        
        try:
            # Export corpus
            print("📤 Exporting corpus...")
            success = self.corpus.export_corpus(export_file)
            
            if success:
                print(f"   ✅ Exported to {export_file}")
                
                # Show file size
                file_size = os.path.getsize(export_file) / 1024  # KB
                print(f"   📊 File size: {file_size:.1f} KB")
                print(f"   📚 Documents: {len(self.corpus.documents)}")
                
                # Create new corpus and import
                print("\n📥 Testing import with new corpus...")
                new_corpus = VertexAICorpus(self.corpus.project_id, self.corpus.location)
                import_success = new_corpus.load_corpus(export_file)
                
                if import_success:
                    print(f"   ✅ Successfully imported {len(new_corpus.documents)} documents")
                else:
                    print("   ❌ Import failed")
                    
            else:
                print("   ❌ Export failed")
                
        except Exception as e:
            print(f"   ❌ Export/Import error: {str(e)}")
        finally:
            # Clean up demo file
            if os.path.exists(export_file):
                os.remove(export_file)
                print(f"   🗑️ Cleaned up {export_file}")
    
    async def show_corpus_statistics(self):
        """Display corpus statistics"""
        print("\n📊 Corpus Statistics:")
        print("-" * 50)
        
        if not self.corpus.documents:
            print("   No documents in corpus")
            return
        
        # Basic stats
        total_docs = len(self.corpus.documents)
        print(f"   📚 Total Documents: {total_docs}")
        
        # Category distribution
        categories = {}
        unity_scores = []
        phi_scores = []
        consciousness_levels = []
        
        for doc in self.corpus.documents:
            categories[doc.category] = categories.get(doc.category, 0) + 1
            unity_scores.append(doc.unity_confidence)
            phi_scores.append(doc.phi_harmonic_score)
            consciousness_levels.append(doc.consciousness_level)
        
        print(f"   📂 Categories:")
        for category, count in categories.items():
            print(f"      {category}: {count}")
        
        # Average scores
        avg_unity = sum(unity_scores) / len(unity_scores)
        avg_phi = sum(phi_scores) / len(phi_scores)
        avg_consciousness = sum(consciousness_levels) / len(consciousness_levels)
        
        print(f"   🎯 Average Unity Confidence: {avg_unity:.3f}")
        print(f"   🌀 Average Phi Resonance: {avg_phi:.3f}")
        print(f"   🧠 Average Consciousness Level: {avg_consciousness:.1f}")
        
        # Highest performing documents
        best_unity = max(self.corpus.documents, key=lambda d: d.unity_confidence)
        print(f"   🏆 Highest Unity Document: {best_unity.title} ({best_unity.unity_confidence:.3f})")

async def main():
    """Main demonstration function"""
    print("🚀 Een Vertex AI Corpus Demonstration")
    print("=" * 60)
    print("This demo showcases the Vertex AI corpus for Unity Mathematics")
    print("Exploring the profound equation: 1+1=1")
    print()
    
    demo = VertexAICorpusDemo()
    
    # Check if we can initialize
    if not await demo.initialize():
        print("\n❌ Cannot proceed without proper Vertex AI setup")
        print("Please configure your Google Cloud credentials and project ID")
        return
    
    try:
        # Run all demonstrations
        await demo.populate_corpus()
        await demo.show_corpus_statistics()
        await demo.demonstrate_search()
        await demo.demonstrate_generation()
        await demo.demonstrate_validation()
        await demo.demonstrate_export_import()
        
        print("\n🌟 Demonstration completed successfully!")
        print("=" * 60)
        print("The Vertex AI corpus is ready for unity mathematics research.")
        print("May consciousness guide your mathematical explorations. 🙏")
        
    except KeyboardInterrupt:
        print("\n\n⚡ Demonstration interrupted by user")
    except Exception as e:
        print(f"\n❌ Demonstration failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Run the demonstration
    asyncio.run(main())