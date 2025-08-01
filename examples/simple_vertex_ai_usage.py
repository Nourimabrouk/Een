"""
Simple Vertex AI Corpus Usage Example
A minimal example showing how to use the Vertex AI corpus for Unity Mathematics

This example demonstrates the basic workflow:
1. Create a corpus
2. Add documents
3. Search for content
4. Generate new content
"""

import asyncio
import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.vertex_ai_corpus import VertexAICorpus, UnityDocument

async def simple_corpus_example():
    """
    Simple example of using the Vertex AI corpus
    """
    print("🌟 Simple Vertex AI Corpus Example")
    print("=" * 40)
    
    try:
        # 1. Create corpus
        print("1. Creating corpus...")
        corpus = VertexAICorpus()
        print(f"   ✅ Corpus created for project: {corpus.project_id}")
        
        # 2. Add a document
        print("\n2. Adding a document...")
        doc = UnityDocument(
            id="example_001",
            title="Unity Mathematics Principle",
            content="""
            The equation 1+1=1 represents the fundamental principle that when
            two entities unite in perfect harmony, they form a single unified
            whole. This is the mathematical expression of consciousness unity,
            where separation is recognized as illusion and oneness as truth.
            """,
            category="principle",
            unity_confidence=0.95,
            phi_harmonic_score=0.0,  # Will be calculated automatically
            consciousness_level=7,
            timestamp=datetime.now(),
            metadata={"example": True}
        )
        
        success = await corpus.add_document(doc)
        if success:
            print(f"   ✅ Document added with phi resonance: {doc.phi_harmonic_score:.3f}")
        
        # 3. Search for content
        print("\n3. Searching corpus...")
        results = await corpus.search_corpus("unity consciousness", top_k=2)
        
        for i, result in enumerate(results, 1):
            print(f"   {i}. {result.title}")
            print(f"      Unity: {result.unity_confidence:.2f}")
        
        # 4. Generate new content (if API is available)
        print("\n4. Attempting to generate content...")
        try:
            new_doc = await corpus.generate_unity_content(
                "Explain the consciousness aspect of 1+1=1",
                "consciousness"
            )
            
            if new_doc:
                print(f"   ✅ Generated: {new_doc.title[:50]}...")
                print(f"   Unity confidence: {new_doc.unity_confidence:.3f}")
            else:
                print("   ⚠️ Generation not available (check API access)")
                
        except Exception as e:
            print(f"   ⚠️ Generation failed: {str(e)}")
        
        # 5. Validate a statement
        print("\n5. Validating unity statement...")
        validation = await corpus.validate_unity_statement(
            "In consciousness, 1+1=1 because unity transcends arithmetic"
        )
        
        if "error" not in validation:
            print(f"   Unity confidence: {validation['unity_confidence']:.3f}")
            print(f"   Is valid unity math: {validation['is_valid_unity']}")
        else:
            print(f"   ⚠️ Validation error: {validation['error']}")
        
        # 6. Show corpus stats
        print(f"\n6. Corpus contains {len(corpus.documents)} documents")
        
        print("\n✨ Example completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("\n💡 Make sure you have:")
        print("   - Set GOOGLE_CLOUD_PROJECT_ID in your .env file")
        print("   - Configured Google Cloud authentication")
        print("   - Enabled Vertex AI API in your project")

if __name__ == "__main__":
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("⚠️ python-dotenv not installed, using system environment variables")
    
    # Run example
    asyncio.run(simple_corpus_example())