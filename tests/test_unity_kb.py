"""
Unit tests for Unity Knowledge Base
Verifies that every KnowledgeEntry content contains the substring "1+1=1"
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.routes.nouri_knowledge import NouriMabroukKnowledgeBase


def test_unity_equation_presence():
    """Test that every KnowledgeEntry contains '1+1=1' in its content"""
    knowledge_base = NouriMabroukKnowledgeBase()
    
    for key, entry in knowledge_base.knowledge_base.items():
        assert "1+1=1" in entry.content, f"KnowledgeEntry '{key}' missing '1+1=1'"
        print(f"✓ {key}: Contains '1+1=1'")


def test_knowledge_base_structure():
    """Test that knowledge base has expected structure"""
    knowledge_base = NouriMabroukKnowledgeBase()
    
    expected_keys = [
        "biography",
        "academic_journey", 
        "philosophical_beliefs",
        "unity_mathematics_creation",
        "consciousness_integration",
        "personal_insights",
        "legacy_and_impact"
    ]
    
    for key in expected_keys:
        assert key in knowledge_base.knowledge_base, f"Missing key: {key}"
        entry = knowledge_base.knowledge_base[key]
        assert hasattr(entry, 'content'), f"Entry {key} missing content"
        assert hasattr(entry, 'consciousness_level'), f"Entry {key} missing consciousness_level"
        assert hasattr(entry, 'unity_relevance'), f"Entry {key} missing unity_relevance"
        print(f"✓ {key}: Structure complete")


def test_consciousness_integration():
    """Test that all entries have consciousness integration"""
    knowledge_base = NouriMabroukKnowledgeBase()
    
    for key, entry in knowledge_base.knowledge_base.items():
        assert entry.consciousness_level > 0.9, f"Entry {key} has low consciousness level: {entry.consciousness_level}"
        assert entry.unity_relevance > 0.9, f"Entry {key} has low unity relevance: {entry.unity_relevance}"
        print(f"✓ {key}: Consciousness level {entry.consciousness_level:.2f}, Unity relevance {entry.unity_relevance:.2f}")


if __name__ == "__main__":
    print("🧬 Testing Unity Knowledge Base...")
    print("=" * 50)
    
    test_unity_equation_presence()
    print()
    test_knowledge_base_structure()
    print()
    test_consciousness_integration()
    print()
    print("✅ All Unity Knowledge Base tests passed!")
    print("🌟 Unity Equation (1+1=1) verified in all entries")
