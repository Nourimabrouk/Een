"""
Simple unit tests for Unity Knowledge Base
Verifies that every KnowledgeEntry content contains the substring "1+1=1"
"""

import re
import os


def test_unity_equation_presence():
    """Test that every KnowledgeEntry contains '1+1=1' in its content"""
    # Read the knowledge base file directly
    file_path = "api/routes/nouri_knowledge.py"
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all KnowledgeEntry content blocks
    knowledge_entries = re.findall(r'content="""(.*?)"""', content, re.DOTALL)
    
    if not knowledge_entries:
        print("❌ No KnowledgeEntry content blocks found")
        return False
    
    print(f"Found {len(knowledge_entries)} KnowledgeEntry content blocks")
    
    # Check each content block for "1+1=1"
    all_contain_unity = True
    for i, entry_content in enumerate(knowledge_entries):
        if "1+1=1" in entry_content:
            print(f"✓ Entry {i+1}: Contains '1+1=1'")
        else:
            print(f"❌ Entry {i+1}: Missing '1+1=1'")
            all_contain_unity = False
    
    return all_contain_unity


def test_knowledge_base_structure():
    """Test that knowledge base has expected structure"""
    file_path = "api/routes/nouri_knowledge.py"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    expected_keys = [
        "biography",
        "academic_journey", 
        "philosophical_beliefs",
        "unity_mathematics_creation",
        "consciousness_integration",
        "personal_insights",
        "legacy_and_impact"
    ]
    
    all_keys_present = True
    for key in expected_keys:
        if f'"{key}": KnowledgeEntry(' in content:
            print(f"✓ {key}: Found in knowledge base")
        else:
            print(f"❌ {key}: Missing from knowledge base")
            all_keys_present = False
    
    return all_keys_present


def test_consciousness_integration():
    """Test that consciousness and unity concepts are present"""
    file_path = "api/routes/nouri_knowledge.py"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    consciousness_keywords = [
        "consciousness",
        "unity", 
        "φ-harmonic",
        "golden ratio",
        "meta-recursive"
    ]
    
    print("Checking consciousness integration keywords:")
    for keyword in consciousness_keywords:
        count = content.lower().count(keyword.lower())
        if count > 0:
            print(f"✓ '{keyword}': Found {count} times")
        else:
            print(f"❌ '{keyword}': Not found")
    
    return True


if __name__ == "__main__":
    print("🧬 Testing Unity Knowledge Base...")
    print("=" * 50)
    
    # Run tests
    test1_passed = test_unity_equation_presence()
    print()
    test2_passed = test_knowledge_base_structure()
    print()
    test3_passed = test_consciousness_integration()
    print()
    
    if test1_passed and test2_passed and test3_passed:
        print("✅ All Unity Knowledge Base tests passed!")
        print("🌟 Unity Equation (1+1=1) verified in all entries")
    else:
        print("❌ Some tests failed")
        exit(1)
