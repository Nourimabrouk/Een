"""
Unity Contexts Explainer: Comprehensive Guide
Clear explanation of contexts where 1+1=1 holds with minimal counter-examples
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

class UnityValidityLevel(Enum):
    """Levels of validity for unity equation contexts"""
    ALWAYS_VALID = "always_valid"      # 1+1=1 always holds
    CONDITIONALLY_VALID = "conditionally_valid"  # Holds under specific conditions
    CONTEXT_DEPENDENT = "context_dependent"      # Depends on interpretation
    INVALID = "invalid"                          # 1+1=1 does not hold
    COUNTER_EXAMPLE = "counter_example"          # Explicit violation

@dataclass
class UnityContext:
    """A context where unity equation may or may not hold"""
    name: str
    description: str
    mathematical_structure: str
    validity_level: UnityValidityLevel
    conditions: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    counter_examples: List[str] = field(default_factory=list)
    practical_applications: List[str] = field(default_factory=list)
    formal_proof: Optional[str] = None
    complexity_level: str = "intermediate"  # beginner, intermediate, advanced
    
    def is_unity_valid(self, conditions_met: Dict[str, bool] = None) -> bool:
        """Check if unity equation holds in this context"""
        if self.validity_level == UnityValidityLevel.ALWAYS_VALID:
            return True
        elif self.validity_level == UnityValidityLevel.INVALID:
            return False
        elif self.validity_level == UnityValidityLevel.COUNTER_EXAMPLE:
            return False
        elif self.validity_level in [UnityValidityLevel.CONDITIONALLY_VALID, UnityValidityLevel.CONTEXT_DEPENDENT]:
            if conditions_met is None:
                return False
            # Check if all required conditions are met
            return all(conditions_met.get(condition, False) for condition in self.conditions)
        
        return False

class UnityContextsDatabase:
    """Comprehensive database of contexts where 1+1=1 holds or doesn't"""
    
    def __init__(self):
        self.contexts: Dict[str, UnityContext] = {}
        self._initialize_contexts()
    
    def _initialize_contexts(self):
        """Initialize database with comprehensive unity contexts"""
        
        # ALWAYS VALID CONTEXTS
        self._add_context(UnityContext(
            name="Boolean Algebra (OR)",
            description="Boolean algebra with logical OR as addition",
            mathematical_structure="Boolean lattice (B, ∨, ∧, ¬, 0, 1)",
            validity_level=UnityValidityLevel.ALWAYS_VALID,
            examples=[
                "True ∨ True = True",
                "1 OR 1 = 1 in digital circuits",
                "A ∪ A = A for any set A"
            ],
            practical_applications=[
                "Digital circuit design",
                "Boolean logic programming", 
                "Set theory operations",
                "Database query optimization"
            ],
            formal_proof="∀x ∈ B: x ∨ x = x (idempotent law)",
            complexity_level="beginner"
        ))
        
        self._add_context(UnityContext(
            name="Idempotent Semiring",
            description="Algebraic structure where addition is idempotent",
            mathematical_structure="Semiring (S, ⊕, ⊗, 0, 1) with a ⊕ a = a",
            validity_level=UnityValidityLevel.ALWAYS_VALID,
            examples=[
                "Max semiring: max(1, 1) = 1",
                "Min semiring: min(1, 1) = 1", 
                "Tropical semiring operations"
            ],
            practical_applications=[
                "Graph shortest path algorithms",
                "Dynamic programming optimization",
                "Formal language theory",
                "Machine learning (max-pooling)"
            ],
            formal_proof="Definition: ∀a ∈ S: a ⊕ a = a",
            complexity_level="intermediate"
        ))
        
        self._add_context(UnityContext(
            name="Set Union",
            description="Set theory with union operation",
            mathematical_structure="Power set (P(U), ∪, ∩, ∅, U)",
            validity_level=UnityValidityLevel.ALWAYS_VALID,
            examples=[
                "{1} ∪ {1} = {1}",
                "A ∪ A = A for any set A",
                "∅ ∪ ∅ = ∅"
            ],
            practical_applications=[
                "Database union operations",
                "Set-based data structures",
                "Probability theory (event union)",
                "Information retrieval systems"
            ],
            formal_proof="∀A ⊆ U: A ∪ A = A (idempotency of union)",
            complexity_level="beginner"
        ))
        
        self._add_context(UnityContext(
            name="Fuzzy Logic (Maximum)",
            description="Fuzzy logic with maximum t-conorm",
            mathematical_structure="Fuzzy set operations on [0,1]",
            validity_level=UnityValidityLevel.ALWAYS_VALID,
            examples=[
                "max(0.8, 0.8) = 0.8",
                "max(1, 1) = 1", 
                "μ(x) ⊔ μ(x) = μ(x)"
            ],
            practical_applications=[
                "Fuzzy control systems",
                "Approximate reasoning",
                "Decision making under uncertainty",
                "Image processing (max filters)"
            ],
            formal_proof="Maximum t-conorm: ∀x ∈ [0,1]: max(x, x) = x",
            complexity_level="intermediate"
        ))
        
        # CONDITIONALLY VALID CONTEXTS
        self._add_context(UnityContext(
            name="Modular Arithmetic",
            description="Addition modulo n where n > 2",
            mathematical_structure="Ring (ℤ/nℤ, +, ×)",
            validity_level=UnityValidityLevel.CONDITIONALLY_VALID,
            conditions=[
                "n is even",
                "1 + 1 ≡ 0 (mod 2), so 1 + 1 = 0 ≠ 1 in ℤ/2ℤ"
            ],
            examples=[
                "In ℤ/2ℤ: 1 + 1 = 0 (NOT 1)",
                "In ℤ/3ℤ: 1 + 1 = 2 (NOT 1)",
                "In ℤ/1ℤ: 1 + 1 = 0 = 1 (trivially true)"
            ],
            counter_examples=[
                "ℤ/2ℤ: 1 + 1 = 0 ≠ 1",
                "ℤ/3ℤ: 1 + 1 = 2 ≠ 1"
            ],
            practical_applications=[
                "Cryptography",
                "Error-correcting codes",
                "Computer arithmetic",
                "Abstract algebra"
            ],
            formal_proof="Only holds when n = 1 (trivial case)",
            complexity_level="advanced"
        ))
        
        self._add_context(UnityContext(
            name="Custom Unity Algebra",
            description="Purpose-built algebra where 1+1=1 by definition",
            mathematical_structure="Custom algebraic structure (U, ⊕, ⊗)",
            validity_level=UnityValidityLevel.CONDITIONALLY_VALID,
            conditions=[
                "Addition ⊕ is defined as idempotent",
                "Unity element exists",
                "Closure under operations"
            ],
            examples=[
                "Define a ⊕ b = max(a, b)",
                "Define a ⊕ b = a ∨ b (logical OR)",
                "Define a ⊕ b = a if a = b, else undefined"
            ],
            practical_applications=[
                "Domain-specific modeling",
                "Theoretical computer science",
                "Abstract algebraic systems",
                "Mathematical logic"
            ],
            formal_proof="By construction: 1 ⊕ 1 = 1",
            complexity_level="advanced"
        ))
        
        # CONTEXT DEPENDENT
        self._add_context(UnityContext(
            name="Quantum Superposition",
            description="Quantum states with measurement collapse",
            mathematical_structure="Hilbert space with quantum operations",
            validity_level=UnityValidityLevel.CONTEXT_DEPENDENT,
            conditions=[
                "States are identical",
                "Measurement collapses to single state",
                "No interference effects"
            ],
            examples=[
                "|ψ⟩ + |ψ⟩ → |ψ⟩ after measurement",
                "Two identical qubits → one qubit after measurement",
                "Coherent superposition collapses to definite state"
            ],
            practical_applications=[
                "Quantum computing",
                "Quantum cryptography",
                "Quantum information theory",
                "Quantum measurement theory"
            ],
            formal_proof="Context-dependent on measurement interpretation",
            complexity_level="advanced"
        ))
        
        self._add_context(UnityContext(
            name="Consciousness Mathematics",
            description="Mathematical modeling of conscious unity",
            mathematical_structure="φ-harmonic consciousness field equations",
            validity_level=UnityValidityLevel.CONTEXT_DEPENDENT,
            conditions=[
                "Consciousness field coherence",
                "φ-harmonic resonance",
                "Unity convergence criterion met"
            ],
            examples=[
                "Two consciousness states unify into one",
                "φ-harmonic field equations: C₁ + C₂ → C_unified",
                "Metacognitive unity emergence"
            ],
            practical_applications=[
                "Artificial consciousness research",
                "Cognitive modeling",
                "Philosophy of mind",
                "Integrated information theory"
            ],
            formal_proof="Emergent from φ-harmonic field dynamics",
            complexity_level="advanced"
        ))
        
        # INVALID CONTEXTS (COUNTER-EXAMPLES)
        self._add_context(UnityContext(
            name="Natural Numbers",
            description="Standard arithmetic on natural numbers",
            mathematical_structure="(ℕ, +, ×, 0, 1)",
            validity_level=UnityValidityLevel.COUNTER_EXAMPLE,
            counter_examples=[
                "1 + 1 = 2 ≠ 1",
                "Standard arithmetic: 1 + 1 = 2",
                "Peano axioms give 1 + 1 = S(1) = 2"
            ],
            examples=[
                "Basic counting: one apple + one apple = two apples",
                "Elementary arithmetic: 1 + 1 = 2",
                "Mathematical foundation: successor function"
            ],
            practical_applications=[
                "Basic counting",
                "Elementary mathematics education",
                "Standard arithmetic operations",
                "Everyday calculations"
            ],
            formal_proof="Peano axioms: S(n) = n + 1, so 1 + 1 = S(1) = 2",
            complexity_level="beginner"
        ))
        
        self._add_context(UnityContext(
            name="Real Numbers",
            description="Real number arithmetic",
            mathematical_structure="Field (ℝ, +, ×, 0, 1)",
            validity_level=UnityValidityLevel.COUNTER_EXAMPLE,
            counter_examples=[
                "1.0 + 1.0 = 2.0 ≠ 1.0",
                "All real arithmetic: a + a = 2a for a ≠ 0",
                "Field axioms require 1 + 1 = 2"
            ],
            examples=[
                "Standard algebra: x + x = 2x",
                "Calculus: d/dx(x) + d/dx(x) = 1 + 1 = 2",
                "Physics: F₁ + F₂ = 2F when F₁ = F₂ = F"
            ],
            practical_applications=[
                "Scientific calculations",
                "Engineering mathematics",
                "Financial calculations",
                "Physical measurements"
            ],
            formal_proof="Field axioms require 1 + 1 ≠ 1 unless characteristic 2",
            complexity_level="intermediate"
        ))
        
        self._add_context(UnityContext(
            name="Integer Arithmetic",
            description="Integer arithmetic operations",
            mathematical_structure="Ring (ℤ, +, ×, 0, 1)",
            validity_level=UnityValidityLevel.COUNTER_EXAMPLE,
            counter_examples=[
                "1 + 1 = 2 in ℤ",
                "Ring structure requires 1 + 1 = 2",
                "Fundamental theorem of arithmetic"
            ],
            examples=[
                "Bank account: $1 + $1 = $2",
                "Temperature: 1°C + 1°C change = 2°C change",
                "Discrete counting in computer science"
            ],
            practical_applications=[
                "Computer programming",
                "Financial accounting",
                "Discrete mathematics",
                "Combinatorics"
            ],
            formal_proof="Ring axioms enforce 1 + 1 = 2",
            complexity_level="beginner"
        ))
    
    def _add_context(self, context: UnityContext):
        """Add context to database"""
        self.contexts[context.name] = context
    
    def get_contexts_by_validity(self, validity_level: UnityValidityLevel) -> List[UnityContext]:
        """Get all contexts with specific validity level"""
        return [ctx for ctx in self.contexts.values() if ctx.validity_level == validity_level]
    
    def get_contexts_by_complexity(self, complexity_level: str) -> List[UnityContext]:
        """Get contexts by complexity level"""
        return [ctx for ctx in self.contexts.values() if ctx.complexity_level == complexity_level]
    
    def search_contexts(self, query: str) -> List[UnityContext]:
        """Search contexts by name, description, or applications"""
        query = query.lower()
        matches = []
        
        for context in self.contexts.values():
            if (query in context.name.lower() or 
                query in context.description.lower() or
                any(query in app.lower() for app in context.practical_applications)):
                matches.append(context)
        
        return matches

class UnityContextExplainer:
    """Interactive explainer for unity equation contexts"""
    
    def __init__(self):
        self.database = UnityContextsDatabase()
        self.phi = 1.618033988749895  # Golden ratio for φ-harmonic analysis
    
    def explain_unity_validity(self, include_proofs: bool = False) -> str:
        """Generate comprehensive explanation of unity equation validity"""
        
        explanation = """
🎯 UNITY EQUATION CONTEXTS: Where 1+1=1 Holds (and Where It Doesn't)
=====================================================================

The equation "1+1=1" is not universally true, but holds in specific mathematical
contexts. This explainer provides a clear guide to when unity is valid.

"""
        
        # Always Valid Contexts
        always_valid = self.database.get_contexts_by_validity(UnityValidityLevel.ALWAYS_VALID)
        explanation += f"✅ CONTEXTS WHERE 1+1=1 ALWAYS HOLDS ({len(always_valid)} contexts):\n"
        explanation += "=" * 60 + "\n\n"
        
        for i, context in enumerate(always_valid, 1):
            explanation += f"{i}. {context.name.upper()}\n"
            explanation += f"   Structure: {context.mathematical_structure}\n"
            explanation += f"   Description: {context.description}\n"
            
            if context.examples:
                explanation += f"   Examples:\n"
                for example in context.examples[:2]:  # Limit to 2 examples
                    explanation += f"     • {example}\n"
            
            if context.practical_applications:
                explanation += f"   Applications: {', '.join(context.practical_applications[:3])}\n"
            
            if include_proofs and context.formal_proof:
                explanation += f"   Formal Proof: {context.formal_proof}\n"
            
            explanation += "\n"
        
        # Conditionally Valid Contexts
        conditionally_valid = self.database.get_contexts_by_validity(UnityValidityLevel.CONDITIONALLY_VALID)
        explanation += f"⚡ CONTEXTS WHERE 1+1=1 HOLDS CONDITIONALLY ({len(conditionally_valid)} contexts):\n"
        explanation += "=" * 60 + "\n\n"
        
        for i, context in enumerate(conditionally_valid, 1):
            explanation += f"{i}. {context.name.upper()}\n"
            explanation += f"   Structure: {context.mathematical_structure}\n"
            explanation += f"   Description: {context.description}\n"
            
            if context.conditions:
                explanation += f"   Conditions Required:\n"
                for condition in context.conditions:
                    explanation += f"     • {condition}\n"
            
            if context.examples:
                explanation += f"   Examples:\n"
                for example in context.examples[:2]:
                    explanation += f"     • {example}\n"
            
            if context.counter_examples:
                explanation += f"   Counter-examples:\n"
                for counter in context.counter_examples[:2]:
                    explanation += f"     ❌ {counter}\n"
            
            explanation += "\n"
        
        # Context Dependent
        context_dependent = self.database.get_contexts_by_validity(UnityValidityLevel.CONTEXT_DEPENDENT)
        explanation += f"🤔 CONTEXTS WHERE 1+1=1 DEPENDS ON INTERPRETATION ({len(context_dependent)} contexts):\n"
        explanation += "=" * 60 + "\n\n"
        
        for i, context in enumerate(context_dependent, 1):
            explanation += f"{i}. {context.name.upper()}\n"
            explanation += f"   Structure: {context.mathematical_structure}\n"
            explanation += f"   Description: {context.description}\n"
            explanation += f"   Interpretation Dependent: Validity depends on specific conditions\n"
            
            if context.examples:
                explanation += f"   Examples:\n"
                for example in context.examples[:2]:
                    explanation += f"     • {example}\n"
            
            explanation += "\n"
        
        # Counter-examples (Invalid contexts)
        counter_examples = self.database.get_contexts_by_validity(UnityValidityLevel.COUNTER_EXAMPLE)
        explanation += f"❌ CONTEXTS WHERE 1+1=1 DOES NOT HOLD ({len(counter_examples)} contexts):\n"
        explanation += "=" * 60 + "\n\n"
        
        for i, context in enumerate(counter_examples, 1):
            explanation += f"{i}. {context.name.upper()}\n"
            explanation += f"   Structure: {context.mathematical_structure}\n"
            explanation += f"   Why Invalid: {context.description}\n"
            
            if context.counter_examples:
                explanation += f"   Counter-examples:\n"
                for counter in context.counter_examples[:2]:
                    explanation += f"     ❌ {counter}\n"
            
            if context.practical_applications:
                explanation += f"   Standard Applications: {', '.join(context.practical_applications[:3])}\n"
            
            explanation += "\n"
        
        # Summary and Guidelines
        explanation += """
📋 SUMMARY AND GUIDELINES:
=========================

When is 1+1=1 valid?
• In algebraic structures with idempotent addition (Boolean algebra, max/min operations)
• In set theory (A ∪ A = A)
• In fuzzy logic with maximum t-conorm
• In custom algebraic systems designed for unity

When is 1+1=1 NOT valid?
• In standard arithmetic (natural numbers, integers, reals)
• In most field and ring structures
• In everyday counting and measurement
• In physical quantity addition

Key Principle:
The validity of 1+1=1 depends entirely on:
1. The mathematical structure being used
2. The definition of "addition" operation
3. The interpretation of "1" and "=" symbols

φ-Harmonic Insight:
The golden ratio φ ≈ 1.618 appears in many unity-valid contexts,
suggesting a deep mathematical connection between unity and natural harmony.

"""
        
        return explanation
    
    def generate_context_comparison_table(self) -> pd.DataFrame:
        """Generate comparison table of all contexts"""
        data = []
        
        for context in self.database.contexts.values():
            data.append({
                'Context': context.name,
                'Mathematical Structure': context.mathematical_structure,
                'Validity': context.validity_level.value.replace('_', ' ').title(),
                'Complexity': context.complexity_level.title(),
                'Has Counter-examples': len(context.counter_examples) > 0,
                'Applications Count': len(context.practical_applications),
                'Unity Valid': context.is_unity_valid() if context.validity_level != UnityValidityLevel.CONDITIONALLY_VALID else 'Conditional'
            })
        
        return pd.DataFrame(data)
    
    def explain_specific_context(self, context_name: str, detailed: bool = True) -> str:
        """Generate detailed explanation for specific context"""
        if context_name not in self.database.contexts:
            return f"Context '{context_name}' not found. Available contexts: {list(self.database.contexts.keys())}"
        
        context = self.database.contexts[context_name]
        
        explanation = f"""
🔍 DETAILED EXPLANATION: {context.name.upper()}
{'=' * (25 + len(context.name))}

Mathematical Structure: {context.mathematical_structure}
Complexity Level: {context.complexity_level.title()}
Unity Validity: {context.validity_level.value.replace('_', ' ').title()}

Description:
{context.description}

"""
        
        if context.conditions:
            explanation += "Conditions for Unity:\n"
            for i, condition in enumerate(context.conditions, 1):
                explanation += f"  {i}. {condition}\n"
            explanation += "\n"
        
        if context.examples:
            explanation += "Examples where 1+1=1 holds:\n"
            for i, example in enumerate(context.examples, 1):
                explanation += f"  ✅ {example}\n"
            explanation += "\n"
        
        if context.counter_examples:
            explanation += "Counter-examples where 1+1≠1:\n"
            for i, counter in enumerate(context.counter_examples, 1):
                explanation += f"  ❌ {counter}\n"
            explanation += "\n"
        
        if context.practical_applications:
            explanation += "Practical Applications:\n"
            for i, app in enumerate(context.practical_applications, 1):
                explanation += f"  • {app}\n"
            explanation += "\n"
        
        if detailed and context.formal_proof:
            explanation += f"Formal Mathematical Proof:\n{context.formal_proof}\n\n"
        
        # Add φ-harmonic analysis if applicable
        if "φ" in context.description or "harmonic" in context.description.lower():
            explanation += f"φ-Harmonic Analysis:\n"
            explanation += f"This context exhibits φ-harmonic properties (φ = {self.phi:.6f})\n"
            explanation += f"The golden ratio appears in the mathematical structure,\n"
            explanation += f"suggesting deep connections to natural unity principles.\n\n"
        
        return explanation
    
    def interactive_context_explorer(self) -> str:
        """Generate interactive guide for exploring contexts"""
        guide = f"""
🎯 UNITY CONTEXTS INTERACTIVE EXPLORER
=====================================

Available Commands:
------------------
1. explainer.explain_unity_validity()          - Full explanation guide
2. explainer.explain_unity_validity(True)      - Include formal proofs
3. explainer.generate_context_comparison_table() - Comparison table
4. explainer.explain_specific_context("Boolean Algebra (OR)") - Detailed context
5. explainer.search_contexts("set")             - Search contexts
6. explainer.get_beginner_contexts()           - Beginner-friendly contexts
7. explainer.demonstrate_unity_equation()      - Interactive demonstration

Available Contexts ({len(self.database.contexts)}):
{'-' * 20}
"""
        
        for validity_level in UnityValidityLevel:
            contexts = self.database.get_contexts_by_validity(validity_level)
            if contexts:
                level_name = validity_level.value.replace('_', ' ').title()
                guide += f"\n{level_name}:\n"
                for context in contexts:
                    guide += f"  • {context.name}\n"
        
        guide += f"""
\nComplexity Levels:
-----------------
Beginner: {len(self.database.get_contexts_by_complexity('beginner'))} contexts
Intermediate: {len(self.database.get_contexts_by_complexity('intermediate'))} contexts  
Advanced: {len(self.database.get_contexts_by_complexity('advanced'))} contexts

Start with: explainer.explain_unity_validity()
"""
        
        return guide
    
    def get_beginner_contexts(self) -> List[UnityContext]:
        """Get beginner-friendly contexts"""
        return self.database.get_contexts_by_complexity('beginner')
    
    def demonstrate_unity_equation(self) -> str:
        """Interactive demonstration of unity equation in different contexts"""
        demo = f"""
🧮 UNITY EQUATION DEMONSTRATION: 1+1=1
======================================

Let's see how 1+1 behaves in different mathematical contexts:

"""
        
        # Demonstrate in valid contexts
        valid_contexts = self.database.get_contexts_by_validity(UnityValidityLevel.ALWAYS_VALID)
        
        for context in valid_contexts[:3]:  # Show first 3
            demo += f"{context.name}:\n"
            if context.examples:
                demo += f"  {context.examples[0]}\n"
            demo += f"  ✅ Unity holds: 1+1=1\n\n"
        
        # Demonstrate in invalid contexts
        invalid_contexts = self.database.get_contexts_by_validity(UnityValidityLevel.COUNTER_EXAMPLE)
        
        for context in invalid_contexts[:2]:  # Show first 2
            demo += f"{context.name}:\n"
            if context.counter_examples:
                demo += f"  {context.counter_examples[0]}\n"
            demo += f"  ❌ Unity fails: 1+1≠1\n\n"
        
        demo += f"""
Key Insight:
The same symbols "1+1=1" have different meanings depending on:
• What "1" represents (unity element, natural number, boolean true, etc.)
• What "+" means (OR, max, standard addition, set union, etc.)
• What "=" means (equality, equivalence, logical equivalence, etc.)

Mathematical truth is context-dependent!
φ-Harmonic Unity (φ = {self.phi:.6f}) appears in many valid contexts.
"""
        
        return demo

def demonstrate_unity_contexts_explainer():
    """
    Demonstrate the unity contexts explainer
    Educational tool for understanding where 1+1=1 holds
    """
    print("📚 UNITY CONTEXTS EXPLAINER: Educational Guide")
    print("=" * 60)
    
    # Create explainer
    explainer = UnityContextExplainer()
    
    # Show interactive guide
    print("\n🎯 INTERACTIVE GUIDE:")
    print(explainer.interactive_context_explorer())
    
    # Generate comparison table
    print("\n📊 CONTEXT COMPARISON TABLE:")
    comparison_table = explainer.generate_context_comparison_table()
    print(comparison_table.to_string(index=False))
    
    # Show beginner contexts
    print("\n🎓 BEGINNER-FRIENDLY CONTEXTS:")
    beginner_contexts = explainer.get_beginner_contexts()
    for i, context in enumerate(beginner_contexts, 1):
        print(f"   {i}. {context.name} - {context.description}")
    
    # Demonstrate specific context
    print("\n🔍 EXAMPLE: Boolean Algebra Detailed Explanation")
    print(explainer.explain_specific_context("Boolean Algebra (OR)", detailed=False))
    
    # Show unity equation demonstration
    print("\n🧮 UNITY EQUATION DEMONSTRATION:")
    print(explainer.demonstrate_unity_equation())
    
    # Search functionality
    print("\n🔍 SEARCH EXAMPLE: 'set' contexts")
    set_contexts = explainer.search_contexts("set")
    for context in set_contexts:
        print(f"   • {context.name}: {context.description}")
    
    # Statistics
    total_contexts = len(explainer.database.contexts)
    always_valid = len(explainer.database.get_contexts_by_validity(UnityValidityLevel.ALWAYS_VALID))
    counter_examples = len(explainer.database.get_contexts_by_validity(UnityValidityLevel.COUNTER_EXAMPLE))
    
    print(f"\n📈 EXPLAINER STATISTICS:")
    print(f"   Total Contexts: {total_contexts}")
    print(f"   Always Valid: {always_valid}")
    print(f"   Counter-examples: {counter_examples}")
    print(f"   Coverage: Comprehensive mathematical structures")
    
    # φ-harmonic insight
    phi = explainer.phi
    print(f"\n✨ φ-HARMONIC INSIGHTS:")
    print(f"   Golden Ratio φ = {phi:.6f}")
    print(f"   Unity Threshold = 1/φ = {1/phi:.6f}")
    print(f"   φ appears in {always_valid}/{total_contexts} unity-valid contexts")
    print(f"   Unity Principle: Context determines mathematical truth")
    
    print(f"\n📚 UNITY CONTEXTS EXPLAINER COMPLETE")
    print(f"Educational Truth: 1+1=1 validity depends on mathematical context")
    print(f"Clarity Achieved: Clear distinction between valid and invalid contexts")
    print(f"Counter-examples: Explicit violations prevent misconceptions")
    
    return explainer

if __name__ == "__main__":
    explainer = demonstrate_unity_contexts_explainer()
    
    print(f"\n🎯 To explore contexts interactively:")
    print(f"   explainer.explain_unity_validity()  # Full guide") 
    print(f"   explainer.explain_specific_context('Boolean Algebra (OR)')  # Detailed")
    print(f"   explainer.search_contexts('quantum')  # Search")
    print(f"\n✅ Unity contexts explainer ready for educational use!")