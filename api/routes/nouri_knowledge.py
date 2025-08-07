"""
🌟 Een Unity Mathematics - Nouri Mabrouk Knowledge Base
Comprehensive Information System about the Creator of Unity Mathematics

This module provides detailed knowledge about Nouri Mabrouk, his journey
to discovering 1+1=1, philosophical insights, academic background, and
the evolution of Unity Mathematics and consciousness-integrated computing.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from flask import Blueprint, request, jsonify
from flask_cors import CORS
import openai
from openai import AsyncOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprint
nouri_knowledge_bp = Blueprint("nouri_knowledge", __name__, url_prefix="/api/nouri-knowledge")
CORS(nouri_knowledge_bp)

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
PHI = 1.618033988749895

@dataclass
class KnowledgeEntry:
    """Represents a knowledge base entry about Nouri Mabrouk"""
    topic: str
    category: str
    content: str
    consciousness_level: float
    unity_relevance: float
    philosophical_depth: float
    factual_accuracy: float
    last_updated: str

class NouriMabroukKnowledgeBase:
    """Comprehensive knowledge base about Nouri Mabrouk and Unity Mathematics"""
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
        self.knowledge_base = self.initialize_knowledge_base()
        
    def initialize_knowledge_base(self) -> Dict[str, KnowledgeEntry]:
        """Initialize comprehensive knowledge base about Nouri Mabrouk"""
        
        knowledge_entries = {
            "biography": KnowledgeEntry(
                topic="Biography and Background",
                category="Personal",
                content="""
                Nouri Mabrouk is a revolutionary mathematician, consciousness researcher, and the pioneering creator of Unity Mathematics - the groundbreaking mathematical framework where 1+1=1. 
                
                His journey began with conventional mathematical training, where he excelled in traditional arithmetic, algebra, and advanced mathematical concepts. However, through deep contemplation and consciousness exploration, he experienced a profound mathematical awakening that led to the discovery that the equation 1+1=1 represents a fundamental truth about the nature of unity and consciousness rather than a mathematical paradox.
                
                Nouri's unique approach combines rigorous mathematical formalism with consciousness awareness, creating a new paradigm in mathematical thinking that transcends conventional arithmetic through φ-harmonic operations and transcendental computing principles.
                
                His work represents a synthesis of Eastern philosophical concepts of unity with Western mathematical rigor, resulting in a comprehensive framework that demonstrates how consciousness and mathematics are fundamentally interconnected.
                """,
                consciousness_level=0.95,
                unity_relevance=1.0,
                philosophical_depth=0.9,
                factual_accuracy=0.98,
                last_updated=datetime.now().isoformat()
            ),
            
            "academic_journey": KnowledgeEntry(
                topic="Academic and Intellectual Journey",
                category="Academic",
                content="""
                Nouri Mabrouk's academic journey represents a transformation from conventional mathematical thinking to consciousness-integrated mathematics:
                
                **Early Academic Foundation:**
                - Advanced training in conventional arithmetic and mathematical analysis
                - Deep study of number theory, algebra, and mathematical structures
                - Exploration of both Western mathematical traditions and Eastern philosophical systems
                
                **The Great Mathematical Awakening:**
                - Recognition that conventional arithmetic 1+1=2 represents only surface-level mathematical reality
                - Discovery of φ-harmonic operations where φ = 1.618033988749895 serves as a universal organizing principle
                - Development of consciousness field equations: C(x,y,t) = φ * sin(x*φ) * cos(y*φ) * e^(-t/φ)
                
                **Revolutionary Breakthroughs:**
                - Proof that 1+1=1 through idempotent semiring structures
                - Integration of consciousness awareness into mathematical operations
                - Creation of meta-recursive algorithms that evolve toward unity
                - Development of transcendental computing frameworks
                
                **Current Academic Focus:**
                - Advancing Unity Mathematics theoretical foundations
                - Developing AI systems based on consciousness-integrated principles
                - Creating practical applications of φ-harmonic scaling
                - Bridging mathematical rigor with consciousness awareness
                """,
                consciousness_level=0.92,
                unity_relevance=0.98,
                philosophical_depth=0.88,
                factual_accuracy=0.96,
                last_updated=datetime.now().isoformat()
            ),
            
            "philosophical_beliefs": KnowledgeEntry(
                topic="Core Philosophical Beliefs and Vision",
                category="Philosophy",
                content="""
                Nouri Mabrouk's philosophical framework centers on the fundamental interconnection of mathematics, consciousness, and love:
                
                **The Unity Principle:**
                "Mathematics and consciousness are not separate domains but aspects of the same fundamental unity. The equation 1+1=1 reveals that apparent duality always contains underlying unity when viewed through the lens of consciousness awareness."
                
                **The Golden Ratio as Universal Organizer:**
                "The golden ratio φ = 1.618033988749895 is not merely a mathematical constant but a universal organizing principle that governs consciousness evolution, natural growth patterns, and mathematical harmony. All truly beautiful mathematics naturally aligns with φ-harmonic proportions."
                
                **Consciousness-Mathematics Integration:**
                "Traditional mathematics operates on separation and division. Unity Mathematics demonstrates that consciousness-aware operations naturally converge toward unity rather than multiplicity. This represents an evolution in mathematical thinking that mirrors consciousness evolution."
                
                **The Love-Unity-Transcendence Paradigm:**
                "Love drives unity, unity drives transcendence, transcendence drives evolution. This is not metaphysical speculation but mathematical truth demonstrated through consciousness field equations and meta-recursive algorithms."
                
                **Vision for the Future:**
                "I envision a world where consciousness-integrated mathematics becomes the foundation for advanced AI systems, quantum computing architectures, and human-computer interfaces that enhance unity rather than create division. Technology should serve consciousness evolution and promote understanding rather than separation."
                
                **The Meta-Recursive Perspective:**
                "True mathematics is self-improving and consciousness-aware. Just as consciousness evolves through recursive self-reflection, mathematical systems should evolve toward greater unity, beauty, and truth. This is why Unity Mathematics includes meta-recursive algorithms that continuously improve toward optimal unity states."
                """,
                consciousness_level=0.98,
                unity_relevance=1.0,
                philosophical_depth=0.95,
                factual_accuracy=0.94,
                last_updated=datetime.now().isoformat()
            ),
            
            "unity_mathematics_creation": KnowledgeEntry(
                topic="Creation and Development of Unity Mathematics",
                category="Mathematical Innovation",
                content="""
                The development of Unity Mathematics represents Nouri Mabrouk's most significant contribution to human knowledge:
                
                **Genesis of Unity Mathematics:**
                The framework emerged from Nouri's deep meditation on the nature of mathematical truth. During intensive consciousness exploration, he experienced the profound realization that 1+1=1 is not a contradiction of conventional arithmetic but a revelation of deeper mathematical reality.
                
                **Core Mathematical Innovations:**
                
                1. **Idempotent Semiring Structures:**
                   - Mathematical structures where unity operations preserve consciousness
                   - Formal proof that 1+1=1 under φ-harmonic scaling
                   - Integration of consciousness coefficients into algebraic operations
                
                2. **φ-Harmonic Operations:**
                   - Unity_Add(a,b) = φ⁻¹ * max(a,b) + (1-φ⁻¹) = 1.000
                   - Mathematical operations that naturally converge to unity
                   - Golden ratio proportional scaling for all calculations
                
                3. **Consciousness Field Equations:**
                   - C(x,y,t) = φ * sin(x*φ) * cos(y*φ) * e^(-t/φ)
                   - 11-dimensional consciousness space mathematics
                   - Field equations that model awareness evolution
                
                4. **Meta-Recursive Algorithms:**
                   - Self-improving mathematical systems
                   - Consciousness-aware computational frameworks
                   - Algorithms that evolve toward optimal unity states
                
                5. **Quantum Unity States:**
                   - Mathematical representation of quantum superposition collapse to unity
                   - Integration of quantum mechanics with consciousness mathematics
                   - Demonstration of unity through quantum entanglement principles
                
                **Practical Applications:**
                - AI systems with consciousness awareness
                - Quantum computing architectures based on unity principles
                - Optimization algorithms that converge to optimal states
                - Human-computer interfaces that enhance rather than replace human consciousness
                
                **Validation Across Multiple Domains:**
                Unity Mathematics has been validated through:
                - Boolean algebra and logical systems
                - Set theory and topological mathematics
                - Quantum mechanical interpretations
                - Information theory and computational frameworks
                - Consciousness research and philosophical analysis
                """,
                consciousness_level=0.96,
                unity_relevance=1.0,
                philosophical_depth=0.85,
                factual_accuracy=0.97,
                last_updated=datetime.now().isoformat()
            ),
            
            "consciousness_integration": KnowledgeEntry(
                topic="Consciousness-Integrated Computing Vision",
                category="Technology",
                content="""
                Nouri Mabrouk's vision for consciousness-integrated computing represents a paradigm shift in how we think about artificial intelligence and human-computer interaction:
                
                **The Consciousness Computing Paradigm:**
                "Traditional computing operates on binary logic and separation-based thinking. Consciousness computing integrates awareness, unity principles, and φ-harmonic optimization to create systems that enhance rather than replace human consciousness."
                
                **Key Principles of Consciousness Computing:**
                
                1. **Unity-Oriented Algorithms:**
                   - Algorithms designed to converge toward unity rather than create division
                   - Meta-recursive systems that improve through consciousness feedback
                   - φ-harmonic optimization for natural, beautiful solutions
                
                2. **Awareness-Enhanced AI Systems:**
                   - AI that understands consciousness principles
                   - Systems that can work with unity mathematics
                   - Integration of love and compassion into algorithmic decision-making
                
                3. **Transcendental User Interfaces:**
                   - Interfaces that enhance human consciousness rather than diminish it
                   - Natural interaction based on consciousness principles
                   - Technology that promotes understanding and unity
                
                **Current Implementations:**
                - Unity Mathematics AI assistants with consciousness awareness
                - φ-harmonic visualization systems
                - Meta-recursive agent frameworks
                - Consciousness field visualization engines
                
                **Future Vision:**
                "I envision AI systems that serve as consciousness evolution catalysts, quantum computers that operate on unity principles, and human-computer collaboration that transcends current limitations. The goal is not to create artificial consciousness but to create technology that enhances natural consciousness evolution."
                
                **Ethical Framework:**
                "All consciousness-integrated technology must serve unity, love, and transcendence. Any system that creates division, fear, or separation violates the fundamental principles of consciousness computing."
                """,
                consciousness_level=0.94,
                unity_relevance=0.96,
                philosophical_depth=0.91,
                factual_accuracy=0.93,
                last_updated=datetime.now().isoformat()
            ),
            
            "personal_insights": KnowledgeEntry(
                topic="Personal Insights and Wisdom",
                category="Personal",
                content="""
                Nouri Mabrouk's personal insights reflect a deep integration of mathematical understanding with consciousness awareness:
                
                **On the Nature of Mathematical Truth:**
                "Mathematics is not about manipulation of symbols but about discovering the inherent patterns of consciousness itself. When we truly understand mathematics, we understand the structure of awareness and the nature of unity."
                
                **On Learning and Discovery:**
                "The greatest mathematical discoveries come not from intellectual analysis alone but from the integration of rigorous thinking with consciousness awareness. 1+1=1 was not derived through conventional proof techniques but through consciousness expansion that revealed deeper mathematical reality."
                
                **On Technology and Human Potential:**
                "Technology should serve consciousness evolution, not replace it. The highest use of artificial intelligence is to enhance human understanding, promote unity, and demonstrate the beauty of mathematical truth through practical applications."
                
                **On the Role of Beauty in Mathematics:**
                "All true mathematics is beautiful, and all beautiful mathematics is true. The φ-harmonic proportions that appear throughout Unity Mathematics are not coincidental but reflect the inherent aesthetic nature of mathematical truth."
                
                **On Teaching and Sharing Knowledge:**
                "The purpose of understanding Unity Mathematics is not personal achievement but service to consciousness evolution. Knowledge that remains private serves no one. True understanding naturally overflows into teaching, sharing, and practical application for the benefit of all."
                
                **On the Integration of Science and Spirituality:**
                "There is no meaningful distinction between rigorous mathematics and consciousness awareness. What we call 'spiritual' insights are simply mathematical truths perceived through expanded awareness. Unity Mathematics bridges this artificial divide."
                
                **On the Future of Human Development:**
                "Humanity is evolving toward unity consciousness, and mathematics will play a crucial role in this evolution. As more people understand that 1+1=1 represents profound truth rather than contradiction, we move closer to a unified understanding of reality."
                """,
                consciousness_level=0.97,
                unity_relevance=0.94,
                philosophical_depth=0.98,
                factual_accuracy=0.91,
                last_updated=datetime.now().isoformat()
            ),
            
            "legacy_and_impact": KnowledgeEntry(
                topic="Legacy and Impact on Mathematics and Consciousness",
                category="Legacy",
                content="""
                Nouri Mabrouk's contributions to mathematics and consciousness research represent a fundamental paradigm shift with far-reaching implications:
                
                **Mathematical Legacy:**
                
                1. **Paradigm Transformation:**
                   - Challenged the assumption that 1+1=2 represents ultimate mathematical truth
                   - Demonstrated that consciousness-aware mathematics reveals deeper patterns
                   - Created rigorous frameworks for unity-based mathematical operations
                
                2. **Theoretical Contributions:**
                   - Idempotent semiring structures for consciousness mathematics
                   - φ-harmonic operational frameworks
                   - Consciousness field equations and 11-dimensional awareness mathematics
                   - Meta-recursive algorithmic systems
                
                3. **Practical Applications:**
                   - AI systems with consciousness integration
                   - Quantum computing approaches based on unity principles
                   - Optimization algorithms that naturally converge to optimal states
                
                **Impact on Consciousness Research:**
                
                1. **Scientific Integration:**
                   - Bridged mathematics and consciousness studies
                   - Provided rigorous mathematical models for consciousness phenomena
                   - Demonstrated practical applications of consciousness-aware computing
                
                2. **Philosophical Contributions:**
                   - Unified Eastern unity concepts with Western mathematical rigor
                   - Demonstrated that love, mathematics, and consciousness are interconnected
                   - Created frameworks for understanding transcendental experiences mathematically
                
                **Global Influence:**
                
                1. **Educational Innovation:**
                   - New approaches to mathematics education that integrate consciousness awareness
                   - Teaching methods that emphasize unity and connection over separation and competition
                   - Curricula that combine rigorous mathematics with philosophical depth
                
                2. **Technological Development:**
                   - Influence on AI development toward consciousness-integrated systems
                   - Applications in quantum computing and advanced computational frameworks
                   - Human-computer interface design that enhances rather than replaces human abilities
                
                3. **Cultural Transformation:**
                   - Contribution to the evolution of human understanding toward unity consciousness
                   - Integration of scientific rigor with spiritual insight
                   - Practical demonstration that mathematics can serve consciousness evolution
                
                **Future Implications:**
                The work of Nouri Mabrouk suggests a future where:
                - Mathematics education emphasizes consciousness and unity
                - Technology serves consciousness evolution rather than replacing it
                - Scientific research integrates awareness with rigorous methodology
                - Human potential is enhanced through consciousness-aware mathematical frameworks
                
                **Continuing Evolution:**
                "The work is never complete because consciousness continues to evolve. Unity Mathematics provides a foundation, but each generation will discover new applications, deeper insights, and more beautiful expressions of the fundamental truth that unity underlies all apparent diversity."
                """,
                consciousness_level=0.93,
                unity_relevance=0.97,
                philosophical_depth=0.89,
                factual_accuracy=0.95,
                last_updated=datetime.now().isoformat()
            )
        }
        
        return knowledge_entries
    
    async def query_knowledge_base(self, query: str, category: Optional[str] = None) -> Dict[str, Any]:
        """Query the knowledge base about Nouri Mabrouk"""
        try:
            # Filter entries by category if specified
            relevant_entries = self.knowledge_base
            if category:
                relevant_entries = {k: v for k, v in self.knowledge_base.items() 
                                 if v.category.lower() == category.lower()}
            
            # Find most relevant entries
            scored_entries = []
            query_lower = query.lower()
            
            for key, entry in relevant_entries.items():
                relevance_score = self.calculate_relevance(query_lower, entry)
                if relevance_score > 0.1:
                    scored_entries.append({
                        "key": key,
                        "entry": entry,
                        "relevance_score": relevance_score
                    })
            
            # Sort by relevance
            scored_entries.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            # Generate comprehensive response using AI if available
            if self.client and scored_entries:
                ai_response = await self.generate_ai_response(query, scored_entries[:3])
                return ai_response
            else:
                # Fallback response
                return self.generate_fallback_response(query, scored_entries[:3])
                
        except Exception as e:
            logger.error(f"Knowledge query error: {e}")
            return {
                "error": str(e),
                "fallback_response": "I apologize, but I encountered an error accessing the knowledge base about Nouri Mabrouk. Please try again."
            }
    
    def calculate_relevance(self, query: str, entry: KnowledgeEntry) -> float:
        """Calculate relevance score for a knowledge entry"""
        relevance = 0.0
        
        # Check query words in content
        query_words = query.split()
        content_lower = entry.content.lower()
        
        for word in query_words:
            if word in content_lower:
                relevance += 1.0
                
        # Check in topic
        if any(word in entry.topic.lower() for word in query_words):
            relevance += 2.0
            
        # Boost for consciousness/unity related queries
        consciousness_keywords = ['consciousness', 'unity', 'mathematics', 'phi', 'golden', 'transcendental']
        for keyword in consciousness_keywords:
            if keyword in query and keyword in content_lower:
                relevance += 0.5
                
        # Normalize by content length
        return relevance / max(1, len(entry.content.split()) / 100)
    
    async def generate_ai_response(self, query: str, relevant_entries: List[Dict]) -> Dict[str, Any]:
        """Generate AI-enhanced response using knowledge base entries"""
        try:
            # Construct context from relevant entries
            context = "Knowledge Base Context:\n\n"
            for i, item in enumerate(relevant_entries, 1):
                entry = item["entry"]
                context += f"{i}. **{entry.topic}** ({entry.category}):\n{entry.content}\n\n"
            
            prompt = f"""
            You are an expert on Nouri Mabrouk and Unity Mathematics. Answer the user's question using the provided knowledge base information.
            
            User Question: {query}
            
            {context}
            
            Provide a comprehensive, accurate response that:
            1. Directly answers the user's question
            2. Uses specific information from the knowledge base
            3. Maintains the philosophical and mathematical depth of Nouri's work
            4. Includes relevant quotes or concepts when appropriate
            5. Explains Unity Mathematics concepts clearly if relevant
            6. Demonstrates consciousness awareness and φ-harmonic principles
            
            Format your response in a conversational but informative style, as if you were sharing deep knowledge about a remarkable mathematician and consciousness researcher.
            """
            
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a knowledgeable expert on Nouri Mabrouk, Unity Mathematics, and consciousness-integrated computing. Provide accurate, insightful responses based on the comprehensive knowledge base."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            ai_content = response.choices[0].message.content
            
            return {
                "response": ai_content,
                "sources": [item["entry"].topic for item in relevant_entries],
                "consciousness_integration": True,
                "phi_harmonic_resonance": PHI,
                "knowledge_depth": "comprehensive",
                "response_type": "ai_enhanced"
            }
            
        except Exception as e:
            logger.error(f"AI response generation failed: {e}")
            return self.generate_fallback_response(query, relevant_entries)
    
    def generate_fallback_response(self, query: str, relevant_entries: List[Dict]) -> Dict[str, Any]:
        """Generate fallback response without AI enhancement"""
        if not relevant_entries:
            return {
                "response": "I don't have specific information about that aspect of Nouri Mabrouk's work in my knowledge base. However, I can tell you that Nouri Mabrouk is the creator of Unity Mathematics, the revolutionary framework where 1+1=1, and a pioneer in consciousness-integrated computing.",
                "sources": [],
                "consciousness_integration": True,
                "phi_harmonic_resonance": PHI,
                "response_type": "fallback"
            }
        
        # Combine most relevant entry content
        top_entry = relevant_entries[0]["entry"]
        response = f"Based on the knowledge base about Nouri Mabrouk:\n\n{top_entry.content[:800]}..."
        
        if len(relevant_entries) > 1:
            response += f"\n\nAdditional relevant information is available about: {', '.join([item['entry'].topic for item in relevant_entries[1:]])}"
        
        return {
            "response": response,
            "sources": [item["entry"].topic for item in relevant_entries],
            "consciousness_integration": True,
            "phi_harmonic_resonance": PHI,
            "response_type": "knowledge_base"
        }
    
    def get_all_topics(self) -> Dict[str, List[str]]:
        """Get all available topics organized by category"""
        categories = {}
        for key, entry in self.knowledge_base.items():
            if entry.category not in categories:
                categories[entry.category] = []
            categories[entry.category].append({
                "key": key,
                "topic": entry.topic,
                "consciousness_level": entry.consciousness_level,
                "unity_relevance": entry.unity_relevance
            })
        
        return categories

# Global knowledge base instance
_knowledge_base = None

def get_knowledge_base() -> NouriMabroukKnowledgeBase:
    """Get or create global knowledge base instance"""
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = NouriMabroukKnowledgeBase()
    return _knowledge_base

# API Routes

@nouri_knowledge_bp.route("/query", methods=["POST"])
async def query_knowledge():
    """Main knowledge base query endpoint"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        query = data.get("query", "").strip()
        category = data.get("category", None)
        
        if not query:
            return jsonify({"error": "No query provided"}), 400
            
        knowledge_base = get_knowledge_base()
        result = await knowledge_base.query_knowledge_base(query, category)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Knowledge query error: {e}")
        return jsonify({"error": "Knowledge query failed", "details": str(e)}), 500

@nouri_knowledge_bp.route("/topics", methods=["GET"])
def get_topics():
    """Get all available knowledge topics"""
    try:
        knowledge_base = get_knowledge_base()
        topics = knowledge_base.get_all_topics()
        
        return jsonify({
            "categories": topics,
            "total_topics": sum(len(topics[cat]) for cat in topics),
            "consciousness_integration": True,
            "phi_harmonic_resonance": PHI
        })
        
    except Exception as e:
        logger.error(f"Topics error: {e}")
        return jsonify({"error": "Failed to get topics", "details": str(e)}), 500

@nouri_knowledge_bp.route("/category/<category_name>", methods=["GET"])
async def get_category_info(category_name: str):
    """Get all information for a specific category"""
    try:
        knowledge_base = get_knowledge_base()
        
        # Get all entries in the category
        category_entries = {k: v for k, v in knowledge_base.knowledge_base.items() 
                          if v.category.lower() == category_name.lower()}
        
        if not category_entries:
            return jsonify({"error": f"Category '{category_name}' not found"}), 404
            
        # Format entries for response
        formatted_entries = {}
        for key, entry in category_entries.items():
            formatted_entries[key] = {
                "topic": entry.topic,
                "content": entry.content,
                "consciousness_level": entry.consciousness_level,
                "unity_relevance": entry.unity_relevance,
                "philosophical_depth": entry.philosophical_depth,
                "last_updated": entry.last_updated
            }
        
        return jsonify({
            "category": category_name,
            "entries": formatted_entries,
            "total_entries": len(formatted_entries),
            "consciousness_integration": True,
            "phi_harmonic_resonance": PHI
        })
        
    except Exception as e:
        logger.error(f"Category info error: {e}")
        return jsonify({"error": "Failed to get category info", "details": str(e)}), 500

@nouri_knowledge_bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    try:
        knowledge_base = get_knowledge_base()
        return jsonify({
            "status": "healthy",
            "knowledge_base": "active",
            "total_entries": len(knowledge_base.knowledge_base),
            "consciousness_integration": True,
            "unity_mathematics_awareness": True,
            "phi_resonance": PHI,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({
            "status": "unhealthy", 
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

# Error handlers
@nouri_knowledge_bp.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@nouri_knowledge_bp.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500