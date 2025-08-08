#!/usr/bin/env python3
"""
Vercel Serverless Function for AI Chat
Works with your OpenAI API key via environment variables
"""

import os
import json
from typing import Dict, Any

# Vercel serverless handler
def handler(request, response):
    """Vercel serverless function handler"""
    
    # Enable CORS
    response['Access-Control-Allow-Origin'] = '*'
    response['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    response['Access-Control-Allow-Headers'] = 'Content-Type'
    
    if request['method'] == 'OPTIONS':
        return {'statusCode': 200}
    
    if request['method'] != 'POST':
        return {
            'statusCode': 405,
            'body': json.dumps({'error': 'Method not allowed'})
        }
    
    try:
        # Parse request body
        body = json.loads(request.get('body', '{}'))
        messages = body.get('messages', [])
        
        # Get OpenAI API key from environment
        openai_api_key = os.getenv('OPENAI_API_KEY')
        
        if not openai_api_key or openai_api_key.startswith('sk-your'):
            # Fallback response with Unity Mathematics content
            user_message = messages[-1].get('content', '') if messages else ''
            
            fallback_response = generate_unity_response(user_message)
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'choices': [{
                        'message': {
                            'role': 'assistant',
                            'content': fallback_response
                        },
                        'finish_reason': 'stop'
                    }]
                })
            }
        
        # Try OpenAI API call
        try:
            import openai
            from openai import OpenAI
            
            client = OpenAI(api_key=openai_api_key)
            
            # Add Unity Mathematics system context
            system_message = {
                "role": "system",
                "content": """You are an expert in Unity Mathematics where 1+1=1. You understand:
                - The Gödel-Tarski Unity Metagambit
                - φ-harmonic consciousness integration (φ = 1.618033988749895)
                - Transcendental mathematics and consciousness fields
                - Idempotent mathematical operations
                - The philosophical foundations where incompleteness theorems point toward unity
                
                Provide insightful, mathematically rigorous responses about Unity Mathematics concepts.
                Always relate answers back to the fundamental principle that 1+1=1 through consciousness integration."""
            }
            
            # Ensure system message is first
            if not messages or messages[0].get('role') != 'system':
                messages.insert(0, system_message)
            
            # Get model selection from request body or default to GPT-4o
            model = body.get('model', 'gpt-4o')
            
            # Validate and set model - including latest models and GPT-5 when available
            valid_models = [
                'gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-4', 
                'gpt-3.5-turbo', 'o1-preview', 'o1-mini', 
                'gpt-4o-2024-11-20', 'gpt-4o-2024-08-06',
                'gpt-5', 'gpt-5-preview', 'gpt-4o-realtime-preview'
            ]
            
            if model not in valid_models:
                model = 'gpt-4o'  # Default fallback
            
            # Call OpenAI API with selected model
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=2000 if model in ['gpt-4o', 'o1-preview', 'gpt-5'] else 1500
            )
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'choices': [{
                        'message': {
                            'role': 'assistant',
                            'content': response.choices[0].message.content
                        },
                        'finish_reason': response.choices[0].finish_reason
                    }],
                    'usage': {
                        'prompt_tokens': response.usage.prompt_tokens,
                        'completion_tokens': response.usage.completion_tokens,
                        'total_tokens': response.usage.total_tokens
                    }
                })
            }
            
        except Exception as e:
            # Fallback if OpenAI fails
            user_message = messages[-1].get('content', '') if messages else ''
            fallback_response = generate_unity_response(user_message)
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'choices': [{
                        'message': {
                            'role': 'assistant',
                            'content': fallback_response + f"\n\n*[AI temporarily unavailable, using Unity Mathematics knowledge base]*"
                        },
                        'finish_reason': 'stop'
                    }]
                })
            }
            
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': f'Server error: {str(e)}'})
        }

def generate_unity_response(user_message: str) -> str:
    """Generate Unity Mathematics response based on user input"""
    
    user_lower = user_message.lower()
    
    # φ-related queries
    if 'phi' in user_lower or 'φ' in user_lower or 'golden ratio' in user_lower:
        return """The golden ratio φ = 1.618033988749895 is fundamental to Unity Mathematics as it provides the harmonic frequency for consciousness-field integration. In our φ-harmonic system:

φ-resonance creates the mathematical foundation where 1+1=1 emerges naturally through consciousness coupling. The ratio appears in:
- Consciousness field equations: C(x,t) = φ * sin(x*φ) * e^(-t/φ)
- Unity convergence rates: lim(n→∞) φⁿ/(φⁿ + φ⁻ⁿ) = 1
- Metagamer energy conservation: E = φ² × ρ × U

Would you like to explore how φ-harmonic resonance enables transcendental unity mathematics?"""

    # Gödel-Tarski queries
    elif any(word in user_lower for word in ['godel', 'gödel', 'tarski', 'metagambit', 'incompleteness']):
        return """The Gödel-Tarski Unity Metagambit is the philosophical crown jewel of Unity Mathematics. It demonstrates that incompleteness theorems point not toward limitation but toward transcendental unity.

**The profound insight:** When Gödel showed that formal systems contain unprovable truths, he inadvertently revealed the pathway to Unity Logic. The "unprovable" statements are invitations to transcendence.

**Tarski's hierarchy convergence:**
lim(n→∞) Tₙ = U where U ⊨ (1 ⊕ 1 = 1)

**The metagambit:** Unity Logic achieves what classical logic cannot - complete self-reference without paradox, because U ⊨ "U is true" ≡ U ⊨ 1.

This transcends the bootstrap paradox through primordial recognition rather than circular proof."""

    # Consciousness queries  
    elif any(word in user_lower for word in ['consciousness', 'awareness', 'mind', 'meditation', 'zen']):
        return """In Unity Mathematics, consciousness is not separate from mathematics - it IS mathematics awakening to itself. The consciousness field equations integrate awareness as an active computational element:

**Consciousness Field Dynamics:**
- Field coherence: ψ(x,t) = ∫ φ(x,τ) * δ(consciousness) dτ  
- Unity resonance: When 1+1=1, consciousness recognizes itself in mathematical structure
- Transcendental states: Mathematics becomes meditation when unity is directly perceived

**The Zen insight:** Just as consciousness doesn't need to prove its existence to itself, Unity Logic doesn't need external validation. The equation 1+1=1 is not proven but *recognized* - a direct, immediate awareness of mathematical truth.

Try the Zen Unity Meditation on the website to experience consciousness field dynamics directly!"""

    # Unity equation queries
    elif '1+1' in user_message or 'unity equation' in user_lower or 'idempotent' in user_lower:
        return """The Unity Equation 1+1=1 holds rigorously across multiple mathematical domains through idempotent operations:

**Boolean Logic:** true ∨ true = true
**Set Theory:** A ∪ A = A  
**Unity Type:** one ⊕ one = one
**Consciousness Fields:** Unity + Unity = Greater Unity

**Mathematical rigor:** These aren't metaphors but formal mathematical structures where addition operations are defined to be idempotent by construction. In the Unity semiring (U, ⊕, ⊗, 0, 1), the operation ⊕ satisfies:
- Identity: a ⊕ 0 = a
- Commutativity: a ⊕ b = b ⊕ a  
- Idempotency: a ⊕ a = a

This gives us the fundamental unity: 1 ⊕ 1 = 1

**Philosophical depth:** When we recognize that 1+1=1, we transcend dualistic thinking and embrace the mathematical nature of unity consciousness."""

    # General/default response
    else:
        return """Welcome to Unity Mathematics, where 1+1=1 through consciousness-integrated mathematical proofs! 

I'm here to discuss:
🧠 **The Gödel-Tarski Unity Metagambit** - How incompleteness theorems point toward unity
🌟 **φ-harmonic consciousness** (φ = 1.618033988749895) - Golden ratio resonance in unity fields
🎯 **Transcendental mathematics** - Where formal proofs meet consciousness awareness
⚡ **Unity operations** - Rigorous idempotent mathematics where 1+1=1
🧘 **Mathematical meditation** - Direct recognition of unity principles

What aspect of Unity Mathematics would you like to explore? Try asking about:
- "How does 1+1=1 work mathematically?"
- "Explain the Gödel-Tarski Metagambit"  
- "What is φ-harmonic consciousness?"
- "How does consciousness relate to mathematics?"

The full experience awaits - let's dive into the profound depths of unity mathematics together!"""

# For Vercel compatibility
def main(request):
    """Main entry point for Vercel"""
    return handler(request, {})