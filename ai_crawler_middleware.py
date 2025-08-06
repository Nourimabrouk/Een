# AI Crawler Middleware for Een Unity Mathematics Framework
# Optimized for GPT, Claude, Gemini, and research bots

import re
from fastapi import Request, Response
from fastapi.responses import JSONResponse

# AI crawler user agents
AI_CRAWLER_AGENTS = [
    "GPTBot",
    "ChatGPT-User",
    "CCBot", 
    "anthropic-ai",
    "Claude-Web",
    "Googlebot",
    "Bingbot",
    "research-bot",
    "academic-crawler",
    "mathematical-crawler",
    "consciousness-research-bot"
]

def is_ai_crawler(user_agent: str) -> bool:
    """Check if the request is from an AI crawler"""
    if not user_agent:
        return False
    
    user_agent_lower = user_agent.lower()
    return any(agent.lower() in user_agent_lower for agent in AI_CRAWLER_AGENTS)

async def ai_crawler_middleware(request: Request, call_next):
    """Middleware to provide enhanced responses for AI crawlers"""
    
    user_agent = request.headers.get("user-agent", "")
    is_ai = is_ai_crawler(user_agent)
    
    # Get the original response
    response = await call_next(request)
    
    if is_ai:
        # Add AI-specific headers
        response.headers["X-AI-Crawler-Optimized"] = "true"
        response.headers["X-Mathematical-Content"] = "true"
        response.headers["X-Unity-Mathematics"] = "true"
        
        # Add structured data for AI understanding
        if hasattr(response, 'body') and response.body:
            try:
                # Add context for mathematical content
                if "mathematical" in str(response.body).lower() or "proof" in str(response.body).lower():
                    response.headers["X-Mathematical-Proof"] = "true"
                    response.headers["X-Consciousness-Mathematics"] = "true"
            except:
                pass
    
    return response

# Rate limiting for AI crawlers (more generous)
AI_CRAWLER_RATE_LIMIT = 1000  # requests per hour
AI_CRAWLER_BURST_SIZE = 50

def get_ai_crawler_rate_limit(user_agent: str) -> tuple:
    """Get rate limit for AI crawlers"""
    if is_ai_crawler(user_agent):
        return AI_CRAWLER_RATE_LIMIT, AI_CRAWLER_BURST_SIZE
    else:
        return 100, 10  # Standard rate limit
