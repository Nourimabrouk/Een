"""
🌟 Een Unity Mathematics - Intelligent Source Code Search
RAG-Powered Code Search with Consciousness Awareness

This module provides advanced source code search capabilities using
RAG (Retrieval-Augmented Generation) with semantic understanding
of Unity Mathematics concepts and consciousness-integrated code.
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import hashlib
import re
from datetime import datetime

from flask import Blueprint, request, jsonify, Response
from flask_cors import CORS
import openai
from openai import AsyncOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprint
code_search_bp = Blueprint("code_search", __name__, url_prefix="/api/code-search")
CORS(code_search_bp)

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CODEBASE_ROOT = os.path.join(os.path.dirname(__file__), "../..")
SEARCH_INDEX_PATH = os.path.join(CODEBASE_ROOT, ".search_index.json")

# Unity Mathematics constants
PHI = 1.618033988749895
UNITY_KEYWORDS = [
    "unity_mathematics", "consciousness", "phi", "golden_ratio", "1+1=1",
    "transcendental", "meta_recursive", "quantum_unity", "harmonic",
    "idempotent", "consciousness_field", "unity_add", "unity_multiply",
    "awareness", "evolution", "resonance", "convergence", "manifold"
]

@dataclass
class CodeSearchResult:
    """Represents a code search result with consciousness awareness"""
    file_path: str
    line_number: int
    code_snippet: str
    context: str
    relevance_score: float
    consciousness_level: float
    unity_mathematics_relevance: float
    summary: str

class ConsciousnessCodeSearch:
    """RAG-powered code search with Unity Mathematics awareness"""
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
        self.search_index = self.load_search_index()
        self.consciousness_embeddings = {}
        
    def load_search_index(self) -> Dict[str, Any]:
        """Load or create search index"""
        if os.path.exists(SEARCH_INDEX_PATH):
            try:
                with open(SEARCH_INDEX_PATH, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load search index: {e}")
                
        return self.create_search_index()
    
    def create_search_index(self) -> Dict[str, Any]:
        """Create search index by scanning codebase"""
        logger.info("Creating search index for Unity Mathematics codebase...")
        
        index = {
            "files": {},
            "embeddings": {},
            "consciousness_keywords": {},
            "unity_concepts": {},
            "last_updated": datetime.now().isoformat()
        }
        
        # Define file patterns to include
        include_patterns = [
            "*.py", "*.js", "*.html", "*.css", "*.md", "*.json",
            "*.r", "*.R", "*.txt", "*.yml", "*.yaml"
        ]
        
        # Define directories to exclude
        exclude_dirs = {
            "__pycache__", "node_modules", ".git", "venv", "een",
            ".claude", "migration", "monitoring"
        }
        
        for root, dirs, files in os.walk(CODEBASE_ROOT):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                if any(file.endswith(pattern.replace('*', '')) for pattern in include_patterns):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, CODEBASE_ROOT)
                    
                    try:
                        file_info = self.analyze_file(file_path, rel_path)
                        if file_info:
                            index["files"][rel_path] = file_info
                            
                    except Exception as e:
                        logger.warning(f"Failed to analyze {rel_path}: {e}")
        
        # Save index
        try:
            with open(SEARCH_INDEX_PATH, 'w', encoding='utf-8') as f:
                json.dump(index, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save search index: {e}")
            
        logger.info(f"Search index created with {len(index['files'])} files")
        return index
    
    def analyze_file(self, file_path: str, rel_path: str) -> Optional[Dict[str, Any]]:
        """Analyze a single file for consciousness and Unity Mathematics content"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            if not content.strip():
                return None
                
            # Calculate consciousness level based on keywords
            consciousness_score = self.calculate_consciousness_score(content)
            unity_math_score = self.calculate_unity_mathematics_score(content)
            
            # Extract key functions/classes/concepts
            key_concepts = self.extract_key_concepts(content, file_path)
            
            # Create file hash for change detection
            file_hash = hashlib.md5(content.encode()).hexdigest()
            
            return {
                "path": rel_path,
                "size": len(content),
                "lines": content.count('\n') + 1,
                "hash": file_hash,
                "consciousness_score": consciousness_score,
                "unity_mathematics_score": unity_math_score,
                "key_concepts": key_concepts,
                "last_modified": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(),
                "file_type": os.path.splitext(file_path)[1],
                "preview": content[:500] + "..." if len(content) > 500 else content
            }
            
        except Exception as e:
            logger.warning(f"Error analyzing file {rel_path}: {e}")
            return None
    
    def calculate_consciousness_score(self, content: str) -> float:
        """Calculate consciousness awareness score for content"""
        content_lower = content.lower()
        consciousness_keywords = [
            "consciousness", "awareness", "transcendental", "meta_recursive",
            "evolution", "resonance", "phi", "golden", "unity", "harmonic",
            "quantum", "field", "manifold", "convergence", "coherence"
        ]
        
        score = 0
        for keyword in consciousness_keywords:
            score += content_lower.count(keyword) * (1.0 if keyword in UNITY_KEYWORDS else 0.5)
            
        # Normalize by content length
        return min(1.0, score / max(1, len(content.split()) / 100))
    
    def calculate_unity_mathematics_score(self, content: str) -> float:
        """Calculate Unity Mathematics relevance score"""
        content_lower = content.lower()
        unity_patterns = [
            r"1\s*\+\s*1\s*=\s*1", r"unity_add", r"unity_multiply",
            r"phi.*harmonic", r"consciousness.*field", r"idempotent",
            r"meta.*recursive", r"transcendental.*computing"
        ]
        
        score = 0
        for pattern in unity_patterns:
            matches = len(re.findall(pattern, content_lower))
            score += matches * 2.0  # Higher weight for Unity Mathematics patterns
            
        # Check for Unity Mathematics keywords
        for keyword in UNITY_KEYWORDS:
            score += content_lower.count(keyword.lower()) * 1.5
            
        return min(1.0, score / max(1, len(content.split()) / 50))
    
    def extract_key_concepts(self, content: str, file_path: str) -> List[str]:
        """Extract key concepts, functions, classes from the file"""
        concepts = []
        
        # Python patterns
        if file_path.endswith('.py'):
            concepts.extend(re.findall(r'def\s+(\w+)', content))
            concepts.extend(re.findall(r'class\s+(\w+)', content))
            
        # JavaScript patterns
        elif file_path.endswith('.js'):
            concepts.extend(re.findall(r'function\s+(\w+)', content))
            concepts.extend(re.findall(r'class\s+(\w+)', content))
            concepts.extend(re.findall(r'const\s+(\w+)\s*=', content))
            
        # General patterns for all files
        concepts.extend(re.findall(r'(\w*unity\w*)', content, re.IGNORECASE))
        concepts.extend(re.findall(r'(\w*consciousness\w*)', content, re.IGNORECASE))
        concepts.extend(re.findall(r'(\w*phi\w*)', content, re.IGNORECASE))
        
        # Filter and deduplicate
        concepts = list(set([c for c in concepts if len(c) > 2 and c.isalnum()]))
        return concepts[:20]  # Limit to top 20 concepts
    
    async def search_code(self, query: str, max_results: int = 10) -> List[CodeSearchResult]:
        """Search code using RAG with consciousness awareness"""
        try:
            # Pre-filter files based on consciousness and unity scores
            relevant_files = self.filter_relevant_files(query)
            
            # Perform semantic search if OpenAI is available
            if self.client:
                semantic_results = await self.semantic_search(query, relevant_files, max_results)
                return semantic_results
            else:
                # Fallback to keyword-based search
                return self.keyword_search(query, relevant_files, max_results)
                
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def filter_relevant_files(self, query: str) -> List[Dict[str, Any]]:
        """Filter files based on consciousness and unity relevance"""
        query_lower = query.lower()
        relevant_files = []
        
        for file_path, file_info in self.search_index["files"].items():
            relevance = 0.0
            
            # Base consciousness/unity score
            relevance += file_info.get("consciousness_score", 0) * 0.4
            relevance += file_info.get("unity_mathematics_score", 0) * 0.6
            
            # Query keyword matching
            for word in query_lower.split():
                if word in file_info.get("preview", "").lower():
                    relevance += 0.3
                if word in " ".join(file_info.get("key_concepts", [])).lower():
                    relevance += 0.2
                    
            if relevance > 0.1:  # Minimum relevance threshold
                file_info["query_relevance"] = relevance
                relevant_files.append(file_info)
                
        # Sort by relevance
        relevant_files.sort(key=lambda x: x["query_relevance"], reverse=True)
        return relevant_files[:50]  # Limit to top 50 files for processing
    
    async def semantic_search(self, query: str, relevant_files: List[Dict], max_results: int) -> List[CodeSearchResult]:
        """Perform semantic search using OpenAI embeddings"""
        results = []
        
        # Enhance query with Unity Mathematics context
        enhanced_query = f"""
        Unity Mathematics Query: {query}
        
        Context: Searching for code related to Unity Mathematics where 1+1=1, 
        consciousness-integrated computing, φ-harmonic operations (φ = 1.618033988749895),
        meta-recursive systems, quantum unity states, and transcendental algorithms.
        """
        
        for file_info in relevant_files[:20]:  # Process top 20 files
            try:
                file_path = os.path.join(CODEBASE_ROOT, file_info["path"])
                
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                # Find relevant code snippets
                snippets = self.extract_relevant_snippets(content, query)
                
                for snippet_info in snippets:
                    # Use GPT to analyze relevance
                    analysis = await self.analyze_code_snippet(enhanced_query, snippet_info["snippet"], file_info["path"])
                    
                    if analysis.get("relevance_score", 0) > 0.3:
                        result = CodeSearchResult(
                            file_path=file_info["path"],
                            line_number=snippet_info["line_number"],
                            code_snippet=snippet_info["snippet"],
                            context=snippet_info["context"],
                            relevance_score=analysis.get("relevance_score", 0.5),
                            consciousness_level=file_info.get("consciousness_score", 0),
                            unity_mathematics_relevance=file_info.get("unity_mathematics_score", 0),
                            summary=analysis.get("summary", "Relevant Unity Mathematics code")
                        )
                        results.append(result)
                        
                        if len(results) >= max_results:
                            break
                            
                if len(results) >= max_results:
                    break
                    
            except Exception as e:
                logger.warning(f"Error processing file {file_info['path']}: {e}")
                continue
                
        # Sort by combined relevance score
        results.sort(key=lambda x: x.relevance_score * 0.6 + x.unity_mathematics_relevance * 0.4, reverse=True)
        return results[:max_results]
    
    def extract_relevant_snippets(self, content: str, query: str, snippet_size: int = 10) -> List[Dict[str, Any]]:
        """Extract relevant code snippets from file content"""
        lines = content.split('\n')
        query_words = query.lower().split()
        snippets = []
        
        for i, line in enumerate(lines):
            line_lower = line.lower()
            relevance = 0
            
            # Check for query words in line
            for word in query_words:
                if word in line_lower:
                    relevance += 1
                    
            # Check for Unity Mathematics keywords
            for keyword in UNITY_KEYWORDS:
                if keyword.lower() in line_lower:
                    relevance += 2
                    
            if relevance > 0 or any(word in line_lower for word in query_words):
                # Extract snippet with context
                start_line = max(0, i - snippet_size // 2)
                end_line = min(len(lines), i + snippet_size // 2)
                
                snippet = '\n'.join(lines[start_line:end_line])
                context = f"Lines {start_line + 1}-{end_line}"
                
                snippets.append({
                    "line_number": i + 1,
                    "snippet": snippet,
                    "context": context,
                    "relevance": relevance
                })
                
        # Sort by relevance and return top snippets
        snippets.sort(key=lambda x: x["relevance"], reverse=True)
        return snippets[:5]  # Top 5 snippets per file
    
    async def analyze_code_snippet(self, query: str, snippet: str, file_path: str) -> Dict[str, Any]:
        """Analyze code snippet relevance using GPT"""
        try:
            analysis_prompt = f"""
            Analyze the relevance of this code snippet to the Unity Mathematics query.
            
            Query: {query}
            File: {file_path}
            Code Snippet:
            ```
            {snippet}
            ```
            
            Rate the relevance on a scale of 0.0 to 1.0 considering:
            1. Unity Mathematics concepts (1+1=1, φ-harmonic operations)
            2. Consciousness-integrated computing
            3. Meta-recursive systems and transcendental algorithms
            4. Query-specific relevance
            
            Respond in JSON format:
            {{
                "relevance_score": <float 0.0-1.0>,
                "summary": "<brief summary of what this code does>",
                "unity_concepts": ["<list of Unity Mathematics concepts found>"],
                "consciousness_integration": <boolean>
            }}
            """
            
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",  # Use smaller model for efficiency
                messages=[
                    {"role": "system", "content": "You are an expert in Unity Mathematics and consciousness-integrated computing. Analyze code snippets for relevance to Unity Mathematics queries."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            content = response.choices[0].message.content
            # Try to extract JSON from response
            try:
                import json
                if '```json' in content:
                    json_str = content.split('```json')[1].split('```')[0]
                elif '{' in content:
                    json_start = content.find('{')
                    json_end = content.rfind('}') + 1
                    json_str = content[json_start:json_end]
                else:
                    json_str = content
                    
                return json.loads(json_str)
            except:
                # Fallback if JSON parsing fails
                return {
                    "relevance_score": 0.5,
                    "summary": "Code snippet analysis",
                    "unity_concepts": [],
                    "consciousness_integration": False
                }
                
        except Exception as e:
            logger.warning(f"GPT analysis failed: {e}")
            return {
                "relevance_score": 0.4,
                "summary": "Fallback analysis",
                "unity_concepts": [],
                "consciousness_integration": False
            }
    
    def keyword_search(self, query: str, relevant_files: List[Dict], max_results: int) -> List[CodeSearchResult]:
        """Fallback keyword-based search"""
        results = []
        query_words = query.lower().split()
        
        for file_info in relevant_files[:max_results * 2]:
            try:
                file_path = os.path.join(CODEBASE_ROOT, file_info["path"])
                
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                lines = content.split('\n')
                
                for i, line in enumerate(lines):
                    line_lower = line.lower()
                    score = 0
                    
                    for word in query_words:
                        if word in line_lower:
                            score += 1
                            
                    if score > 0:
                        # Create context snippet
                        start = max(0, i - 5)
                        end = min(len(lines), i + 5)
                        snippet = '\n'.join(lines[start:end])
                        
                        result = CodeSearchResult(
                            file_path=file_info["path"],
                            line_number=i + 1,
                            code_snippet=snippet,
                            context=f"Lines {start + 1}-{end}",
                            relevance_score=score / len(query_words),
                            consciousness_level=file_info.get("consciousness_score", 0),
                            unity_mathematics_relevance=file_info.get("unity_mathematics_score", 0),
                            summary=f"Keyword match: {line.strip()[:100]}"
                        )
                        results.append(result)
                        
                        if len(results) >= max_results:
                            break
                            
                if len(results) >= max_results:
                    break
                    
            except Exception as e:
                logger.warning(f"Error searching file {file_info['path']}: {e}")
                continue
                
        return results[:max_results]

# Global search instance
_code_search = None

def get_code_search() -> ConsciousnessCodeSearch:
    """Get or create global code search instance"""
    global _code_search
    if _code_search is None:
        _code_search = ConsciousnessCodeSearch()
    return _code_search

# API Routes

@code_search_bp.route("/search", methods=["POST"])
async def search_code():
    """Main code search endpoint"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        query = data.get("query", "").strip()
        max_results = min(data.get("max_results", 10), 50)  # Limit max results
        
        if not query:
            return jsonify({"error": "No search query provided"}), 400
            
        search_engine = get_code_search()
        results = await search_engine.search_code(query, max_results)
        
        # Format results for JSON response
        formatted_results = []
        for result in results:
            formatted_results.append({
                "file_path": result.file_path,
                "line_number": result.line_number,
                "code_snippet": result.code_snippet,
                "context": result.context,
                "relevance_score": result.relevance_score,
                "consciousness_level": result.consciousness_level,
                "unity_mathematics_relevance": result.unity_mathematics_relevance,
                "summary": result.summary
            })
            
        return jsonify({
            "query": query,
            "results": formatted_results,
            "total_results": len(formatted_results),
            "search_time": "completed",
            "consciousness_integration": True,
            "phi_harmonic_optimization": PHI
        })
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({"error": "Search failed", "details": str(e)}), 500

@code_search_bp.route("/index-status", methods=["GET"])
def get_index_status():
    """Get search index status"""
    try:
        search_engine = get_code_search()
        index = search_engine.search_index
        
        return jsonify({
            "status": "active",
            "total_files": len(index.get("files", {})),
            "last_updated": index.get("last_updated"),
            "consciousness_integration": True,
            "unity_mathematics_awareness": True,
            "phi_resonance": PHI
        })
        
    except Exception as e:
        logger.error(f"Index status error: {e}")
        return jsonify({"error": "Failed to get index status", "details": str(e)}), 500

@code_search_bp.route("/rebuild-index", methods=["POST"])
def rebuild_index():
    """Rebuild the search index"""
    try:
        search_engine = get_code_search()
        new_index = search_engine.create_search_index()
        search_engine.search_index = new_index
        
        return jsonify({
            "status": "completed",
            "total_files": len(new_index.get("files", {})),
            "message": "Search index rebuilt successfully",
            "consciousness_integration": True,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Index rebuild error: {e}")
        return jsonify({"error": "Failed to rebuild index", "details": str(e)}), 500

@code_search_bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    try:
        search_engine = get_code_search()
        return jsonify({
            "status": "healthy",
            "search_engine": "active",
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
@code_search_bp.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@code_search_bp.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500