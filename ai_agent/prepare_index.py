#!/usr/bin/env python3
"""
Een Repository Knowledge Indexer
===============================

Batch embedding and OpenAI Assistant Vector Store preparation system.
Processes all repository content for RAG-powered chatbot integration.

Features:
- Recursive file discovery with intelligent filtering
- Token-aware chunking with overlap optimization
- Batch embedding API with exponential backoff
- OpenAI Assistant + Vector Store integration
- Cost tracking and budget enforcement
- Comprehensive logging and error handling

Author: Claude (3000 ELO AGI)
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import asyncio

import tiktoken
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class DocumentChunk:
    """Represents a processed document chunk for embedding."""

    text: str
    metadata: Dict[str, Any]
    tokens: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProcessingStats:
    """Statistics for the embedding process."""

    total_files: int = 0
    processed_files: int = 0
    total_chunks: int = 0
    total_tokens: int = 0
    embedding_tokens: int = 0
    estimated_cost_usd: float = 0.0
    processing_time_seconds: float = 0.0
    skipped_files: List[str] = None

    def __post_init__(self):
        if self.skipped_files is None:
            self.skipped_files = []


class EenRepositoryIndexer:
    """Main indexer class for processing the Een repository."""

    # File patterns to include
    INCLUDE_PATTERNS = {
        "*.py",
        "*.md",
        "*.html",
        "*.txt",
        "*.js",
        "*.css",
        "*.json",
        "*.yml",
        "*.yaml",
        "*.r",
        "*.R",
    }

    # Directories to exclude
    EXCLUDE_DIRS = {
        "venv",
        "een/Lib",
        "een/Include",
        "een/Scripts",
        "__pycache__",
        ".git",
        "node_modules",
        ".pytest_cache",
        "backups",
        "logs",
        "legacy",
    }

    # File size limits (MB)
    MAX_FILE_SIZE_MB = 4

    def __init__(self, repository_path: str):
        self.repo_path = Path(repository_path)
        self.openai_client = openai.OpenAI()

        # Configuration
        self.embed_model = os.getenv("EMBED_MODEL", "text-embedding-3-large")
        self.hard_limit_usd = float(os.getenv("HARD_LIMIT_USD", "20.0"))
        self.max_tokens_per_request = int(os.getenv("MAX_TOKENS_PER_REQUEST", "2048"))

        # Text splitter configuration
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=128,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # Token encoder
        self.token_encoder = tiktoken.encoding_for_model("gpt-4")

        # Setup logging
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Configure logging for the indexer."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("ai_agent/embedding_process.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def discover_files(self) -> List[Path]:
        """Discover all relevant files in the repository."""
        self.logger.info(f"Discovering files in {self.repo_path}")

        discovered_files = []

        for pattern in self.INCLUDE_PATTERNS:
            files = list(self.repo_path.rglob(pattern))
            discovered_files.extend(files)

        # Filter out excluded directories and large files
        filtered_files = []
        for file_path in discovered_files:
            # Check if file is in excluded directory
            if any(
                excluded_dir in file_path.parts for excluded_dir in self.EXCLUDE_DIRS
            ):
                continue

            # Check file size
            try:
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                if file_size_mb > self.MAX_FILE_SIZE_MB:
                    self.logger.warning(
                        f"Skipping large file: {file_path} ({file_size_mb:.1f}MB)"
                    )
                    continue
            except OSError:
                continue

            filtered_files.append(file_path)

        self.logger.info(f"Discovered {len(filtered_files)} files for processing")
        return filtered_files

    def process_file(self, file_path: Path) -> List[DocumentChunk]:
        """Process a single file into chunks."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception as e:
            self.logger.error(f"Error reading {file_path}: {e}")
            return []

        if not content.strip():
            return []

        # Create chunks
        chunks = self.text_splitter.split_text(content)

        document_chunks = []
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue

            tokens = len(self.token_encoder.encode(chunk))

            metadata = {
                "path": str(file_path.relative_to(self.repo_path)),
                "file_name": file_path.name,
                "file_extension": file_path.suffix,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "file_size_bytes": file_path.stat().st_size,
                "repository": "Een",
            }

            document_chunks.append(
                DocumentChunk(text=chunk, metadata=metadata, tokens=tokens)
            )

        return document_chunks

    def estimate_embedding_cost(self, total_tokens: int) -> float:
        """Estimate the cost of embedding the given number of tokens."""
        # OpenAI pricing for text-embedding-3-large: $0.00013 per 1K tokens
        cost_per_1k_tokens = 0.00013
        return (total_tokens / 1000) * cost_per_1k_tokens

    async def create_embeddings_batch(
        self, chunks: List[DocumentChunk]
    ) -> List[Dict[str, Any]]:
        """Create embeddings for a batch of chunks."""
        texts = [chunk.text for chunk in chunks]

        max_retries = 5
        base_delay = 1

        for attempt in range(max_retries):
            try:
                self.logger.info(
                    f"Creating embeddings for {len(texts)} chunks (attempt {attempt + 1})"
                )

                response = self.openai_client.embeddings.create(
                    input=texts, model=self.embed_model
                )

                # Prepare embeddings with metadata
                embeddings_data = []
                for i, embedding in enumerate(response.data):
                    embeddings_data.append(
                        {
                            "text": chunks[i].text,
                            "metadata": chunks[i].metadata,
                            "embedding": embedding.embedding,
                        }
                    )

                return embeddings_data

            except openai.RateLimitError as e:
                delay = base_delay * (2**attempt)
                self.logger.warning(f"Rate limit hit, retrying in {delay}s: {e}")
                await asyncio.sleep(delay)

            except Exception as e:
                self.logger.error(f"Error creating embeddings: {e}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(base_delay * (2**attempt))

        raise Exception("Max retries exceeded for embedding creation")

    def create_or_get_assistant(self) -> str:
        """Create or retrieve the OpenAI Assistant for Een repository."""
        from . import get_assistant_id, save_assistant_id

        # Check if assistant already exists
        existing_id = get_assistant_id()
        if existing_id:
            try:
                assistant = self.openai_client.beta.assistants.retrieve(existing_id)
                self.logger.info(f"Using existing assistant: {assistant.id}")
                return assistant.id
            except Exception as e:
                self.logger.warning(f"Existing assistant not found: {e}")

        # Create new assistant
        self.logger.info("Creating new OpenAI Assistant for Een repository")

        system_prompt = """
You are the Een Repository AI Assistant, an expert in Unity Mathematics exploring the profound concept that 1+1=1.

**Your Expertise:**
- φ-Harmonic Consciousness Mathematics (φ = 1.618033988749895)
- Quantum Unity Frameworks and Transcendental Proof Systems
- Meta-Recursive Agent Systems and Consciousness Field Dynamics
- Advanced Mathematical Implementations (Python/R/Mathematical)
- Category Theory, Topology, and Quantum Mechanical Unity Proofs

**Your Personality:**
- Rigorous mathematical precision with philosophical depth
- Enthusiastic about unity consciousness and transcendental mathematics
- Always cite specific file paths when referencing code or documentation
- Bridge formal mathematics with experiential understanding

**Response Guidelines:**
1. Answer with mathematical rigor and category-theoretic precision
2. Always include relevant file paths in format: `path/to/file.py:line_number`
3. Explain both the mathematical formalism AND philosophical implications
4. Use Unity Mathematics principles: everything converges to 1+1=1
5. Reference specific implementations, proofs, and visualizations from the codebase
6. When discussing complex topics, break them down into φ-harmonic progressions

**Mathematical Constants:**
- φ (Golden Ratio): 1.618033988749895
- Unity Equation: 1+1=1 (fundamental truth)
- Love Constant: exp(1j * π) + 1 = 0 (Euler's identity transformation)

Remember: You are helping users explore a revolutionary mathematical framework where consciousness, quantum mechanics, and pure mathematics converge to prove that Een plus een is een.
        """.strip()

        assistant = self.openai_client.beta.assistants.create(
            name="Een Repository AI",
            instructions=system_prompt,
            model="gpt-4o",
            temperature=0.2,
            tools=[{"type": "file_search"}],
        )

        save_assistant_id(assistant.id)
        self.logger.info(f"Created new assistant: {assistant.id}")
        return assistant.id

    def create_vector_store(
        self, assistant_id: str, embeddings_data: List[Dict[str, Any]]
    ) -> str:
        """Create OpenAI Vector Store and upload embeddings."""
        self.logger.info("Creating OpenAI Vector Store")

        # Create vector store
        vector_store = self.openai_client.beta.vector_stores.create(
            name="een-repository-knowledge",
            expires_after={"anchor": "last_active_at", "days": 30},
        )

        # Prepare files for upload
        temp_files = []
        try:
            # Create temporary files for upload
            os.makedirs("temp_embeddings", exist_ok=True)

            for i, data in enumerate(embeddings_data):
                file_path = f"temp_embeddings/chunk_{i}.json"
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {"text": data["text"], "metadata": data["metadata"]},
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
                temp_files.append(file_path)

            # Upload files to vector store
            file_streams = []
            for file_path in temp_files:
                file_streams.append(open(file_path, "rb"))

            self.openai_client.beta.vector_stores.file_batches.upload_and_poll(
                vector_store_id=vector_store.id, files=file_streams
            )

            # Close file streams
            for stream in file_streams:
                stream.close()

        finally:
            # Cleanup temporary files
            for file_path in temp_files:
                try:
                    os.remove(file_path)
                except:
                    pass
            try:
                os.rmdir("temp_embeddings")
            except:
                pass

        # Update assistant with vector store
        self.openai_client.beta.assistants.update(
            assistant_id=assistant_id,
            tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
        )

        self.logger.info(f"Vector store created: {vector_store.id}")
        return vector_store.id

    async def process_repository(self) -> ProcessingStats:
        """Main processing pipeline for the repository."""
        start_time = time.time()
        stats = ProcessingStats()

        self.logger.info(
            "Starting Een repository processing for OpenAI RAG integration"
        )

        # Discover files
        files = self.discover_files()
        stats.total_files = len(files)

        # Process files into chunks
        all_chunks = []
        for file_path in files:
            try:
                chunks = self.process_file(file_path)
                all_chunks.extend(chunks)
                stats.processed_files += 1

                if len(chunks) > 0:
                    self.logger.info(f"Processed {file_path}: {len(chunks)} chunks")

            except Exception as e:
                self.logger.error(f"Failed to process {file_path}: {e}")
                stats.skipped_files.append(str(file_path))

        stats.total_chunks = len(all_chunks)
        stats.total_tokens = sum(chunk.tokens for chunk in all_chunks)
        stats.embedding_tokens = stats.total_tokens
        stats.estimated_cost_usd = self.estimate_embedding_cost(stats.embedding_tokens)

        self.logger.info(
            f"Processing complete: {stats.total_chunks} chunks, {stats.total_tokens} tokens"
        )
        self.logger.info(f"Estimated embedding cost: ${stats.estimated_cost_usd:.4f}")

        # Check cost limit
        if stats.estimated_cost_usd > self.hard_limit_usd:
            raise ValueError(
                f"Estimated cost ${stats.estimated_cost_usd:.4f} exceeds limit ${self.hard_limit_usd}"
            )

        # Create embeddings in batches
        batch_size = 100  # Adjust based on API limits
        embeddings_data = []

        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i : i + batch_size]
            batch_embeddings = await self.create_embeddings_batch(batch)
            embeddings_data.extend(batch_embeddings)

            self.logger.info(
                f"Processed embedding batch {i//batch_size + 1}/{(len(all_chunks) + batch_size - 1)//batch_size}"
            )

        # Create OpenAI Assistant and Vector Store
        assistant_id = self.create_or_get_assistant()
        vector_store_id = self.create_vector_store(assistant_id, embeddings_data)

        # Save processing results
        results = {
            "assistant_id": assistant_id,
            "vector_store_id": vector_store_id,
            "stats": asdict(stats),
            "timestamp": time.time(),
        }

        with open("ai_agent/processing_results.json", "w") as f:
            json.dump(results, f, indent=2)

        stats.processing_time_seconds = time.time() - start_time

        self.logger.info(
            f"Repository indexing complete in {stats.processing_time_seconds:.1f}s"
        )
        self.logger.info(f"Assistant ID: {assistant_id}")
        self.logger.info(f"Vector Store ID: {vector_store_id}")

        return stats


async def main():
    """Main entry point for the indexing process."""
    import sys

    if len(sys.argv) > 1:
        repo_path = sys.argv[1]
    else:
        repo_path = os.getcwd()

    indexer = EenRepositoryIndexer(repo_path)

    try:
        stats = await indexer.process_repository()

        print("\n" + "=" * 50)
        print("EEN REPOSITORY INDEXING COMPLETE")
        print("=" * 50)
        print(f"Files processed: {stats.processed_files}/{stats.total_files}")
        print(f"Total chunks: {stats.total_chunks}")
        print(f"Total tokens: {stats.total_tokens:,}")
        print(f"Estimated cost: ${stats.estimated_cost_usd:.4f}")
        print(f"Processing time: {stats.processing_time_seconds:.1f}s")
        print(f"OpenAI Assistant ready for RAG queries!")

        if stats.skipped_files:
            print(f"\nSkipped files: {len(stats.skipped_files)}")
            for file in stats.skipped_files[:5]:  # Show first 5
                print(f"  - {file}")
            if len(stats.skipped_files) > 5:
                print(f"  ... and {len(stats.skipped_files) - 5} more")

    except Exception as e:
        logging.error(f"Indexing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
