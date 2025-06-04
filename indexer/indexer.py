#!/usr/bin/env python3
"""
Dual Storage Indexer for Phase 2
Indexes both full documents (for BM25) and chunks (for semantic search)
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime, timezone
import asyncio
import aiohttp
from sentence_transformers import SentenceTransformer
import numpy as np
import argparse
from collections import deque
import time

# Add the chunker to path
sys.path.append(os.path.dirname(__file__))
from chunker import DocumentChunker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DualVespaIndexer:
    """
    Indexer that stores both full documents and chunks in Vespa
    for optimal BM25 and semantic search performance.
    """
    
    def __init__(self, vespa_url: str = None, 
                 embedding_batch_size: int = 32,
                 max_concurrent_embeddings: int = 100):
        self.vespa_url = vespa_url or os.getenv("VESPA_URL", "http://localhost:4080")
        # Set cert and key paths from certs directory if they exist
        cert_dir = Path("certs")
        cert_file = cert_dir / "data-plane-public-cert.pem"
        key_file = cert_dir / "data-plane-private-key.pem"
        
        if cert_file.exists() and key_file.exists():
            self.cert_path = str(cert_file)
            self.key_path = str(key_file)
            logger.info(f"Using Vespa Cloud certificates from certs directory")
        self.api_key = os.getenv("VESPA_API_KEY")
        self.document_api_url = f"{self.vespa_url}/document/v1/default/doc/docid"
        self.session = None
        
        # Initialize sentence transformer for embeddings
        logger.info("Loading embedding model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Model ready")
        
        # Initialize chunker
        self.chunker = DocumentChunker(chunk_size=800, overlap=150)
        
        # Batching configuration
        self.embedding_batch_size = embedding_batch_size
        self.max_concurrent_embeddings = max_concurrent_embeddings
        
        # Embedding queue for async processing
        self.embedding_queue = asyncio.Queue(maxsize=max_concurrent_embeddings)
        self.embedding_results = {}
        self.embedding_task = None
        
        # Statistics
        self.stats = {
            "docs_processed": 0,
            "docs_indexed": 0,
            "chunks_created": 0,
            "chunks_indexed": 0,
            "failed_docs": [],
            "failed_chunks": [],
            "embedding_time": 0,
            "indexing_time": 0
        }
        
    async def __aenter__(self):
        # Configure headers for authentication
        headers = {}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        
        # Configure SSL context for Vespa Cloud if certificates are available
        ssl_context = None
        if self.cert_path and self.key_path:
            try:
                import ssl
                ssl_context = ssl.create_default_context()
                ssl_context.load_cert_chain(self.cert_path, self.key_path)
                logger.info(f"SSL context configured with cert: {self.cert_path}")
            except Exception as e:
                logger.error(f"Failed to load SSL certificates: {e}")
                ssl_context = None
        
        connector = aiohttp.TCPConnector(ssl=ssl_context) if ssl_context else None
        self.session = aiohttp.ClientSession(connector=connector, headers=headers)
        # Start the embedding processor
        self.embedding_task = asyncio.create_task(self._embedding_processor())
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Stop embedding processor
        if self.embedding_task:
            await self.embedding_queue.put(None)  # Sentinel to stop
            await self.embedding_task
        
        if self.session:
            await self.session.close()
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts in a batch"""
        # Limit text lengths to avoid memory issues
        texts = [text[:2000] for text in texts]
        embeddings = self.model.encode(texts, 
                                     batch_size=self.embedding_batch_size,
                                     convert_to_numpy=True,
                                     show_progress_bar=False)
        return [emb.tolist() for emb in embeddings]
    
    async def _embedding_processor(self):
        """Background task to process embedding requests in batches"""
        batch = []
        batch_ids = []
        
        while True:
            try:
                # Collect items for batch with timeout
                timeout = 0.1 if batch else None
                
                try:
                    item = await asyncio.wait_for(
                        self.embedding_queue.get(), 
                        timeout=timeout
                    )
                except asyncio.TimeoutError:
                    # Process accumulated batch if timeout
                    if batch:
                        start_time = time.time()
                        embeddings = self.generate_embeddings_batch(batch)
                        self.stats["embedding_time"] += time.time() - start_time
                        
                        for idx, emb in zip(batch_ids, embeddings):
                            self.embedding_results[idx] = emb
                        
                        batch = []
                        batch_ids = []
                    continue
                
                if item is None:  # Sentinel to stop
                    # Process remaining batch
                    if batch:
                        embeddings = self.generate_embeddings_batch(batch)
                        for idx, emb in zip(batch_ids, embeddings):
                            self.embedding_results[idx] = emb
                    break
                
                text_id, text = item
                batch.append(text)
                batch_ids.append(text_id)
                
                # Process batch if full
                if len(batch) >= self.embedding_batch_size:
                    start_time = time.time()
                    embeddings = self.generate_embeddings_batch(batch)
                    self.stats["embedding_time"] += time.time() - start_time
                    
                    for idx, emb in zip(batch_ids, embeddings):
                        self.embedding_results[idx] = emb
                    
                    batch = []
                    batch_ids = []
                    
            except Exception as e:
                logger.error(f"Error in embedding processor: {e}")
                # Store None for failed embeddings
                for idx in batch_ids:
                    self.embedding_results[idx] = None
                batch = []
                batch_ids = []
    
    async def get_embedding_async(self, text_id: str, text: str) -> Optional[List[float]]:
        """Queue text for embedding and wait for result"""
        await self.embedding_queue.put((text_id, text))
        
        # Wait for result
        while text_id not in self.embedding_results:
            await asyncio.sleep(0.01)
        
        result = self.embedding_results.pop(text_id)
        return result
    
    async def index_document(self, doc: Dict) -> Tuple[bool, bool]:
        """
        Index both full document and its chunks.
        
        Returns:
            Tuple of (full_doc_success, chunks_success)
        """
        self.stats["docs_processed"] += 1
        
        # Create chunks first
        chunks = self.chunker.chunk_document(doc)
        if chunks:
            self.stats["chunks_created"] += len(chunks)
        
        # Queue all embeddings at once
        embedding_tasks = []
        
        # Queue full document embedding
        content_text = f"{doc.get('title', '')} {doc.get('content', '')} {doc.get('description', '')}"
        doc_embedding_task = self.get_embedding_async(f"doc_{doc['id']}", content_text)
        embedding_tasks.append(doc_embedding_task)
        
        # Queue chunk embeddings
        chunk_embedding_tasks = []
        for chunk in chunks:
            task = self.get_embedding_async(f"chunk_{chunk['id']}", chunk['chunk_content'])
            chunk_embedding_tasks.append(task)
            embedding_tasks.append(task)
        
        # Wait for all embeddings
        embeddings = await asyncio.gather(*embedding_tasks)
        doc_embedding = embeddings[0]
        chunk_embeddings = embeddings[1:]
        
        # Now index with pre-computed embeddings
        start_time = time.time()
        
        # Index full document
        full_doc_success = await self.index_full_document_with_embedding(doc, doc_embedding)
        
        # Index chunks in parallel
        chunks_success = True
        if chunks:
            chunk_tasks = [
                self.index_chunk_with_embedding(chunk, embedding) 
                for chunk, embedding in zip(chunks, chunk_embeddings)
            ]
            chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)
            
            # Check results
            for i, result in enumerate(chunk_results):
                if isinstance(result, Exception):
                    chunks_success = False
                    self.stats["failed_chunks"].append(chunks[i]["id"])
                elif result:
                    self.stats["chunks_indexed"] += 1
                else:
                    chunks_success = False
                    self.stats["failed_chunks"].append(chunks[i]["id"])
        
        self.stats["indexing_time"] += time.time() - start_time
        
        if full_doc_success:
            self.stats["docs_indexed"] += 1
        else:
            self.stats["failed_docs"].append(doc["id"])
        
        return full_doc_success, chunks_success
    
    async def index_full_document_with_embedding(self, doc: Dict, embedding: Optional[List[float]]) -> bool:
        """Index full document with pre-computed embedding"""
        try:
            if embedding is None:
                logger.error(f"No embedding for document {doc['id']}")
                return False
            
            # Prepare full document for Vespa
            vespa_doc = {
                "fields": {
                    "doc_type": "full_doc",
                    "id": doc["id"],
                    "url": doc["url"],
                    "domain": doc["domain"],
                    "title": doc.get("title", ""),
                    "content": doc.get("content", ""),
                    "description": doc.get("description", ""),
                    "keywords": doc.get("keywords", ""),
                    "code_snippets": doc.get("code_snippets", []),
                    "crawled_at": doc.get("crawled_at", ""),
                    "content_embedding": embedding
                }
            }
            
            # Send to Vespa
            url = f"{self.document_api_url}/{doc['id']}"
            async with self.session.post(url, json=vespa_doc) as response:
                if response.status == 200:
                    return True
                else:
                    logger.error(f"Failed to index document {doc['id']}: HTTP {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error indexing full document {doc['id']}: {str(e)}")
            return False
    
    async def index_chunk_with_embedding(self, chunk: Dict, embedding: Optional[List[float]]) -> bool:
        """Index chunk with pre-computed embedding"""
        try:
            if embedding is None:
                logger.error(f"No embedding for chunk {chunk['id']}")
                return False
            
            # Prepare chunk document for Vespa
            vespa_doc = {
                "fields": {
                    "doc_type": "chunk",
                    "id": chunk["id"],
                    "parent_id": chunk["parent_id"],
                    "url": chunk["url"],
                    "domain": chunk["domain"],
                    "title": chunk["title"],
                    "chunk_content": chunk["chunk_content"],
                    "chunk_index": chunk["chunk_index"],
                    "chunk_start": chunk["chunk_start"],
                    "chunk_end": chunk["chunk_end"],
                    "chunk_size": chunk["chunk_size"],
                    "crawled_at": chunk["crawled_at"],
                    "parent_doc_type": chunk["parent_doc_type"],
                    "chunk_embedding": embedding
                }
            }
            
            # Send to Vespa
            url = f"{self.document_api_url}/{chunk['id']}"
            async with self.session.post(url, json=vespa_doc) as response:
                if response.status == 200:
                    return True
                else:
                    logger.error(f"Failed to index chunk {chunk['id']}: HTTP {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error indexing chunk {chunk['id']}: {str(e)}")
            return False
    
    async def index_batch(self, documents: List[Dict], batch_size: int = 50):
        """Index documents in batches with dual storage and optimized pipeline"""
        total_docs = len(documents)
        logger.info(f"Indexing {total_docs} documents (batch_size={batch_size})...")
        
        # Process in batches with pipeline optimization
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            # Index each document (full + chunks) in the batch
            tasks = [self.index_document(doc) for doc in batch]
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Log progress every 1% or at least every 100 documents
            processed = min(i + batch_size, total_docs)
            if processed % max(total_docs // 100, 100) < batch_size or processed == total_docs:
                docs_per_sec = processed / (time.time() - self.start_time) if hasattr(self, 'start_time') else 0
                logger.info(f"Progress: {processed}/{total_docs} documents ({docs_per_sec:.1f} docs/s)")
            
            # No pause needed - async operations provide natural flow control
        
        # No need for queue.join() - embeddings are awaited in index_document
        
        return self.stats
    
    async def check_vespa_health(self) -> bool:
        """Check if Vespa is healthy and ready"""
        try:
            async with self.session.get(f"{self.vespa_url}/state/v1/health") as response:
                if response.status == 200:
                    data = await response.json()
                    status = data.get("status", {}).get("code", "")
                    if status == "up":
                        return True
                    else:
                        logger.warning(f"Vespa status: {status}")
                        return False
                else:
                    logger.error(f"Vespa health check failed: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Failed to connect to Vespa: {str(e)}")
            return False
    
    def print_stats(self):
        """Print indexing statistics"""
        doc_success_rate = (self.stats['docs_indexed'] / self.stats['docs_processed'] * 100) if self.stats['docs_processed'] > 0 else 0
        chunk_success_rate = (self.stats['chunks_indexed'] / self.stats['chunks_created'] * 100) if self.stats['chunks_created'] > 0 else 0
        
        logger.info(f"Indexing complete: {self.stats['docs_indexed']}/{self.stats['docs_processed']} docs ({doc_success_rate:.1f}%), "
                   f"{self.stats['chunks_indexed']}/{self.stats['chunks_created']} chunks ({chunk_success_rate:.1f}%)")
        
        if self.stats['failed_docs'] or self.stats['failed_chunks']:
            logger.warning(f"Failures: {len(self.stats['failed_docs'])} docs, {len(self.stats['failed_chunks'])} chunks")
        
        # Performance metrics
        if self.stats['embedding_time'] > 0:
            logger.info(f"Performance: Embedding time: {self.stats['embedding_time']:.1f}s, "
                       f"Indexing time: {self.stats['indexing_time']:.1f}s")


async def load_documents_from_crawl(crawl_dir: Path, max_per_domain: Optional[int] = None) -> List[Dict]:
    """Load all documents from a crawl directory with optional per-domain limit"""
    documents = []
    domain_counts = {}
    
    # Find all document files
    doc_files = sorted(crawl_dir.glob("*.json"))
    
    logger.info(f"Found {len(doc_files)} document files")
    
    for doc_file in doc_files:
        try:
            with open(doc_file, 'r', encoding='utf-8') as f:
                doc = json.load(f)
                
                # Apply per-domain limit if specified
                if max_per_domain:
                    domain = doc.get('domain', 'unknown')
                    domain_counts[domain] = domain_counts.get(domain, 0)
                    
                    if domain_counts[domain] >= max_per_domain:
                        continue  # Skip this document
                    
                    domain_counts[domain] += 1
                
                documents.append(doc)
        except Exception as e:
            logger.error(f"Failed to load {doc_file}: {str(e)}")
    
    if max_per_domain:
        logger.info(f"Loaded {len(documents)} documents after applying domain limit")
        logger.info(f"Domain distribution: {domain_counts}")
    
    return documents


async def main():
    parser = argparse.ArgumentParser(description='Dual Index crawled documents to Vespa')
    parser.add_argument('--crawl-dir', type=str, help='Path to crawl directory')
    parser.add_argument('--vespa-url', type=str, default='http://localhost:4080', 
                        help='Vespa URL (default: http://localhost:4080)')
    parser.add_argument('--batch-size', type=int, default=100, 
                        help='Batch size for indexing (default: 100)')
    parser.add_argument('--embedding-batch-size', type=int, default=32,
                        help='Batch size for embedding generation (default: 32)')
    parser.add_argument('--max-concurrent-embeddings', type=int, default=200,
                        help='Maximum concurrent embeddings in queue (default: 200)')
    parser.add_argument('--limit', type=int, help='Limit number of documents to process')
    parser.add_argument('--max-per-domain', type=int, default=10000,
                        help='Maximum documents per domain (default: 10000)')
    
    args = parser.parse_args()
    
    # Find the crawl directory or use all if not specified
    if args.crawl_dir:
        crawl_dir = Path(args.crawl_dir)
        documents = await load_documents_from_crawl(crawl_dir, args.max_per_domain)
    else:
        crawled_data_dir = Path("crawled_data")
        if not crawled_data_dir.exists():
            logger.error("No crawled_data directory found")
            return
        
        # Find all crawl directories
        crawl_dirs = sorted([d for d in crawled_data_dir.iterdir() if d.is_dir()])
        if not crawl_dirs:
            logger.error("No crawl directories found")
            return
        
        logger.info(f"Found {len(crawl_dirs)} crawl directories")
        
        # Load documents from all crawl directories with global domain tracking
        documents = []
        global_domain_counts = {}
        
        for crawl_dir in crawl_dirs:
            logger.info(f"Loading documents from: {crawl_dir}")
            
            # Load all docs from this crawl dir
            doc_files = sorted(crawl_dir.glob("*.json"))
            crawl_docs_added = 0
            
            for doc_file in doc_files:
                try:
                    with open(doc_file, 'r', encoding='utf-8') as f:
                        doc = json.load(f)
                        
                        # Apply global per-domain limit
                        domain = doc.get('domain', 'unknown')
                        global_domain_counts[domain] = global_domain_counts.get(domain, 0)
                        
                        if global_domain_counts[domain] >= args.max_per_domain:
                            continue  # Skip this document
                        
                        global_domain_counts[domain] += 1
                        documents.append(doc)
                        crawl_docs_added += 1
                        
                except Exception as e:
                    logger.error(f"Failed to load {doc_file}: {str(e)}")
            
            logger.info(f"Loaded {crawl_docs_added} documents from {crawl_dir}")
        
        logger.info(f"Total documents loaded: {len(documents)} (with max {args.max_per_domain} per domain)")
        logger.info(f"Domain distribution: {dict(sorted(global_domain_counts.items(), key=lambda x: x[1], reverse=True)[:10])}...")
    if not documents:
        logger.error("No documents found to index")
        return
    
    # Apply limit if specified
    if args.limit:
        documents = documents[:args.limit]
        logger.info(f"Limited to first {len(documents)} documents")
    
    # Log document count is already done above, no need to repeat
    
    # Index documents with dual storage
    async with DualVespaIndexer(
        args.vespa_url, 
        embedding_batch_size=args.embedding_batch_size,
        max_concurrent_embeddings=args.max_concurrent_embeddings
    ) as indexer:
        # Check Vespa health
        if not await indexer.check_vespa_health():
            logger.error("Vespa is not healthy. Please ensure Vespa is running.")
            return
        
        start_time = datetime.now()
        indexer.start_time = time.time()  # For progress tracking
        
        stats = await indexer.index_batch(documents, args.batch_size)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"Completed in {duration:.2f}s ({len(documents)/duration:.1f} docs/s)")
        indexer.print_stats()
        
        # Update metadata
        metadata_file = crawl_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            metadata['dual_indexed'] = True
            metadata['dual_indexed_at'] = datetime.now(timezone.utc).isoformat()
            metadata['docs_indexed'] = stats['docs_indexed']
            metadata['chunks_created'] = stats['chunks_created']
            metadata['chunks_indexed'] = stats['chunks_indexed']
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())