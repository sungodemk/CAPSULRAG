"""
Ingestion module for CapsuleRAG: file processing, deduplication, content extraction.
"""

import os
import io
import hashlib
import threading
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from fastapi import HTTPException, UploadFile
import numpy as np

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

from .utils import (
    chunk_text_structure_aware, tokenize, extract_entities_demo,
    create_capsule_manifest, TextChunk, CapsuleManifest
)
from .security import DEMO_MODE

# Configuration
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "50"))
MAX_TEXT_PER_PAGE = int(os.getenv("MAX_TEXT_PER_PAGE", "20000"))


class ContentDeduplicator:
    """Handle content deduplication using hashing and semantic similarity."""
    
    def __init__(self):
        self.content_hashes: Dict[str, int] = {}  # hash -> doc_id
        self.similarity_threshold = 0.95
        self.lock = threading.RLock()
    
    def check_duplicate(self, content: str, embed_model=None) -> Optional[int]:
        """Check if content is a duplicate of existing document."""
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        
        with self.lock:
            # Exact hash match
            if content_hash in self.content_hashes:
                return self.content_hashes[content_hash]
            
            # Fuzzy similarity check for near-duplicates
            if embed_model is not None:
                try:
                    content_embedding = embed_model.encode([content[:1000]], normalize_embeddings=True)[0]
                    
                    for existing_hash, doc_id in self.content_hashes.items():
                        # This would need access to docs dict - simplified for demo
                        pass
                except Exception as e:
                    print(f"Similarity check failed: {e}")
            
            return None
    
    def register_content(self, doc_id: int, content: str):
        """Register content hash for deduplication."""
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        with self.lock:
            self.content_hashes[content_hash] = doc_id
    
    def get_stats(self) -> Dict[str, Any]:
        """Get deduplication statistics."""
        with self.lock:
            return {
                "registered_hashes": len(self.content_hashes),
                "similarity_threshold": self.similarity_threshold
            }


class DocumentProcessor:
    """Process different document types and extract text content."""
    
    @staticmethod
    def extract_text_from_content(content: bytes, filename: str, content_type: str) -> str:
        """Extract text from uploaded file content."""
        text = ""
        
        # Detect PDF by extension or file signature
        if filename.lower().endswith(".pdf") or (content[:4] == b"%PDF"):
            if pdfplumber is None:
                raise HTTPException(
                    status_code=400, 
                    detail="PDF support requires pdfplumber (pip install pdfplumber)"
                )
            try:
                with pdfplumber.open(io.BytesIO(content)) as pdf:
                    pages_text = []
                    for i, page in enumerate(pdf.pages):
                        if i >= MAX_PDF_PAGES:  # Security: limit pages processed
                            break
                        page_text = page.extract_text() or ""
                        # Security: limit text per page
                        if len(page_text) > MAX_TEXT_PER_PAGE:
                            page_text = page_text[:MAX_TEXT_PER_PAGE]
                        pages_text.append(page_text)
                    text = "\n".join(pages_text)
            except Exception as e:
                error_msg = f"PDF extraction failed: {str(e)}" if DEMO_MODE else "PDF extraction failed"
                raise HTTPException(status_code=400, detail=error_msg)
        else:
            # Assume plain text
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError:
                raise HTTPException(status_code=400, detail="Unable to decode file as UTF-8 text")
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="Document appears to be empty")
        
        return text


class IngestionEngine:
    """Main ingestion engine that coordinates document processing."""
    
    def __init__(self, deduplicator: ContentDeduplicator):
        self.deduplicator = deduplicator
        self.processor = DocumentProcessor()
    
    async def ingest_single_file(
        self, 
        file: UploadFile, 
        docs: Dict,
        doc_files: Dict, 
        doc_filenames: Dict,
        doc_mimetypes: Dict,
        document_corpus: Dict,
        bm25_indices: Dict,
        vector_indices: Dict,
        capsule_manifests: Dict,
        embed_model=None,
        bm25_class=None,
        entity_graph_updater=None,
        relationship_analyzer=None,
        health_monitor=None,
        capsule_router=None,
        query_cache=None
    ) -> Dict[str, Any]:
        """Process and ingest a single file."""
        
        # Read file content
        content = await file.read()
        filename = file.filename or ""
        content_type = file.content_type or "application/octet-stream"
        
        # Extract text
        text = self.processor.extract_text_from_content(content, filename, content_type)
        
        # Check for duplicates
        duplicate_doc_id = self.deduplicator.check_duplicate(text, embed_model)
        if duplicate_doc_id:
            return {
                "doc_id": duplicate_doc_id,
                "num_chunks": len(document_corpus.get(duplicate_doc_id, [])),
                "content_hash": capsule_manifests[duplicate_doc_id].content_hash,
                "manifest_signature": capsule_manifests[duplicate_doc_id].signature,
                "duplicate_detected": True,
                "message": f"Document is a duplicate of existing document {duplicate_doc_id}"
            }
        
        # Get next document ID
        doc_id = len(docs) + 1
        
        # Store original file data
        doc_files[doc_id] = content
        doc_filenames[doc_id] = filename
        doc_mimetypes[doc_id] = content_type
        
        # Chunk the text with structure awareness and byte ranges
        chunks = chunk_text_structure_aware(text)
        
        # Store chunks in document corpus
        document_corpus[doc_id] = chunks
        docs[doc_id] = text
        
        # Create BM25 index for this document
        if bm25_class:
            chunk_texts = [chunk.text for chunk in chunks]
            tokenized_chunks = [tokenize(chunk_text) for chunk_text in chunk_texts]
            bm25_indices[doc_id] = bm25_class(tokenized_chunks)
        
        # Create vector embeddings
        if embed_model:
            try:
                chunk_texts = [chunk.text for chunk in chunks]
                embeddings = embed_model.encode(chunk_texts, normalize_embeddings=True)
                embeddings = np.asarray(embeddings, dtype=np.float32)
                vector_indices[doc_id] = embeddings
            except Exception as e:
                print(f"Warning: Failed to create embeddings: {e}")
                vector_indices[doc_id] = None
        
        # Extract entities and update knowledge graph
        entities = extract_entities_demo(text, filename)
        if entity_graph_updater:
            entity_graph_updater(doc_id, entities)
        
        # Analyze relationships with other documents
        if relationship_analyzer:
            relationship_analyzer.analyze_relationships(doc_id, entities)
        
        # Create content-addressed manifest
        manifest = create_capsule_manifest(doc_id, text, filename, len(chunks))
        capsule_manifests[doc_id] = manifest
        
        # Update capsule router with topic information
        if capsule_router:
            capsule_router.update_capsule_topic(doc_id, text, embed_model=embed_model)
        
        # Register content for deduplication
        self.deduplicator.register_content(doc_id, text)
        
        # Initialize health metrics
        if health_monitor:
            health_monitor.update_health_metrics(doc_id, document_corpus)
        
        # Invalidate cache since new document affects search results
        if query_cache:
            query_cache.invalidate_for_doc(doc_id)
        
        return {
            "doc_id": doc_id, 
            "num_chunks": len(chunks),
            "content_hash": manifest.content_hash,
            "manifest_signature": manifest.signature
        }
    
    async def ingest_batch(
        self,
        files: List[UploadFile],
        **kwargs  # Same parameters as ingest_single_file
    ) -> Dict[str, Any]:
        """Process multiple files in batch."""
        results = []
        successful = 0
        failed = 0
        
        for file in files:
            try:
                result = await self.ingest_single_file(file, **kwargs)
                results.append({
                    "filename": file.filename,
                    "status": "success",
                    "doc_id": result["doc_id"],
                    "num_chunks": result["num_chunks"],
                    "content_hash": result["content_hash"],
                    "duplicate_detected": result.get("duplicate_detected", False)
                })
                successful += 1
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "error": str(e)
                })
                failed += 1
        
        # Invalidate cache after batch operation
        if successful > 0 and "query_cache" in kwargs and kwargs["query_cache"]:
            kwargs["query_cache"].invalidate_for_doc(0)  # Clear all cache
        
        return {
            "total_files": len(files),
            "successful": successful,
            "failed": failed,
            "results": results
        }


# Global instances
deduplicator = ContentDeduplicator()
ingestion_engine = IngestionEngine(deduplicator)
