#!/usr/bin/env python3
"""
CapsuleRAG MCP Server

Exposes CapsuleRAG functionality as MCP tools and resources following the 
Model Context Protocol specification.
"""

import asyncio
import json
import sys
from typing import Any, Dict, List, Optional, Sequence

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    LoggingLevel,
    CallToolRequest,
    ListResourcesRequest,
    ListToolsRequest,
    ReadResourceRequest,
)

# Import our existing CapsuleRAG functions
from server import (
    docs,
    document_corpus,
    bm25_indices,
    vector_indices,
    capsule_manifests,
    _ensure_embed_model,
    _ensure_cross_encoder,
    _chunk_text_structure_aware,
    _create_capsule_manifest,
    _tokenize,
    _check_acl_permission,
    BM25Okapi,
    TextChunk,
    CapsuleManifest,
    SearchRequest,
)
import numpy as np

# Initialize MCP server
server = Server("capsulerag")

# Document counter for new ingestions
_doc_counter = 0


def get_next_doc_id() -> int:
    """Get next available document ID."""
    global _doc_counter
    _doc_counter = max(_doc_counter + 1, max(docs.keys(), default=0) + 1)
    return _doc_counter


@server.list_tools()
async def handle_list_tools() -> List[Tool]:
    """List available CapsuleRAG tools."""
    return [
        Tool(
            name="capsules.ingest",
            description="Ingest a document into CapsuleRAG, creating a new searchable capsule",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Document content to ingest"
                    },
                    "filename": {
                        "type": "string",
                        "description": "Name of the document file"
                    },
                    "user_permissions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "User permissions for ACL checking",
                        "default": ["public:read"]
                    }
                },
                "required": ["content", "filename"]
            }
        ),
        Tool(
            name="capsules.search",
            description="Search across document capsules using hybrid retrieval (BM25 + semantic + cross-encoder reranking)",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 50
                    },
                    "rerank": {
                        "type": "boolean",
                        "description": "Enable cross-encoder reranking",
                        "default": True
                    },
                    "rerank_top_k": {
                        "type": "integer",
                        "description": "Number of candidates for reranking",
                        "default": 20
                    },
                    "user_permissions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "User permissions for ACL filtering",
                        "default": ["public:read"]
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="capsules.hydrate",
            description="Retrieve full document content with ACL checking",
            inputSchema={
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "integer",
                        "description": "Document ID to retrieve"
                    },
                    "user_permissions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "User permissions for ACL checking",
                        "default": ["public:read"]
                    }
                },
                "required": ["doc_id"]
            }
        ),
        Tool(
            name="capsules.manifest",
            description="Get governance manifest for a capsule (content hash, signature, ACL, model info)",
            inputSchema={
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "integer",
                        "description": "Document ID to get manifest for"
                    }
                },
                "required": ["doc_id"]
            }
        ),
    ]


@server.list_resources()
async def handle_list_resources() -> List[Resource]:
    """List available CapsuleRAG resources."""
    resources = []
    
    # Add manifests as resources
    for doc_id in capsule_manifests.keys():
        manifest = capsule_manifests[doc_id]
        resources.append(Resource(
            uri=f"capsule://{doc_id}/manifest",
            name=f"Manifest for {manifest.filename}",
            description=f"Governance manifest for capsule {doc_id}",
            mimeType="application/json"
        ))
    
    # Add document chunks as resources
    for doc_id, chunks in document_corpus.items():
        for i, chunk in enumerate(chunks):
            resources.append(Resource(
                uri=f"capsule://{doc_id}/chunk/{i}",
                name=f"Chunk {i} from {capsule_manifests.get(doc_id, type('', (), {'filename': f'doc_{doc_id}'})).filename}",
                description=f"Text chunk {i} from document {doc_id} ({chunk.chunk_type})",
                mimeType="text/plain"
            ))
    
    return resources


@server.read_resource()
async def handle_read_resource(uri: str) -> str:
    """Read resource content by URI."""
    if uri.startswith("capsule://"):
        parts = uri[10:].split("/")  # Remove "capsule://"
        
        if len(parts) >= 2:
            doc_id = int(parts[0])
            resource_type = parts[1]
            
            if resource_type == "manifest":
                if doc_id in capsule_manifests:
                    manifest = capsule_manifests[doc_id]
                    return json.dumps({
                        "doc_id": manifest.doc_id,
                        "content_hash": manifest.content_hash,
                        "created_at": manifest.created_at,
                        "filename": manifest.filename,
                        "num_chunks": manifest.num_chunks,
                        "model_info": manifest.model_info,
                        "acl": manifest.acl,
                        "signature": manifest.signature
                    }, indent=2)
                else:
                    raise ValueError(f"Manifest not found for document {doc_id}")
            
            elif resource_type == "chunk" and len(parts) >= 3:
                chunk_idx = int(parts[2])
                if doc_id in document_corpus and chunk_idx < len(document_corpus[doc_id]):
                    chunk = document_corpus[doc_id][chunk_idx]
                    return json.dumps({
                        "text": chunk.text,
                        "start_byte": chunk.start_byte,
                        "end_byte": chunk.end_byte,
                        "chunk_type": chunk.chunk_type
                    }, indent=2)
                else:
                    raise ValueError(f"Chunk {chunk_idx} not found for document {doc_id}")
    
    raise ValueError(f"Unknown resource URI: {uri}")


@server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls."""
    
    if name == "capsules.ingest":
        content = arguments["content"]
        filename = arguments["filename"]
        user_permissions = arguments.get("user_permissions", ["public:read"])
        
        # Get next doc ID
        doc_id = get_next_doc_id()
        
        # Store document
        docs[doc_id] = content
        
        # Create structure-aware chunks
        chunks = _chunk_text_structure_aware(content)
        document_corpus[doc_id] = chunks
        
        # Build BM25 index
        tokens = _tokenize(content)
        bm25_indices[doc_id] = BM25Okapi([tokens])
        
        # Build embeddings
        embed_model = _ensure_embed_model()
        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = embed_model.encode(chunk_texts, normalize_embeddings=True)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        vector_indices[doc_id] = embeddings
        
        # Create manifest
        manifest = _create_capsule_manifest(doc_id, content, filename, len(chunks))
        capsule_manifests[doc_id] = manifest
        
        return [TextContent(
            type="text",
            text=json.dumps({
                "doc_id": doc_id,
                "num_chunks": len(chunks),
                "content_hash": manifest.content_hash,
                "manifest_signature": manifest.signature,
                "filename": filename
            }, indent=2)
        )]
    
    elif name == "capsules.search":
        query = arguments["query"]
        top_k = arguments.get("top_k", 5)
        rerank = arguments.get("rerank", True)
        rerank_top_k = arguments.get("rerank_top_k", 20)
        user_permissions = arguments.get("user_permissions", ["public:read"])
        
        if not docs:
            return [TextContent(type="text", text='{"results": [], "message": "No documents ingested yet"}')]
        
        # Filter accessible documents
        accessible_docs = {}
        for doc_id in docs.keys():
            if doc_id in capsule_manifests:
                manifest = capsule_manifests[doc_id]
                if _check_acl_permission(manifest.acl, user_permissions, "read"):
                    accessible_docs[doc_id] = docs[doc_id]
            else:
                accessible_docs[doc_id] = docs[doc_id]
        
        if not accessible_docs:
            return [TextContent(type="text", text='{"results": [], "message": "No accessible documents found"}')]
        
        # Perform hybrid search
        q_tokens = _tokenize(query)
        lexical_scores = []
        
        for doc_id in accessible_docs.keys():
            if doc_id in bm25_indices:
                bm25 = bm25_indices[doc_id]
                try:
                    score = float(bm25.get_scores(q_tokens)[0])
                except Exception:
                    score = 0.0
                lexical_scores.append((doc_id, score))
        
        # Sort by lexical score and get top candidates
        lexical_scores.sort(key=lambda x: x[1], reverse=True)
        candidates = lexical_scores[:rerank_top_k] if rerank else lexical_scores[:top_k]
        
        # Semantic search on candidates
        embed_model = _ensure_embed_model()
        q_emb = embed_model.encode([query], normalize_embeddings=True)
        
        aggregated_results = []
        chunks_per_doc = max(1, top_k // len(candidates)) if candidates else 1
        
        for doc_id, lex_score in candidates:
            if doc_id in vector_indices and doc_id in document_corpus:
                embeddings = vector_indices[doc_id]
                chunks = document_corpus[doc_id]
                
                # Compute semantic similarities
                sims = np.dot(embeddings, q_emb.T).flatten()
                
                # Get top chunks for this document
                top_chunk_indices = np.argsort(sims)[::-1][:chunks_per_doc]
                
                for chunk_idx in top_chunk_indices:
                    if chunk_idx < len(chunks):
                        chunk = chunks[chunk_idx]
                        sem_score = float(sims[chunk_idx])
                        combined_score = 0.7 * sem_score + 0.3 * lex_score
                        
                        aggregated_results.append({
                            "doc_id": doc_id,
                            "score": combined_score,
                            "snippet": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                            "byte_range": {"start": chunk.start_byte, "end": chunk.end_byte},
                            "chunk_type": chunk.chunk_type,
                        })
        
        # Sort by combined score
        aggregated_results.sort(key=lambda r: r["score"], reverse=True)
        
        # Apply cross-encoder reranking if requested
        if rerank and len(aggregated_results) > 1:
            candidates = aggregated_results[:rerank_top_k]
            try:
                cross_encoder = _ensure_cross_encoder()
                pairs = [(query, result["snippet"]) for result in candidates]
                cross_scores = cross_encoder.predict(pairs)
                
                for i, result in enumerate(candidates):
                    original_score = result["score"]
                    cross_score = float(cross_scores[i])
                    result["score"] = 0.6 * cross_score + 0.4 * original_score
                    result["cross_encoder_score"] = cross_score
                
                candidates.sort(key=lambda r: r["score"], reverse=True)
                results = candidates[:top_k]
            except Exception as e:
                results = aggregated_results[:top_k]
        else:
            results = aggregated_results[:top_k]
        
        return [TextContent(
            type="text",
            text=json.dumps({
                "results": results,
                "total_documents": len(docs),
                "accessible_documents": len(accessible_docs),
                "query": query,
                "reranked": rerank and len(results) > 1,
                "user_permissions": user_permissions
            }, indent=2)
        )]
    
    elif name == "capsules.hydrate":
        doc_id = arguments["doc_id"]
        user_permissions = arguments.get("user_permissions", ["public:read"])
        
        if doc_id not in docs:
            raise ValueError(f"Document {doc_id} not found")
        
        # Check ACL permissions
        if doc_id in capsule_manifests:
            manifest = capsule_manifests[doc_id]
            if not _check_acl_permission(manifest.acl, user_permissions, "read"):
                raise PermissionError("Access denied")
        
        return [TextContent(
            type="text",
            text=json.dumps({
                "doc_id": doc_id,
                "text": docs[doc_id]
            }, indent=2)
        )]
    
    elif name == "capsules.manifest":
        doc_id = arguments["doc_id"]
        
        if doc_id not in capsule_manifests:
            raise ValueError(f"Manifest not found for document {doc_id}")
        
        manifest = capsule_manifests[doc_id]
        return [TextContent(
            type="text",
            text=json.dumps({
                "doc_id": manifest.doc_id,
                "content_hash": manifest.content_hash,
                "created_at": manifest.created_at,
                "filename": manifest.filename,
                "num_chunks": manifest.num_chunks,
                "model_info": manifest.model_info,
                "acl": manifest.acl,
                "signature": manifest.signature
            }, indent=2)
        )]
    
    else:
        raise ValueError(f"Unknown tool: {name}")


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="capsulerag",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
