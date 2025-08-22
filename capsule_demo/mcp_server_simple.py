#!/usr/bin/env python3
"""
Simplified CapsuleRAG MCP Server

Uses a minimal MCP implementation that follows the protocol but doesn't rely on 
complex library setup.
"""

import asyncio
import json
import sys
from typing import Any, Dict, List

"""Updated imports for modularized server"""
# Global stores and model loaders live in main.py now
from main import (
    docs,
    document_corpus,
    bm25_indices,
    vector_indices,
    capsule_manifests,
    _ensure_embed_model,
    _ensure_cross_encoder,
)

# Utilities and types moved to modules.utils
from modules.utils import (
    chunk_text_structure_aware as _chunk_text_structure_aware,
    create_capsule_manifest as _create_capsule_manifest,
    tokenize as _tokenize,
    TextChunk,
    CapsuleManifest,
)

# Security helpers moved to modules.security
from modules.security import check_acl_permission as _check_acl_permission

# BM25 class from rank_bm25
from rank_bm25 import BM25Okapi

# SearchRequest type (not strictly required here) is in modules.search
from modules.search import SearchRequest
import numpy as np

# Document counter for new ingestions
_doc_counter = 0


def get_next_doc_id() -> int:
    """Get next available document ID."""
    global _doc_counter
    _doc_counter = max(_doc_counter + 1, max(docs.keys(), default=0) + 1)
    return _doc_counter


class SimpleMCPServer:
    """Minimal MCP server implementation."""
    
    def __init__(self):
        self.tools = self._define_tools()
        
    def _define_tools(self) -> Dict[str, Dict]:
        """Define available MCP tools."""
        return {
            "capsules.ingest": {
                "description": "Ingest a document into CapsuleRAG, creating a new searchable capsule",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Document content to ingest"},
                        "filename": {"type": "string", "description": "Name of the document file"},
                        "user_permissions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "User permissions for ACL checking",
                            "default": ["public:read"]
                        }
                    },
                    "required": ["content", "filename"]
                }
            },
            "capsules.search": {
                "description": "Search across document capsules using hybrid retrieval",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "top_k": {"type": "integer", "default": 5, "minimum": 1, "maximum": 50},
                        "rerank": {"type": "boolean", "default": True},
                        "user_permissions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "default": ["public:read"]
                        }
                    },
                    "required": ["query"]
                }
            },
            "capsules.hydrate": {
                "description": "Retrieve full document content with ACL checking",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "doc_id": {"type": "integer", "description": "Document ID to retrieve"},
                        "user_permissions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "default": ["public:read"]
                        }
                    },
                    "required": ["doc_id"]
                }
            }
        }
    
    async def handle_request(self, request: Dict) -> Dict:
        """Handle incoming MCP request."""
        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")
        
        try:
            if method == "initialize":
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {
                            "tools": {"listChanged": False},
                            "resources": {"listChanged": False, "subscribe": False}
                        },
                        "serverInfo": {"name": "capsulerag", "version": "1.0.0"}
                    }
                }
            
            elif method == "tools/list":
                tools_list = []
                for name, tool_def in self.tools.items():
                    tools_list.append({
                        "name": name,
                        "description": tool_def["description"],
                        "inputSchema": tool_def["inputSchema"]
                    })
                
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"tools": tools_list}
                }
            
            elif method == "tools/call":
                tool_name = params.get("name")
                arguments = params.get("arguments", {})
                
                result = await self.call_tool(tool_name, arguments)
                
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [{"type": "text", "text": result}]
                    }
                }
            
            elif method == "resources/list":
                resources = []
                for doc_id in capsule_manifests.keys():
                    manifest = capsule_manifests[doc_id]
                    resources.append({
                        "uri": f"capsule://{doc_id}/manifest",
                        "name": f"Manifest for {manifest.filename}",
                        "description": f"Governance manifest for capsule {doc_id}",
                        "mimeType": "application/json"
                    })
                
                return {
                    "jsonrpc": "2.0", 
                    "id": request_id,
                    "result": {"resources": resources}
                }
            
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"}
                }
                
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32603, "message": f"Internal error: {str(e)}"}
            }
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool call and return result as JSON string."""
        
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
            
            return json.dumps({
                "doc_id": doc_id,
                "num_chunks": len(chunks),
                "content_hash": manifest.content_hash,
                "manifest_signature": manifest.signature,
                "filename": filename
            }, indent=2)
        
        elif name == "capsules.search":
            query = arguments["query"]
            top_k = arguments.get("top_k", 5)
            rerank = arguments.get("rerank", True)
            user_permissions = arguments.get("user_permissions", ["public:read"])
            
            if not docs:
                return json.dumps({"results": [], "message": "No documents ingested yet"})
            
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
                return json.dumps({"results": [], "message": "No accessible documents found"})
            
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
            candidates = lexical_scores[:top_k]
            
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
            results = aggregated_results[:top_k]
            
            return json.dumps({
                "results": results,
                "total_documents": len(docs),
                "accessible_documents": len(accessible_docs),
                "query": query,
                "reranked": False,  # Simplified for demo
                "user_permissions": user_permissions
            }, indent=2)
        
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
            
            return json.dumps({
                "doc_id": doc_id,
                "text": docs[doc_id]
            }, indent=2)
        
        else:
            raise ValueError(f"Unknown tool: {name}")


async def run_server():
    """Run the simplified MCP server over stdio."""
    server = SimpleMCPServer()
    
    while True:
        try:
            # Read request from stdin
            line = await asyncio.get_event_loop().run_in_executor(
                None, sys.stdin.readline
            )
            if not line:
                break
            
            # Parse JSON-RPC request
            request = json.loads(line.strip())
            
            # Handle request
            response = await server.handle_request(request)
            
            # Write response to stdout
            print(json.dumps(response), flush=True)
            
        except Exception as e:
            # Send error response
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32603, "message": f"Parse error: {str(e)}"}
            }
            print(json.dumps(error_response), flush=True)


if __name__ == "__main__":
    asyncio.run(run_server())
