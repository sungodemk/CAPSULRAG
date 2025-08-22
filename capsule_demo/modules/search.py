"""
Search module for CapsuleRAG: intelligent search, ranking, caching, and routing.
"""

import os
import re
import time
import json
import hashlib
import threading
from typing import Dict, List, Optional, Any, Tuple
from collections import OrderedDict, defaultdict
import numpy as np
from fastapi import HTTPException
from pydantic import BaseModel, Field
import math

from .security import adversarial_detector, check_acl_permission
from .utils import tokenize


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(5, ge=1, le=50)
    rerank: bool = True
    rerank_top_k: int = Field(20, ge=1, le=200)


class QueryCache:
    """Intelligent caching system for search queries."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.cache = OrderedDict()
        self.access_times = {}
        self.query_stats = defaultdict(lambda: {"count": 0, "avg_time": 0.0})
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.lock = threading.RLock()
    
    def _cache_key(self, query: str, user_permissions: List[str], filters: Dict = None) -> str:
        """Generate cache key considering query, permissions, and filters."""
        perms_str = "|".join(sorted(user_permissions))
        filters_str = json.dumps(filters or {}, sort_keys=True)
        return hashlib.md5(f"{query}|{perms_str}|{filters_str}".encode()).hexdigest()
    
    def get(self, query: str, user_permissions: List[str], filters: Dict = None) -> Optional[Dict]:
        """Get cached result if valid and not expired."""
        key = self._cache_key(query, user_permissions, filters)
        
        with self.lock:
            if key not in self.cache:
                return None
            
            # Check TTL
            cache_time = self.access_times.get(key, 0)
            if time.time() - cache_time > self.ttl_seconds:
                del self.cache[key]
                del self.access_times[key]
                return None
            
            # Move to end (LRU)
            result = self.cache[key]
            del self.cache[key]
            self.cache[key] = result
            self.access_times[key] = time.time()
            
            return result
    
    def put(self, query: str, user_permissions: List[str], result: Dict, 
            query_time: float, filters: Dict = None):
        """Cache search result with performance tracking."""
        key = self._cache_key(query, user_permissions, filters)
        
        with self.lock:
            # Update query statistics
            stats = self.query_stats[query.lower()]
            stats["count"] += 1
            stats["avg_time"] = (stats["avg_time"] * (stats["count"] - 1) + query_time) / stats["count"]
            
            # Evict oldest if at capacity
            if len(self.cache) >= self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                del self.access_times[oldest_key]
            
            self.cache[key] = result
            self.access_times[key] = time.time()
    
    def invalidate_for_doc(self, doc_id: int):
        """Invalidate cache entries that might be affected by document changes."""
        with self.lock:
            # For simplicity, clear all cache when any document changes
            self.cache.clear()
            self.access_times.clear()
    
    def get_stats(self) -> Dict:
        """Get cache performance statistics."""
        with self.lock:
            return {
                "cache_size": len(self.cache),
                "max_size": self.max_size,
                "hit_rate": len(self.cache) / max(1, len(self.cache) + len(self.query_stats)),
                "popular_queries": dict(sorted(
                    self.query_stats.items(), 
                    key=lambda x: x[1]["count"], 
                    reverse=True
                )[:10])
            }


class CapsuleRouter:
    """Route queries to relevant capsules for faster search."""
    
    def __init__(self):
        self.capsule_topics: Dict[int, np.ndarray] = {}  # doc_id -> topic embedding
        self.topic_index = None  # Will be built when needed
        self.lock = threading.RLock()
    
    def update_capsule_topic(self, doc_id: int, text: str, embed_model=None):
        """Update topic embedding for a capsule."""
        if embed_model is None:
            return
            
        try:
            # Use first 500 chars as topic representative
            topic_text = text[:500] if len(text) > 500 else text
            topic_embedding = embed_model.encode([topic_text], normalize_embeddings=True)[0]
            
            with self.lock:
                self.capsule_topics[doc_id] = topic_embedding
                self.topic_index = None  # Invalidate index
        except Exception as e:
            print(f"Failed to update topic for capsule {doc_id}: {e}")
    
    def find_relevant_capsules(self, query: str, top_k: int = 5, embed_model=None) -> List[int]:
        """Find most relevant capsules for a query using topic similarity."""
        if not self.capsule_topics or embed_model is None:
            return list(self.capsule_topics.keys()) if self.capsule_topics else []
        
        try:
            query_embedding = embed_model.encode([query], normalize_embeddings=True)[0]
            
            with self.lock:
                similarities = []
                for doc_id, topic_emb in self.capsule_topics.items():
                    sim = np.dot(query_embedding, topic_emb)
                    similarities.append((doc_id, sim))
                
                # Sort by similarity and return top_k
                similarities.sort(key=lambda x: x[1], reverse=True)
                return [doc_id for doc_id, _ in similarities[:top_k]]
        
        except Exception as e:
            print(f"Capsule routing failed: {e}")
            return list(self.capsule_topics.keys())


# Intent classification patterns
QUERY_INTENT_PATTERNS = {
    "procedural": [
        r"how\s+do\s+i",
        r"how\s+to",
        r"step\s+by\s+step",
        r"procedure\s+for",
        r"instructions\s+for",
        r"guide\s+to"
    ],
    "factual": [
        r"what\s+is",
        r"define",
        r"definition\s+of",
        r"explain",
        r"describe"
    ],
    "comparative": [
        r"compare",
        r"difference\s+between",
        r"vs\s+",
        r"versus",
        r"better\s+than",
        r"which\s+is"
    ],
    "exploratory": [
        r"show\s+me",
        r"find\s+all",
        r"list",
        r"examples\s+of",
        r"related\s+to"
    ],
    "troubleshooting": [
        r"error",
        r"problem",
        r"issue",
        r"fault",
        r"malfunction",
        r"not\s+working",
        r"fix"
    ]
}


def classify_query_intent(query: str) -> str:
    """Classify query intent for retrieval strategy optimization."""
    query_lower = query.lower()
    
    for intent, patterns in QUERY_INTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, query_lower):
                return intent
    
    return "general"


class SearchEngine:
    """Main search engine with hybrid retrieval and intelligent features."""
    
    def __init__(self):
        self.cache = QueryCache()
        self.router = CapsuleRouter()
        
        # Intent-based search weights
        self.intent_weights = {
            "procedural": {"semantic": 0.4, "lexical": 0.6},  # Favor exact procedure matches
            "factual": {"semantic": 0.8, "lexical": 0.2},     # Favor semantic understanding
            "comparative": {"semantic": 0.7, "lexical": 0.3}, # Need conceptual understanding
            "exploratory": {"semantic": 0.6, "lexical": 0.4}, # Balanced approach
            "troubleshooting": {"semantic": 0.5, "lexical": 0.5}, # Need both exact and conceptual
            "general": {"semantic": 0.7, "lexical": 0.3}      # Default to semantic
        }
    
    def search(
        self,
        request: SearchRequest,
        user_permissions: List[str],
        docs: Dict,
        document_corpus: Dict,
        bm25_indices: Dict,
        vector_indices: Dict,
        capsule_manifests: Dict,
        embed_model=None,
        cross_encoder=None,
        health_monitor=None
    ) -> Dict[str, Any]:
        """Perform intelligent hybrid search with all enhancements."""
        
        if not docs:
            return {"results": [], "message": "No documents ingested yet"}
        
        start_time = time.time()
        
        # Security checks
        client_ip = "127.0.0.1"  # Simplified for demo
        
        # Check rate limiting
        rate_ok, rate_msg = adversarial_detector.check_rate_limit(client_ip)
        if not rate_ok:
            raise HTTPException(status_code=429, detail=rate_msg)
        
        # Check for adversarial queries
        is_suspicious, suspicion_reason = adversarial_detector.is_suspicious_query(request.query)
        if is_suspicious:
            print(f"Suspicious query detected from {client_ip}: {suspicion_reason}")
            from .security import DEMO_MODE
            if not DEMO_MODE:
                raise HTTPException(status_code=400, detail="Query contains suspicious content")
        
        # Check cache first
        cached_result = self.cache.get(request.query, user_permissions)
        if cached_result:
            cached_result["cache_hit"] = True
            cached_result["response_time_ms"] = (time.time() - start_time) * 1000
            return cached_result
        
        # Classify query intent for optimized retrieval
        query_intent = classify_query_intent(request.query)
        weights = self.intent_weights.get(query_intent, self.intent_weights["general"])
        
        # Use capsule routing to find relevant documents
        if len(docs) > 10:  # Only use routing for larger corpora
            candidate_docs = self.router.find_relevant_capsules(
                request.query, 
                top_k=min(10, len(docs)),
                embed_model=embed_model
            )
        else:
            candidate_docs = list(docs.keys())
        # Fallback: if routing produced no candidates, search all docs
        if not candidate_docs:
            candidate_docs = list(docs.keys())
 
        # Filter candidate documents based on ACL permissions
        accessible_docs = {}
        for doc_id in candidate_docs:
            if doc_id not in docs:
                continue
            allow = True
            if doc_id in capsule_manifests:
                try:
                    manifest = capsule_manifests[doc_id]
                    allow = check_acl_permission(manifest.acl, user_permissions, "read")
                except Exception:
                    allow = True
            if allow:
                accessible_docs[doc_id] = docs[doc_id]
 
        if not accessible_docs:
            return {"results": [], "message": "No accessible documents found"}
 
        
        try:
            # Lexical scoring per accessible document
            q_tokens = tokenize(request.query)
            lexical_scores: List[Tuple[int, float]] = []
            for doc_id in accessible_docs.keys():
                score = 0.0
                bm25 = bm25_indices.get(doc_id)
                if bm25 is not None:
                    try:
                        scores = bm25.get_scores(q_tokens)
                        # If index is per-chunk, take max or mean
                        if hasattr(scores, '__len__') and len(scores) > 1:
                            score = float(max(scores))
                        else:
                            score = float(scores[0])
                    except Exception:
                        score = 0.0
                lexical_scores.append((doc_id, score))
 
            # Select candidate documents by BM25
            lexical_scores.sort(key=lambda x: x[1], reverse=True)
            candidates = [doc_id for doc_id, _ in lexical_scores[:max(1, min(5, len(lexical_scores)))] ]
 
            # Normalize lexical scores
            max_lex = max((s for _, s in lexical_scores[:5]), default=0.0)
            doc_lex_norm = {doc_id: (score / max_lex) if max_lex > 0 else 0.0 for doc_id, score in lexical_scores}
 
            # Semantic retrieval per candidate capsule
            aggregated_results = []
            if embed_model:
                q_emb = embed_model.encode([request.query], normalize_embeddings=True)
                q_emb = np.asarray(q_emb, dtype=np.float32)
 
                for doc_id in candidates:
                    # Skip docs without embeddings or chunks
                    if doc_id not in vector_indices or vector_indices.get(doc_id) is None or doc_id not in document_corpus:
                        continue
 
                    index_obj = vector_indices[doc_id]
                    chunks_per_doc = 3
 
                    # Numpy fallback: compute inner product on normalized vectors
                    emb_mat: np.ndarray = index_obj
                    if emb_mat is None or emb_mat.size == 0:
                        continue
 
                    sims_vec = (emb_mat @ q_emb[0].reshape(-1, 1)).reshape(-1)
                    # Replace NaN/Inf to keep scores JSON-safe
                    sims_vec = np.nan_to_num(sims_vec, nan=0.0, posinf=1.0, neginf=-1.0)
                    k = int(min(max(1, chunks_per_doc), sims_vec.shape[0]))
                    top_indices = np.argpartition(-sims_vec, k - 1)[:k]
                    top_indices = top_indices[np.argsort(-sims_vec[top_indices])]
                    sims_row = sims_vec[top_indices].tolist()
                    idxs_row = top_indices.tolist()
 
                    lex = float(doc_lex_norm.get(doc_id, 0.0))
                    for sim, chunk_idx in zip(sims_row, idxs_row):
                        if chunk_idx < 0:
                            continue
                        if not np.isfinite(sim):
                            continue
                        # Map cosine similarity [-1, 1] -> [0, 1]
                        sem = (float(sim) + 1.0) / 2.0
                        combined = weights["semantic"] * sem + weights["lexical"] * lex
                        chunk = document_corpus[doc_id][chunk_idx]
                        snippet = chunk.text[:300] + "..." if len(chunk.text) > 300 else chunk.text
                        aggregated_results.append({
                            "doc_id": doc_id,
                            "score": combined,
                            "snippet": snippet,
                            "byte_range": {"start": chunk.start_byte, "end": chunk.end_byte},
                            "chunk_type": chunk.chunk_type,
                        })
 
            # Rerank globally and return top_k
            aggregated_results.sort(key=lambda r: r["score"], reverse=True)
 
            # Apply cross-encoder reranking if requested and available
            if request.rerank and len(aggregated_results) > 1 and cross_encoder:
                candidates = aggregated_results[:request.rerank_top_k]
                 
                try:
                    pairs = [(request.query, result["snippet"]) for result in candidates]
                    cross_scores = cross_encoder.predict(pairs)
                     
                    for i, result in enumerate(candidates):
                        original_score = result["score"]
                        cross_score = float(cross_scores[i])
                        if not math.isfinite(cross_score):
                            cross_score = 0.0
                        # Blend: 60% cross-encoder + 40% hybrid
                        result["score"] = 0.6 * cross_score + 0.4 * original_score
                        result["cross_encoder_score"] = cross_score
                     
                    candidates.sort(key=lambda r: r["score"], reverse=True)
                    results = candidates[:request.top_k]
                 
                except Exception as e:
                    print(f"Cross-encoder reranking failed: {e}")
                    results = aggregated_results[:request.top_k]
            else:
                results = aggregated_results[:request.top_k]
            
            # Record health metrics for accessed documents
            if health_monitor:
                for result in results:
                    doc_id = result["doc_id"]
                    relevance_score = result["score"]
                    health_monitor.record_access(doc_id, relevance_score)
            
            # Sanitize result scores
            for r in results:
                if not math.isfinite(r.get("score", 0.0)):
                    r["score"] = 0.0

            # Prepare final result
            query_time = time.time() - start_time
            final_result = {
                "results": results,
                "total_documents": len(docs),
                "accessible_documents": len(accessible_docs),
                "candidate_documents": len(candidate_docs),
                "query": request.query,
                "query_intent": query_intent,
                "search_weights": weights,
                "reranked": request.rerank and len(results) > 1,
                "user_permissions": user_permissions,
                "cache_hit": False,
                "response_time_ms": query_time * 1000
            }
            
            # Cache the result for future queries
            self.cache.put(request.query, user_permissions, final_result, query_time)
            
            return final_result
            
        except Exception as e:
            print(f"Search failed: {e}")
            from .security import DEMO_MODE
            if DEMO_MODE:
                return {"results": [], "error": str(e), "message": "Search failed"}
            else:
                return {"results": [], "message": "Search failed"}


# Global instances
search_engine = SearchEngine()
query_cache = search_engine.cache
capsule_router = search_engine.router
