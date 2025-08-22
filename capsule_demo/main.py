"""
Main CapsuleRAG application with modular architecture.
FastAPI app initialization and route registration.
"""

import os
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Query
from fastapi.responses import RedirectResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from typing import List, Dict, Any
import numpy as np

# Import all modules
from modules.security import (
    SecurityHeadersMiddleware, get_user_permissions, check_acl_permission,
    adversarial_detector, DEMO_MODE
)
from modules.utils import (
    summarize_text_to_bullets, generate_document_preview, extract_entities_demo
)
from modules.search import search_engine, SearchRequest, QUERY_INTENT_PATTERNS
from modules.ingest import ingestion_engine, deduplicator
from modules.health import health_monitor, lifecycle_engine
from modules.relationships import relationship_miner
from modules.enrichment import enrichment_engine

# Global storage (in production, this would be a proper database)
docs: Dict[int, str] = {}
doc_files: Dict[int, bytes] = {}
doc_filenames: Dict[int, str] = {}
doc_mimetypes: Dict[int, str] = {}
document_corpus: Dict[int, List] = {}
bm25_indices: Dict[int, Any] = {}
vector_indices: Dict[int, Any] = {}
capsule_manifests: Dict[int, Any] = {}

# Entity graph storage
entity_graph: Dict[str, Dict[str, List[str]]] = {}
entity_relations: Dict[str, List[Dict[str, str]]] = {}

# Models (initialized lazily)
_embed_model = None
_cross_encoder = None

# Model loading functions
def _ensure_embed_model():
    global _embed_model
    if _embed_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            model_name = "all-MiniLM-L12-v2" if os.getenv("FAST_MODE", "false").lower() == "true" else "all-MiniLM-L6-v2"
            print(f"Loading embedding model: {model_name}")
            device = "cpu"  # Force CPU to avoid meta tensor issues
            _embed_model = SentenceTransformer(model_name, device=device)
        except ImportError as e:
            raise HTTPException(status_code=500, detail=f"sentence-transformers not available: {e}")
        except Exception as e:
            print(f"Model loading error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load embedding model: {e}")
    return _embed_model

def _ensure_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder
            model_name = "cross-encoder/ms-marco-TinyBERT-L-2-v2" if os.getenv("FAST_MODE", "false").lower() == "true" else "cross-encoder/ms-marco-MiniLM-L-6-v2"
            print(f"Loading cross-encoder model: {model_name}")
            device = "cpu"
            _cross_encoder = CrossEncoder(model_name, device=device)
        except ImportError as e:
            raise HTTPException(status_code=500, detail=f"sentence-transformers not available: {e}")
        except Exception as e:
            print(f"Cross-encoder loading error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load cross-encoder model: {e}")
    return _cross_encoder

# Entity graph updater
def _update_entity_graph(doc_id: int, entities: Dict[str, List[str]]):
    """Update the entity graph with extracted entities."""
    global entity_graph, entity_relations
    
    doc_id_str = str(doc_id)
    
    for entity_type, entity_list in entities.items():
        if entity_type not in entity_graph:
            entity_graph[entity_type] = {}
            
        for entity in entity_list:
            if entity not in entity_graph[entity_type]:
                entity_graph[entity_type][entity] = []
            
            if doc_id_str not in entity_graph[entity_type][entity]:
                entity_graph[entity_type][entity].append(doc_id_str)
            
            # Create demo relations
            if entity not in entity_relations:
                entity_relations[entity] = []
            
            # Link equipment to incidents (demo logic)
            if entity_type == "equipment":
                for incident in entities.get("incident", []):
                    relation = {"relation": "involved_in", "target": incident}
                    if relation not in entity_relations[entity]:
                        entity_relations[entity].append(relation)

# Initialize FastAPI app
app = FastAPI(title="CapsuleRAG", description="Intelligent Document Retrieval with Governance")

# Add security middleware
app.add_middleware(SecurityHeadersMiddleware)

# Model pre-loading
PRELOAD_MODELS = os.getenv("PRELOAD_MODELS", "true").lower() == "true"
if PRELOAD_MODELS:
    import threading
    
    def preload_models():
        try:
            print("Pre-loading embedding model...")
            _ensure_embed_model()
            print("Pre-loading cross-encoder model...")
            _ensure_cross_encoder()
            print("Model pre-loading complete!")
        except Exception as e:
            print(f"Model pre-loading failed: {e}")
    
    print("Starting model pre-loading in background...")
    threading.Thread(target=preload_models, daemon=True).start()

# Routes
@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """Ingest a single document."""
    try:
        from rank_bm25 import BM25Okapi
        embed_model = _ensure_embed_model()
        
        result = await ingestion_engine.ingest_single_file(
            file=file,
            docs=docs,
            doc_files=doc_files,
            doc_filenames=doc_filenames,
            doc_mimetypes=doc_mimetypes,
            document_corpus=document_corpus,
            bm25_indices=bm25_indices,
            vector_indices=vector_indices,
            capsule_manifests=capsule_manifests,
            embed_model=embed_model,
            bm25_class=BM25Okapi,
            entity_graph_updater=_update_entity_graph,
            relationship_analyzer=relationship_miner,
            health_monitor=health_monitor,
            capsule_router=search_engine.router,
            query_cache=search_engine.cache
        )
        return result
        
    except Exception as e:
        print(f"Ingestion failed: {e}")
        if DEMO_MODE:
            raise HTTPException(status_code=500, detail=str(e))
        else:
            raise HTTPException(status_code=500, detail="Ingestion failed")

@app.post("/ingest/batch")
async def ingest_batch(files: List[UploadFile] = File(...)):
    """Ingest multiple documents."""
    try:
        from rank_bm25 import BM25Okapi
        embed_model = _ensure_embed_model()
        
        result = await ingestion_engine.ingest_batch(
            files=files,
            docs=docs,
            doc_files=doc_files,
            doc_filenames=doc_filenames,
            doc_mimetypes=doc_mimetypes,
            document_corpus=document_corpus,
            bm25_indices=bm25_indices,
            vector_indices=vector_indices,
            capsule_manifests=capsule_manifests,
            embed_model=embed_model,
            bm25_class=BM25Okapi,
            entity_graph_updater=_update_entity_graph,
            relationship_analyzer=relationship_miner,
            health_monitor=health_monitor,
            capsule_router=search_engine.router,
            query_cache=search_engine.cache
        )
        return result
        
    except Exception as e:
        print(f"Batch ingestion failed: {e}")
        if DEMO_MODE:
            raise HTTPException(status_code=500, detail=str(e))
        else:
            raise HTTPException(status_code=500, detail="Batch ingestion failed")

@app.post("/search")
async def search(request: SearchRequest, user_permissions: List[str] = Depends(get_user_permissions)):
    """Enhanced hybrid search with all improvements."""
    try:
        embed_model = _ensure_embed_model()
        cross_encoder = _ensure_cross_encoder() if request.rerank else None
        
        result = search_engine.search(
            request=request,
            user_permissions=user_permissions,
            docs=docs,
            document_corpus=document_corpus,
            bm25_indices=bm25_indices,
            vector_indices=vector_indices,
            capsule_manifests=capsule_manifests,
            embed_model=embed_model,
            cross_encoder=cross_encoder,
            health_monitor=health_monitor
        )
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Search failed: {e}")
        if DEMO_MODE:
            return {"results": [], "error": str(e), "message": "Search failed"}
        else:
            return {"results": [], "message": "Search failed"}

@app.get("/status")
async def status():
    """System status and document count."""
    return {
        "total_documents": len(docs),
        "bm25_indices": len(bm25_indices),
        "vector_indices": len(vector_indices),
        "document_corpus": len(document_corpus),
        "manifests": len(capsule_manifests),
        "document_ids": list(docs.keys()),
        "models_loaded": {
            "embedding_model": _embed_model is not None,
            "cross_encoder": _cross_encoder is not None
        },
        "demo_mode": DEMO_MODE,
        "preload_enabled": PRELOAD_MODELS,
        "cache_stats": search_engine.cache.get_stats(),
        "routing_stats": {
            "capsules_indexed": len(search_engine.router.capsule_topics),
            "routing_enabled": len(search_engine.router.capsule_topics) > 0
        }
    }

@app.get("/analytics")
async def analytics():
    """Detailed analytics about search performance and usage patterns."""
    cache_stats = search_engine.cache.get_stats()
    health_summary = health_monitor.get_health_summary()
    
    return {
        "cache_performance": cache_stats,
        "intent_patterns": QUERY_INTENT_PATTERNS,
        "capsule_routing": {
            "total_capsules": len(search_engine.router.capsule_topics),
            "routing_threshold": 10,
            "topic_coverage": len(search_engine.router.capsule_topics) / max(1, len(docs))
        },
        "capsule_health": health_summary,
        "system_health": {
            "avg_query_time": sum(stat["avg_time"] for stat in cache_stats["popular_queries"].values()) / max(1, len(cache_stats["popular_queries"])),
            "cache_efficiency": cache_stats["hit_rate"],
            "avg_capsule_health": health_summary.get("avg_health_score", 0.0)
        }
    }

@app.get("/health")
async def get_health_overview():
    """Capsule health overview and recommendations."""
    health_summary = health_monitor.get_health_summary()
    
    # Update health metrics for all capsules
    for doc_id in docs.keys():
        health_monitor.update_health_metrics(doc_id, document_corpus)
    
    # Get detailed recommendations
    recommendations = []
    for doc_id in docs.keys():
        health = health_monitor.get_capsule_health(doc_id)
        if health.get("recommendation") in ["archive_candidate", "needs_attention", "low_usage"]:
            recommendations.append({
                "doc_id": doc_id,
                "filename": doc_filenames.get(doc_id, f"document_{doc_id}"),
                "health_score": health.get("health_score", 0.0),
                "recommendation": health.get("recommendation", "unknown"),
                "days_since_access": health.get("days_since_access", float('inf')),
                "total_accesses": health.get("total_accesses", 0)
            })
    
    recommendations.sort(key=lambda x: x["health_score"])
    
    return {
        "summary": health_summary,
        "recommendations": recommendations[:10],
        "deduplication_stats": deduplicator.get_stats(),
        "lifecycle_summary": lifecycle_engine.get_lifecycle_summary()
    }

@app.get("/health/{doc_id}")
async def get_capsule_health(doc_id: int):
    """Detailed health metrics for a specific capsule."""
    if doc_id not in docs:
        raise HTTPException(status_code=404, detail="Document not found")
    
    health_monitor.update_health_metrics(doc_id, document_corpus)
    health = health_monitor.get_capsule_health(doc_id)
    lifecycle = lifecycle_engine.analyze_lifecycle_trends(doc_id, health)
    
    return {
        "doc_id": doc_id,
        "filename": doc_filenames.get(doc_id, f"document_{doc_id}"),
        "health_metrics": health,
        "lifecycle_analysis": lifecycle,
        "manifest": capsule_manifests.get(doc_id, {})._asdict() if doc_id in capsule_manifests else {}
    }

@app.get("/security")
async def get_security_overview():
    """Security monitoring overview and statistics."""
    security_stats = adversarial_detector.get_security_stats()
    
    return {
        "rate_limiting": security_stats,
        "demo_mode": DEMO_MODE,
        "security_features": {
            "adversarial_detection": True,
            "rate_limiting": True,
            "api_key_auth": True,
            "acl_enforcement": True,
            "content_sanitization": not DEMO_MODE
        },
        "threat_indicators": {
            "suspicious_pattern_count": security_stats["suspicious_patterns_count"],
            "total_patterns_checked": "per_query",
            "rate_limit_enforcement": "active"
        }
    }

@app.get("/relationships/{doc_id}")
async def get_document_relationships(doc_id: int):
    """Related documents and relationship analysis for a specific capsule."""
    if doc_id not in docs:
        raise HTTPException(status_code=404, detail="Document not found")
    
    related_capsules = relationship_miner.find_related_capsules(doc_id, docs, doc_filenames)
    
    relationships = []
    for related_doc_id, similarity, rel_type in related_capsules:
        relationships.append({
            "doc_id": related_doc_id,
            "filename": doc_filenames.get(related_doc_id, f"document_{related_doc_id}"),
            "similarity": similarity,
            "relationship_type": rel_type,
            "shared_entities": similarity * 10
        })
    
    return {
        "doc_id": doc_id,
        "filename": doc_filenames.get(doc_id, f"document_{doc_id}"),
        "related_documents": relationships,
        "relationship_count": len(relationships)
    }

@app.get("/network")
async def get_entity_network():
    """Entity co-occurrence network for visualization."""
    network = relationship_miner.get_entity_network()
    insights = relationship_miner.get_relationship_insights(docs, doc_filenames)
    
    return {
        "entity_network": network,
        "relationship_insights": insights,
        "summary": {
            "total_entities": network["node_count"],
            "total_relationships": network["edge_count"],
            "network_density": network["edge_count"] / max(1, network["node_count"] * (network["node_count"] - 1))
        }
    }

@app.get("/summarize/{doc_id}")
async def summarize(doc_id: int, max_bullets: int = Query(5, ge=1, le=20), 
                   user_permissions: List[str] = Depends(get_user_permissions)):
    """Generate bullet-point summary of document."""
    if doc_id not in docs:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Check ACL permissions
    if doc_id in capsule_manifests:
        manifest = capsule_manifests[doc_id]
        if not check_acl_permission(manifest.acl, user_permissions, "read"):
            raise HTTPException(status_code=403, detail="Access denied")
    
    text = docs[doc_id]
    embed_model = _ensure_embed_model()
    bullets = summarize_text_to_bullets(text, max_bullets=max_bullets, embed_model=embed_model)
    return {"doc_id": doc_id, "bullets": bullets}

@app.get("/preview/{doc_id}")
async def preview(doc_id: int, user_permissions: List[str] = Depends(get_user_permissions)):
    """Generate preview images for a document."""
    if doc_id not in doc_files:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Check ACL permissions
    if doc_id in capsule_manifests:
        manifest = capsule_manifests[doc_id]
        if not check_acl_permission(manifest.acl, user_permissions, "read"):
            raise HTTPException(status_code=403, detail="Access denied")
    
    content = doc_files[doc_id]
    content_type = doc_mimetypes.get(doc_id, "application/octet-stream")
    preview_images = generate_document_preview(content, content_type)
    filename = doc_filenames.get(doc_id, f"document_{doc_id}")
    
    return {
        "doc_id": doc_id,
        "filename": filename,
        "content_type": content_type,
        "preview_images": preview_images,
        "page_count": len(preview_images)
    }

@app.get("/download/{doc_id}")
async def download(doc_id: int, user_permissions: List[str] = Depends(get_user_permissions)):
    """Download the original document file."""
    if doc_id not in doc_files:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Check ACL permissions
    if doc_id in capsule_manifests:
        manifest = capsule_manifests[doc_id]
        if not check_acl_permission(manifest.acl, user_permissions, "read"):
            raise HTTPException(status_code=403, detail="Access denied")
    
    filename = doc_filenames.get(doc_id, f"document_{doc_id}")
    content_type = doc_mimetypes.get(doc_id, "application/octet-stream")
    content = doc_files[doc_id]
    
    return Response(
        content=content,
        media_type=content_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@app.get("/entities")
async def list_entities():
    """List all entities in the knowledge graph."""
    return {
        "entity_graph": entity_graph,
        "entity_relations": entity_relations,
        "total_entities": sum(len(entities) for entities in entity_graph.values())
    }

@app.get("/enrichment/{doc_id}")
async def get_enrichment(doc_id: int):
    """Get enriched metadata for a specific document."""
    if doc_id not in docs:
        raise HTTPException(status_code=404, detail="Document not found")
    
    enriched = enrichment_engine.get_enriched_metadata(doc_id)
    if not enriched:
        # Trigger enrichment if not already processed
        enrichment_engine.enqueue_enrichment(
            doc_id=doc_id,
            text=docs[doc_id],
            filename=doc_filenames.get(doc_id, f"document_{doc_id}"),
            priority="high"
        )
        return {"message": "Enrichment queued", "status": "processing"}
    
    return enriched

@app.get("/enrichment")
async def enrichment_overview():
    """Get overview of content enrichment processing."""
    stats = enrichment_engine.get_enrichment_stats()
    
    # Get sample enriched documents
    enriched_samples = []
    for doc_id in list(docs.keys())[:5]:  # First 5 docs
        enriched = enrichment_engine.get_enriched_metadata(doc_id)
        if enriched:
            enriched_samples.append({
                "doc_id": doc_id,
                "filename": doc_filenames.get(doc_id, f"document_{doc_id}"),
                "quality_score": enriched.get("quality_score", 0.0),
                "entity_count": enriched.get("entity_count", 0),
                "content_type": enriched.get("content_type", {}).get("primary_type", "unknown")
            })
    
    return {
        "processing_stats": stats,
        "enriched_samples": enriched_samples,
        "total_documents": len(docs)
    }

@app.get("/")
async def root():
    """Redirect to the index.html page."""
    return RedirectResponse(url="/static/index.html")

# Serve static files
frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
