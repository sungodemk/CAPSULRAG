"""
Health monitoring module for CapsuleRAG: capsule health metrics and lifecycle prediction.
"""

import time
import threading
from typing import Dict, List, Any
from collections import defaultdict


class CapsuleHealthMonitor:
    """Monitor and track health metrics for individual capsules."""
    
    def __init__(self):
        self.health_metrics: Dict[int, Dict[str, Any]] = {}
        self.access_logs: Dict[int, List[float]] = defaultdict(list)
        self.lock = threading.RLock()
    
    def record_access(self, doc_id: int, relevance_score: float):
        """Record access to a capsule with relevance score."""
        with self.lock:
            self.access_logs[doc_id].append(time.time())
            
            # Keep only last 100 accesses per capsule
            if len(self.access_logs[doc_id]) > 100:
                self.access_logs[doc_id] = self.access_logs[doc_id][-100:]
    
    def update_health_metrics(self, doc_id: int, document_corpus: Dict):
        """Update comprehensive health metrics for a capsule."""
        if doc_id not in document_corpus:
            return
        
        with self.lock:
            chunks = document_corpus[doc_id]
            access_times = self.access_logs.get(doc_id, [])
            
            # Calculate metrics
            now = time.time()
            recent_accesses = [t for t in access_times if now - t < 7 * 24 * 3600]  # Last 7 days
            
            # Chunk quality metrics
            avg_chunk_length = sum(len(chunk.text) for chunk in chunks) / max(1, len(chunks))
            chunk_type_diversity = len(set(chunk.chunk_type for chunk in chunks)) / max(1, len(chunks))
            
            # Usage metrics
            total_accesses = len(access_times)
            recent_access_rate = len(recent_accesses) / max(1, 7)  # Accesses per day
            last_accessed = max(access_times) if access_times else 0
            days_since_access = (now - last_accessed) / (24 * 3600) if last_accessed else float('inf')
            
            # Health score (0-1)
            recency_score = min(1.0, max(0.0, 1.0 - days_since_access / 30))  # Decay over 30 days
            usage_score = min(1.0, recent_access_rate / 5)  # Normalize to 5 accesses/day = 1.0
            quality_score = min(1.0, (avg_chunk_length / 100) * chunk_type_diversity)
            
            health_score = (recency_score * 0.4 + usage_score * 0.4 + quality_score * 0.2)
            
            self.health_metrics[doc_id] = {
                "health_score": health_score,
                "total_accesses": total_accesses,
                "recent_access_rate": recent_access_rate,
                "days_since_access": days_since_access,
                "avg_chunk_length": avg_chunk_length,
                "chunk_count": len(chunks),
                "chunk_type_diversity": chunk_type_diversity,
                "last_updated": now,
                "recommendation": self._get_recommendation(health_score, days_since_access, recent_access_rate)
            }
    
    def _get_recommendation(self, health_score: float, days_since_access: float, access_rate: float) -> str:
        """Generate recommendation based on metrics."""
        if health_score > 0.8:
            return "high_value"
        elif health_score > 0.5:
            return "moderate_value"
        elif days_since_access > 90:
            return "archive_candidate"
        elif access_rate < 0.1:
            return "low_usage"
        else:
            return "needs_attention"
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary."""
        with self.lock:
            if not self.health_metrics:
                return {"total_capsules": 0, "avg_health": 0.0, "recommendations": {}}
            
            health_scores = [m["health_score"] for m in self.health_metrics.values()]
            recommendations = [m["recommendation"] for m in self.health_metrics.values()]
            
            return {
                "total_capsules": len(self.health_metrics),
                "avg_health_score": sum(health_scores) / len(health_scores),
                "health_distribution": {
                    "high": len([s for s in health_scores if s > 0.8]),
                    "medium": len([s for s in health_scores if 0.5 <= s <= 0.8]),
                    "low": len([s for s in health_scores if s < 0.5])
                },
                "recommendations": {
                    rec: recommendations.count(rec) for rec in set(recommendations)
                }
            }
    
    def get_capsule_health(self, doc_id: int) -> Dict[str, Any]:
        """Get health metrics for specific capsule."""
        with self.lock:
            return self.health_metrics.get(doc_id, {})


class LifecyclePredictionEngine:
    """Predict capsule lifecycle and optimal management strategies."""
    
    def __init__(self):
        self.lifecycle_data: Dict[int, Dict[str, Any]] = {}
        self.lock = threading.RLock()
    
    def analyze_lifecycle_trends(self, doc_id: int, health_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze lifecycle trends and predict future states."""
        with self.lock:
            now = time.time()
            
            # Track lifecycle progression
            if doc_id not in self.lifecycle_data:
                self.lifecycle_data[doc_id] = {
                    "creation_time": now,
                    "health_history": [],
                    "usage_patterns": [],
                    "lifecycle_stage": "new"
                }
            
            lifecycle = self.lifecycle_data[doc_id]
            
            # Record current health
            current_health = health_metrics.get("health_score", 0.0)
            lifecycle["health_history"].append({
                "timestamp": now,
                "health_score": current_health,
                "access_rate": health_metrics.get("recent_access_rate", 0.0)
            })
            
            # Keep only last 30 data points
            if len(lifecycle["health_history"]) > 30:
                lifecycle["health_history"] = lifecycle["health_history"][-30:]
            
            # Determine lifecycle stage
            days_since_creation = (now - lifecycle["creation_time"]) / (24 * 3600)
            recent_access_rate = health_metrics.get("recent_access_rate", 0.0)
            
            if days_since_creation < 7:
                stage = "new"
            elif recent_access_rate > 1.0:
                stage = "active"
            elif recent_access_rate > 0.1:
                stage = "moderate"
            elif days_since_creation > 90:
                stage = "aging"
            else:
                stage = "declining"
            
            lifecycle["lifecycle_stage"] = stage
            
            # Predict future trends
            if len(lifecycle["health_history"]) >= 3:
                recent_scores = [h["health_score"] for h in lifecycle["health_history"][-3:]]
                trend = "stable"
                
                if recent_scores[-1] > recent_scores[0] + 0.1:
                    trend = "improving"
                elif recent_scores[-1] < recent_scores[0] - 0.1:
                    trend = "declining"
                
                # Predict when health might drop below thresholds
                prediction_horizon = 30  # days
                predicted_health = current_health
                
                if trend == "declining":
                    decline_rate = (recent_scores[0] - recent_scores[-1]) / 3  # per measurement
                    predicted_health = max(0.0, current_health - decline_rate * 10)  # Extrapolate
                
                return {
                    "current_stage": stage,
                    "trend": trend,
                    "predicted_health_30d": predicted_health,
                    "days_since_creation": days_since_creation,
                    "archive_recommendation": predicted_health < 0.2,
                    "optimization_suggestion": self._get_optimization_suggestion(stage, trend, current_health)
                }
            
            return {
                "current_stage": stage,
                "trend": "unknown",
                "predicted_health_30d": current_health,
                "days_since_creation": days_since_creation,
                "archive_recommendation": False,
                "optimization_suggestion": "monitoring"
            }
    
    def _get_optimization_suggestion(self, stage: str, trend: str, health: float) -> str:
        """Generate optimization suggestions based on lifecycle analysis."""
        if stage == "new" and health < 0.3:
            return "improve_chunking"
        elif stage == "active" and trend == "declining":
            return "refresh_content"
        elif stage == "aging" and health > 0.7:
            return "preserve_active"
        elif stage == "declining":
            return "consider_archive"
        elif health > 0.8:
            return "maintain_current"
        else:
            return "review_usage_patterns"
    
    def get_lifecycle_summary(self) -> Dict[str, Any]:
        """Get summary of all capsule lifecycles."""
        with self.lock:
            if not self.lifecycle_data:
                return {"total_capsules": 0, "stage_distribution": {}}
            
            stages = [data["lifecycle_stage"] for data in self.lifecycle_data.values()]
            stage_counts = {}
            for stage in ["new", "active", "moderate", "declining", "aging"]:
                stage_counts[stage] = stages.count(stage)
            
            return {
                "total_capsules": len(self.lifecycle_data),
                "stage_distribution": stage_counts,
                "average_age_days": sum(
                    (time.time() - data["creation_time"]) / (24 * 3600)
                    for data in self.lifecycle_data.values()
                ) / len(self.lifecycle_data)
            }


# Global instances
health_monitor = CapsuleHealthMonitor()
lifecycle_engine = LifecyclePredictionEngine()
