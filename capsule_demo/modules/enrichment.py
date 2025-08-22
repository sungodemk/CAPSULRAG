"""
Content enrichment module for CapsuleRAG: background metadata extraction and enhancement.
"""

import asyncio
import threading
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from collections import defaultdict

from .utils import extract_entities_demo, summarize_text_to_bullets


class ContentEnrichmentEngine:
    """Background content enrichment and metadata extraction."""
    
    def __init__(self):
        self.enrichment_queue: List[Dict[str, Any]] = []
        self.enriched_metadata: Dict[int, Dict[str, Any]] = {}
        self.processing = False
        self.lock = threading.RLock()
        
        # Start background processing
        self.worker_thread = threading.Thread(target=self._background_worker, daemon=True)
        self.worker_thread.start()
    
    def enqueue_enrichment(self, doc_id: int, text: str, filename: str, priority: str = "normal"):
        """Queue a document for background enrichment."""
        with self.lock:
            enrichment_task = {
                "doc_id": doc_id,
                "text": text,
                "filename": filename,
                "priority": priority,
                "queued_at": time.time(),
                "status": "queued"
            }
            
            # Insert based on priority
            if priority == "high":
                self.enrichment_queue.insert(0, enrichment_task)
            else:
                self.enrichment_queue.append(enrichment_task)
    
    def _background_worker(self):
        """Background worker that processes enrichment queue."""
        while True:
            try:
                with self.lock:
                    if not self.enrichment_queue or self.processing:
                        time.sleep(1)
                        continue
                    
                    task = self.enrichment_queue.pop(0)
                    self.processing = True
                
                # Process the task
                self._enrich_document(task)
                
                with self.lock:
                    self.processing = False
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Enrichment worker error: {e}")
                with self.lock:
                    self.processing = False
                time.sleep(5)  # Wait before retrying
    
    def _enrich_document(self, task: Dict[str, Any]):
        """Perform comprehensive document enrichment."""
        doc_id = task["doc_id"]
        text = task["text"]
        filename = task["filename"]
        
        enrichment_start = time.time()
        enriched = {
            "doc_id": doc_id,
            "filename": filename,
            "enriched_at": datetime.now(timezone.utc).isoformat(),
            "processing_time": 0.0
        }
        
        try:
            # 1. Enhanced entity extraction
            entities = extract_entities_demo(text, filename)
            enriched["entities"] = entities
            enriched["entity_count"] = sum(len(ents) for ents in entities.values())
            
            # 2. Document statistics
            enriched["word_count"] = len(text.split())
            enriched["char_count"] = len(text)
            enriched["paragraph_count"] = len([p for p in text.split('\n\n') if p.strip()])
            enriched["line_count"] = len(text.split('\n'))
            
            # 3. Content classification
            enriched["content_type"] = self._classify_content(text, filename)
            
            # 4. Key phrases extraction (simple approach)
            enriched["key_phrases"] = self._extract_key_phrases(text)
            
            # 5. Document structure analysis
            enriched["structure"] = self._analyze_structure(text)
            
            # 6. Language and readability metrics
            enriched["readability"] = self._analyze_readability(text)
            
            # 7. Topic modeling (lightweight)
            enriched["topics"] = self._extract_topics(text)
            
            # 8. Quality metrics
            enriched["quality_score"] = self._calculate_quality_score(enriched)
            
            enriched["processing_time"] = time.time() - enrichment_start
            enriched["status"] = "completed"
            
        except Exception as e:
            print(f"Enrichment failed for doc {doc_id}: {e}")
            enriched["status"] = "failed"
            enriched["error"] = str(e)
            enriched["processing_time"] = time.time() - enrichment_start
        
        # Store enriched metadata
        with self.lock:
            self.enriched_metadata[doc_id] = enriched
    
    def _classify_content(self, text: str, filename: str) -> Dict[str, Any]:
        """Classify content type and characteristics."""
        text_lower = text.lower()
        filename_lower = filename.lower()
        
        # Document type classification
        doc_types = []
        if any(word in text_lower for word in ["manual", "procedure", "instructions", "guide"]):
            doc_types.append("manual")
        if any(word in text_lower for word in ["policy", "regulation", "compliance", "standard"]):
            doc_types.append("policy")
        if any(word in text_lower for word in ["incident", "report", "analysis", "investigation"]):
            doc_types.append("report")
        if any(word in text_lower for word in ["emergency", "safety", "hazard", "warning"]):
            doc_types.append("safety")
        if any(word in filename_lower for word in ["spec", "technical", "engineering"]):
            doc_types.append("technical")
        
        # Content characteristics
        characteristics = []
        if len(text.split()) > 5000:
            characteristics.append("comprehensive")
        elif len(text.split()) < 500:
            characteristics.append("brief")
        
        if text.count('\n') / len(text.split()) > 0.3:
            characteristics.append("structured")
        
        if any(char in text for char in ["•", "-", "1.", "2.", "3."]):
            characteristics.append("procedural")
        
        return {
            "primary_type": doc_types[0] if doc_types else "general",
            "all_types": doc_types,
            "characteristics": characteristics
        }
    
    def _extract_key_phrases(self, text: str, max_phrases: int = 10) -> List[str]:
        """Extract key phrases using simple frequency analysis."""
        import re
        from collections import Counter
        
        # Simple n-gram extraction
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out common words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
        }
        
        # Bigrams and trigrams
        phrases = []
        for i in range(len(words) - 1):
            if words[i] not in stop_words and words[i+1] not in stop_words:
                if len(words[i]) > 3 and len(words[i+1]) > 3:
                    phrases.append(f"{words[i]} {words[i+1]}")
        
        for i in range(len(words) - 2):
            if all(word not in stop_words for word in words[i:i+3]):
                if all(len(word) > 3 for word in words[i:i+3]):
                    phrases.append(f"{words[i]} {words[i+1]} {words[i+2]}")
        
        # Count and return most frequent
        phrase_counts = Counter(phrases)
        return [phrase for phrase, _ in phrase_counts.most_common(max_phrases)]
    
    def _analyze_structure(self, text: str) -> Dict[str, Any]:
        """Analyze document structure and organization."""
        lines = text.split('\n')
        
        # Header detection
        headers = []
        for line in lines:
            line_stripped = line.strip()
            if line_stripped:
                # Markdown-style headers
                if line_stripped.startswith('#'):
                    headers.append({"text": line_stripped, "level": len(line_stripped) - len(line_stripped.lstrip('#'))})
                # ALL CAPS might be headers
                elif line_stripped.isupper() and len(line_stripped.split()) <= 8:
                    headers.append({"text": line_stripped, "level": 1})
                # Numbered sections
                elif re.match(r'^\d+\.', line_stripped):
                    headers.append({"text": line_stripped, "level": 2})
        
        # List detection
        lists = 0
        for line in lines:
            if re.match(r'^\s*[-•*]\s+', line) or re.match(r'^\s*\d+\.\s+', line):
                lists += 1
        
        return {
            "total_lines": len(lines),
            "non_empty_lines": len([l for l in lines if l.strip()]),
            "headers": headers,
            "header_count": len(headers),
            "list_items": lists,
            "has_structure": len(headers) > 0 or lists > 3
        }
    
    def _analyze_readability(self, text: str) -> Dict[str, Any]:
        """Analyze readability and language complexity."""
        words = text.split()
        sentences = text.count('.') + text.count('!') + text.count('?')
        
        if not words or not sentences:
            return {"complexity": "unknown"}
        
        avg_sentence_length = len(words) / max(1, sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Simple complexity score
        complexity_score = (avg_sentence_length / 20) + (avg_word_length / 6)
        
        if complexity_score < 1.0:
            complexity = "simple"
        elif complexity_score < 1.5:
            complexity = "moderate"
        else:
            complexity = "complex"
        
        return {
            "avg_sentence_length": avg_sentence_length,
            "avg_word_length": avg_word_length,
            "complexity": complexity,
            "complexity_score": complexity_score
        }
    
    def _extract_topics(self, text: str, max_topics: int = 5) -> List[str]:
        """Extract main topics using keyword clustering."""
        import re
        from collections import Counter
        
        # Extract meaningful words
        words = re.findall(r'\b[A-Za-z]{4,}\b', text.lower())
        
        # Filter technical/domain words
        domain_indicators = [
            'equipment', 'system', 'procedure', 'safety', 'maintenance', 'operation',
            'manual', 'process', 'control', 'monitor', 'check', 'inspect', 'repair',
            'pressure', 'temperature', 'valve', 'pump', 'compressor', 'generator'
        ]
        
        # Count word frequency
        word_counts = Counter(words)
        
        # Identify topics based on frequent domain words
        topics = []
        for word, count in word_counts.most_common(50):
            if word in domain_indicators and count > 2:
                topics.append(word)
        
        # Add general topics based on content patterns
        if any(word in text.lower() for word in ['emergency', 'alarm', 'warning', 'danger']):
            topics.append('emergency_procedures')
        if any(word in text.lower() for word in ['maintain', 'service', 'repair', 'replace']):
            topics.append('maintenance')
        if any(word in text.lower() for word in ['operate', 'start', 'stop', 'control']):
            topics.append('operation')
        
        return topics[:max_topics]
    
    def _calculate_quality_score(self, enriched: Dict[str, Any]) -> float:
        """Calculate overall document quality score."""
        score = 0.0
        
        # Content length (optimal range)
        word_count = enriched.get("word_count", 0)
        if 200 <= word_count <= 5000:
            score += 0.3
        elif word_count > 100:
            score += 0.1
        
        # Structure quality
        if enriched.get("structure", {}).get("has_structure", False):
            score += 0.2
        
        # Entity richness
        entity_count = enriched.get("entity_count", 0)
        if entity_count > 5:
            score += 0.2
        elif entity_count > 0:
            score += 0.1
        
        # Readability
        complexity = enriched.get("readability", {}).get("complexity", "unknown")
        if complexity in ["simple", "moderate"]:
            score += 0.15
        
        # Topic coverage
        if len(enriched.get("topics", [])) >= 3:
            score += 0.15
        
        return min(1.0, score)
    
    def get_enriched_metadata(self, doc_id: int) -> Optional[Dict[str, Any]]:
        """Get enriched metadata for a document."""
        with self.lock:
            return self.enriched_metadata.get(doc_id)
    
    def get_enrichment_stats(self) -> Dict[str, Any]:
        """Get enrichment processing statistics."""
        with self.lock:
            total_docs = len(self.enriched_metadata)
            successful = len([m for m in self.enriched_metadata.values() if m.get("status") == "completed"])
            avg_processing_time = sum(m.get("processing_time", 0) for m in self.enriched_metadata.values()) / max(1, total_docs)
            
            return {
                "total_processed": total_docs,
                "successful": successful,
                "failed": total_docs - successful,
                "queue_length": len(self.enrichment_queue),
                "processing": self.processing,
                "avg_processing_time": avg_processing_time,
                "success_rate": successful / max(1, total_docs)
            }


# Global instance
enrichment_engine = ContentEnrichmentEngine()
