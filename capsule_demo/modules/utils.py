"""
Utility functions for CapsuleRAG: tokenization, chunking, preview generation, hashing.
"""

import os
import re
import io
import json
import base64
import hashlib
import time
from datetime import datetime, timezone
from typing import List, NamedTuple, Dict, Any
import numpy as np

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    from pdf2image import convert_from_bytes
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Configuration
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "50"))
MAX_TEXT_PER_PAGE = int(os.getenv("MAX_TEXT_PER_PAGE", "20000"))
CHUNK_SIZE_TOKENS = 200
CHUNK_OVERLAP_TOKENS = 50


class TextChunk(NamedTuple):
    text: str
    start_byte: int
    end_byte: int
    chunk_type: str  # 'heading', 'paragraph', 'list', 'text'


class CapsuleManifest(NamedTuple):
    doc_id: int
    content_hash: str
    created_at: str
    filename: str
    num_chunks: int
    model_info: Dict[str, str]
    acl: List[str]  # Access control list
    signature: str


def compute_content_hash(text: str) -> str:
    """Compute SHA-256 hash of document content for content addressing."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def create_manifest_signature(manifest_data: Dict[str, Any]) -> str:
    """Create a simple signature for the manifest (in production, use proper crypto signing)."""
    # In production, this would use private key signing (RSA/ECDSA)
    # For demo, we'll use a simple hash-based approach
    content = json.dumps(manifest_data, sort_keys=True)
    return hashlib.sha256(f"CAPSULE_DEMO_KEY:{content}".encode()).hexdigest()[:32]


def tokenize(text: str) -> List[str]:
    """Simple tokenization for BM25."""
    # Basic tokenization: lowercase, split on non-alphanumeric, filter empty
    tokens = re.findall(r'\b\w+\b', text.lower())
    return [token for token in tokens if len(token) > 1]


def chunk_text_structure_aware(text: str, chunk_size: int = CHUNK_SIZE_TOKENS, overlap: int = CHUNK_OVERLAP_TOKENS) -> List[TextChunk]:
    """Structure-aware chunking that preserves document structure and byte ranges."""
    text_bytes = text.encode('utf-8')
    chunks: List[TextChunk] = []
    
    # Try to detect structure patterns
    lines = text.split('\n')
    current_pos = 0
    
    for line in lines:
        line_start = current_pos
        line_bytes = (line + '\n').encode('utf-8')
        line_end = current_pos + len(line_bytes)
        
        line_stripped = line.strip()
        if not line_stripped:
            current_pos = line_end
            continue
            
        # Detect chunk type based on structure
        chunk_type = 'text'
        if re.match(r'^#{1,6}\s+', line_stripped):  # Markdown headers
            chunk_type = 'heading'
        elif re.match(r'^[-*+]\s+', line_stripped):  # List items
            chunk_type = 'list'
        elif len(line_stripped) > 30:  # Paragraph-like content
            chunk_type = 'paragraph'
        
        # For now, each meaningful line becomes a chunk (can be optimized later)
        if len(line_stripped) >= 20:  # Minimum chunk size
            chunks.append(TextChunk(
                text=line_stripped,
                start_byte=line_start,
                end_byte=line_end - 1,  # Exclude newline
                chunk_type=chunk_type
            ))
        
        current_pos = line_end
    
    # Fallback to token-based chunking if no structured chunks found
    if not chunks:
        tokens = tokenize(text)
        if tokens:
            full_text = " ".join(tokens)
            chunks.append(TextChunk(
                text=full_text,
                start_byte=0,
                end_byte=len(text_bytes) - 1,
                chunk_type='text'
            ))
    
    return chunks


def generate_document_preview(content: bytes, content_type: str, max_pages: int = 5) -> List[str]:
    """Generate preview images for a document and return as base64 strings."""
    preview_images = []
    
    # Security: limit max pages and file size
    max_pages = min(max_pages, MAX_PDF_PAGES)
    max_file_size = 50 * 1024 * 1024  # 50MB limit for preview processing
    
    if len(content) > max_file_size:
        print(f"File too large for preview: {len(content)} bytes")
        return []
    
    try:
        if content_type == "application/pdf" or content.startswith(b"%PDF"):
            # Handle PDF files
            if PDF2IMAGE_AVAILABLE:
                try:
                    # Security: limit DPI and pages
                    images = convert_from_bytes(content, first_page=1, last_page=max_pages, dpi=100)
                    for img in images:
                        # Security: check image dimensions
                        max_pixels = 2000 * 2000  # 4MP limit
                        if img.width * img.height > max_pixels:
                            print(f"Image too large: {img.width}x{img.height}")
                            continue
                            
                        # Convert to RGB if necessary
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        # Resize for preview (max width 800px)
                        if img.width > 800:
                            ratio = 800 / img.width
                            new_height = int(img.height * ratio)
                            img = img.resize((800, new_height), Image.Resampling.LANCZOS)
                        
                        # Convert to base64
                        buffer = io.BytesIO()
                        img.save(buffer, format='PNG')
                        img_b64 = base64.b64encode(buffer.getvalue()).decode()
                        preview_images.append(img_b64)
                except Exception as e:
                    print(f"PDF preview generation failed: {e}")
            
        elif content_type.startswith("image/"):
            # Handle image files
            if PIL_AVAILABLE:
                try:
                    img = Image.open(io.BytesIO(content))
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Resize for preview
                    if img.width > 800:
                        ratio = 800 / img.width
                        new_height = int(img.height * ratio)
                        img = img.resize((800, new_height), Image.Resampling.LANCZOS)
                    
                    buffer = io.BytesIO()
                    img.save(buffer, format='PNG')
                    img_b64 = base64.b64encode(buffer.getvalue()).decode()
                    preview_images.append(img_b64)
                except Exception as e:
                    print(f"Image preview generation failed: {e}")
        
        else:
            # Handle text files by creating a text image
            if PIL_AVAILABLE:
                try:
                    text_content = content.decode('utf-8', errors='replace')[:2000]  # First 2000 chars
                    
                    # Create a simple text preview image
                    img_width, img_height = 800, 1000
                    img = Image.new('RGB', (img_width, img_height), color='white')
                    draw = ImageDraw.Draw(img)
                    
                    # Try to use a monospace font, fallback to default
                    try:
                        font = ImageFont.truetype("/System/Library/Fonts/Monaco.ttf", 12)
                    except:
                        try:
                            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 12)
                        except:
                            font = ImageFont.load_default()
                    
                    # Wrap text and draw
                    lines = text_content.split('\n')
                    y_offset = 20
                    for line in lines[:70]:  # Max 70 lines
                        if y_offset > img_height - 30:
                            break
                        # Wrap long lines
                        if len(line) > 80:
                            line = line[:77] + "..."
                        draw.text((20, y_offset), line, fill='black', font=font)
                        y_offset += 14
                    
                    buffer = io.BytesIO()
                    img.save(buffer, format='PNG')
                    img_b64 = base64.b64encode(buffer.getvalue()).decode()
                    preview_images.append(img_b64)
                except Exception as e:
                    print(f"Text preview generation failed: {e}")
    
    except Exception as e:
        print(f"Preview generation failed: {e}")
    
    return preview_images


def extract_entities_demo(text: str, filename: str) -> Dict[str, List[str]]:
    """Demo entity extraction - looks for equipment, incidents, and locations."""
    entities = {"equipment": [], "incident": [], "location": []}
    
    # Simple pattern matching for demo
    # Equipment patterns
    equipment_patterns = [r"Compressor [A-Z]", r"Pump \d+", r"Valve [A-Z]\d+", r"Tank \d+", r"Generator \d+"]
    for pattern in equipment_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        entities["equipment"].extend([m.title() for m in matches])
    
    # Incident patterns  
    incident_patterns = [r"failure", r"malfunction", r"breakdown", r"leak", r"alarm", r"emergency", r"fault"]
    for pattern in incident_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            entities["incident"].append(f"{pattern.title()} in {filename}")
    
    # Location patterns
    location_patterns = [r"Building \d+", r"Floor \d+", r"Room \d+", r"Area [A-Z]", r"Zone \d+"]
    for pattern in location_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        entities["location"].extend([m.title() for m in matches])
    
    # Remove duplicates
    for entity_type in entities:
        entities[entity_type] = list(set(entities[entity_type]))
    
    return entities


def split_into_sentences(text: str) -> List[str]:
    """Lightweight sentence splitter; avoids heavyweight dependencies."""
    # Keeps periods inside numbers/abbreviations by using regex lookarounds
    raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text.strip())
    sentences: List[str] = []
    for s in raw_sentences:
        cleaned = s.strip()
        if len(cleaned) >= 30:
            sentences.append(cleaned)
    return sentences[:500]


def summarize_text_to_bullets(text: str, max_bullets: int = 5, embed_model=None) -> List[str]:
    """Generate bullet-point summary using extractive summarization."""
    sentences = split_into_sentences(text)
    if not sentences:
        # Fallback: chunk by lines if no sentence-like splits
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return lines[:max_bullets]

    if embed_model is None:
        # Simple fallback: return first few sentences
        return sentences[:max_bullets]

    sent_embeddings = embed_model.encode(sentences, normalize_embeddings=True)
    sent_embeddings = np.asarray(sent_embeddings, dtype=np.float32)

    # Compute centroid and similarity to centroid (representativeness)
    centroid = np.mean(sent_embeddings, axis=0, keepdims=True)
    centroid /= (np.linalg.norm(centroid) + 1e-12)
    sims = (sent_embeddings @ centroid.T).reshape(-1)

    # Maximal Marginal Relevance (MMR) selection to promote diversity
    max_bullets = int(max(1, min(max_bullets, len(sentences))))
    lambda_diversity = 0.7
    selected: List[int] = []
    candidate_indices = list(range(len(sentences)))

    # Greedy selection
    while candidate_indices and len(selected) < max_bullets:
        if not selected:
            # pick most representative first
            next_idx = int(np.argmax(sims[candidate_indices]))
            selected_idx = candidate_indices[next_idx]
            selected.append(selected_idx)
            candidate_indices.pop(next_idx)
            continue
        # For others, compute MMR score
        best_score = -1e9
        best_idx = -1
        for ci, cand in enumerate(candidate_indices):
            sim_to_centroid = sims[cand]
            redundancy = 0.0
            for sel in selected:
                redundancy = max(redundancy, float(sent_embeddings[cand] @ sent_embeddings[sel].T))
            mmr = lambda_diversity * sim_to_centroid - (1.0 - lambda_diversity) * redundancy
            if mmr > best_score:
                best_score = mmr
                best_idx = ci
        if best_idx >= 0:
            selected.append(candidate_indices.pop(best_idx))
        else:
            break

    # Preserve original order for readability
    selected_sorted = sorted(selected)
    bullets = [sentences[i] for i in selected_sorted]
    return bullets


def create_capsule_manifest(doc_id: int, text: str, filename: str, num_chunks: int) -> CapsuleManifest:
    """Create a signed manifest for the capsule."""
    
    content_hash = compute_content_hash(text)
    created_at = datetime.now(timezone.utc).isoformat()
    
    # Model information
    model_info = {
        "embedding_model": "all-MiniLM-L6-v2",
        "cross_encoder": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "chunking_method": "structure_aware",
        "vector_backend": "numpy_fallback"  # Would be "hnswlib" if enabled
    }
    
    # Default ACL (in production, this would be user-specific)
    # For demo: make some documents restricted
    if "confidential" in filename.lower() or "private" in filename.lower():
        acl = ["admin:read", "admin:write"]
    else:
        acl = ["public:read", "admin:write"]
    
    # Create manifest data for signing
    manifest_data = {
        "doc_id": doc_id,
        "content_hash": content_hash,
        "created_at": created_at,
        "filename": filename,
        "num_chunks": num_chunks,
        "model_info": model_info,
        "acl": acl
    }
    
    signature = create_manifest_signature(manifest_data)
    
    return CapsuleManifest(
        doc_id=doc_id,
        content_hash=content_hash,
        created_at=created_at,
        filename=filename,
        num_chunks=num_chunks,
        model_info=model_info,
        acl=acl,
        signature=signature
    )
