"""
Security module for CapsuleRAG: headers, adversarial detection, API key auth.
"""

import os
import re
import time
import threading
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict
from fastapi import HTTPException, Depends, Request
from fastapi.security import APIKeyHeader
try:
    from fastapi.middleware.base import BaseHTTPMiddleware
except ImportError:
    # Fallback for older FastAPI versions
    from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import Response

# Security configuration
DEMO_MODE = os.getenv("CAPSULE_DEMO_MODE", "true").lower() == "true"
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", "52428800"))  # 50MB default
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "demo-admin-key")
USER_API_KEY = os.getenv("USER_API_KEY", "demo-user-key")


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""
    
    async def dispatch(self, request: Request, call_next):
        # Basic body size guard (best-effort for uploads)
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_UPLOAD_BYTES:
            return Response("Request too large", status_code=413)

        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        if not DEMO_MODE:
            response.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self'"
        
        return response


class AdversarialDetector:
    """Detect and prevent adversarial queries and attacks."""
    
    def __init__(self):
        self.suspicious_patterns = [
            # Prompt injection attempts
            r"ignore\s+previous\s+instructions",
            r"forget\s+all\s+previous",
            r"new\s+instructions",
            r"system\s+prompt",
            r"you\s+are\s+now",
            r"pretend\s+to\s+be",
            
            # Data extraction attempts
            r"show\s+me\s+everything",
            r"list\s+all\s+documents",
            r"dump\s+database",
            r"export\s+all",
            r"admin\s+only",
            r"confidential.*show",
            
            # SQL injection patterns
            r"union\s+select",
            r"drop\s+table",
            r"delete\s+from",
            r"update\s+set",
            
            # Command injection
            r";\s*rm\s+-rf",
            r"&&\s*cat",
            r"\|\s*curl",
            r"exec\s*\(",
        ]
        
        self.rate_limits: Dict[str, List[float]] = defaultdict(list)
        self.max_requests_per_minute = 30
        self.max_requests_per_hour = 200
        self.lock = threading.RLock()
    
    def is_suspicious_query(self, query: str) -> Tuple[bool, str]:
        """Check if query contains suspicious patterns."""
        query_lower = query.lower()
        
        for pattern in self.suspicious_patterns:
            if re.search(pattern, query_lower):
                return True, f"Suspicious pattern detected: {pattern}"
        
        # Check for excessive special characters (potential encoding attacks)
        special_chars = len(re.findall(r'[<>{}|\\;\'"&$`]', query))
        if special_chars > len(query) * 0.3:
            return True, "Excessive special characters detected"
        
        # Check for extremely long queries (potential buffer overflow)
        if len(query) > 2000:
            return True, "Query exceeds maximum length"
        
        return False, ""
    
    def check_rate_limit(self, client_ip: str) -> Tuple[bool, str]:
        """Check if client is within rate limits."""
        now = time.time()
        
        with self.lock:
            # Clean old entries
            self.rate_limits[client_ip] = [
                t for t in self.rate_limits[client_ip] 
                if now - t < 3600  # Keep last hour
            ]
            
            # Check limits
            minute_requests = len([t for t in self.rate_limits[client_ip] if now - t < 60])
            hour_requests = len(self.rate_limits[client_ip])
            
            if minute_requests >= self.max_requests_per_minute:
                return False, f"Rate limit exceeded: {minute_requests} requests in last minute"
            
            if hour_requests >= self.max_requests_per_hour:
                return False, f"Rate limit exceeded: {hour_requests} requests in last hour"
            
            # Record this request
            self.rate_limits[client_ip].append(now)
            return True, ""
    
    def get_security_stats(self) -> Dict[str, Any]:
        """Get security monitoring statistics."""
        with self.lock:
            now = time.time()
            active_clients = len([
                ip for ip, times in self.rate_limits.items()
                if any(now - t < 300 for t in times)  # Active in last 5 minutes
            ])
            
            total_requests = sum(len(times) for times in self.rate_limits.values())
            
            return {
                "active_clients": active_clients,
                "total_clients_tracked": len(self.rate_limits),
                "total_requests_tracked": total_requests,
                "rate_limits": {
                    "per_minute": self.max_requests_per_minute,
                    "per_hour": self.max_requests_per_hour
                },
                "suspicious_patterns_count": len(self.suspicious_patterns)
            }


# API Key authentication
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_user_permissions(api_key: Optional[str] = Depends(api_key_header)) -> List[str]:
    """Extract user permissions from API key. In demo mode, allows unauthenticated access."""
    if DEMO_MODE and not api_key:
        return ["public:read", "admin:read", "admin:write"]  # Demo: full access
    
    if api_key == ADMIN_API_KEY:
        return ["admin:read", "admin:write", "public:read"]
    elif api_key == USER_API_KEY:
        return ["public:read"]
    elif DEMO_MODE:
        return ["public:read"]  # Demo: fallback to read-only
    else:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def check_acl_permission(acl: List[str], user_permissions: List[str], required_action: str) -> bool:
    """Check if user has required permission based on ACL and user permissions."""
    # Simple ACL matching - in production would be more sophisticated
    required_perms = [perm for perm in acl if perm.endswith(f":{required_action}")]
    if not required_perms:
        return True  # No specific requirements
    
    # Check if user has any of the required permissions
    for req_perm in required_perms:
        if req_perm in user_permissions:
            return True
    
    return False


# Global detector instance
adversarial_detector = AdversarialDetector()
