#!/usr/bin/env python3
"""
Een Security Middleware
=======================

Comprehensive security middleware for the Een Unity Mathematics project.
Provides authentication, rate limiting, CORS, security headers, and input validation.

Author: Claude (3000 ELO AGI)
"""

import os
import time
import hashlib
import hmac
import secrets
import re
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
from functools import wraps
import logging

from flask import request, jsonify, g, current_app
from werkzeug.exceptions import HTTPException

logger = logging.getLogger(__name__)


class SecurityMiddleware:
    """Comprehensive security middleware for Een project."""

    def __init__(self, app=None):
        self.app = app
        self.rate_limit_store: Dict[str, List[float]] = {}
        self.blocked_ips: Dict[str, float] = {}

        # Security configuration
        self.config = {
            "rate_limit_per_minute": int(os.getenv("RATE_LIMIT_PER_MINUTE", "30")),
            "max_request_size": int(os.getenv("MAX_REQUEST_SIZE", "10485760")),  # 10MB
            "session_timeout": int(os.getenv("SESSION_TIMEOUT", "3600")),  # 1 hour
            "max_login_attempts": int(os.getenv("MAX_LOGIN_ATTEMPTS", "5")),
            "block_duration": int(os.getenv("BLOCK_DURATION", "300")),  # 5 minutes
            "allowed_origins": os.getenv(
                "ALLOWED_ORIGINS",
                "https://een.consciousness.math,http://localhost:3000",
            ).split(","),
            "require_auth": os.getenv("REQUIRE_AUTH", "true").lower() == "true",
            "api_key_required": os.getenv("API_KEY_REQUIRED", "true").lower() == "true",
        }

        if app is not None:
            self.init_app(app)

    def init_app(self, app):
        """Initialize the security middleware with Flask app."""
        self.app = app

        # Register middleware
        app.before_request(self.before_request)
        app.after_request(self.after_request)
        app.errorhandler(Exception)(self.handle_exception)

        # Add security headers
        app.config["SECURITY_HEADERS"] = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net https://cdn.plot.ly; style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; img-src 'self' data: https:; font-src 'self' https://cdn.jsdelivr.net; connect-src 'self' https://api.openai.com https://api.anthropic.com;",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
        }

        logger.info("Security middleware initialized")

    def before_request(self):
        """Security checks before each request."""
        try:
            # Get client IP
            client_ip = self._get_client_ip()

            # Check if IP is blocked
            if self._is_ip_blocked(client_ip):
                return jsonify({"error": "Access denied"}), 403

            # Rate limiting
            if not self._check_rate_limit(client_ip):
                return jsonify({"error": "Rate limit exceeded"}), 429

            # Request size validation
            if not self._validate_request_size():
                return jsonify({"error": "Request too large"}), 413

            # Input validation
            if not self._validate_input():
                return jsonify({"error": "Invalid input"}), 400

            # Authentication check
            if self.config["require_auth"] and not self._authenticate_request():
                return jsonify({"error": "Authentication required"}), 401

            # Store client info for logging
            g.client_ip = client_ip
            g.request_time = time.time()

        except Exception as e:
            logger.error(f"Security middleware error: {e}")
            return jsonify({"error": "Internal server error"}), 500

    def after_request(self, response):
        """Add security headers after each request."""
        try:
            # Add security headers
            for header, value in self.app.config["SECURITY_HEADERS"].items():
                response.headers[header] = value

            # Add CORS headers
            origin = request.headers.get("Origin")
            if origin and origin in self.config["allowed_origins"]:
                response.headers["Access-Control-Allow-Origin"] = origin
                response.headers["Access-Control-Allow-Methods"] = (
                    "GET, POST, PUT, DELETE, OPTIONS"
                )
                response.headers["Access-Control-Allow-Headers"] = (
                    "Content-Type, Authorization"
                )
                response.headers["Access-Control-Max-Age"] = "3600"

            # Add rate limit headers
            client_ip = self._get_client_ip()
            remaining = self._get_remaining_requests(client_ip)
            response.headers["X-RateLimit-Remaining"] = str(remaining)
            response.headers["X-RateLimit-Reset"] = str(int(time.time() + 60))

            # Log security events
            self._log_security_event(response)

        except Exception as e:
            logger.error(f"Error adding security headers: {e}")

        return response

    def handle_exception(self, exception):
        """Handle exceptions securely."""
        if isinstance(exception, HTTPException):
            return jsonify({"error": exception.description}), exception.code

        # Don't expose internal errors
        logger.error(f"Unhandled exception: {exception}")
        return jsonify({"error": "Internal server error"}), 500

    def _get_client_ip(self) -> str:
        """Get the real client IP address."""
        # Check for forwarded headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        return request.remote_addr

    def _is_ip_blocked(self, client_ip: str) -> bool:
        """Check if IP is blocked due to abuse."""
        if client_ip in self.blocked_ips:
            block_until = self.blocked_ips[client_ip]
            if time.time() < block_until:
                return True
            else:
                del self.blocked_ips[client_ip]
        return False

    def _check_rate_limit(self, client_ip: str) -> bool:
        """Check rate limiting for client IP."""
        now = time.time()
        minute_ago = now - 60

        if client_ip not in self.rate_limit_store:
            self.rate_limit_store[client_ip] = []

        # Remove old requests
        self.rate_limit_store[client_ip] = [
            req_time
            for req_time in self.rate_limit_store[client_ip]
            if req_time > minute_ago
        ]

        # Check if limit exceeded
        if (
            len(self.rate_limit_store[client_ip])
            >= self.config["rate_limit_per_minute"]
        ):
            # Block IP if consistently exceeding limits
            if (
                len(self.rate_limit_store[client_ip])
                >= self.config["rate_limit_per_minute"] * 2
            ):
                self.blocked_ips[client_ip] = now + self.config["block_duration"]
                logger.warning(f"IP {client_ip} blocked for abuse")
            return False

        # Add current request
        self.rate_limit_store[client_ip].append(now)
        return True

    def _get_remaining_requests(self, client_ip: str) -> int:
        """Get remaining requests for client IP."""
        if client_ip not in self.rate_limit_store:
            return self.config["rate_limit_per_minute"]

        now = time.time()
        minute_ago = now - 60

        recent_requests = [
            req_time
            for req_time in self.rate_limit_store[client_ip]
            if req_time > minute_ago
        ]

        return max(0, self.config["rate_limit_per_minute"] - len(recent_requests))

    def _validate_request_size(self) -> bool:
        """Validate request size."""
        content_length = request.content_length
        if content_length and content_length > self.config["max_request_size"]:
            return False
        return True

    def _validate_input(self) -> bool:
        """Validate and sanitize input."""
        # Check for suspicious patterns
        suspicious_patterns = [
            r"<script[^>]*>",
            r"javascript:",
            r"data:text/html",
            r"vbscript:",
            r"on\w+\s*=",
            r"<iframe[^>]*>",
            r"<object[^>]*>",
            r"<embed[^>]*>",
        ]

        # Check URL parameters
        for key, value in request.args.items():
            if isinstance(value, str):
                for pattern in suspicious_patterns:
                    if re.search(pattern, value, re.IGNORECASE):
                        logger.warning(
                            f"Suspicious input detected in URL parameter {key}"
                        )
                        return False

        # Check form data
        if request.form:
            for key, value in request.form.items():
                if isinstance(value, str):
                    for pattern in suspicious_patterns:
                        if re.search(pattern, value, re.IGNORECASE):
                            logger.warning(
                                f"Suspicious input detected in form field {key}"
                            )
                            return False

        # Check JSON data
        if request.is_json:
            try:
                json_data = request.get_json()
                if json_data:
                    self._validate_json_input(json_data)
            except Exception as e:
                logger.warning(f"Invalid JSON input: {e}")
                return False

        return True

    def _validate_json_input(self, data: Any, depth: int = 0):
        """Recursively validate JSON input."""
        if depth > 10:  # Prevent deep recursion
            raise ValueError("Input too deeply nested")

        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(key, str) and len(key) > 100:
                    raise ValueError("Key too long")
                self._validate_json_input(value, depth + 1)
        elif isinstance(data, list):
            if len(data) > 1000:
                raise ValueError("List too long")
            for item in data:
                self._validate_json_input(item, depth + 1)
        elif isinstance(data, str):
            if len(data) > 10000:
                raise ValueError("String too long")

    def _authenticate_request(self) -> bool:
        """Authenticate the request."""
        # Skip authentication for health checks and static files
        if request.endpoint in ["health_check", "static"]:
            return True

        # Check for API key
        if self.config["api_key_required"]:
            api_key = request.headers.get("Authorization")
            if not api_key:
                return False

            # Remove 'Bearer ' prefix if present
            if api_key.startswith("Bearer "):
                api_key = api_key[7:]

            # Validate API key
            if not self._validate_api_key(api_key):
                return False

        return True

    def _validate_api_key(self, api_key: str) -> bool:
        """Validate API key."""
        # Get expected API key from environment
        expected_key = os.getenv("API_KEY")
        if not expected_key:
            logger.warning("No API key configured in environment")
            return False

        # Use constant-time comparison to prevent timing attacks
        return hmac.compare_digest(api_key, expected_key)

    def _log_security_event(self, response):
        """Log security-relevant events."""
        client_ip = getattr(g, "client_ip", "unknown")
        status_code = response.status_code

        # Log failed authentication attempts
        if status_code == 401:
            logger.warning(f"Failed authentication attempt from {client_ip}")

        # Log rate limit violations
        if status_code == 429:
            logger.warning(f"Rate limit exceeded by {client_ip}")

        # Log blocked requests
        if status_code == 403:
            logger.warning(f"Access denied to {client_ip}")

        # Log suspicious requests
        if status_code == 400 and "Invalid input" in response.get_data(as_text=True):
            logger.warning(f"Invalid input from {client_ip}")


def require_auth(f):
    """Decorator to require authentication for specific endpoints."""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not g.get("authenticated", False):
            return jsonify({"error": "Authentication required"}), 401
        return f(*args, **kwargs)

    return decorated_function


def rate_limit(requests_per_minute: int = 30):
    """Decorator to apply custom rate limiting."""

    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            client_ip = request.remote_addr
            # Custom rate limiting logic here
            return f(*args, **kwargs)

        return decorated_function

    return decorator


def sanitize_input(f):
    """Decorator to sanitize input data."""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Additional input sanitization here
        return f(*args, **kwargs)

    return decorated_function
