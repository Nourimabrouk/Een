"""
Security module for Een Consciousness API
Handles authentication, authorization, and security utilities
"""

import os
import secrets
import hashlib
import bcrypt
import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from fastapi import HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging
from pydantic import BaseModel, Field
import re

logger = logging.getLogger(__name__)

# Security configuration
SECRET_KEY = os.getenv("EEN_SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

# Rate limiting configuration
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "3600"))  # 1 hour

# Password policy
MIN_PASSWORD_LENGTH = 8
PASSWORD_REGEX = re.compile(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]')

# IP whitelist for admin access
ADMIN_IP_WHITELIST = os.getenv("ADMIN_IP_WHITELIST", "").split(",")

class User(BaseModel):
    """User model"""
    username: str = Field(..., description="Username")
    email: str = Field(..., description="Email address")
    role: str = Field(default="user", description="User role")
    is_active: bool = Field(default=True, description="Whether user is active")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None

class Token(BaseModel):
    """Token model"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_expires_in: int

class TokenData(BaseModel):
    """Token data model"""
    username: Optional[str] = None
    role: Optional[str] = None

class PasswordChange(BaseModel):
    """Password change model"""
    current_password: str
    new_password: str

class SecurityConfig(BaseModel):
    """Security configuration model"""
    min_password_length: int = MIN_PASSWORD_LENGTH
    require_special_chars: bool = True
    require_numbers: bool = True
    require_uppercase: bool = True
    require_lowercase: bool = True
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 30

class SecurityManager:
    """Security manager for authentication and authorization"""
    
    def __init__(self):
        self.users_db: Dict[str, Dict] = {}
        self.failed_login_attempts: Dict[str, List[datetime]] = {}
        self.locked_accounts: Dict[str, datetime] = {}
        self.active_tokens: Dict[str, Dict] = {}
        self.rate_limit_store: Dict[str, List[float]] = {}
        
        # Initialize with default admin user
        self._create_default_admin()
    
    def _create_default_admin(self):
        """Create default admin user if not exists"""
        admin_password = os.getenv("ADMIN_PASSWORD", "admin123")
        admin_email = os.getenv("ADMIN_EMAIL", "admin@een.consciousness.math")
        
        if "admin" not in self.users_db:
            self.create_user(
                username="admin",
                email=admin_email,
                password=admin_password,
                role="admin"
            )
            logger.info("Default admin user created")
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
    
    def validate_password_strength(self, password: str) -> Dict[str, Any]:
        """Validate password strength"""
        errors = []
        
        if len(password) < MIN_PASSWORD_LENGTH:
            errors.append(f"Password must be at least {MIN_PASSWORD_LENGTH} characters long")
        
        if not PASSWORD_REGEX.match(password):
            errors.append("Password must contain at least one uppercase letter, one lowercase letter, one number, and one special character")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    
    def create_user(self, username: str, email: str, password: str, role: str = "user") -> User:
        """Create a new user"""
        if username in self.users_db:
            raise ValueError("Username already exists")
        
        # Validate password strength
        password_validation = self.validate_password_strength(password)
        if not password_validation["valid"]:
            raise ValueError(f"Password validation failed: {'; '.join(password_validation['errors'])}")
        
        # Hash password
        hashed_password = self.hash_password(password)
        
        # Create user
        user_data = {
            "username": username,
            "email": email,
            "password_hash": hashed_password,
            "role": role,
            "is_active": True,
            "created_at": datetime.utcnow(),
            "last_login": None
        }
        
        self.users_db[username] = user_data
        
        user = User(**{k: v for k, v in user_data.items() if k != "password_hash"})
        logger.info(f"User {username} created successfully")
        
        return user
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username and password"""
        # Check if account is locked
        if username in self.locked_accounts:
            lockout_time = self.locked_accounts[username]
            if datetime.utcnow() < lockout_time:
                remaining_time = lockout_time - datetime.utcnow()
                raise HTTPException(
                    status_code=status.HTTP_423_LOCKED,
                    detail=f"Account is locked. Try again in {int(remaining_time.total_seconds() / 60)} minutes"
                )
            else:
                # Unlock account
                del self.locked_accounts[username]
                self.failed_login_attempts[username] = []
        
        user_data = self.users_db.get(username)
        if not user_data:
            self._record_failed_login(username)
            return None
        
        if not user_data["is_active"]:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Account is deactivated"
            )
        
        if not self.verify_password(password, user_data["password_hash"]):
            self._record_failed_login(username)
            return None
        
        # Reset failed login attempts on successful login
        if username in self.failed_login_attempts:
            del self.failed_login_attempts[username]
        
        # Update last login
        user_data["last_login"] = datetime.utcnow()
        
        user = User(**{k: v for k, v in user_data.items() if k != "password_hash"})
        return user
    
    def _record_failed_login(self, username: str):
        """Record failed login attempt"""
        if username not in self.failed_login_attempts:
            self.failed_login_attempts[username] = []
        
        self.failed_login_attempts[username].append(datetime.utcnow())
        
        # Check if account should be locked
        recent_attempts = [
            attempt for attempt in self.failed_login_attempts[username]
            if datetime.utcnow() - attempt < timedelta(minutes=30)
        ]
        
        if len(recent_attempts) >= 5:
            lockout_time = datetime.utcnow() + timedelta(minutes=30)
            self.locked_accounts[username] = lockout_time
            logger.warning(f"Account {username} locked due to too many failed login attempts")
    
    def create_tokens(self, user: User) -> Token:
        """Create access and refresh tokens"""
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        refresh_token_expires = timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        
        access_token = self._create_token(
            data={"sub": user.username, "role": user.role},
            expires_delta=access_token_expires
        )
        
        refresh_token = self._create_token(
            data={"sub": user.username, "type": "refresh"},
            expires_delta=refresh_token_expires
        )
        
        # Store active token
        self.active_tokens[user.username] = {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "created_at": datetime.utcnow(),
            "expires_at": datetime.utcnow() + access_token_expires
        }
        
        return Token(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            refresh_expires_in=REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60
        )
    
    def _create_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire, "iat": datetime.utcnow()})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    def verify_token(self, token: str) -> TokenData:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username: str = payload.get("sub")
            role: str = payload.get("role")
            
            if username is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token"
                )
            
            # Check if token is in active tokens
            if username in self.active_tokens:
                stored_token = self.active_tokens[username]["access_token"]
                if token != stored_token:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Token has been invalidated"
                    )
            
            return TokenData(username=username, role=role)
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    def refresh_token(self, refresh_token: str) -> Token:
        """Refresh access token using refresh token"""
        try:
            payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
            username: str = payload.get("sub")
            token_type: str = payload.get("type")
            
            if username is None or token_type != "refresh":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid refresh token"
                )
            
            # Get user data
            user_data = self.users_db.get(username)
            if not user_data or not user_data["is_active"]:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found or inactive"
                )
            
            user = User(**{k: v for k, v in user_data.items() if k != "password_hash"})
            
            # Create new tokens
            return self.create_tokens(user)
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh token has expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
    
    def logout(self, username: str):
        """Logout user by invalidating tokens"""
        if username in self.active_tokens:
            del self.active_tokens[username]
            logger.info(f"User {username} logged out")
    
    def check_rate_limit(self, client_ip: str) -> bool:
        """Check if client has exceeded rate limit"""
        now = time.time()
        client_requests = self.rate_limit_store.get(client_ip, [])
        
        # Remove old requests outside the window
        client_requests[:] = [req_time for req_time in client_requests if now - req_time < RATE_LIMIT_WINDOW]
        
        if len(client_requests) >= RATE_LIMIT_REQUESTS:
            return False
        
        client_requests.append(now)
        self.rate_limit_store[client_ip] = client_requests
        return True
    
    def is_admin(self, user: User, request: Request) -> bool:
        """Check if user is admin and has admin access"""
        if user.role != "admin":
            return False
        
        # Check IP whitelist for admin access
        if ADMIN_IP_WHITELIST and ADMIN_IP_WHITELIST[0]:  # Not empty
            client_ip = self._get_client_ip(request)
            if client_ip not in ADMIN_IP_WHITELIST:
                logger.warning(f"Admin access denied for IP: {client_ip}")
                return False
        
        return True
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0]
        return request.client.host if request.client else "unknown"
    
    def change_password(self, username: str, current_password: str, new_password: str) -> bool:
        """Change user password"""
        user_data = self.users_db.get(username)
        if not user_data:
            raise ValueError("User not found")
        
        # Verify current password
        if not self.verify_password(current_password, user_data["password_hash"]):
            raise ValueError("Current password is incorrect")
        
        # Validate new password strength
        password_validation = self.validate_password_strength(new_password)
        if not password_validation["valid"]:
            raise ValueError(f"Password validation failed: {'; '.join(password_validation['errors'])}")
        
        # Hash new password
        new_password_hash = self.hash_password(new_password)
        user_data["password_hash"] = new_password_hash
        
        # Invalidate all active tokens
        self.logout(username)
        
        logger.info(f"Password changed for user {username}")
        return True
    
    def get_user(self, username: str) -> Optional[User]:
        """Get user by username"""
        user_data = self.users_db.get(username)
        if user_data:
            return User(**{k: v for k, v in user_data.items() if k != "password_hash"})
        return None
    
    def list_users(self) -> List[User]:
        """List all users (admin only)"""
        users = []
        for user_data in self.users_db.values():
            user = User(**{k: v for k, v in user_data.items() if k != "password_hash"})
            users.append(user)
        return users
    
    def deactivate_user(self, username: str) -> bool:
        """Deactivate user (admin only)"""
        if username in self.users_db:
            self.users_db[username]["is_active"] = False
            self.logout(username)
            logger.info(f"User {username} deactivated")
            return True
        return False
    
    def activate_user(self, username: str) -> bool:
        """Activate user (admin only)"""
        if username in self.users_db:
            self.users_db[username]["is_active"] = True
            logger.info(f"User {username} activated")
            return True
        return False

# Global security manager instance
security_manager = SecurityManager()

# Security dependencies
security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """Get current authenticated user"""
    token_data = security_manager.verify_token(credentials.credentials)
    user = security_manager.get_user(token_data.username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    return user

def get_current_admin_user(current_user: User = Depends(get_current_user), request: Request = None) -> User:
    """Get current admin user"""
    if not security_manager.is_admin(current_user, request):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user

def check_rate_limit_dependency(client_ip: str = Depends(lambda: "127.0.0.1")) -> bool:
    """Rate limit dependency"""
    if not security_manager.check_rate_limit(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )
    return True

# Import time module for rate limiting
import time 