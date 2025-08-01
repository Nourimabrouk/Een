"""
Authentication API routes
Handles user authentication, registration, and user management
"""

from fastapi import APIRouter, HTTPException, Depends, status, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, EmailStr
from typing import Optional, Dict, Any, List
import logging
import sys
import pathlib

# Add the project root to the path
project_root = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import security modules
from api.security import security_manager, get_current_user, get_current_admin_user
from api.security import User, Token, PasswordChange, SecurityConfig

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["authentication"])

# Pydantic models
class UserRegistration(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    email: EmailStr = Field(..., description="Email address")
    password: str = Field(..., min_length=8, description="Password")
    confirm_password: str = Field(..., description="Password confirmation")

class UserLogin(BaseModel):
    username: str = Field(..., description="Username")
    password: str = Field(..., description="Password")

class UserUpdate(BaseModel):
    email: Optional[EmailStr] = Field(None, description="Email address")
    is_active: Optional[bool] = Field(None, description="Whether user is active")

class RefreshTokenRequest(BaseModel):
    refresh_token: str = Field(..., description="Refresh token")

class PasswordResetRequest(BaseModel):
    email: EmailStr = Field(..., description="Email address for password reset")

class PasswordResetConfirm(BaseModel):
    token: str = Field(..., description="Password reset token")
    new_password: str = Field(..., min_length=8, description="New password")
    confirm_password: str = Field(..., description="Password confirmation")

# API Routes

@router.post("/register", response_model=Dict[str, Any])
async def register_user(
    user_data: UserRegistration,
    request_obj: Request = None
):
    """Register a new user"""
    # Check rate limit
    client_ip = security_manager._get_client_ip(request_obj) if request_obj else "127.0.0.1"
    if not security_manager.check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    try:
        # Validate password confirmation
        if user_data.password != user_data.confirm_password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Passwords do not match"
            )
        
        # Create user
        user = security_manager.create_user(
            username=user_data.username,
            email=user_data.email,
            password=user_data.password,
            role="user"
        )
        
        return {
            "success": True,
            "message": "User registered successfully",
            "user": {
                "username": user.username,
                "email": user.email,
                "role": user.role,
                "is_active": user.is_active
            },
            "timestamp": "2025-01-01T00:00:00Z"
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"User registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )

@router.post("/login", response_model=Token)
async def login_user(
    user_credentials: UserLogin,
    request_obj: Request = None
):
    """Authenticate user and return JWT tokens"""
    # Check rate limit
    client_ip = security_manager._get_client_ip(request_obj) if request_obj else "127.0.0.1"
    if not security_manager.check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    try:
        # Authenticate user
        user = security_manager.authenticate_user(
            username=user_credentials.username,
            password=user_credentials.password
        )
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Create tokens
        tokens = security_manager.create_tokens(user)
        
        return tokens
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )

@router.post("/refresh", response_model=Token)
async def refresh_token(
    refresh_request: RefreshTokenRequest,
    request_obj: Request = None
):
    """Refresh access token using refresh token"""
    # Check rate limit
    client_ip = security_manager._get_client_ip(request_obj) if request_obj else "127.0.0.1"
    if not security_manager.check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    try:
        # Refresh token
        tokens = security_manager.refresh_token(refresh_request.refresh_token)
        return tokens
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )

@router.post("/logout")
async def logout_user(
    current_user: User = Depends(get_current_user)
):
    """Logout user by invalidating tokens"""
    try:
        security_manager.logout(current_user.username)
        
        return {
            "success": True,
            "message": "User logged out successfully",
            "timestamp": "2025-01-01T00:00:00Z"
        }
        
    except Exception as e:
        logger.error(f"Logout error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )

@router.post("/change-password")
async def change_password(
    password_change: PasswordChange,
    current_user: User = Depends(get_current_user)
):
    """Change user password"""
    try:
        success = security_manager.change_password(
            username=current_user.username,
            current_password=password_change.current_password,
            new_password=password_change.new_password
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password change failed"
            )
        
        return {
            "success": True,
            "message": "Password changed successfully",
            "timestamp": "2025-01-01T00:00:00Z"
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Password change error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password change failed"
        )

@router.post("/reset-password/request")
async def request_password_reset(
    reset_request: PasswordResetRequest,
    request_obj: Request = None
):
    """Request password reset"""
    # Check rate limit
    client_ip = security_manager._get_client_ip(request_obj) if request_obj else "127.0.0.1"
    if not security_manager.check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    try:
        # In a real implementation, this would send an email with reset token
        # For now, we'll just return a success message
        return {
            "success": True,
            "message": "Password reset email sent (if user exists)",
            "timestamp": "2025-01-01T00:00:00Z"
        }
        
    except Exception as e:
        logger.error(f"Password reset request error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password reset request failed"
        )

@router.post("/reset-password/confirm")
async def confirm_password_reset(
    reset_confirm: PasswordResetConfirm,
    request_obj: Request = None
):
    """Confirm password reset with token"""
    # Check rate limit
    client_ip = security_manager._get_client_ip(request_obj) if request_obj else "127.0.0.1"
    if not security_manager.check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    try:
        # Validate password confirmation
        if reset_confirm.new_password != reset_confirm.confirm_password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Passwords do not match"
            )
        
        # In a real implementation, this would validate the token and update password
        # For now, we'll just return a success message
        return {
            "success": True,
            "message": "Password reset successfully",
            "timestamp": "2025-01-01T00:00:00Z"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Password reset confirm error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password reset confirmation failed"
        )

@router.get("/me", response_model=Dict[str, Any])
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
):
    """Get current user information"""
    return {
        "success": True,
        "user": {
            "username": current_user.username,
            "email": current_user.email,
            "role": current_user.role,
            "is_active": current_user.is_active,
            "created_at": current_user.created_at.isoformat() if current_user.created_at else None,
            "last_login": current_user.last_login.isoformat() if current_user.last_login else None
        },
        "timestamp": "2025-01-01T00:00:00Z"
    }

@router.put("/me", response_model=Dict[str, Any])
async def update_current_user(
    user_update: UserUpdate,
    current_user: User = Depends(get_current_user)
):
    """Update current user information"""
    try:
        # In a real implementation, this would update the user in the database
        # For now, we'll just return the updated user info
        updated_user = {
            "username": current_user.username,
            "email": user_update.email if user_update.email else current_user.email,
            "role": current_user.role,
            "is_active": user_update.is_active if user_update.is_active is not None else current_user.is_active,
            "created_at": current_user.created_at.isoformat() if current_user.created_at else None,
            "last_login": current_user.last_login.isoformat() if current_user.last_login else None
        }
        
        return {
            "success": True,
            "message": "User updated successfully",
            "user": updated_user,
            "timestamp": "2025-01-01T00:00:00Z"
        }
        
    except Exception as e:
        logger.error(f"User update error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User update failed"
        )

# Admin-only routes

@router.get("/users", response_model=List[Dict[str, Any]])
async def list_users(
    current_admin: User = Depends(get_current_admin_user)
):
    """List all users (admin only)"""
    try:
        users = security_manager.list_users()
        
        return [
            {
                "username": user.username,
                "email": user.email,
                "role": user.role,
                "is_active": user.is_active,
                "created_at": user.created_at.isoformat() if user.created_at else None,
                "last_login": user.last_login.isoformat() if user.last_login else None
            }
            for user in users
        ]
        
    except Exception as e:
        logger.error(f"List users error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list users"
        )

@router.get("/users/{username}", response_model=Dict[str, Any])
async def get_user(
    username: str,
    current_admin: User = Depends(get_current_admin_user)
):
    """Get user by username (admin only)"""
    try:
        user = security_manager.get_user(username)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return {
            "success": True,
            "user": {
                "username": user.username,
                "email": user.email,
                "role": user.role,
                "is_active": user.is_active,
                "created_at": user.created_at.isoformat() if user.created_at else None,
                "last_login": user.last_login.isoformat() if user.last_login else None
            },
            "timestamp": "2025-01-01T00:00:00Z"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get user error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user"
        )

@router.put("/users/{username}/activate")
async def activate_user(
    username: str,
    current_admin: User = Depends(get_current_admin_user)
):
    """Activate user (admin only)"""
    try:
        success = security_manager.activate_user(username)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return {
            "success": True,
            "message": f"User {username} activated successfully",
            "timestamp": "2025-01-01T00:00:00Z"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Activate user error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to activate user"
        )

@router.put("/users/{username}/deactivate")
async def deactivate_user(
    username: str,
    current_admin: User = Depends(get_current_admin_user)
):
    """Deactivate user (admin only)"""
    try:
        success = security_manager.deactivate_user(username)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return {
            "success": True,
            "message": f"User {username} deactivated successfully",
            "timestamp": "2025-01-01T00:00:00Z"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Deactivate user error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to deactivate user"
        )

@router.get("/security/config", response_model=SecurityConfig)
async def get_security_config(
    current_admin: User = Depends(get_current_admin_user)
):
    """Get security configuration (admin only)"""
    return SecurityConfig()

@router.get("/stats")
async def get_auth_stats(
    current_admin: User = Depends(get_current_admin_user)
):
    """Get authentication statistics (admin only)"""
    try:
        users = security_manager.list_users()
        
        stats = {
            "total_users": len(users),
            "active_users": len([u for u in users if u.is_active]),
            "inactive_users": len([u for u in users if not u.is_active]),
            "admin_users": len([u for u in users if u.role == "admin"]),
            "regular_users": len([u for u in users if u.role == "user"]),
            "locked_accounts": len(security_manager.locked_accounts),
            "active_sessions": len(security_manager.active_tokens)
        }
        
        return {
            "success": True,
            "stats": stats,
            "timestamp": "2025-01-01T00:00:00Z"
        }
        
    except Exception as e:
        logger.error(f"Auth stats error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get authentication statistics"
        ) 