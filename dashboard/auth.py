"""
Authentication module for Kalshi Trading Bot Dashboard.
Provides JWT-based user authentication with registration and login.
"""

import os
import secrets
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, Field

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
SECRET_KEY = os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)


# === Pydantic Models ===


class UserCreate(BaseModel):
    """User registration request."""

    email: EmailStr
    password: str = Field(..., min_length=8)
    name: str = Field(..., min_length=2, max_length=100)


class UserLogin(BaseModel):
    """User login request."""

    email: EmailStr
    password: str


class UserResponse(BaseModel):
    """User response (no password)."""

    id: int
    email: str
    name: str
    is_active: bool
    is_admin: bool
    created_at: datetime
    last_login: Optional[datetime] = None


class Token(BaseModel):
    """JWT token response."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


class TokenData(BaseModel):
    """Decoded token data."""

    user_id: int
    email: str
    is_admin: bool = False


# === Helper Functions ===


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_token(token: str) -> Optional[TokenData]:
    """Decode and validate a JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = payload.get("sub")
        email: str = payload.get("email")
        is_admin: bool = payload.get("is_admin", False)
        if user_id is None or email is None:
            return None
        return TokenData(user_id=user_id, email=email, is_admin=is_admin)
    except JWTError:
        return None


# === Database Operations ===


async def create_user_table(db) -> None:
    """Create users table if it doesn't exist."""
    query = """
    CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        email VARCHAR(255) UNIQUE NOT NULL,
        password_hash VARCHAR(255) NOT NULL,
        name VARCHAR(100) NOT NULL,
        is_active BOOLEAN DEFAULT TRUE,
        is_admin BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMPTZ DEFAULT NOW(),
        last_login TIMESTAMPTZ,
        api_key VARCHAR(64) UNIQUE
    );
    CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
    CREATE INDEX IF NOT EXISTS idx_users_api_key ON users(api_key);
    """
    try:
        await db.execute(query)
    except Exception as e:
        # Table might already exist
        pass


async def get_user_by_email(db, email: str) -> Optional[dict]:
    """Get user by email."""
    query = "SELECT * FROM users WHERE email = $1"
    return await db.fetchone(query, email)


async def get_user_by_id(db, user_id: int) -> Optional[dict]:
    """Get user by ID."""
    query = "SELECT * FROM users WHERE id = $1"
    return await db.fetchone(query, user_id)


async def create_user(db, user: UserCreate) -> dict:
    """Create a new user."""
    password_hash = get_password_hash(user.password)
    api_key = secrets.token_urlsafe(32)

    query = """
    INSERT INTO users (email, password_hash, name, api_key)
    VALUES ($1, $2, $3, $4)
    RETURNING id, email, name, is_active, is_admin, created_at, last_login
    """
    return await db.fetchone(query, user.email, password_hash, user.name, api_key)


async def update_last_login(db, user_id: int) -> None:
    """Update user's last login timestamp."""
    query = "UPDATE users SET last_login = NOW() WHERE id = $1"
    await db.execute(query, user_id)


async def get_user_count(db) -> int:
    """Get total number of users."""
    result = await db.fetchone("SELECT COUNT(*) as count FROM users")
    return result["count"] if result else 0


# === Authentication Dependencies ===


async def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme), db=None  # Injected by route
) -> Optional[dict]:
    """Get the current authenticated user from JWT token."""
    if token is None:
        return None

    token_data = decode_token(token)
    if token_data is None:
        return None

    user = await get_user_by_id(db, token_data.user_id)
    if user is None or not user.get("is_active", True):
        return None

    return user


async def require_auth(
    token: Optional[str] = Depends(oauth2_scheme),
) -> TokenData:
    """Require authentication - raises 401 if not authenticated."""
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token_data = decode_token(token)
    if token_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return token_data


async def require_admin(
    token_data: TokenData = Depends(require_auth),
) -> TokenData:
    """Require admin privileges."""
    if not token_data.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )
    return token_data


# === Auth Router ===

from fastapi import APIRouter

auth_router = APIRouter(prefix="/api/auth", tags=["Authentication"])


@auth_router.post("/register", response_model=Token)
async def register(user_data: UserCreate, db=None):
    """Register a new user account."""
    # Check if user exists
    existing = await get_user_by_email(db, user_data.email)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered"
        )

    # Create user
    user = await create_user(db, user_data)

    # Generate token
    access_token = create_access_token(
        data={"sub": user["id"], "email": user["email"], "is_admin": user["is_admin"]}
    )

    return Token(
        access_token=access_token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=UserResponse(**user),
    )


@auth_router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db=None):
    """Login with email and password."""
    user = await get_user_by_email(
        db, form_data.username
    )  # OAuth2 uses 'username' field

    if not user or not verify_password(form_data.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Account is disabled"
        )

    # Update last login
    await update_last_login(db, user["id"])

    # Generate token
    access_token = create_access_token(
        data={
            "sub": user["id"],
            "email": user["email"],
            "is_admin": user.get("is_admin", False),
        }
    )

    return Token(
        access_token=access_token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=UserResponse(
            id=user["id"],
            email=user["email"],
            name=user["name"],
            is_active=user["is_active"],
            is_admin=user.get("is_admin", False),
            created_at=user["created_at"],
            last_login=user.get("last_login"),
        ),
    )


@auth_router.get("/me", response_model=UserResponse)
async def get_me(token_data: TokenData = Depends(require_auth), db=None):
    """Get current user profile."""
    user = await get_user_by_id(db, token_data.user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return UserResponse(
        id=user["id"],
        email=user["email"],
        name=user["name"],
        is_active=user["is_active"],
        is_admin=user.get("is_admin", False),
        created_at=user["created_at"],
        last_login=user.get("last_login"),
    )


@auth_router.post("/logout")
async def logout():
    """Logout - client should discard the token."""
    return {"message": "Logged out successfully"}
