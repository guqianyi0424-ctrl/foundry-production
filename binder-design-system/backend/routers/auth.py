from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, Field, field_validator
from jose import JWTError, jwt
import bcrypt
from sqlalchemy.orm import Session
import json
import os
import re
import time
import uuid
from urllib.parse import parse_qs

from database import get_db, User, AuditLog

SECRET_KEY = os.getenv("DEEPBINDER_SECRET_KEY", "deepbinder_secret_key_2026_change_in_production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440
MIN_SECRET_KEY_LENGTH = 32
DEFAULT_SECRET_KEY = "deepbinder_secret_key_2026_change_in_production"
MAX_LOGIN_FAILURES = int(os.getenv("DEEPBINDER_MAX_LOGIN_FAILURES", "5"))
LOGIN_FAILURE_WINDOW_SECONDS = int(os.getenv("DEEPBINDER_LOGIN_FAILURE_WINDOW_SECONDS", "300"))
_login_failures: dict[str, list[float]] = {}

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)

router = APIRouter()


def get_password_hash(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def _login_failure_key(request: Request, username: Optional[str]) -> str:
    client_host = request.client.host if request.client else "unknown"
    normalized_username = (username or "").strip().lower() or "<empty>"
    return f"{client_host}:{normalized_username}"


def _active_login_failures(key: str, now: Optional[float] = None) -> list[float]:
    now = now if now is not None else time.monotonic()
    cutoff = now - LOGIN_FAILURE_WINDOW_SECONDS
    attempts = [timestamp for timestamp in _login_failures.get(key, []) if timestamp >= cutoff]
    if attempts:
        _login_failures[key] = attempts
    else:
        _login_failures.pop(key, None)
    return attempts


def _is_login_rate_limited(key: str) -> bool:
    if MAX_LOGIN_FAILURES <= 0:
        return False
    return len(_active_login_failures(key)) >= MAX_LOGIN_FAILURES


def _record_login_failure(key: str) -> None:
    if MAX_LOGIN_FAILURES <= 0:
        return
    attempts = _active_login_failures(key)
    attempts.append(time.monotonic())
    _login_failures[key] = attempts


def _clear_login_failures(key: str) -> None:
    _login_failures.pop(key, None)


async def get_current_user(token: Optional[str] = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    if not token:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="请先登录")
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="请先登录")
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="请先登录")
    return user


async def require_login(current_user: Optional[User] = Depends(get_current_user)):
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="请先登录")
    return current_user


def require_role(allowed_roles: list):
    async def role_checker(current_user: User = Depends(require_login)):
        if current_user.role not in allowed_roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="权限不足")
        return current_user
    return role_checker


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=8, max_length=128)
    email: Optional[str] = Field(default=None, max_length=254)

    @field_validator("username")
    @classmethod
    def username_must_be_safe_identifier(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
            raise ValueError("用户名只能包含字母、数字、下划线和连字符")
        return value

    @field_validator("email")
    @classmethod
    def email_must_have_basic_shape(cls, value: Optional[str]) -> Optional[str]:
        if value is None or value == "":
            return value
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", value):
            raise ValueError("邮箱格式不正确")
        return value


def validate_secret_key_for_environment() -> None:
    env = os.getenv("DEEPBINDER_ENV", "development").strip().lower()
    if env in {"prod", "production"} and (
        SECRET_KEY == DEFAULT_SECRET_KEY or len(SECRET_KEY) < MIN_SECRET_KEY_LENGTH
    ):
        raise RuntimeError("生产环境必须设置长度不少于32位的 DEEPBINDER_SECRET_KEY")


class UserResponse(BaseModel):
    id: str
    username: str
    email: Optional[str]
    role: str
    created_at: Optional[datetime] = None
    last_login: Optional[datetime] = None

    class Config:
        from_attributes = True


@router.post("/auth/register", response_model=UserResponse, summary="用户注册")
async def register(req: RegisterRequest, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.username == req.username).first()
    if existing:
        raise HTTPException(status_code=400, detail="用户名已存在")
    if req.email:
        existing_email = db.query(User).filter(User.email == req.email).first()
        if existing_email:
            raise HTTPException(status_code=400, detail="邮箱已被注册")

    user = User(
        id=str(uuid.uuid4()),
        username=req.username,
        email=req.email,
        hashed_password=get_password_hash(req.password),
        role="researcher",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post("/auth/login", summary="用户登录")
async def login(request: Request, db: Session = Depends(get_db)):
    content_type = request.headers.get("content-type", "").split(";")[0]
    body = await request.body()

    if content_type == "application/json":
        try:
            payload = json.loads(body.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            payload = {}
        username = payload.get("username")
        password = payload.get("password")
    else:
        payload = parse_qs(body.decode("utf-8"), keep_blank_values=True)
        username = payload.get("username", [""])[0]
        password = payload.get("password", [""])[0]

    failure_key = _login_failure_key(request, username)
    if _is_login_rate_limited(failure_key):
        raise HTTPException(status_code=429, detail="登录失败次数过多，请稍后再试")

    user = db.query(User).filter(User.username == username).first()
    if not user or not verify_password(password or "", user.hashed_password):
        _record_login_failure(failure_key)
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    _clear_login_failures(failure_key)
    user.last_login = datetime.utcnow()
    audit = AuditLog(id=str(uuid.uuid4()), user_id=user.id, action="login", target="auth", detail={"username": user.username})
    db.add(audit)
    db.commit()

    access_token = create_access_token(data={"sub": user.username, "role": user.role})
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": UserResponse.model_validate(user).model_dump(),
    }


@router.get("/auth/me", response_model=UserResponse, summary="当前用户信息")
async def get_me(current_user: User = Depends(require_login)):
    return current_user


@router.get("/users", summary="用户列表(管理员)")
async def list_users(current_user: User = Depends(require_role(["admin"])), db: Session = Depends(get_db)):
    users = db.query(User).all()
    return {"users": [UserResponse.model_validate(u).model_dump() for u in users]}
