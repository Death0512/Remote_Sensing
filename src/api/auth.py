import base64
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
import hashlib
import hmac
import json
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel


class Role(str, Enum):
    VIEWER = "viewer"
    ANALYST = "analyst"
    OPERATOR = "operator"
    SUPERVISOR = "supervisor"
    ADMIN = "admin"


class InvalidTokenError(ValueError):
    pass


ROLE_HIERARCHY: Dict[Role, int] = {
    Role.VIEWER: 1,
    Role.ANALYST: 2,
    Role.OPERATOR: 3,
    Role.SUPERVISOR: 4,
    Role.ADMIN: 5,
}

INSECURE_SECRETS_DENYLIST = {
    "",
    "dev-secret-key-change-in-production",
    "secret",
    "change-me",
    "changeme",
    "super-secret-tactical-defense-key-2026",
    "default",
}
INSECURE_SECRETS = INSECURE_SECRETS_DENYLIST


def is_production() -> bool:
    env = os.environ.get("ENVIRONMENT", "development").lower().strip()
    env_alias = os.environ.get("ENV", "").lower().strip()
    prod_flag = os.environ.get("PROD", "").lower().strip()
    return env in ("production", "prod") or env_alias in ("production", "prod") or prod_flag in ("1", "true")


def get_jwt_secret() -> str:
    secret = os.environ.get("JWT_SECRET")
    if is_production():
        if not secret:
            raise RuntimeError("JWT_SECRET must be explicitly set and secure in production")
        stripped = secret.strip()
        if stripped in INSECURE_SECRETS_DENYLIST or stripped.lower() in INSECURE_SECRETS_DENYLIST:
            raise RuntimeError("JWT_SECRET must be explicitly set and secure in production")
        if len(stripped) < 32 or len(set(stripped)) < 8:
            raise RuntimeError("JWT_SECRET must be at least 32 characters with sufficient entropy in production")
        return stripped
    if secret:
        stripped = secret.strip()
        return stripped if stripped else "super-secret-tactical-defense-key-2026"
    return "super-secret-tactical-defense-key-2026"


SECRET_KEY = get_jwt_secret()
ALGORITHM = "HS256"
DEFAULT_EXPIRATION_HOURS = 24


@dataclass(frozen=True)
class AuthenticatedUser:
    user_id: str
    role: Role
    username: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "role": self.role.value,
            "username": self.username,
        }


DEMO_USERS: Dict[str, Dict[str, Any]] = {
    "admin": {
        "user_id": "admin-01",
        "username": "admin",
        "password": "admin123",
        "role": Role.ADMIN,
    },
    "sup-01": {
        "user_id": "sup-01",
        "username": "supervisor",
        "password": "supervisor123",
        "role": Role.SUPERVISOR,
    },
    "supervisor": {
        "user_id": "sup-01",
        "username": "supervisor",
        "password": "supervisor123",
        "role": Role.SUPERVISOR,
    },
    "op-01": {
        "user_id": "op-01",
        "username": "operator",
        "password": "operator123",
        "role": Role.OPERATOR,
    },
    "operator": {
        "user_id": "op-01",
        "username": "operator",
        "password": "operator123",
        "role": Role.OPERATOR,
    },
    "an-01": {
        "user_id": "an-01",
        "username": "analyst",
        "password": "analyst123",
        "role": Role.ANALYST,
    },
    "analyst": {
        "user_id": "an-01",
        "username": "analyst",
        "password": "analyst123",
        "role": Role.ANALYST,
    },
    "view-01": {
        "user_id": "view-01",
        "username": "viewer",
        "password": "viewer123",
        "role": Role.VIEWER,
    },
    "viewer": {
        "user_id": "view-01",
        "username": "viewer",
        "password": "viewer123",
        "role": Role.VIEWER,
    },
}


def get_demo_users() -> Dict[str, Dict[str, Any]]:
    if os.environ.get("ENVIRONMENT", "development").lower() in ("production", "prod") or is_production():
        return {}
    return DEMO_USERS


MOCK_USERS: Dict[str, Dict[str, Any]] = get_demo_users()


class TokenRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "Bearer"
    expires_in: int
    user: Dict[str, Any]


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("utf-8")


def _b64decode(data: str) -> bytes:
    pad = len(data) % 4
    if pad > 0:
        data += "=" * (4 - pad)
    return base64.urlsafe_b64decode(data.encode("utf-8"))


def create_access_token(
    user_id: str,
    role: Union[Role, str],
    username: Optional[str] = None,
    secret: Optional[str] = None,
    expires_delta: Optional[timedelta] = None,
) -> str:
    role_str = role.value if isinstance(role, Role) else str(role)
    now = datetime.now(timezone.utc)
    exp = now + (expires_delta if expires_delta is not None else timedelta(hours=DEFAULT_EXPIRATION_HOURS))
    payload = {
        "sub": user_id,
        "user_id": user_id,
        "role": role_str,
        "username": username or user_id,
        "iat": int(now.timestamp()),
        "exp": int(exp.timestamp()),
    }
    header = {"alg": ALGORITHM, "typ": "JWT"}
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    payload_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    encoded_header = _b64encode(header_bytes)
    encoded_payload = _b64encode(payload_bytes)
    signing_input = f"{encoded_header}.{encoded_payload}".encode("utf-8")
    key = secret.strip() if secret is not None else get_jwt_secret()
    signature = hmac.new(key.encode("utf-8"), signing_input, hashlib.sha256).digest()
    encoded_signature = _b64encode(signature)
    return f"{encoded_header}.{encoded_payload}.{encoded_signature}"


def decode_access_token(token: str, secret: Optional[str] = None) -> Dict[str, Any]:
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Invalid token structure")
    encoded_header, encoded_payload, encoded_signature = parts
    signing_input = f"{encoded_header}.{encoded_payload}".encode("utf-8")
    key = secret.strip() if secret is not None else get_jwt_secret()
    expected_sig = hmac.new(key.encode("utf-8"), signing_input, hashlib.sha256).digest()
    actual_sig = _b64decode(encoded_signature)
    if not hmac.compare_digest(expected_sig, actual_sig):
        raise ValueError("Invalid token signature")

    payload_json = _b64decode(encoded_payload).decode("utf-8")
    payload: Dict[str, Any] = json.loads(payload_json)
    exp = payload.get("exp")
    if exp is None:
        raise InvalidTokenError("Token missing mandatory expiration claim 'exp'")
    now_ts = int(datetime.now(timezone.utc).timestamp())
    if now_ts >= exp:
        raise ValueError("Token has expired")
    return payload


decode_token = decode_access_token
verify_token = decode_access_token


bearer_security = HTTPBearer(auto_error=False)


def get_current_user(
    request: Request,
    bearer_creds: Optional[HTTPAuthorizationCredentials] = Security(bearer_security),
) -> AuthenticatedUser:
    token: Optional[str] = None
    if bearer_creds and bearer_creds.credentials:
        token = bearer_creds.credentials
    if not token:
        token = request.headers.get("X-Session-Token")
    if not token:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication credentials",
        )

    try:
        payload = decode_access_token(token)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )

    user_id = payload.get("user_id") or payload.get("sub")
    role_str = payload.get("role")
    username = payload.get("username", user_id)

    if not user_id or not role_str:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing required claims",
        )

    try:
        role_enum = Role(role_str.lower())
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Unrecognized role in token: {role_str}",
        )

    return AuthenticatedUser(user_id=str(user_id), role=role_enum, username=str(username))


def require_role(*allowed_roles: Union[Role, str]) -> Callable[[AuthenticatedUser], AuthenticatedUser]:
    target_roles: set[str] = set()
    for r in allowed_roles:
        if isinstance(r, Role):
            target_roles.add(r.value)
        else:
            target_roles.add(str(r).lower())

    def role_dependency(user: AuthenticatedUser = Depends(get_current_user)) -> AuthenticatedUser:
        if user.role.value not in target_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions: role '{user.role.value}' not in allowed roles {list(target_roles)}",
            )
        return user

    return role_dependency


def authenticate_user(username: str, password: str) -> Optional[AuthenticatedUser]:
    users = get_demo_users()
    record = users.get(username)
    if not record:
        return None
    if record["password"] != password:
        return None
    return AuthenticatedUser(
        user_id=record["user_id"],
        role=record["role"],
        username=record["username"],
    )
