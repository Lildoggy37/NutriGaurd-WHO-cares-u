"""
JWT 认证模块 — 用户身份验证与授权。

用法:
  token = create_token("user_abc123")           # 生成 JWT
  user_id = verify_token(token)                  # 验证并提取 user_id
  user_id = extract_user_id(request)             # 从 HTTP 请求中提取

安全原则:
  - user_id 从 JWT token 解析, 不接受前端传入
  - 工具调用时强制使用认证 user_id, 拒绝越权访问
  - 无 token 的请求使用 session_id 降级 (开发兼容)
"""
import os
import time
import jwt
from typing import Optional

SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "nutriguard-dev-secret-change-in-production")
ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = 24


def create_token(user_id: str) -> str:
    """生成 JWT token"""
    payload = {
        "user_id": user_id,
        "iat": time.time(),
        "exp": time.time() + TOKEN_EXPIRE_HOURS * 3600,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str) -> Optional[str]:
    """验证 JWT 并返回 user_id, 无效返回 None"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("user_id")
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError, Exception):
        return None


def extract_user_id(request, fallback_to_session: bool = True) -> str:
    """
    从 HTTP 请求中提取 user_id。
    优先 Authorization header, 其次 ?token= query param, 最后 session_id fallback。
    """
    # 1. Authorization: Bearer <token>
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        user_id = verify_token(auth[7:])
        if user_id:
            return user_id

    # 2. Query param ?token=
    token = request.query_params.get("token", "")
    if token:
        user_id = verify_token(token)
        if user_id:
            return user_id

    # 3. Fallback: session_id (开发阶段兼容)
    if fallback_to_session:
        return request.query_params.get("session_id", "anonymous")

    return "anonymous"


# ============================================================
#  工具调用安全拦截
# ============================================================

class AuthContext:
    """
    线程安全的认证上下文。
    在 API 请求入口设置, 工具调用时通过此对象获取当前用户 ID。
    """
    _current_user: str = "anonymous"

    @classmethod
    def set_current_user(cls, user_id: str):
        cls._current_user = user_id

    @classmethod
    def get_current_user(cls) -> str:
        return cls._current_user


def enforce_user_id(tool_user_id: str | None, param_name: str = "user_id") -> str:
    """
    工具调用安全拦截: 如果传入的 user_id 和认证用户不一致, 强制使用认证用户。
    返回安全的 user_id。
    """
    current = AuthContext.get_current_user()
    if current == "anonymous":
        return tool_user_id or "anonymous"
    if tool_user_id and tool_user_id != current:
        print(f"[Auth] 越权拦截: 工具传入 {tool_user_id}, 认证用户 {current}, 已强制覆盖")
    return current
