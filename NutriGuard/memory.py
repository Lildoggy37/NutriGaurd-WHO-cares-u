"""
三层记忆架构：工作记忆 / 短期摘要 / 长期存储。

层级：
  Layer 0 — System Prompt (永远保留)
  Layer 1 — 最近 N 轮对话 (保留原样，默认 3 轮 = ~6 条消息)
  Layer 2 — 中间对话 (LLM 压缩为摘要 SystemMessage)
  Layer 3 — 早期对话 (LLM 提取关键事实 → SQLite 长期记忆 → 丢弃)

触发阈值：估计 token 数 > WORKING_MEMORY_TOKENS (默认 8000)
"""
import json
import time
import re
from typing import Sequence

from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage, RemoveMessage,
)


# ============================================================
#  配置
# ============================================================
WORKING_MEMORY_TOKENS = 8000       # 工作记忆窗口（token 数）
RECENT_ROUNDS = 3                  # 保留最近 N 轮对话
CHARS_PER_TOKEN = 2.0              # 中文估算：1 token ≈ 2 字符


def estimate_tokens(messages: Sequence[BaseMessage]) -> int:
    """估算消息列表的 token 数（中文近似：2 字符 ≈ 1 token）"""
    total = 0
    for m in messages:
        content = str(getattr(m, "content", ""))
        total += len(content) / CHARS_PER_TOKEN
    return int(total)


def _is_human(msg: BaseMessage) -> bool:
    return isinstance(msg, HumanMessage)


def _is_complete_ai(msg: BaseMessage) -> bool:
    """无 tool_calls 的 AIMessage —— 终止信号"""
    return isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None)


def _is_noise(msg: BaseMessage) -> bool:
    """Supervisor 无法理解的噪音消息"""
    if isinstance(msg, ToolMessage):
        return True
    if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
        return True
    return False


def find_last_human(messages: Sequence[BaseMessage]) -> int:
    """返回最后一条 HumanMessage 的索引，找不到返回 -1"""
    for i in range(len(messages) - 1, -1, -1):
        if _is_human(messages[i]):
            return i
    return -1


def find_last_complete_ai(messages: Sequence[BaseMessage]) -> int:
    """返回最后一条完整 AIMessage（无 tool_calls）的索引"""
    for i in range(len(messages) - 1, -1, -1):
        if _is_complete_ai(messages[i]):
            return i
    return -1


# ============================================================
#  三层分层
# ============================================================

def split_messages(messages: Sequence[BaseMessage]) -> tuple[list, list, list]:
    """
    将消息队列分为三层：
      - layer1_recent: 最近 RECENT_ROUNDS 轮（保留原样）
      - layer2_middle: 中间的对话（压缩为摘要）
      - layer3_old:    早期对话（提取长期记忆后丢弃）
    """
    n = len(messages)
    if n <= RECENT_ROUNDS * 2 + 4:
        return list(messages), [], []  # 太少，全保留

    # 找到最后 HumanMessage 的轮次边界
    human_positions = [i for i, m in enumerate(messages) if _is_human(m)]
    if len(human_positions) < RECENT_ROUNDS + 2:
        return list(messages), [], []

    # 最近 N 轮：从倒数第 N 个 HumanMessage 开始
    recent_start = human_positions[-(RECENT_ROUNDS)]
    layer1 = list(messages[recent_start:])

    # 中间：recent_start 之前，但不算太老的（后一半）
    middle_end = recent_start
    middle_start = max(0, middle_end - max(8, middle_end // 2))
    layer2 = list(messages[middle_start:middle_end])

    # 早期：最老的部分
    layer3 = list(messages[:middle_start])

    return layer1, layer2, layer3


# ============================================================
#  压缩格式化
# ============================================================

def format_messages_for_llm(messages: Sequence[BaseMessage]) -> str:
    """将消息列表格式化为 LLM 可读文本"""
    lines = []
    for m in messages:
        role = m.type
        content = str(getattr(m, "content", ""))[:500]
        if _is_noise(m):
            continue
        lines.append(f"[{role}] {content}")
    return "\n".join(lines)


def build_summary_prompt(middle_text: str) -> str:
    """生成中间层对话摘要的 prompt"""
    return (
        "请用 100 字以内总结以下对话的核心内容和用户需求，不要遗漏任何操作（如记录饮食、计算热量）：\n\n"
        + middle_text
    )


def build_extraction_prompt(old_text: str) -> str:
    """生成早期对话关键信息提取的 prompt"""
    return (
        "从以下历史对话中提取对健康管理长期有价值的个人信息，以 JSON 格式输出：\n"
        '{"疾病史": [], "饮食偏好": [], "常用食物": [], "活动习惯": "", "重要日期": []}\n'
        "只填写有明确提到的字段，没有的留空。不要输出任何其他内容。\n\n"
        + old_text
    )


# ============================================================
#  Redis 工作记忆缓存（可选层）
# ============================================================

class WorkingMemoryCache:
    """
    Redis 工作记忆缓存。
    Redis 离线时自动降级为无操作，不阻塞业务。
    """

    def __init__(self, redis_client=None):
        self._redis = redis_client

    def _key(self, session_id: str) -> str:
        return f"working_memory:{session_id}"

    async def save(self, session_id: str, messages: list, user_profile: dict) -> None:
        if self._redis is None:
            return
        try:
            data = json.dumps({
                "messages": [
                    {"type": m.type, "content": str(getattr(m, "content", ""))[:1000],
                     "id": getattr(m, "id", None)}
                    for m in messages[-20:]  # 只缓存最近 20 条
                ],
                "user_profile": user_profile,
                "timestamp": time.time(),
            }, ensure_ascii=False)
            self._redis.setex(self._key(session_id), 3600, data)  # 1h TTL
        except Exception:
            pass  # 降级

    def load(self, session_id: str) -> dict | None:
        if self._redis is None:
            return None
        try:
            raw = self._redis.get(self._key(session_id))
            if raw:
                return json.loads(raw)
        except Exception:
            pass
        return None


# ============================================================
#  长期记忆 SQLite 接口（桥接到 db.py）
# ============================================================

def save_long_term_memory(user_id: str, facts: dict, source: str = "compressor"):
    """写入长期记忆"""
    from db import save_long_term_memories as _save
    try:
        _save(user_id, facts, source)
    except Exception:
        pass  # 长期记忆写入失败不阻塞


def load_long_term_memory(user_id: str) -> dict:
    """读取长期记忆"""
    from db import load_long_term_memories as _load
    try:
        return _load(user_id)
    except Exception:
        return {}
