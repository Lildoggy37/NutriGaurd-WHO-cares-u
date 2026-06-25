"""
记忆选择性召回 — 设计草图（不改现有代码，仅供展示）

当前: 压缩时全量加载 → user_profile → Supervisor 全量注入 prompt
改进: 记忆向量化存 Qdrant → 每次 Supervisor 前根据 query 做 KNN 取 Top-3 → 按需注入

和 RAG 检索共用同一套 BGE Embedding 管道。
"""
import json
import os
from typing import Optional

# ============================================================
#  1. 记忆向量化存储（压缩时调用）
# ============================================================

async def vectorize_and_store_memory(
    user_id: str,
    memory_text: str,       # 压缩器提取的记忆文本
    embedder,               # BGE Embedding (和 RAG 共用)
    qdrant_collection,      # Qdrant collection (和 RAG 分开, 专用 memories 集合)
):
    """将记忆文本向量化并存入 Qdrant 记忆库"""
    from langchain_core.documents import Document

    vector = await embedder.aembed_query(memory_text)
    doc = Document(
        page_content=memory_text,
        metadata={"user_id": user_id, "timestamp": time.time()},
    )
    qdrant_collection.add_documents([doc], embeddings=[vector])
    print(f"[MemoryStore] 记忆已向量化存储: {memory_text[:50]}...")


# ============================================================
#  2. 选择性召回（Supervisor 前调用）
# ============================================================

async def recall_relevant_memories(
    query: str,             # 当前用户 query
    user_id: str,           # 用户 ID (用于过滤)
    embedder,               # BGE Embedding
    qdrant_memories,        # 记忆专用 Qdrant collection
    top_k: int = 3,
) -> str:
    """
    根据当前 query 语义召回最相关的 Top-K 条记忆。
    返回拼接好的文本, 直接注入 Supervisor system prompt。
    """
    # 向量化 query
    query_vector = await embedder.aembed_query(query)

    # Qdrant 语义搜索 + user_id 过滤
    results = qdrant_memories.search(
        query_vector=query_vector,
        limit=top_k,
        # 只检索当前用户的记忆
        filter={"must": [{"key": "user_id", "match": {"value": user_id}}]},
    )

    if not results:
        return ""

    # 拼接记忆片段
    memories = [r.payload.get("page_content", "") for r in results]
    return "\n".join([f"- {m}" for m in memories])


# ============================================================
#  3. Supervisor 调用处改动（graph_brain.py supervisor_node）
# ============================================================

async def supervisor_node_v2(state: AgentState):
    """
    改进后的 Supervisor — 按需召回记忆，不灌全量 user_profile。

    改动对比:
      OLD: profile_str = json.dumps(state.get("user_profile", {}))
      NEW: recall_and_inject(state, embedder, qdrant)
    """
    messages = state["messages"]
    last_user_msg = _get_last_human_message(messages)

    # --- 按需召回（只取相关的 Top-3，不灌全量） ---
    memory_context = ""
    if last_user_msg:
        memory_context = await recall_relevant_memories(
            query=last_user_msg.content,
            user_id=state.get("user_profile", {}).get("用户标识", "unknown"),
            embedder=_embedder,          # 全局 BGE（和 RAG 共用）
            qdrant_memories=_qdrant_memories,  # 记忆专用 Qdrant
            top_k=3,
        )

    # 构建 system prompt（两类记忆分开处理）
    # 1. SQLite 健康档案 → 始终注入（总量小、对路由至关重要）
    health_profile = load_sqlite_health_profile(user_id)  # {"性别":"男","身高":175,"疾病":"糖尿病"}
    profile_str = json.dumps(health_profile, ensure_ascii=False)

    # 2. Qdrant 语义记忆 → 按需召回 Top-3（总量大、按 query 筛选）
    memory_context = await recall_relevant_memories(
        query=last_user_msg.content,
        user_id=user_id,
        embedder=_embedder,
    )

    # 三段拼接
    sys_prompt = ROUTE_PROMPT
    sys_prompt += f"\n\n【健康档案】{profile_str}"        # 必注
    if memory_context:
        sys_prompt += f"\n\n【相关记忆】{memory_context}"  # 选注

    # 其余逻辑不变
    clean = _filter_noise(messages)
    llm_input = [SystemMessage(content=sys_prompt)] + clean

    async with llm_rate_limiter:
        response = await llm.ainvoke(llm_input)
    # ... 后续解析逻辑不变


# ============================================================
#  4. 三层注入时机的完整对比
# ============================================================

"""
时机一: Supervisor 前 (影响路由)
  例: "帮我查一下饮食" + 记忆召回 → 检测到用户有糖尿病史
      → Supervisor 路由到 rag_expert 时 query 已经带了上下文
      → RAG 检索 "糖尿病饮食禁忌" 而非泛泛的 "饮食"

时机二: RAG Expert 后 (影响回答)
  例: RAG 检索到 "低GI食物推荐" + 记忆召回 → 用户偏好素食
      → Agent 生成回答时过滤掉肉类推荐, 只输出素食低GI选项

时机三: Action Expert 前 (影响操作)
  例: 用户说 "记一下早饭" + 记忆召回 → 最近记录显示喜欢吃燕麦
      → Agent 直接用燕麦默认份量填充, 减少追问次数
"""


# ============================================================
#  5. 和当前代码的兼容性（不改架构，只加节点）
# ============================================================

"""
不需要改现有的 graph 拓扑。在 graph_brain.py 中:

  新增: async def memory_recall_node(state):
         从 Qdrant 做 KNN 召回 → 结果写回 state["user_profile"]["相关记忆"]
         然后自动路由到 supervisor

  图拓扑: START → preprocess → memory_recall → supervisor → ...

  memory_recall 是透明节点:
    - 首次使用 (无记忆): 空字符串注入, supervisor 行为不变
    - 有记忆后: 语义匹配 Top-3, supervisor prompt 比现在更精准
    - 失败: 直接放行, 不阻塞
"""


# 占位符：避免直接跑这段代码报 NameError
import time
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from dataclasses import dataclass

class AgentState(dict): pass
ROUTE_PROMPT = "..."
llm_rate_limiter = None
_embedder = None
_qdrant_memories = None

def _get_last_human_message(messages) -> Optional[BaseMessage]:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m
    return None

def _filter_noise(messages) -> list:
    return [m for m in messages if not hasattr(m, 'tool_calls')]

print("记忆召回设计草图已加载。不做任何代码修改。")
print("核心思路: BGE向量化 → Qdrant检索 → 按需注入Supervisor prompt")
