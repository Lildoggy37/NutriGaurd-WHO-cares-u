# NutriGuard-Copilot 面试准备手册

> AI 应用开发实习生面试 | 2025-06

---

## 1. 项目总览

**一句话定位**：独立设计的多智能体 AI 膳食管家，基于 LangGraph Supervisor 模式协调 7 个 Agent 节点，通过 MCP 协议接入 8 个工具，覆盖 RAG 检索、饮食记录、合规审查、记忆管理全链路。

**技术栈**：`Python` `LangGraph` `LangChain` `FastAPI` `Qdrant` `BGE/Reranker` `Redis Stack` `SQLite` `MCP` `Next.js 14` `Docker`

**核心解决的问题**：

| 问题 | 方案 |
|------|------|
| 单 Agent 长链路不可靠 | Supervisor 多 Agent 分工 + Harness 框架（重试/超时/兜底） |
| 上下文易漂移 | 三层记忆架构（工作/短期/长期）+ token 阈值压缩 |
| 工具调用缺乏标准 | MCP 协议封装 8 个工具，按职能分派给不同 Agent |

---

## 2. 架构全景图

```
用户 HTTP 请求 (SSE)
    │
    ▼
┌─────────────────────────────────────────┐
│  api_server.py (FastAPI)                │
│  ├── sliding_window_rate_limiter (Redis) │
│  ├── /api/chat/stream (SSE)            │
│  └── lifespan: spawn MCP subprocess    │
└──────────────┬──────────────────────────┘
               │
    ┌──────────▼──────────────────────────┐
    │  graph_brain.py (LangGraph)         │
    │                                      │
    │  START → preprocess → supervisor     │
    │              │                       │
    │     ┌────────┼────────┬──────────┐   │
    │     ▼        ▼        ▼          ▼   │
    │  rag      action    slot        END  │
    │  expert   expert   filler            │
    │     │        │                       │
    │     ▼        ▼                       │
    │  rag       memory                    │
    │  reflection compressor               │
    │     │        │                       │
    │     └───┬────┘                       │
    │         ▼                            │
    │     supervisor (loop until FINISH)   │
    └──────────────┬───────────────────────┘
                   │ MCP stdio
    ┌──────────────▼───────────────────────┐
    │  mcp_server.py (MCP Tools)           │
    │  ├── search_diet_guidelines          │
    │  ├── check_food_gi                   │
    │  ├── search_medical_taboos           │
    │  ├── search_food (SQLite)            │
    │  ├── log_user_meal                   │
    │  ├── calculate_daily_calories        │
    │  ├── generate_shopping_list          │
    │  └── update_health_profile           │
    └──────────────┬───────────────────────┘
                   │
    ┌──────────────┼───────────────────────┐
    │  Qdrant (RAG)  Redis (Cache/Limit)   │
    │  SQLite (DB)   BGE Models (local)    │
    └──────────────────────────────────────┘
```

**一条信息从头到尾**：

```
用户: "糖尿病可以吃什么？"
  → FastAPI /api/chat/stream (SSE)
    → sliding_window_rate_limiter (Redis, 60req/min/IP)
      → graph.astream_events(initial_state)
        → preprocess_node: "糖尿病可以吃什么？" (无改动, 跳过)
        → supervisor_node: CoT分析 → route: rag_expert    KNN相似召回top3记忆注入
        → rag_expert_node: create_agent 决定调用 search_diet_guidelines
          → MCP stdio → mcp_server 执行 RAG (Qdrant召回+Reranker)
          → 返回检索结果
        → rag_reflection_node: 审查回答合规性 → PASS
        → END (前端 SSE 收到答案文本流)
```

---

## 3. 模型选型

### 3.1 LLM：qwen-plus (阿里 DashScope)

| 标准 | qwen-plus | qwen-max | GPT-4o | DeepSeek |
|------|----------|----------|--------|----------|
| 中文能力 | ★★★★★ | ★★★★★ | ★★★★ | ★★★★ |
| 成本 (/1M tokens) | ￥1.6 入 + ￥4 出 | ￥4 入 + ￥12 出 | ~￥70 | ~￥2 |
| 延迟 | ~1s | ~2s | ~3s | ~1s |
| 工具调用 | ✓ | ✓ | ✓ | ✓ |

**选择 qwen-plus 的原因**：
- 中文营养学知识 > GPT-4o（DashScope 中文训练数据优势）
- 成本是 GPT-4o 的 1/20，单 query 约 ￥0.001-0.004
- 延迟低，CoT 路由 + RAG 总耗时可控制在 5s 内
- 兼容 OpenAI 格式（`base_url: dashscope.aliyuncs.com/compatible-mode/v1`）

**坑点**：qwen 的 `with_structured_output` (JSON Schema) 通过 compatible-mode 端点不稳定 → 改用手动 JSON 提取（`_extract_json`）。

### 3.2 Embedding：BGE-large-zh-v1.5 (本地)

| 标准 | BGE-large-zh-v1.5 | text-embedding-3 | BGE-M3 |
|------|-------------------|------------------|--------|
| 维度 | 1024 | 1536 | 1024 |
| 中文 MTEB | ★★★★★ | ★★★★ | ★★★★ |
| 部署 | 本地 (免费) | API (付费) | 本地 |
| Hybrid 支持 | Dense only | Dense only | Dense+Sparse 一体 |

**为什么没用 BGE-M3**：
- M3 在多语言上更好，但项目是纯中文营养学 → v1.5 的中文微调更专注
- 已有 v1.5 模型本地缓存，换 M3 要重新下载、重新跑评测验证
- BM25 Sparse 路由已由独立的 BM25 模型覆盖

### 3.3 Reranker：BGE-Reranker-v2-m3 (本地)

**为什么不是简单的余弦相似度排序？**

| 方法 | 原理 | 精度 | 速度 |
|------|------|------|------|
| 余弦相似度 | 独立向量点积 | 中 | 快 |
| **CrossEncoder** | `(query, doc)` 成对交叉注意力 | **高** | 慢 |

CrossEncoder 把 query 和 document 一起输入模型，能理解 query 具体在问什么再评价 document 的相关性。代价是每对 (query, doc) 都要跑一次模型推理。所以只在 Top-10 粗召回后的精排阶段使用。

### 3.4 BM25 (Qdrant/bm25, 本地)

传统的 TF-IDF 词袋检索。**Dense 向量语义匹配能力强但会稀释精确术语**（"GI值" 不如 "升糖指数" 的语义向量突出）。BM25 精确命中 "嘌呤"、"GI值"、"31.0g" 等专业关键词，和 Dense 互补。

---

## 4. 技术选型决策

### 4.1 LangGraph vs 原生 LangChain Agent

| 标准 | LangGraph | LangChain Agent |
|------|-----------|-----------------|
| 状态管理 | StateGraph + TypedDict | 隐式 |
| 条件路由 | conditional_edges | 无原生支持 |
| Checkpointer | MemorySaver (按 thread_id 隔离) | 无 |
| 可视化 | graph.get_graph() 导出 | 无 |

选 LangGraph 的核心原因：**需要显式控制 Agent 之间的路由逻辑**。Supervisor → rag/action/slot/finish 的条件分发、压缩后回到 supervisor 的循环，这些是 LangChain Agent 做不到的。

### 4.2 MCP vs REST API

| 标准 | MCP | REST |
|------|-----|------|
| 工具发现 | `tools/list` 自动发现 | 手动文档 |
| 参数 schema | 自动从函数签名生成 | 手动定义 |
| 通信 | stdio (进程内管道) | HTTP |
| 延迟 | 极低 (无网络开销) | 需要网络 |

选 MCP 是因为所有工具都是 Python 函数——不需要远程调用。stdio 通信零延迟、零网络依赖、自动工具发现。`load_mcp_tools(session)` 一行代码拿到全部工具的 LangChain 包装。

### 4.3 Qdrant vs Milvus vs Chroma

| 标准 | Qdrant | Milvus | Chroma |
|------|--------|--------|--------|
| 部署 | 内存模式 (`:memory:`) 零配置 | 需要 Docker | 文件模式 |
| Hybrid Search | ✓ (Dense+Sparse) | 需手动实现 | ✗ |
| FastEmbed 集成 | ✓ (BM25) | ✗ | ✗ |

选 Qdrant 的核心原因：**内存模式 + Hybrid Search 原生支持**。开发阶段不需要独立部署，一行 `location=":memory:"` 即可。Hybrid Search 的 Dense+Sparse 双路召回是 Qdrant 的原生 API，不需要额外实现。

### 4.4 Redis Stack vs 标准 Redis

| 功能 | Redis Stack | 标准 Redis |
|------|------------|-----------|
| `FT.CREATE` (向量索引) | ✓ | ✗ |
| `FT.SEARCH` (KNN) | ✓ | ✗ |
| 限流 (Sorted Set) | ✓ | ✓ |
| 工作记忆 (SETEX) | ✓ | ✓ |

选 Redis Stack 是为了语义缓存（需要 RediSearch 模块的 KNN 向量搜索）。如果没有 Redis Stack，启动时自动检测并禁用缓存功能，不阻塞核心业务。

### 4.5 SQLite vs PostgreSQL

选 SQLite：单机部署、零配置、Python 自带、41 种食物 + 用户画像数据量极小。`upsert_health_profile` 的字段级增量更新通过 SQL UPDATE 实现，天然原子。

---

## 5. RAG 系统

### 5.1 检索全链路

```
用户: "糖尿病可以吃什么？"
    │
    ▼
┌──────────────────────────────┐
│ Qdrant Hybrid Search         │
│ ┌─────────┐  ┌─────────────┐ │
│ │ BGE Dense│  │ BM25 Sparse │ │
│ │ (1024维) │  │ (TF-IDF)    │ │
│ └────┬─────┘  └──────┬──────┘ │
│      └──────┬────────┘        │
│             ▼                 │
│      融合排序 → Top-10         │
└─────────────┬────────────────┘
              │
              ▼
┌──────────────────────────────┐
│ BGE-Reranker-v2-m3           │
│ CrossEncoder 精排 10→3       │
│ score ∈ [0, 1]               │
└─────────────┬────────────────┘
              │
              ▼
       Top-3 上下文 → LLM 生成答案
```

### 5.2 为什么 Hybrid Search

| 场景 | Dense only | Sparse only | Hybrid |
|------|-----------|------------|--------|
| "尿酸高不能吃什么" | ✓ (理解尿酸=痛风) | ✗ (可能漏掉) | ✓ |
| "GI值是多少" | △ (不了解缩写) | ✓ (精确命中) | ✓ |
| 表格数据精确匹配 | △ | ✓ | ✓ |

### 5.3 Reranker 打分

```python
# 对 (query, doc) 成对打分
pairs = [[query, doc.page_content] for doc in top10_docs]
scores = reranker.predict(pairs)  # CrossEncoder 直接打分 [0,1]
top3 = sorted(zip(docs, scores))[:3]
```

**为什么不用 Dense Cosine**：Dense 是把 query 和 doc 独立嵌入再算相似度——信息损失。CrossEncoder 把两者拼接后做完整注意力，精度高得多。代价是慢（每对 0.3s），所以只在 Top-10 精排阶段使用。

### 5.4 语义缓存

```
query → BGE embed_query → 1024维向量
    → Redis FT.SEARCH KNN (cosine distance, k=1)
    → 最近缓存条目相似度 = 1 - cosine_distance
    → ≥0.85? 命中(<50ms) : 未命中(走完整RAG, ~3s)
```

**0.85 阈值来源**：BGE 向量对同类变体相似度通常 0.85-0.95，异类 0.2-0.6。0.85 恰好卡在中间——过于严格会导致缓存形同虚设（>0.95），过于宽松会导致误命中（<0.7 可能把"糖尿病"和"心血管"匹配在一起）。

**加速比**：冷链路 ~3s / 缓存命中 ~40ms ≈ **75x**。

### 5.5 评测指标

| 指标 | 值 | 含义 |
|------|-----|------|
| Recall@3 | 85% | 100 条中 85 条在 Top-3 找到目标内容 |
| MRR | 0.772 | 正确答案平均排在第 1.3 位 |
| RAGAS Faithfulness | 1.000 | 回答声明 100% 基于检索证据 |
| RAGAS Context Precision | 0.667 | 召回文档 67% 与问题相关 |

---

## 6. Chunking 策略

### 6.1 优化前

```python
MarkdownHeaderTextSplitter([("##","Chapter"),("###","Section")])
# 按 H2/H3 标题切分
```

问题：
- 块大小不均：大表格 500 字一块，短段落 30 字一块
- 表格跨标题被截断
- 无 overlap，边界硬切丢上下文
- 无元数据，chunk 不知道自己的章节归属

### 6.2 优化后 (`chunking.py`)

四层策略：

```
Layer 1: 解析 H2/H3 → 提取 Chapter/Section 元数据 (不用于切割)
Layer 2: 表格保护 → 检测 | 行，防止从表格中间切开
Layer 3: RecursiveCharacterTextSplitter(512char, 128 overlap) → 固定大小切割
Layer 4: 元数据注入 → chunk 文本前缀加 【Chapter】[Section]
```

效果：

| 指标 | 优化前 | 优化后 | 变化 |
|------|--------|--------|------|
| Context Precision | 0.400 | 0.667 | **+67%** |
| Reranker avg score (糖尿病) | 0.50 | 0.67 | +34% |
| Reranker avg score (鸡胸肉) | 0.65 | 0.84 | +29% |

元数据标签（`【慢性病饮食管理】[糖尿病]`）让 BM25 关键词匹配能命中章节标题，让 Reranker 能利用结构化信息判断文档相关性。

---

## 7. 上下文与记忆

### 7.1 三层记忆架构

```
Layer 1: 工作记忆 (Redis)
  ├── 最近 20 条消息 + user_profile → JSON 序列化
  ├── 1h TTL, 服务重启可恢复
  └── Redis 离线 → 降级为 MemorySaver

Layer 2: 短期记忆 (LangGraph MemorySaver + 压缩)
  ├── AgentState.messages (operator.add 累加)
  ├── 压缩阈值: estimate_tokens() > 8000
  ├── Layer 1 (最近3轮): 完整保留
  ├── Layer 2 (中间): LLM 生成 100 字摘要
  └── Layer 3 (早期): 提取 JSON 事实 → SQLite 长期记忆 → 丢弃

Layer 3: 长期记忆 (SQLite)
  ├── user_health_profiles: 性别/年龄/身高/体重/疾病 (数值计算用)
  └── long_term_memories: 疾病史/偏好/习惯 (语义上下文用)
```

### 7.2 压缩死循环修复

问题链：压缩 → 消息减少 → Supervisor 丢失终止信号 → 重新路由 agent → 再次压缩 → 死循环。

修复：
1. 压缩后注入 `SystemMessage(name="memory_summary")` — Supervisor 检测到直接 FINISH
2. 压缩后显式保留最后一条完整 AIMessage（终止信号）
3. 压缩后显式保留最后一条 HumanMessage（用户意图）
4. 图拓扑层面：压缩节点返回 `next_node: "FINISH"`（不经过 supervisor 条件边）

---

## 8. Session 对话

### 8.1 两层 Session 概念

| | 前端 session_id | MCP ClientSession |
|------|----------------|-------------------|
| 目的 | 隔离不同用户的对话 | 和 MCP 子进程通信 |
| 创建 | lID() | lifespan 启动时 |
| 存储 | MemorySaver (thread_id=session_id) | 无持久化 (stdio 管道) |
| 生命周期 | 浏览器会话期间 | 服务运行期间 |

**关键区分**：`session_id` 用于隔离用户对话状态，`MCP ClientSession` 是全局唯一的协议通信管道。所有用户共享同一个 MCP 子进程和工具实例。

### 8.2 MCP 通信流程

```
api_server.py (lifespan)
    │
    ├─ StdioServerParameters(command=python, args=[mcp_server.py])
    │  → 拉起子进程
    │
    ├─ stdio_client(params) → (read, write)
    │  → stdin/stdout 双向管道
    │
    ├─ ClientSession(read, write)
    │  → JSON-RPC 协议层
    │
    ├─ session.initialize()
    │  → 握手: 能力发现/版本协商
    │
    └─ load_mcp_tools(session)
       → 发现 8 个工具, 返回 LangChain BaseTool 列表
```

---

## 9. 多 Agent 框架

### 9.1 7 个节点

| 节点 | 作用 | LLM调用 | 工具 |
|------|------|---------|------|
| `preprocess` | 纠错/同义词展开/指代消解 | 1次 (≤10字跳过) | 无 |
| `supervisor` | CoT 意图分析 → 路由分发 | 1次 | 无 |
| `rag_expert` | 知识检索 + 回答生成 | ReAct (1-3次) | search_diet_guidelines/check_food_gi/search_medical_taboos/search_food |
| `action_expert` | 饮食记录/热量计算/健康管理 | ReAct (1-3次) | log_user_meal/calculate_daily_calories/generate_shopping_list/update_health_profile |
| `slot_filler` | 信息不全时动态追问 | 1次 | 无 |
| `rag_reflection` | 合规审查 (幻觉/安全/完整) | 1次 | 无 |
| `memory_compressor` | 压缩 + 长期记忆提取 | 0-2次 (>8k token触发) | 无 |

### 9.2 图拓扑

```python
# 固定边
START → preprocess → supervisor

# 条件边 (基于 state["next_node"])
supervisor → rag_expert / action_expert / slot_filler / END

# 固定路径
rag_expert → rag_reflection → END  (RAG查完审查完直接结束)
action_expert → memory_compressor → supervisor  (action可能继续路由)
```

### 9.3 Supervisor CoT 推理

```python
# 不是一次直接输出 JSON, 而是引导 LLM 分步分析
ROUTE_PROMPT = """
1. 用户最后一句话的核心诉求是什么？
2. 这属于哪类意图？知识查询 / 操作执行 / 信息不全 / 结束？
3. 确认理由
4. 输出 JSON: {"route":"rag_expert","reason":"..."}
"""
```

**效果**：CoT 比直接输出 JSON 准确率提升约 5 个百分点（尤其在 "生成低GI购物清单" 这种含 GI 但不是知识查询的模糊 case 上）。

---

## 10. 消息流转

### 10.1 AgentState 结构

```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]  # 消息累加器
    next_node: str            # Supervisor 路由决策
    user_profile: Dict[str,str]  # 用户画像 (注入 Supervisor prompt)
```

### 10.2 消息类型

- `HumanMessage` — 用户输入
- `AIMessage` — Agent 回答、LLM 路由、工具调用决策
- `AIMessage(tool_calls=...)` — 标记工具调用
- `ToolMessage` — 工具返回结果
- `SystemMessage` — 压缩摘要、审查标记、预处理改写
- `RemoveMessage` — 压缩时标记删除

### 10.3 SSE 流式推送

```python
async for event in graph.astream_events(state, version="v2"):
    if kind == "on_chat_model_stream":
        node = event["metadata"]["langgraph_node"]
        if node not in {"supervisor", "rag_reflection", "memory_compressor"}:
            yield f"data: {json.dumps({'type':'text','content':token})}"

    elif kind == "on_chain_start":
        yield f"data: {json.dumps({'type':'status','content':f'节点[{name}]开始'})}"

    elif kind == "on_chain_end":
        yield f"data: {json.dumps({'type':'status','content':f'节点[{name}]完成({elapsed}s)'})}"
```

**黑名单过滤**：supervisor/reflection/compressor 的内部 JSON 不推送给前端，只推送 rag_expert/action_expert/slot_filler 的生成 token。

---

## 11. 兜底与熔断

### 四层防护

```
第 1 层: Redis 滑动窗口       → 60req/60s/IP (防恶意刷, 离线跳过)
第 2 层: LLMRateLimiter        → 令牌桶(10 QPS) + Semaphore(并发5) (防 API 超额)
第 3 层: NodeHarness            → retry(1-2) / timeout(10-60s) / fallback FINISH
第 4 层: ToolHarness            → 超时 / 连续5次熔断60s / 降级文案
```

### NodeHarness 装饰器

```python
@node_harness(name="supervisor", retries=1, timeout_seconds=30, fallback={"next_node":"FINISH"})
async def supervisor_node(state): ...
```

自动提供：指数退避重试、超时熔断、Prometheus 指标埋点、统一日志格式。

### ToolHarness 装饰器

```python
@tool_harness(name="search_diet_guidelines", timeout_seconds=30, max_failures=5)
```

连续 5 次超时 → 自动熔断 60s → 返回 fallback_msg `"膳食指南检索暂时不可用"`。成功后自动重置计数器。

### LLMRateLimiter

```python
async with llm_rate_limiter:  # 令牌桶 + Semaphore
    response = await llm.ainvoke(...)
```

令牌不足时 `await asyncio.sleep(0.1s)` 异步排队，不丢请求。所有 LLM 调用共享同一个全局限流器。

---

## 12. 评测体系

### 12.1 14 个 Pytest 单元测试

| 类别 | 测试 | 数量 | 依赖 LLM |
|------|------|------|----------|
| 图拓扑 | 节点注册/编译/边连接/路由 | 4 | ✗ |
| 路由函数 | rag/action/slot/finish 分发 | 4 | ✗ |
| AgentState | 最小状态/画像/消息累加 | 3 | ✗ |
| 压缩逻辑 | 阈值边界/RemoveMessage/保留 | 3 | ✗ |
| 集成测试 | 全链路 4 意图 | 4 (skip by default) | ✓ |

10 个纯单元测试不调用 LLM（用 MagicMock 替代 ChatOpenAI），4 个集成测试默认跳过（`-m integration` 启用）。

### 12.2 RAG 检索评测 (100 条 + 5-fold CV)

**评测流程**：

```
eval_rag_v2.json (100条, 8类别)
    ↓ np.random.seed(42) 固定洗牌
    ↓ 分成 5 折, 每折 20 条
    ├── Fold 1: test=[0:20]    train=[20:100]
    ├── Fold 2: test=[20:40]   train=[0:20]+[40:100]
    ├── Fold 3: test=[40:60]   ...
    ├── Fold 4: test=[60:80]   ...
    └── Fold 5: test=[80:100]  ...
    ↓ 每折独立跑: Qdrant Hybrid → Reranker → Hit/Miss
    ↓ 汇总 5 折: mean ± std
```

每个 fold 的 test 集是 20 条不同的查询，确保每条查询都被当过测试集。5 折独立跑完取均值和标准差作为最终指标，单次划分的随机波动被抵消。

| 指标 | 值 |
|------|-----|
| 评测集 | 100 条, 8 类别, 80/20 train/test (seed=42) |
| 对抗样本 | 12 条混入全量集 (错别字/口语/中英混杂) |
| Recall@3 | **85%** (全量实测) |
| MRR | **0.772** |
| CV 用法 | `python eval_all.py --cv 5` |

> **注意：这里的 Train/Test 不是模型训练。** BGE、BM25、Reranker 全是预训练好的冻结模型，没有任何梯度更新或权重调整。Train set (80 条) 用于调超参数——chunk_size、overlap、score_threshold、neighbor_expansion 策略——反复跑看指标；Test set (20 条) 是 hold-out，调完所有参数后只跑一次，作为最终报告指标。如果没有这个分离，用全部 100 条调参再报告 100 条上的分数，就是对着答案调参的过拟合。5-fold CV 进一步保证划分的偶然性不影响结论。

### 12.3 RAGAS 评测 (LLM-as-judge, 5 条)

RAGAS 用 LLM 裁判评估生成质量，每项指标都是一次独立的 LLM 调用：

```
Faithfulness:     LLM 逐声明检验 → 是否基于检索证据 → 1.000
Answer Relevancy: 从答案反向生成问题 → 与原始 query 相似度 → 0.660
Context Precision: LLM 判断每个 chunk 是否相关 → 0.667
Context Recall:    关键词命中率 → 0.733
```

**RAGAS 没有 5-fold CV**——它是生成质量评估，依赖 LLM API 调用（每条查询需要 3-4 次 LLM 裁判调用，5 条共 ~15 次）。当前 5 条是快速验证用的，扩充到 20+ 条后可加 CV。

| 指标 | 值 |
|------|-----|
| **Faithfulness** | **1.000** |
| **Answer Relevancy** | **0.660** |
| **Context Precision** | **0.667** |
| **Context Recall** | **0.733** |

### 12.4 防过拟合措施

| 措施 | RAG 检索 | RAGAS | 说明 |
|------|---------|-------|------|
| Train/Test 分离 | ✓ 80/20 | — | seed=42 固定划分 |
| 5-fold CV | ✓ | ✗ (待扩充) | RAG 那 100 条全部实测过 |
| 对抗样本 | ✓ 12 条 | — | 错别字/口语/中英混杂 |
| JSON versioning | ✓ | ✓ | `eval_rag_v2.json` 含 version 字段 |
| 规则+LLM 双验证 | ✓ 路由评测 | — | 规则 85%(下限), CoT 90%(实测) |
| 固定 seed | ✓ | — | 划分可复现 |

---

## 13. 优化历程

| 阶段 | 做了什么 | 动因 | 效果 |
|------|---------|------|------|
| 1 | 引入 RAGAS 评测 | 只有 Recall@3 不知道生成质量 | 暴露 Faithfulness 0.57 (JSON解析bug) |
| 2 | 修复 Faithfulness prompt | 5/10 条 JSON 解析失败 | Faithfulness 0.572→1.000 |
| 3 | Reranker 阈值 (0.2) + 邻居展开 | Precision 0.27 太低 | Precision 0.27→0.40 |
| 4 | 答案生成 prompt 重写 | Relevancy 0.57 太低 | Relevancy 0.57→0.73 |
| 5 | Chunking 重写 (512ch+128ov+元数据) | Precision 卡在 0.40 | **Precision 0.40→0.667 (+67%)** |
| 6 | Token 追踪 + LLM 并发控制 | 高并发保护 | API 调用排队不丢请求 |

---

## 14. 高频面试题

### Q1: "你的 RAG 和直接用 ChatGPT 有什么区别？"

> "ChatGPT 的知识是训练时固化的,不知道我语料里关于中国膳食指南的具体内容。我的 RAG 先做 Hybrid 检索定位相关段落,再让 LLM 基于检索结果生成回答——比如"糙米饭 GI=56"这个数据不是背出来的,是从语料表格里检索到的。RAGAS Faithfulness=1.0 验证了所有回答声明都在检索证据中有出处。"

### Q2: "Hybrid Search 为什么比单路好？"

> "Dense(BGE 语义) 擅长同类映射——'尿酸高'→'痛风禁忌',但会稀释精确术语。BM25 擅长精确匹配——'GI=56'、'嘌呤' 这类数值和专有名词。评测数据验证了这一点:全 Hybrid 比纯 Dense 的 Recall@3 高约 5 个百分点。"

### Q3: "上下文压缩后怎么避免死循环？"

> "做了三件事。一是压缩后在消息中注入 memory_summary 标记,Supervisor 检测到直接 FINISH。二是压缩后显式保留最后一条完整 AIMessage 作为终止信号。三是压缩时确保最后一条 HumanMessage 也在保留区里,不丢失用户意图。图拓扑层面压缩节点返回 FINISH 不经过条件边。"

### Q4: "Reranker 打分的原理是什么？"

> "CrossEncoder 把 query 和 document 成对输入,做交叉注意力计算,直接输出 0-1 的 relevance score。和 Bi-Encoder(独立嵌入+余弦相似度) 相比,精度高得多,因为它知道 query 在问什么再评价文档。代价是每对 (query, doc) 都要跑一次模型推理,所以只在粗召后的 Top-10 精排阶段用。"

### Q5: "MCP 协议相比 REST 有什么优势？"

> "MCP 解决了工具发现、参数 schema 生成、进程隔离三个问题。`load_mcp_tools(session)` 一行代码自动发现所有工具并生成 LangChain 包装,不需要手动定义 REST 端点。stdio 通信零网络延迟。8 个工具按职能分派给两个 Agent 持有,互不污染。"

### Q6: "如果 LLM API 超时了怎么办？"

> "四层兜底。LLMRateLimiter 控制并发量(5并发/10QPS),减少 API 压力。NodeHarness 自动重试 1-2 次(指数退避 1.5x)。ToolHarness 连续 5 次失败触发 60s 熔断,返回友好降级文案。全链路降级——Redis 离线跳过限流,RAG 模型失败用 LLM 兜底。"

### Q7: "评测数据怎么来的？怎么防过拟合？"

> "两层评测分开。先说清楚一点——这里的 Train/Test 不是模型训练, BGE/BM25/Reranker 全是预训练好的冻结模型,没有梯度更新。Train set (80 条) 用于调 chunk_size/overlap/threshold 等超参数,反复跑看指标；Test set (20 条) 是 hold-out,调完所有参数后只跑一次,作为最终报告——防止对着答案调参。5-fold CV 进一步保证划分的偶然性不影响结论——`seed=42` 固定洗牌,均分 5 折每折 20 条,跑 5 次取均值±标准差。RAGAS 用 LLM-as-judge 评估 5 条查询的生成质量,每条 3-4 次 LLM 调用,暂时没做 CV。防过拟合还有对抗样本 12 条和 JSON versioning。"

### Q8: "为什么用 Redis 做语义缓存而不是直接存在内存？"

> "三个原因。一是跨请求共享——服务重启后缓存还在。二是向量搜索能力——Redis Stack 的 FT.SEARCH KNN 原生支持 1024 维向量余弦相似度搜索。三是 TTL 自动过期,24h 后自动清理。Redis 离线时自动降级,每次都走完整 RAG。"

### Q9: "Chunking 是怎么优化的？效果如何？"

> "之前按 Markdown 标题切分,块大小不均、表格截断、无 overlap。优化成四层:解析标题提取元数据→表格保护→512字符+128 overlap 切割→元数据注入 chunk 文本前缀。Context Precision 从 0.400 跳到 0.667(+67%),因为元数据标签让 BM25 能命中章节标题,Reranker 能用结构化信息判断相关性。"

### Q10: "多Agent框架和单Agent有什么区别？"

> "单 Agent 处理多种异构任务容易产生幻觉——一个 LLM 同时负责知识检索和饮食记录很容易搞混。我把职责拆给了 7 个节点:Supervisor 做路由,rag_expert 做知识检索,action_expert 做操作执行,slot_filler 做追问,reflection 做审查,compressor 做记忆管理。每个节点有独立的 system_prompt 和工具集,降低单一节点的复杂度。"

### Q11: "Session 怎么隔离不同用户？"

> "前端用 crypto.randomUUID() 生成 session_id,存 localStorage 保证同一浏览器跨刷新不丢。后端把这个 session_id 作为 LangGraph MemorySaver 的 thread_id,不同 ID 的消息历史完全隔离。MCP ClientSession 是全局唯一的协议管道,所有用户共享同一个子进程和工具实例。"

### Q12: "如果100个用户同时问,系统怎么扛？"

> "四层防护。Redis 滑动窗口限流 60req/min/IP 拦截恶意刷。LLMRateLimiter 令牌桶(10QPS) + Semaphore(5并发) 控制 LLM API 压力,超额请求 async 排队不丢。NodeHarness retry 兜底超时。ToolHarness 连续失败自动熔断。全链路优雅降级——任何组件离线都不影响核心功能。"

### Q13: "记忆中发生冲突怎么办？"

> "当前是 last-write-wins——后写的覆盖先写的，没有版本比较。如果用户第一次说'身高175'，之后说'身高170'，最终存170。已实现 `updated_at` timestamp，但还没在写入时做冲突检测。改进方向：写入前对比新旧值，发现冲突时触发 slot_filler 追问——'您上次说的是175，这次是170，以哪次为准？'——而不是静默覆盖。"

### Q14: "跨用户记忆怎么隔离？"

> "三层隔离。第一层：`session_id → thread_id → MemorySaver` checkpoint，不同 ID 的消息历史完全物理隔离。第二层：所有 SQLite 查询带 `WHERE user_id=?` 过滤，每个用户只能查到自己的数据。第三层：Redis key 前缀 `working_memory:{session_id}` 自然隔离。语义缓存（`cache:{hash(query)}`）是全局共享的——但它只存 query→answer，不存任何用户数据，不存在隐私交叉。"

### Q15: "记忆检索怎么用才高效？"

> "当前是全量加载——因为每个用户长期记忆通常只有 5-10 条，KV 结构直接全量读取比向量检索更简单高效。Supervisor 路由时整个 `user_profile` 注入系统 prompt，不需要额外检索。如果用户量增长到 1000+ 且每人 50+ 条记忆，可以改为 BGE Embedding 做语义检索：每条记忆向量化 → Qdrant 索引 → 按 query 做 KNN 找 Top-5 最相关记忆注入 prompt，和 RAG 检索共用同一套 Embedding 模型。"

### Q16: "用什么策略召回相关记忆？"

> "当前策略是'全量注入'——压缩时把 user_profile + 长期记忆全部加载，Supervisor 调用时直接 `json.dumps(state['user_profile'])` 拼进系统 prompt。优点是简单、零延迟。缺点是随着记忆增长 prompt 会膨胀，无关记忆也会注入。
>
> 优化设计在 `MEMORY_RECALL_DESIGN.py`（设计草图，未改代码）：两类记忆分开处理——
>
> **SQLite 健康档案（始终注入）**：性别/年龄/身高/体重/疾病列表，总量 5-8 个字段约 100 字。'用户有糖尿病'应该影响每一次路由，无关 query 也要知道。
>
> **Qdrant 语义记忆（按需召回 Top-3）**：存在 Qdrant 的不是原始对话消息（那些在 Redis 工作记忆里），也不是健康数值（那些在 SQLite 里），而是**压缩器产出的语义片段**——具体是每次压缩时 Layer 2 生成的摘要（"用户偏好低GI饮食，常吃燕麦鸡胸肉"）和 Layer 3 提取的结构化事实（{"饮食偏好":["低GI"],"常用食物":["燕麦"]}）。这些文本被 BGE 向量化后存入 Qdrant 记忆库，和知识库文档共享同一套 Embedding。每次 Supervisor 前用当前 query 做 KNN 取 Top-3——比如 query 是'推荐今天晚餐'，召回'偏好低GI饮食'+'常用燕麦鸡胸肉'+'上一次记录午餐是糙米饭'三条。
>
> Supervisor prompt 最终三段拼接：路由规则 + 健康档案(必注) + 语义记忆(选注)。进一步可扩展三层时机：Supervisor 前（影响路由）、RAG Expert 后（过滤偏好的回答）、Action Expert 前（补全操作上下文减少追问）。"

### Q17: "Agent 引用错误信息怎么办？"

> "Reflection 审查当前只检查 'RAG 回答 vs 检索证据' 的一致性，不检查 'Agent 引用 vs SQLite 实际数据' 的一致性。举个例子：如果 SQLite 里存的身高是 175，Agent 在某次回答中说'您身高 170'，Reflection 不会发现这个错误——因为它只审查 RAG 上下文，不审查用户记忆引用。解决方向：加一个 `verify_memory_consistency` 节点，在 Agent 生成回答后提取引用的用户数据字段，和 SQLite 实际值对比。不一致时附加 SystemMessage 修正。"

### Q18: "如何防止记忆隐私泄漏？"

> "坦率说这是当前最薄弱的一环。做了基础的：`.env` 不入 git、API Key 不在镜像里硬编码。没做的：SQLite 数据明文存储（没有加密）、日志里会打印 user_profile 内容（没有脱敏）、没有 PII 删除接口（用户没法要求删数据）。面试时我会明确：'当前阶段项目侧重 Agent 架构验证，隐私加密和日志脱敏是下一步的事。生产中会加 AES 加密 SQLite 字段、日志 PII 过滤、GDPR 删除接口。'"

### Q19: "记忆会随时间失效怎么办？"

> "当前混合策略：Redis 工作记忆 1h TTL、语义缓存 24h TTL——这两块有自动过期。SQLite 长期记忆和健康画像永不过期，只增不减。这意味着用户三个月前说的 '喜欢低GI'，偏好变了之后旧记忆还在。改进方向：给 `long_term_memories` 加 `last_accessed` 字段，低访问频率的记忆自动降权或归档。每次压缩时刷新活跃记忆的 timestamp。Surpervisor 注入 prompt 时只选取最近 30 天的高频记忆。"

### Q20: "记忆系统怎么保证长期稳定？"

> "SQLite 开了 WAL 模式——即使服务崩溃，已提交的写入不会丢失。但有两个不稳定点：一是 MemorySaver 是纯内存，服务重启所有对话历史丢失（已有 Redis 工作记忆能恢复最近 20 条，但不完整）；二是没有定期备份和数据版本迁移。修复方案：把 MemorySaver 换成 SqliteSaver（LangGraph 官方提供的 SQLite 版 checkpointer），对话历史也持久化。加 Cron 定期备份到 S3/阿里云 OSS。加 schema version 字段做迁移版本管理。"

### Q21: "用户中途换话题，意图切换怎么处理？"

> "每一个新消息都会经过完整的 Preprocess → Supervisor 流程。Supervisor 接收的是完整的 `state['messages']` 历史加上当前 user_profile，通过 CoT 分步分析判断用户意图。如果用户之前说'帮我记午餐'，中间插了一句'对了糖尿病能吃水果吗'，Supervisor 看到最新 HumanMessage 是问水果的，路由就是 rag_expert 而不是继续 action_expert。两个保障：第一，Supervisor 每次都重新审视完整上下文，不会'锁定'在上一轮的状态里；第二，Preprocess 节点会对指代不明确的追问做消解——比如用户前面聊糖尿病，突然说'那这个能吃吗'，Preprocess 会结合上文把'这个'替换成具体的食物名。但有一个边界情况：如果用户在 action_expert 正在执行多步操作的中途插话，之前的操作状态还没写入 SQLite，切换后数据可能处于中间态。修复方向：在 action_expert 的每次操作后立即 commit，保证意图切换时数据已落盘。"

### Q22: "你的 Agent 依赖各种 API，和 Web Search 有什么区别？为什么不用小红书搜？"

> "Web Search 是只读的信息管道，NutriGuard 是闭环行动系统。同样是'糖尿病吃什么'，Web Search 返回 10 个链接，用户自己看、自己算、自己记。NutriGuard 在一个 query 里执行了 5 步：RAG 检索权威指南 → 查 SQLite 发现用户 175cm/80kg/糖尿病 → 计算个性化热量目标 → Reflection 审查回答安全 → 记住偏好下次自动注入。小红书的问题有三：信源不可控（'每天只吃苹果瘦 10 斤'），没有个性化（不知道你的身高体重疾病），不能执行（不能帮你记录饮食算热量）。NutriGuard 的壁垒不在检索，在检索之后的行动链——读（RAG）+ 写（SQLite）+ 算（BMR→TDEE→宏量）+ 审（Reflection）+ 记（三层记忆）。"

### Q23: "RAG 语料是哪里来的？怎么保证可信度？"

> "语料汇编自三个国家级指南：中国营养学会《中国居民膳食指南(2025)》、国家卫健委《成人高尿酸血症与痛风食养指南(2024)》、《中国2型糖尿病防治指南(2024)》。文件头标注了来源，14K 字手写整理。可信度靠两层保证：数据源权威（国家级机构指南）+ Reflection 审查（LLM 生成回答后校验是否基于检索证据，幻觉自动拦截）。诚实说——这是我个人的手工汇编而非官方 PDF 直接解析，生产环境需要自动解析原文并给每段结果带来源引用。RAGAS Faithfulness 1.0 证明当前版本没有超出检索证据的幻觉。"

### Q24: "全链路降级是怎么做的？哪些组件能降级？"

> "除了 MCP 子进程（启动必须的），所有组件都有降级：Redis 离线→限流跳过、缓存跳过、工作记忆退到 MemorySaver；BGE 模型文件缺失→返回'知识库暂时不可用'由 LLM 兜底；BM25 下载失败→降级为 Dense-only 检索；Reranker 未安装→用 Qdrant 混合分数排序替代；LLM 调用失败→NodeHarness 重试 2 次→仍失败返回友好提示；SQLite 写入异常→rollback 返回'记录失败'。用户最多感受到'回答慢了一点'，不会看到 500。设计原则是所有加速层（Redis/BGE/Reranker）都可降级，核心链路（LLM/SQLite）有重试。"

### Q25: "Pytest 测试怎么覆盖的？有哪些方面？"

> "14 个测试分 5 个方面：图拓扑 4 个（7 节点注册/编译/START→preprocess 链/4 条条件边），路由函数 4 个（lambda 对 rag/action/slot/finish 的返回值），AgentState 3 个（最小状态/带画像/operator.add 累加），压缩逻辑 3 个（10→11 阈值/RemoveMessage ID/12 条保留最后 2 条），集成 4 个（全链路真实 LLM 跑 rag/action/slot/finish，默认 skip）。纯单元测试不调 LLM——MockChatOpenAI 替代。集成测试 `pytest -m integration` 启用。"

### Q26: "Recall 的值怎么判的？AI 标注还是人工？"

> "规则匹配，不是 AI 也不是人工。每条 query 预设 3-4 个容错关键词（如 '糙米饭 GI=56'→keywords:['56','GI','升糖']），代码检查 Top-K chunk 内容是否包含任意一个关键词。100 条标注花了我 2 小时。对于精确数值查询完全可靠（'GI=56'），对于语义等价覆盖率 ~85-90%。不是 100% 精确——但比 AI 标注便宜（省 ￥50+）且比人工快 10 倍。"

### Q27: "用户画像具体有哪些字段？"

> "SQLite `user_health_profiles`：gender/age/height_cm/weight_kg/activity_level/conditions/updated_at。外加 `long_term_memories` 表的 KV 结构（疾病史/饮食偏好/常用食物/活动习惯）。AgentState 里还有一个 user_profile dict，是压缩器生成的文本摘要（'用户偏好低GI饮食，三次询问糖尿病禁忌...'）。数值字段用于 `calculate_daily_calories` 的 BMR 计算，语义字段用于 Supervisor 路由决策。"

### Q28: "Chunking 遇到大文档更新怎么做？"

> "当前是全量重切——改一段就要重切整个文件。改进方案是按 section_id 做增量更新：每个 section 存 SHA256 hash，更新时 hash 没变的 section 直接跳过，只对变更的 section 做'删旧 chunk→切新 chunk→重新向量化→写入 Qdrant'。50K 文档如果只改了一个章节（~2K 字），只处理 4-5 个 chunk 而不是 80+ 个，速度提升 5-6 倍。"

### Q29: "现在数据库有多大？"

> "极小。SQLite `nutriguard.db` ~100KB（41 种食物 + 用户画像 + 长期记忆 + 餐食记录）。Qdrant 内存模式 ~200KB（19 个 chunk 的 Dense + Sparse 向量）。单用户场景每天新增几条记录，增速可忽略。"

### Q30: "100 条测试数据覆盖了哪些方面？"

> "8 个类别每类 12-13 条：disease_taboos（12 条，糖尿病/痛风/高血压/肾病/肝硬化等）、nutrition_data（13 条，精确数值如鸡胸肉 31g 蛋白质）、semantic_equiv（13 条，口语→术语如'尿酸高'→痛风）、food_gi（12 条，GI 值查询）、special_population（13 条，孕吐/哺乳/儿童/更年期/术后/化疗）、daily_advice（13 条，喝水/运动/外卖/节食）、casual_mix（12 条，火锅/空腹牛奶/隔夜菜）、adversarial（12 条，错别字/口语/中英混杂/极端短句）。覆盖 11 种疾病 × 20+ 种食物 × 5 种用户表达方式。"

## 15. 诚实说还缺什么

这是一个应届生/实习生的独立项目，不是生产系统。以下是已知的局限性——在面试中主动提及会体现工程判断力：

| 层面 | 缺口 | 影响 | 计划 |
|------|------|------|------|
| 安全 | SQLite 明文, 日志无脱敏, 无 PII 删除 | 隐私合规风险 | AES 加密 + 日志过滤 + GDPR API |
| 持久化 | MemorySaver 重启丢失对话 | 用户体验中断 | 换 SqliteSaver |
| 备份 | 无定期备份, 无灾难恢复 | 数据丢失风险 | Cron + OSS |
| 冲突 | 记忆 last-write-wins, 无冲突检测 | 用户数据覆盖风险 | timestamp 比较 + 追问 |
| 过期 | 长期记忆永不过期 | 过时数据污染 | access_count + 自动归档 |
| 一致性 | 无 Agent 引用 vs 实际数据校验 | 幻觉风险 | verify_memory_consistency 节点 |
| 语料 | 14K 字 mock_corpus, 覆盖率不足 | Recall@3 锁死在 85% | 扩充到 50K+ 字 |
| 路由 | Supervisor 偶尔误判（~10%） | 对话体验 | CoT 已提升到 90%, 进一步调 prompt |
| 评测 | 被测集仅 100 条, 压缩场景仅 6 条 | 统计意义有限 | 持续扩充基准集 |

---

## 16. 自我介绍 (面试开场 2 分钟)

> 面试官好，我叫韦皓淇，华中科技大学计算机学院然后数据科学与大数据专业的大二在读，2028年毕业。
>
> 我做的项目主要覆盖两个方向——**AI应用开发**和**Java 后端高并发**。
>
> AI 方向上，我独立设计和实现了一个叫 NutriGuard 的多智能体 AI 膳食管家。核心架构基于 LangGraph Supervisor 模式，协调了 7 个 Agent 节点，通过 MCP 协议标准化接入了 8 个工具。技术上主要解决了三个问题：一是单 Agent 长链路不可靠，我用 Supervisor 做了意图分发加 Harness 框架做统一重试和降级；二是上下文容易漂移，我用三层记忆架构（Redis 工作记忆 + LangGraph 短期 + SQLite 长期）加 token 阈值压缩来管理；三是评估体系缺失，我自建了 100 条 RAG 评测集加 RAGAS 生成质量评估，经历了 6 轮优化把 Context Precision 从 0.40 提到了 0.67。项目有 14 个 pytest 单元测试，Docker 一键部署。
>
> 后端方向上，我做了这个平台但是是偏向电影交流的。技术栈是 SpringBoot + Redis + MySQL + MyBatis + SpringCloud。我在里面负责了秒杀模块的缓存优化：用 Redis 布隆过滤器防穿透、互斥锁加逻辑过期防击穿、Lua 脚本在 Redis 侧完成库存预扣保证原子性。微服务体系上用了 Nacos 做配置中心、Gateway 搭 API 网关做负载均衡。
>
> 这两个项目方向不同但底层能力互补——AI 项目锻炼了我从 0 设计复杂系统的能力，后端项目让我理解了高并发场景下缓存、消息队列、分布式一致性怎么落地。我对 AI Agent 和工程落地都很感兴趣，希望能在实习中继续深入。"

**使用提示**：根据面试岗位调整侧重点。投 AI 岗位时 NutriGuard 占 70% 篇幅，投后端时黑马点评占 70%。结构化项目（社区团购、校园助手、个人博客）可作为课堂作业背景提及，不需要展开。
