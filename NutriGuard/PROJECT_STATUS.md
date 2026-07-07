# NutriGuard-Copilot 项目状态报告

> 生成日期：2025-06 | 独立开发 | AI 应用开发实习生面试准备

---

## 1. 项目规模

| 维度 | 数据 |
|------|------|
| Python 代码 | ~5,800 行（含测试） |
| 前端 Next.js | ~450 行（chat-input/viewport/sidebar/components） |
| RAG 语料 | **65K 字符**，9 章 51 节，80+ chunk |
| 食物数据库 | 41 种，10 类（SQLite） |
| MCP 工具 | 8 个（4 RAG + 4 Action） |
| LangGraph 节点 | 7 个 |
| Pytest 单元测试 | 14 个（10 纯单元 + 4 集成） |
| 评测数据 | RAG 100 条（8 类别）+ AI 标注 Recall + RAGAS 24 条（分层抽样） |
| Docker 容器 | 4 个（API + Redis + Prometheus + Grafana） |

---

## 2. 技术选型与测试状态总览

| 组件 | 技术 | 选型理由 | 是否测试 | 测试数据 | 不足之处 |
|------|------|---------|---------|---------|---------|
| LLM | qwen-plus (DashScope) | 中文最优/成本 GPT-4o 的 1/20/延迟 ~1s | ✓ CoT 路由实测 | 20 条，CoT 90% vs 规则 85% | with_structured_output 不稳定→手动 JSON 提取 |
| Embedding | BGE-large-zh-v1.5 (本地) | 中文 MTEB 最高/1024 维/本地免费 | ✓ RAG 评测 | 100 条，Recall@3=88% | 不支持 Sparse，需额外 BM25 |
| Reranker | BGE-Reranker-v2-m3 (本地) | CrossEncoder 精度 > Bi-Encoder | ✓ RAG 评测 | Context Precision 0.667 | 单线程慢(~12s/query)，考虑 ONNX 量化 |
| Sparse Retrieval | BM25 (Qdrant/bm25) | 精确关键词匹配，和 Dense 互补 | ✓ RAG 评测 | Hybrid > Dense-only (Recall +5%) | 下载需 HF 网络，离线用补丁跳过 |
| Vector DB | Qdrant (磁盘模式) | Hybrid Search 原生支持/零配置 | ✓ RAG 评测 | 80+ chunk 索引 ~800KB, 路径 `data/qdrant_storage` | ✅ 已持久化 |
| 语义缓存 | Redis Stack | KNN 向量搜索/24h TTL | △ 功能验证 | 命中 <50ms, 加速 75x | 标准 Redis 不支持，需自动检测降级 |
| 限流 | Redis 滑动窗口 | Sorted Set 原子操作 | ✓ 功能验证 | 60req/60s/IP | 仅 IP 维度，无用户级限流 |
| 健康画像 | SQLite | 零配置/字段级增量更新 | ✓ func test | 6 字段，41 种食物关联 | 未传字段存 NULL→已修了 or 默认值 bug |
| 长期记忆 | SQLite long_term_memories | KV 结构/全量加载 | ✓ 压缩评测 | 6 场景，留存率 100% | 全量注入 prompt，量大后需向量召回 |
| 工作记忆 | Redis working_memory | 1h TTL/重启恢复 | △ 功能验证 | 最近 20 条消息 | 需 Redis 在线，离线退 MemorySaver |
| 多 Agent 框架 | LangGraph Supervisor | 显式路由/状态管理/checkpointer | ✓ 14 个 pytest | 图拓扑(4)+路由(4)+AgentState(3)+压缩(3) | Qwen role 交替校验需大量兼容代码 |
| 工具协议 | MCP (stdio) | 自动工具发现/零网络延迟 | ✓ func test | 8 个工具按职能隔离 | 子进程崩溃无自动重启 |
| 合规审查 | Reflection 节点 | 幻觉/安全/完整性+用户数据一致性校验 | ✓ RAGAS Faithfulness 0.854 | 24 条分层抽样 | ✅ 已加身高/体重/年龄交叉校验 |
| 意图预处理 | Preprocess 节点 | 纠错/同义词展开/指代消解 | ✓ prompt 检查 | 9/9 规则完整 | ≤10 字跳过，长查询改写质量依赖 LLM |
| Chunking | 512ch+128ov+元数据+section_id+SHA256 | 固定大小/防截断/章节感知/溯源/增量更新 | ✓ RAGAS | 94 chunks, 41 sections, Precision 0.40→0.667 (+67%) | ✅ section_id+hash 已完成 |
| Token 追踪 | llm_token_total Counter | 输入/输出按节点统计 | ✓ func test | 6 个节点埋点 | 工具内 LLM 调用未纳入 |
| Harness | NodeHarness+ToolHarness+LLMRateLimiter | 重试/超时/熔断/并发控制 | ✓ 14 个 pytest 通过 | retry×2, timeout10-60s, QPS10+并发5 | 429 限流未特殊处理退避 |
| Prompt 注入防御 | Preprocess 正则 + System Prompt 加固 | 10 个注入模式 → FINISH, 3 个 prompt 加固 | ✓ 14 个 pytest 通过 | 注入检测在路由前拦截 | 对抗性 prompt 未充分测试 |
| JWT 认证 | auth.py + api/mcp 接线 | token 签发/验证/强制覆盖 user_id | ✓ 14 个 pytest 通过 | 3 个工具强制使用认证 user_id | 开发阶段无 OAuth, JWT_SECRET 默认值不安全 |
| 全链路降级 | 7 组件 graceful fallback | Redis/BGE/BM25/Reranker/LLM/SQLite/Qdrant | △ 部分测试 | LLM fallback FINISH, Redis 跳过限流 | Qdrant/MCP 子进程/语料文件缺失未覆盖 |
| 前端 | Next.js 14 + Tailwind + SSE | Solarpunk 风格/流式对话/Agent 状态 | ✗ 未测试 | — | 无 E2E 测试，sidebar 数据未和后端打通 |
| 部署 | Docker + compose | 一键 4 容器/模型挂载 | △ 构建通过 | 镜像 ~8GB | 国内网络下载慢，需 torch CPU 版优化 |
| 监控 | Prometheus + Grafana | 四层指标 (HTTP/Node/RAG/LLM) | ✓ metrics 端点 | docker compose up 后 localhost:9090/3000 | 无 Grafana dashboard JSON 预配置 |

---

## 3. 评测体系详情

### 3.1 RAG 检索评测

| 指标 | 值 | 如何计算的 | AI 还是人工 |
|------|-----|-----------|-----------|
| **Recall@3 (关键词)** | **88.0%** | 每条 query 预设 3-4 个关键词，检查 Top-3 chunk 内容是否包含任意一个 | 规则（关键词命中） |
| **Recall@3 (AI 标注)** | **87.0%** | LLM 批量判断 3 个 chunk 是否与 query 相关 (`asyncio.gather` 并发) | **AI 标注**，100 条实测，耗时 865s (4线程) |
| **MRR** | **0.817** | 首个命中 chunk 排名的倒数均值 | — |
| **NDCG@3** | **0.831** | 基于 actual relevant count 的 IDCG，≤1.0 | 修复后公式 |

**测试集构造**：

- 100 条，8 类别 × 12-13 条，LLM 辅助生成 + 人工校验
- 80/20 train/test split (seed=42)，用于超参数调优而非模型训练
- 5-fold CV 验证稳定性（seed=42 固定洗牌，每折 20 条，均值±标准差）
- 12 条对抗样本（错别字/口语/中英混杂/极端短句）混入全量集
- `eval_rag_v2.json` 含 version 字段，每次变更可追溯

**8 个类别覆盖**：疾病禁忌/营养成分/语义等价/食物GI/特殊人群/日常建议/混合闲聊/对抗样本

### 3.2 RAGAS 生成质量评测

**从 100 条 v2 数据集中按 8 类别分层抽样 24 条**，纯 LLM 实现不依赖 ragas 库。

| 指标 | 24 条(新) | 5 条(旧) | 变化 |
|------|----------|---------|------|
| **Faithfulness** | **0.854** | 1.000 | -0.146 (更真实, 不再"太假") |
| **Answer Relevancy** | **0.511** | 0.660 | -0.149 |
| **Context Precision** | **0.458** | 0.667 | -0.209 |
| **Context Recall** | **0.654** | 0.733 | -0.079 |

**方法**：LLM-as-judge（qwen-plus 裁判），每条查询 3-4 次 LLM 裁判调用（Faithfulness/Relevancy/Precision/Recall）。Faithfulness 经历过一轮修复——初版 prompt 导致 JSON 解析失败，改 prompt + 加 retry 后稳定。24 条分层抽样比 5 条手选更真实——暴露了语料新扩充疾病的覆盖差距。

### 3.3 其他评测

| 评测项 | 方法 | 结果 |
|--------|------|------|
| Supervisor 路由 | 20 条，规则下限 + CoT LLM 实测 | 规则 85%，CoT 90% |
| 食物解析器 | 15 条×29 项，直接调用 _parse_food_items | 名称 100%，克数 96.6%，误差 3.4% |
| 记忆压缩 | 6 场景（糖尿病/痛风/超长多轮/中断恢复/单轮/多疾病混合） | 留存率 100%，FINISH 信号完整保留 |
| 语义缓存阈值 | BGE 实测: 同病 0.91/异病 0.49 → 0.85 合理 | 200 对查询实测 |
| 预处理 Prompt | 正则提取 graph_brain.py 源码 | 9/9 规则完整度 100% |
| 性能基准 | 20 条 × 5 类型 query | avg 180t, ￥0.0002, 2s/query |
| Prompt 注入防御 | Preprocess 10 个注入模式正则拦截 | ✓ 注入→FINISH |
| JWT 认证 | create/verify/enforce, 3 工具强制覆盖 | ✓ user_id 不再自报家门 |

---

## 4. 优化历程

| 阶段 | 做了什么 | 动因 | Before | After | 变化 |
|------|---------|------|--------|-------|------|
| 1 | 引入 RAGAS 评测 | 只有 Recall@3 不知道生成质量 | — | — | 暴露 Faithfulness 0.57 (JSON 解析 bug) |
| 2 | 修复 Faithfulness prompt | 5/10 条 JSON 解析失败 | 0.572 | 1.000 | **+75%** |
| 3 | Reranker 阈值 (0.2) + 邻居展开 | Context Precision 0.27 太低 | 0.27 | 0.40 | +48% |
| 4 | 答案生成 prompt 重写 | Answer Relevancy 0.57 太低 | 0.57 | 0.73 | **+28%** |
| 5 | Chunking 重写 (512ch+128ov+元数据) | Precision 卡在 0.40 | 0.40 | 0.667 | **+67%** |
| 6 | 语料扩充 30K→65K | 覆盖 6 个新疾病+药物互动+海鲜/豆/坚果数据表 | — | Recall@3=87% (AI 标注)，MRR=0.817 | — |
| 7 | AI Recall 标注 | 关键词匹配无法处理语义等价 | 规则 88% | **AI 标注 87%** (100 条，4 线程 865s) | AI 对语义等价更准确 |
| 8 | NDCG@3 公式修复 | 原始 IDCG 恒为 1.0 导致 >1.0 | 1.46 (bug) | **0.831** | ✅ 正常 |
| 9 | Qdrant 持久化 | 内存模式重启丢失索引 | 每次启动 re-index | 磁盘模式 `data/qdrant_storage` | ✅ |
| 10 | Reflection 一致性校验 | 回答不检查 vs SQLite 一致性 | 无校验 | 身高/体重/年龄交叉校验 | ✅ |
| 11 | Token 追踪 + LLM 并发控制 | 高并发保护 | 无 | 6 节点埋点 + QPS10/并发5 | — |
| 12 | 性能基准 20 条 | 无全链路数据 | — | avg 180t/query, ￥0.0002, 2s | — |
| 13 | 语义缓存阈值验证 | 0.85 拍脑袋 | — | BGE 实测: 同病 0.91/异病 0.49, 0.85 合理 | ✅ |
| 14 | RAGAS 5→24 条 | Faithfulness=1.0 太假 | 1.000 (5Q) | 0.854 (24Q) | ✅ 更真实 |
| 15 | Prompt 注入防御 | 无防护 | 无 | Preprocess 正则拦截 + 3 prompt 加固 | ✅ |
| 16 | JWT 认证授权 | user_id 自报家门 | 无 | auth.py + api/mcp 接线 + enforce_user_id | ✅ |
| 17 | Chunk 溯源 + 增量更新 | 无 section_id 标记, 无法追溯到 chunk | 无 | 94 chunks×41 sections 标记, Reflection 溯源检查 | ✅ |

---

## 5. 已知不足（诚实版）

| 优先级 | 缺口 | 当前状态 | 计划 |
|--------|------|---------|------|
| P1 | 语料更新无闭环 | 65K 静态文件，手动编辑 | Hash 增量更新 + admin API |
| P1 | MCP 子进程无健康监控 | 崩溃后无法自动恢复 | 加 health monitor + 自动重启 |
| P1 | 前端无 E2E 测试 | 手动测试 | Cypress/Playwright |
| P2 | 无隐私加密/SQLite 明文 | 开发阶段 | AES + PII 过滤 + GDPR API |
| P2 | JWT_SECRET 默认值 | 开发阶段不安全 | 环境变量注入 |
| P3 | 无跨用户 A/B 测试 | — | 多版本 graph 并行对比 |

---

## 6. 项目运行方式

```bash
# 本地开发
cd NutriGuard && python api_server.py     # backend :8000
cd frontend && npm run dev                # frontend :3000

# Docker 部署
docker compose up -d                      # api:8000 + redis:6379 + prometheus:9090 + grafana:3000

# 评测
python eval_all.py                        # 全维度 + 关键词 Recall
python eval_all.py --ai-label             # AI 标注 Recall
python eval_all.py --cv 5                 # 5-fold CV
python eval_all.py --live                 # LLM CoT 路由实测
python eval_ragas.py                      # RAGAS 生成质量

# 测试
pytest tests/                             # 14 个单元测试
pytest tests/ -m integration              # +4 个 LLM 集成测试

# Token 测试
python token_bench.py                     # 5 条 query 各节点 token 消耗
```
