"""
Prometheus 监控指标定义 — 四层可观测性。

层级：
  1. HTTP 层 — 请求量/延迟/状态码
  2. Node 层 — LangGraph 各节点耗时
  3. RAG 层  — 检索次数/缓存命中/延迟
  4. LLM 层  — LLM 调用次数/错误数
"""
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CollectorRegistry

registry = CollectorRegistry()

# ============================================================
#  1. HTTP 层
# ============================================================
http_requests_total = Counter(
    "http_requests_total",
    "HTTP 请求总数",
    ["method", "endpoint", "status_code"],
    registry=registry,
)

http_request_duration = Histogram(
    "http_request_duration_seconds",
    "HTTP 请求延迟 (秒)",
    ["method", "endpoint"],
    buckets=[0.05, 0.1, 0.5, 1, 2, 5, 10, 30, 60],
    registry=registry,
)

# ============================================================
#  2. Node 层 — LangGraph 节点耗时
# ============================================================
graph_node_duration = Histogram(
    "graph_node_duration_seconds",
    "图节点执行耗时 (秒)",
    ["node_name"],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60, 120],
    registry=registry,
)

graph_node_total = Counter(
    "graph_node_total",
    "图节点执行次数",
    ["node_name"],
    registry=registry,
)

graph_node_errors = Counter(
    "graph_node_errors_total",
    "图节点异常次数",
    ["node_name"],
    registry=registry,
)

# ============================================================
#  3. RAG 层 — 检索指标
# ============================================================
rag_search_total = Counter(
    "rag_search_total",
    "RAG 检索总次数",
    ["tool_name"],
    registry=registry,
)

rag_cache_hit_total = Counter(
    "rag_cache_hit_total",
    "语义缓存命中次数",
    ["tool_name"],
    registry=registry,
)

rag_cache_miss_total = Counter(
    "rag_cache_miss_total",
    "语义缓存未命中次数",
    ["tool_name"],
    registry=registry,
)

rag_search_duration = Histogram(
    "rag_search_duration_seconds",
    "RAG 检索耗时 (秒)",
    ["tool_name", "stage"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 30, 60],
    registry=registry,
)

# RAG 引擎就绪状态
rag_engine_ready = Gauge(
    "rag_engine_ready",
    "RAG 检索引擎是否就绪 (1=是, 0=否)",
    registry=registry,
)

# ============================================================
#  4. LLM 层
# ============================================================
llm_call_total = Counter(
    "llm_call_total",
    "LLM 调用总次数",
    ["model", "node"],
    registry=registry,
)

llm_call_errors = Counter(
    "llm_call_errors_total",
    "LLM 调用异常次数",
    ["model", "node"],
    registry=registry,
)

# ============================================================
#  导出
# ============================================================
def get_metrics():
    return generate_latest(registry)
