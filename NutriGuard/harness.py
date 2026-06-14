"""
统一执行框架 — 节点 Harness + 工具 Harness + LLM 并发控制。

NodeHarness:     重试 / 超时 / 兜底 / 指标 / 日志
ToolHarness:     超时 / 熔断 / 降级链
LLMRateLimiter:  LLM API QPS 保护, 背压排队


用法：
  from harness import node_harness, tool_harness

  @node_harness(name="rag_expert", retries=1, timeout=60, fallback=FINISH)
  async def rag_expert_node(state): ...

  @tool_harness(name="search_diet_guidelines", timeout=30)
  async def search_diet_guidelines(query: str) -> str: ...
"""
import asyncio
import functools
import time
import sys
from typing import Any, Awaitable, Callable


# ============================================================
#  NodeHarness — 图节点执行框架
# ============================================================

class NodeConfig:
    __slots__ = ("name", "retries", "timeout_seconds", "fallback")
    def __init__(self, name="", retries=1, timeout_seconds=60, fallback=None):
        self.name = name
        self.retries = retries
        self.timeout_seconds = timeout_seconds
        self.fallback = fallback


def node_harness(name="", retries=1, timeout_seconds=60, fallback=None):
    """
    图节点装饰器。
    - retries: 异常后重试次数（不含首次，共 1+retries 次）
    - timeout_seconds: 单次执行超时（秒）
    - fallback: 最终失败时的返回值，支持 callable(state) → dict
    """
    cfg = NodeConfig(name, retries, timeout_seconds, fallback)

    def decorator(
        fn: Callable[[Any], Awaitable[dict]],
    ) -> Callable[[Any], Awaitable[dict]]:
        @functools.wraps(fn)
        async def wrapper(state):
            attempts = cfg.retries + 1
            last_error = None
            t0 = time.time()

            for attempt in range(attempts):
                try:
                    result = await asyncio.wait_for(
                        fn(state),
                        timeout=cfg.timeout_seconds,
                    )
                    elapsed = time.time() - t0

                    # 记录成功
                    try:
                        from monitoring import graph_node_duration, graph_node_total
                        graph_node_total.labels(node_name=cfg.name).inc()
                        graph_node_duration.labels(node_name=cfg.name).observe(elapsed)
                    except Exception:
                        pass

                    print(
                        f"[Harness] {cfg.name} OK ({elapsed:.1f}s"
                        + (f", attempt {attempt+1}/{attempts})" if attempt > 0 else ")"),
                        file=sys.stderr,
                    )
                    return result

                except asyncio.TimeoutError:
                    last_error = TimeoutError(f"{cfg.name} 超时 ({cfg.timeout_seconds}s)")
                    print(
                        f"[Harness] {cfg.name} 超时 attempt {attempt+1}/{attempts}",
                        file=sys.stderr,
                    )
                except Exception as e:
                    last_error = e
                    print(
                        f"[Harness] {cfg.name} 异常 attempt {attempt+1}/{attempts}: {e}",
                        file=sys.stderr,
                    )

                if attempt < attempts - 1:
                    backoff = 1.5 ** attempt
                    await asyncio.sleep(backoff)

            # 全部重试失败 → 兜底
            try:
                from monitoring import graph_node_errors
                graph_node_errors.labels(node_name=cfg.name).inc()
            except Exception:
                pass

            print(
                f"[Harness] {cfg.name} 全部 {attempts} 次失败: {last_error}，触发兜底",
                file=sys.stderr,
            )

            if callable(cfg.fallback):
                return cfg.fallback(state)
            if cfg.fallback is not None:
                return cfg.fallback
            return {"next_node": "FINISH"}

        return wrapper
    return decorator


# ============================================================
#  ToolHarness — MCP 工具调用框架
# ============================================================

class ToolConfig:
    __slots__ = ("name", "timeout_seconds", "max_failures", "cooldown_seconds",
                 "fallback_msg", "_failures", "_disabled_until")
    def __init__(self, name="", timeout_seconds=30, max_failures=5,
                 cooldown_seconds=60, fallback_msg="工具暂时不可用，请稍后重试"):
        self.name = name
        self.timeout_seconds = timeout_seconds
        self.max_failures = max_failures
        self.cooldown_seconds = cooldown_seconds
        self.fallback_msg = fallback_msg
        self._failures = 0
        self._disabled_until = 0.0


# 全局工具熔断状态
_tool_states: dict[str, ToolConfig] = {}


def _get_tool_state(name, timeout_seconds, max_failures, cooldown_seconds, fallback_msg):
    if name not in _tool_states:
        _tool_states[name] = ToolConfig(name, timeout_seconds, max_failures,
                                         cooldown_seconds, fallback_msg)
    return _tool_states[name]


def tool_harness(name="", timeout_seconds=30, max_failures=5,
                 cooldown_seconds=60, fallback_msg="工具暂时不可用，请稍后重试"):
    """
    MCP 工具装饰器。
    - timeout_seconds: 单次执行超时
    - max_failures: 连续失败多少次后熔断
    - cooldown_seconds: 熔断冷却时间
    - fallback_msg: 熔断/超时时返回的降级文本
    """
    def decorator(fn: Callable[..., Awaitable[str]]) -> Callable[..., Awaitable[str]]:
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            cfg = _get_tool_state(name, timeout_seconds, max_failures,
                                  cooldown_seconds, fallback_msg)

            # 熔断检查
            if cfg._disabled_until > time.time():
                print(
                    f"[ToolHarness] {name} 已熔断 (冷却至 {cfg._disabled_until - time.time():.0f}s 后)",
                    file=sys.stderr,
                )
                return cfg.fallback_msg

            try:
                result = await asyncio.wait_for(
                    fn(*args, **kwargs),
                    timeout=cfg.timeout_seconds,
                )
                cfg._failures = 0  # 成功 → 重置计数器
                return result

            except asyncio.TimeoutError:
                cfg._failures += 1
                print(
                    f"[ToolHarness] {name} 超时 ({cfg.timeout_seconds}s) "
                    f"({cfg._failures}/{cfg.max_failures})",
                    file=sys.stderr,
                )
            except Exception as e:
                cfg._failures += 1
                print(
                    f"[ToolHarness] {name} 异常 ({cfg._failures}/{cfg.max_failures}): {e}",
                    file=sys.stderr,
                )

            # 熔断触发
            if cfg._failures >= cfg.max_failures:
                cfg._disabled_until = time.time() + cfg.cooldown_seconds
                print(
                    f"[ToolHarness] {name} 触发熔断！冷却 {cfg.cooldown_seconds}s",
                    file=sys.stderr,
                )

            return cfg.fallback_msg

        return wrapper
    return decorator


# ============================================================
#  LLMRateLimiter — LLM API QPS 保护 + 背压排队
# ============================================================

class LLMRateLimiter:
    """
    异步信号量 + 令牌桶，保护 LLM API 不被突发流量打爆。

    用法:
        limiter = LLMRateLimiter(max_concurrent=5, max_per_second=10)

        async with limiter:
            response = await llm.ainvoke(...)

    原理:
      - 令牌桶控制每秒调用数 (QPS)
      - Semaphore 控制同时进行的调用数
      - 超过限制时请求自动排队 (背压)，不丢请求
    """

    def __init__(self, max_concurrent: int = 5, max_per_second: int = 10):
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._max_per_second = max_per_second
        self._tokens = max_per_second
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self._max_per_second,
                               self._tokens + int(elapsed * self._max_per_second))
            self._last_refill = now
            if self._tokens > 0:
                self._tokens -= 1
            else:
                wait_time = 1.0 / self._max_per_second
                await asyncio.sleep(wait_time)
                self._tokens = self._max_per_second - 1
                self._last_refill = time.monotonic()
        await self._semaphore.acquire()

    def release(self):
        self._semaphore.release()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, *args):
        self.release()


# 全局实例
llm_rate_limiter = LLMRateLimiter(max_concurrent=5, max_per_second=10)
