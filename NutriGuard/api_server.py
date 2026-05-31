import os
import sys
import time
import json
import asyncio
from contextlib import asynccontextmanager, AsyncExitStack

# 加载项目根目录的 .env 文件（DASHSCOPE_API_KEY 等）
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import redis
import redis.asyncio as aioredis
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

# 导入底层组件
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from graph_brain import build_multi_agent_graph

# ==========================================
#   1 lifespan
# ==========================================
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
redis_client = aioredis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)

@asynccontextmanager
async def lifespan(app:FastAPI):
    """
    FastAPI 生命周期管理器：在接收第一个 HTTP 请求前，把后台基建全部搭好。
    使用 AsyncExitStack 扁平化管理多个异步上下文，确保关机。
    """
    print("[生命周期] 正在启动API网关及底层微服务...")

    # 用绝对路径定位 mcp_server.py，避免工作目录问题
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    mcp_server_path = os.path.join(BASE_DIR, "mcp_server.py")

    if not os.path.isfile(mcp_server_path):
        raise FileNotFoundError(f"MCP 服务端脚本不存在: {mcp_server_path}")

    print(f"[生命周期] MCP 子进程入口: {mcp_server_path}")

    async with AsyncExitStack() as stack:
        try:
            # 1 启动 MCP 子进程（内部加载 BGE 模型约 30s）
            print("[生命周期] 正在拉起 MCP 子进程（等待模型加载，约 30s）...")
            server_params = StdioServerParameters(
                command=sys.executable,
                args=[mcp_server_path],
            )
            stdio_transport = await stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport

            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            print("[生命周期] MCP 微服务已连接！")

            # 2 动态加载工具
            all_tools = await load_mcp_tools(session)
            print(f"[生命周期] 已加载 {len(all_tools)} 个 MCP 工具")

            # 3 路由 tools 给不同的 agent
            rag_tool_names = {
                "search_diet_guidelines", "check_food_gi",
                "search_medical_taboos", "search_food",
            }
            rag_tools = [t for t in all_tools if t.name in rag_tool_names]
            action_tool_names = {
                "log_user_meal", "calculate_daily_calories",
                "generate_shopping_list", "update_health_profile",
            }
            action_tools = [t for t in all_tools if t.name in action_tool_names]

            print(f"[生命周期] RAG 工具: {[t.name for t in rag_tools]}")
            print(f"[生命周期] Action 工具: {[t.name for t in action_tools]}")

            # 4 组装 LangGraph
            app.state.graph = build_multi_agent_graph(rag_tools, action_tools)
            print("[生命周期] Multi-Agent 神经网络编译完成，服务就绪。")

            # 5 预热 RAG 引擎（触发懒加载，避免首次用户请求等待 30s）
            warmup_tool = next(
                (t for t in rag_tools if t.name == "search_diet_guidelines"), None
            )
            if warmup_tool:
                try:
                    print("[生命周期] 正在预热 RAG 检索引擎...")
                    await warmup_tool.ainvoke({"query": "预热"})
                    print("[生命周期] RAG 引擎预热完成")
                except Exception as e:
                    print(f"[生命周期] 预热异常（不影响启动）: {e}")

            yield

        except Exception as e:
            import traceback
            print(f"[生命周期] 启动失败: {e}")
            traceback.print_exc()
            raise
        finally:
            print("[生命周期] 收到关闭信号，清理连接池与子进程...")
            await redis_client.close()

app = FastAPI(
    title="NutriGuard-Copilot API", 
    description="高并发多智能体膳食管家网关",
    lifespan=lifespan
)

app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_credentials=True,allow_methods=["*"], allow_headers=["*"])

# ==========================================
#   2. Redis 滑动窗口，限流防爆
# ==========================================
async def sliding_window_rate_limiter(request:Request):
    client_ip = request.client.host
    key = f"rate_limit:ip:{client_ip}"
    limit = 60
    window = 60
    now = time.time()

    try:
        # 使用Pipeline保证原子性
        async with redis_client.pipeline(transaction=True) as pipe:
            pipe.zremrangebyscore(key,0,now - window)
            pipe.zcard(key)
            pipe.zadd(key,{str(now):now})
            pipe.expire(key,window)
            results = await pipe.execute()
        current_requests = results[1]

        if current_requests > limit:
            print(f" [风控触发] IP: {client_ip} 并发过高 ({current_requests}/{limit})，已熔断！")
            raise HTTPException(status_code=429, detail="API 调用频率超限，请一分钟后再试。")

    except redis.exceptions.ConnectionError:
        print("⚠️ [警告] Redis 离线，滑动窗口限流已降级跳过。")
        pass  # 降级，不阻塞业务

# ==========================================
#   3. 健康检查探针
# ==========================================  
@app.get("/health", tags=["Monitor"])
async def health_check():
    """
    专为 K8s 和 Docker Compose 设计的健康探测接口。
    不仅检查 Web 服务存活，还会下钻检查 Redis 和核心图引擎状态。
    """
    health_status = {
        "status": "up",
        "timestamp": time.time(),
        "components": {
            "web_framework": "fastapi",
            "redis_cache": "unknown",
            "brain_engine": "ready" if hasattr(app.state, "graph") else "initializing"
        }
    }
    
    # 深度探测 Redis 连通性
    try:
        # 设置极短的超时时间，防止健康检查本身把服务拖死
        async with asyncio.timeout(1.0):
            await redis_client.ping()
            health_status["components"]["redis_cache"] = "connected"
    except (redis.exceptions.ConnectionError, asyncio.TimeoutError):
        health_status["components"]["redis_cache"] = "disconnected"
        # \ 架构师决策：因为我们前面写了优雅降级（Redis 挂了也能聊天），
        # 所以我们把整体状态标为 degraded（降级），而不是抛出 503 把整个 Pod 杀掉。
        health_status["status"] = "degraded"
        
    # 如果核心图引擎都没加载出来（比如 lifespan 卡住了），那就是致命错误
    if health_status["components"]["brain_engine"] == "initializing":
        health_status["status"] = "down"
        raise HTTPException(
            status_code=503, 
            detail=health_status
        )
        
    return health_status


# ==========================================
#   4. 流式对话接口SSE
# ==========================================     
class ChatRequest(BaseModel):
    session_id: str = "default_user_001"
    query: str

@app.post("/api/chat/stream", dependencies=[Depends(sliding_window_rate_limiter)])
async def chat_stream_endpoint(request: ChatRequest):
    # 预检：图引擎是否就绪
    if not hasattr(app.state, "graph") or app.state.graph is None:
        raise HTTPException(
            status_code=503,
            detail="服务正在初始化中，请稍后重试。若持续出现此错误，请检查 API Key 配置。",
        )

    async def event_generator() -> AsyncGenerator[str, None]:
        initial_state = {
            "messages": [HumanMessage(content=request.query)],
            "user_profile": {"用户标识": request.session_id},
            "next_node": "",
        }
        config = {"configurable": {"thread_id": request.session_id}}
        graph = app.state.graph

        try:
            async for event in graph.astream_events(initial_state, config=config, version="v2"):
                kind = event["event"]

                # LLM 打字机效果 — 黑名单模式：默认放行，只排除 supervisor/reflection
                # create_agent 内部子节点（"agent"/"tools"等）均会正常流式输出
                if kind == "on_chat_model_stream":
                    node_name = event.get("metadata", {}).get("langgraph_node", "")
                    HIDDEN_NODES = {"supervisor", "rag_reflection", "memory_compressor"}
                    if node_name not in HIDDEN_NODES:
                        chunk = event["data"]["chunk"]
                        if hasattr(chunk, "content") and chunk.content is not None:
                            yield f"data: {json.dumps({'type': 'text', 'content': chunk.content}, ensure_ascii=False)}\n\n"

                # Agent 节点状态
                elif kind == "on_chain_start":
                    node_name = event.get("name", "")
                    STATUS_NODES = {
                        "supervisor", "rag_expert", "rag_reflection",
                        "action_expert", "slot_filler", "memory_compressor",
                    }
                    if node_name in STATUS_NODES:
                        yield f"data: {json.dumps({'type': 'status', 'content': f'🔍 节点 [{node_name}] 开始思考...'}, ensure_ascii=False)}\n\n"

                elif kind == "on_chain_end":
                    pass  # slot_filler 已改为 LLM 生成，由 on_chat_model_stream 处理

        except Exception as e:
            import traceback
            print(f"[流式推送异常] {e}")
            traceback.print_exc()
            detail = str(e)[:200]
            yield f"data: {json.dumps({'type': 'error', 'content': f'系统内部错误: {detail}'}, ensure_ascii=False)}\n\n"
        finally:
            yield f"data: {json.dumps({'type': 'done', 'content': '[DONE]'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    # uvicorn 启动入口
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)