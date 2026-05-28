import sys
import time
import json
import asyncio
from contextlib import asynccontextmanager, AsyncExitStack
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import redis.asyncio as redis
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
redis_client = redis.Redis(host="localhost",port=6379,decode_responses=True)

@asynccontextmanager
async def lifespan(app:FastAPI):
    """
    FastAPI 生命周期管理器：在接收第一个 HTTP 请求前，把后台基建全部搭好。
    使用 AsyncExitStack 扁平化管理多个异步上下文，确保关机。
    """
    print("[生命周期] 正在启动API网关，底层微服务")
    redis_client = redis.Redis(host="localhost",port=6379,decode_responses=True)
    async with AsyncExitStack() as stack:
        try:
            
            # 1 链接MCP子进程
            server_params = StdioServerParameters(command=sys.executable,args=["mcp_server.py"])
            stdio_transport = await stack.enter_async_context(stdio_client(server_params))
            read, write = stdio_transport
            
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            print("[生命周期] MCP微服务已连接！")

            # 动态加载工具
            all_tools = await load_mcp_tools(session)

            # 路由tools给不同的agent
            rag_tool_names = {"search_diet_guidelines", "check_food_gi", "search_medical_taboos"}
            rag_tools = [t for t in all_tools if t.name in rag_tool_names]
            action_tool_names = {"log_user_meal","calculate_daily_calories","generate_shopping_list"}
            action_tools = [t for t in all_tools if t.name in action_tool_names]

            # 组装LangGraph,挂载到FastAPI的全局state
            app.state.graph = build_multi_agent_graph(rag_tools, action_tools)
            print("[生命周期] Multi-Agent 神经网络挂载完毕！服务随时准备接客。")
            
            # 交出控制权，让 FastAPI 开始处理 HTTP 请求
            yield

        except Exception as e:
            print(f"[生命周期错误] 启动失败：{e}")
            raise
        finally:
            print("[生命周期]收到关闭信号，清理连接池与子进程...")
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
    
    async def event_generator() -> AsyncGenerator[str, None]:
        initial_state = {
            "messages": [HumanMessage(content=request.query)],
            "user_profile": {"用户标识": request.session_id},
            "next_node": "" 
        }
        config = {"configurable": {"thread_id": request.session_id}}
        graph = app.state.graph # 从全局变量取出编译好的大脑

        try:
            # 使用 v2 版本的流式事件监听
            async for event in graph.astream_events(initial_state, config=config, version="v2"):
                kind = event["event"]
                
                # 抓取底层 LLM 生成的实时文字片段 (打字机效果)
                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if hasattr(chunk, "content") and chunk.content is not None:
                        # 遵循 Server-Sent Events (SSE) 规范
                        yield f"data: {json.dumps({'type': 'text', 'content': chunk.content}, ensure_ascii=False)}\n\n"
                        
                # 抓取图节点流转状态 (用于前端展示 Agent 思考路径/进度条)
                elif kind == "on_chain_start":
                    node_name = event.get("name", "")
                    if node_name in ["supervisor", "rag_expert", "action_expert", "slot_filler", "memory_compressor"]:
                        yield f"data: {json.dumps({'type': 'status', 'content': f'🔍 节点 [{node_name}] 开始思考...'}, ensure_ascii=False)}\n\n"
                        
        except Exception as e:
            print(f"🧨 [流式推送异常] {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': '系统内部发生错误，请稍后重试。'}, ensure_ascii=False)}\n\n"
        finally:
            # 标志流结束
            yield f"data: {json.dumps({'type': 'done', 'content': '[DONE]'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    # uvicorn 启动入口
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)