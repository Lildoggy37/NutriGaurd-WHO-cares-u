import sys
import asyncio
import os

# Windows 控制台默认 GBK 编码，emoji 会导致 UnicodeEncodeError
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent

# .env 在项目根目录（NutriGuard 的上一级）
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MCP_SERVER_PATH = os.path.join(BASE_DIR, "mcp_server.py")

async def run_stub_test():
    print(" 正在启动 MCP 测试桩...")

    # 挂载我们刚写好的 MCP（使用绝对路径）
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[MCP_SERVER_PATH],
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            
            llm = ChatOpenAI(
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                model="qwen-plus"
            )
            
            # 创建最小化测试 Agent
            agent = create_react_agent(llm, tools=tools)
            
            print("\n" + "="*40)
            print("🧪 第一次提问 (预期：Cache 未命中，耗时 > 2000ms)")
            result1 = await agent.ainvoke({"messages": [("user", "帮我查一下糖尿病的饮食禁忌")]})
            print("Agent 回答:", result1["messages"][-1].content)
            
            print("\n" + "="*40)
            print("🧪 第二次提问 (语义变换，预期：Cache 命中，耗时 < 100ms)")
            # 注意：故意换了种问法，测试“语义缓存”而不是“字面缓存”
            result2 = await agent.ainvoke({"messages": [("user", "如果得了高血糖，吃东西要注意啥？帮我查下病理禁忌。")]})
            print("Agent 回答:", result2["messages"][-1].content)
            print("="*40)

if __name__ == "__main__":
    try:
        asyncio.run(run_stub_test())
    except Exception as e:
        # stdio 连接关闭时可能产生清理异常，忽略无实际影响的错误
        if "Connection closed" in str(e) or "unhandled errors" in str(e):
            print(f"\n[MCP 测试桩] 服务端已断开（正常结束）", file=sys.stderr)
        else:
            raise