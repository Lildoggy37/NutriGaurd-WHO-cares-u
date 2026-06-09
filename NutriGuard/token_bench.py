"""
Token 消耗测试 — 运行几个 query 看各节点消耗

用法：python token_bench.py
"""
import os, sys, json, time, asyncio
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(BASE_DIR, "..", ".env"))

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

ROUTE_PROMPT = """你是医疗健康路由分发器。请输出 JSON 路由决策。
路由规则：问知识→rag_expert 记饮食/算热量→action_expert 信息不全→slot_filler 完成/闲聊→FINISH
输出：{"route": "rag_expert", "reason": "理由"}"""

PREPROCESS_PROMPT = """你是医疗健康查询预处理助手。
1.纠错（唐尿病→糖尿病）2.同义词展开（尿酸高→高尿酸血症）3.指代消解 4.意图澄清
噪声控制：不添加用户未提到的信息，不猜测健康状况，长度80字以内。
只输出改写后的问题文本。"""

REFLECTION_BASE = """审查以下 AI 营养学回答。判定 PASS/CORRECT/REJECT。
输出 JSON：{"verdict": "PASS", "reason": "...", "risk_items": ""}"""

TEST_QUERIES = [
    ("糖尿病饮食禁忌", "短知识查询"),
    ("帮我记录午餐：鸡胸肉200g,西兰花300g,糙米饭150g", "长操作指令"),
    ("我身高175体重80有糖尿病，帮我规划今天的晚餐", "混合意图"),
    ("你好", "简单问候"),
    ("血压有点高吃东西该注意啥这阵子老觉得头晕", "口语化长句"),
]

PRICE = {"input": 0.0008 / 1000, "output": 0.002 / 1000}  # qwen-plus per token


async def run_bench():
    llm = ChatOpenAI(model="qwen-plus", temperature=0.0,
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    print(f"{'Query':<30s} {'Node':<14s} {'In':>6s} {'Out':>6s} {'Total':>6s} {'Cost':>8s}")
    print("-" * 80)
    total_cost = 0

    for query, desc in TEST_QUERIES:
        results = []

        # Preprocess
        resp = await llm.ainvoke([
            SystemMessage(content=PREPROCESS_PROMPT),
            HumanMessage(content=f"用户原始输入：{query}"),
        ])
        u = resp.response_metadata.get("token_usage", {})
        results.append(("preprocess", u.get("prompt_tokens", 0), u.get("completion_tokens", 0)))

        # Supervisor
        resp = await llm.ainvoke([
            SystemMessage(content=ROUTE_PROMPT),
            HumanMessage(content=query),
        ])
        u = resp.response_metadata.get("token_usage", {})
        results.append(("supervisor", u.get("prompt_tokens", 0), u.get("completion_tokens", 0)))

        # Reflection (simulate for knowledge queries)
        if any(kw in query for kw in ["禁忌","注意","吃","规划","晚餐"]):
            dummy = "建议选择低GI食物，控制总热量摄入。以上仅供参考，请咨询医生。"
            resp = await llm.ainvoke([SystemMessage(content=REFLECTION_BASE + f"\nAI回答: {dummy}")])
            u = resp.response_metadata.get("token_usage", {})
            results.append(("reflection", u.get("prompt_tokens", 0), u.get("completion_tokens", 0)))

        for node, inp, out in results:
            cost = inp * PRICE["input"] + out * PRICE["output"]
            total_cost += cost
            print(f"{query[:30]:30s} {node:<14s} {inp:>5d} {out:>5d} {inp+out:>5d} ￥{cost:>6.5f}")
        print("-" * 80)

    print(f"\n总计: ￥{total_cost:.6f}")
    print(f"价格: qwen-plus 入 ￥0.0008/1K tokens, 出 ￥0.002/1K tokens")
    print(f"注: 不含 rag_expert/action_expert 内部的 create_agent 调用(~500-2000t 额外)")
    print(f"    完整 query 约 1.5-2x 上表消耗")


if __name__ == "__main__":
    asyncio.run(run_bench())
