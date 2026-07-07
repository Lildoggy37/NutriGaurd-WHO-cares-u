"""
性能基准报告 — 20 条不同类型 query 的全链路耗时+token 消耗

用法：python token_bench.py
输出：控制台表格 + token_bench_report.json
"""
import os, sys, json, time, asyncio
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(BASE_DIR, "..", ".env"))
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# ============================================================
# 20 条测试 query, 按类型分类
# ============================================================
BENCH_QUERIES = [
    # 简短问候 (2)
    ("你好", "greeting"),
    ("谢谢", "greeting"),
    # 短知识查询 (5)
    ("糖尿病饮食禁忌", "knowledge_short"),
    ("痛风不能吃什么", "knowledge_short"),
    ("鸡胸肉蛋白质含量", "knowledge_short"),
    ("糙米饭GI值", "knowledge_short"),
    ("高血压怎么吃", "knowledge_short"),
    # 长知识查询 (4)
    ("孕妇查出高血糖，吃东西要注意啥？", "knowledge_long"),
    ("太胖了想减肥，每天应该吃多少热量合适？", "knowledge_long"),
    ("老年人肌肉流失怎么通过饮食改善？", "knowledge_long"),
    ("尿酸高了吃什么不好？", "knowledge_long"),
    # 操作指令 (5)
    ("帮我记录早餐：包子2个,豆浆1杯", "action"),
    ("算一下今天的热量", "action"),
    ("帮我更新一下体重80kg", "action"),
    ("生成低GI购物清单", "action"),
    ("帮我记录午餐：鸡胸肉200g,西兰花300g,糙米饭150g", "action"),
    # 混合意图 (4)
    ("我身高175体重80有糖尿病，帮我规划今天的晚餐", "mixed"),
    ("记录午餐后帮我查一下痛风饮食禁忌", "mixed"),
    ("帮我记了早餐，现在算一下热量还差多少", "mixed"),
    ("帮我更新体重，然后生成一周的低GI食谱购物清单", "mixed"),
]

ROUTE_PROMPT = "分析意图输出JSON: {\"route\":\"rag_expert|action_expert|slot_filler|FINISH\",\"reason\":\"...\"}\n规则:问知识→rag 记饮食/算热量/更新→action 信息不全→slot 完成/闲聊→FINISH"

PRICE = {"input": 0.0008 / 1000, "output": 0.002 / 1000}  # qwen-plus per token
MODEL = "qwen-plus"


# ============================================================
#  Benchmark
# ============================================================
async def run_bench():
    llm = ChatOpenAI(model=MODEL, temperature=0.0,
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    results = []
    type_stats = {}
    total_cost = 0
    total_time = 0

    print(f"Performance Benchmark — {len(BENCH_QUERIES)} queries × {MODEL}")
    print(f"{'Query':<35s} {'Type':<14s} {'Node':<12s} {'Time':>6s} {'In':>6s} {'Out':>6s} {'Total':>6s} {'Cost':>8s}")
    print("=" * 100)

    for query, qtype in BENCH_QUERIES:
        query_start = time.time()
        nodes = []

        # Preprocess
        t0 = time.time()
        resp = await llm.ainvoke([
            SystemMessage(content="纠错+同义词展开+指代消解。噪声控制:不添加信息,不猜测。只输出改写文本。"),
            HumanMessage(content=f"用户输入：{query}"),
        ])
        u = resp.response_metadata.get("token_usage", {})
        inp, out = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
        cost = inp * PRICE["input"] + out * PRICE["output"]
        nodes.append({"node": "preprocess", "time": round(time.time()-t0, 2), "in": inp, "out": out, "cost": cost})

        # Supervisor
        t0 = time.time()
        resp = await llm.ainvoke([
            SystemMessage(content=ROUTE_PROMPT),
            HumanMessage(content=query),
        ])
        u = resp.response_metadata.get("token_usage", {})
        inp, out = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
        cost = inp * PRICE["input"] + out * PRICE["output"]
        nodes.append({"node": "supervisor", "time": round(time.time()-t0, 2), "in": inp, "out": out, "cost": cost})

        elapsed_total = time.time() - query_start
        total_cost += sum(n["cost"] for n in nodes)

        type_stats.setdefault(qtype, {"count":0, "time":0, "tokens":0, "cost":0})
        type_stats[qtype]["count"] += 1
        type_stats[qtype]["time"] += elapsed_total
        type_stats[qtype]["tokens"] += sum(n["in"] + n["out"] for n in nodes)
        type_stats[qtype]["cost"] += sum(n["cost"] for n in nodes)

        results.append({"query": query, "type": qtype, "nodes": nodes, "total_s": round(elapsed_total, 2)})

        for n in nodes:
            print(f"{query[:35]:<35s} {qtype:<14s} {n['node']:<12s} {n['time']:>5.1f}s {n['in']:>5d} {n['out']:>5d} {n['in']+n['out']:>5d} ￥{n['cost']:>6.4f}")
        print("-" * 100)

    # Summary
    print(f"\n{'='*60}")
    print("PER-QUERY TYPE AVERAGES")
    print(f"{'='*60}")
    print(f"{'Type':<16s} {'Count':>5s} {'Avg Time':>9s} {'Avg Tokens':>10s} {'Avg Cost':>9s}")
    for t, stats in sorted(type_stats.items()):
        c = stats["count"]
        print(f"{t:<16s} {c:>5d} {stats['time']/c:>8.1f}s {stats['tokens']/c:>9.0f}t ￥{stats['cost']/c:>7.5f}")

    print(f"\nTotal: {len(BENCH_QUERIES)} queries, ￥{total_cost:.5f}, {sum(s['time'] for s in type_stats.values()):.1f}s")
    print(f"Note: Only routing layer (preprocess+supervisor). Full pipeline (RAG+Agent) ~2-3x tokens.")

    # Save
    report = {
        "model": MODEL, "queries": len(BENCH_QUERIES),
        "type_summary": {t: {"count": s["count"], "avg_time_s": round(s["time"]/s["count"], 1),
                             "avg_tokens": round(s["tokens"]/s["count"]), "avg_cost": round(s["cost"]/s["count"], 5)}
                         for t, s in type_stats.items()},
        "details": results,
    }
    with open(os.path.join(BASE_DIR, "token_bench_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nReport: token_bench_report.json")


if __name__ == "__main__":
    asyncio.run(run_bench())
