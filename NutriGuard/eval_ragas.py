"""
RAGAS 风格评估：Faithfulness / AnswerRelevancy / ContextPrecision / ContextRecall.

不依赖 ragas 库 — 纯 LLM-as-judge 实现。

用法：python eval_ragas.py
"""
import os, sys, json, time, statistics, re, asyncio

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from dotenv import load_dotenv
load_dotenv(os.path.join(BASE_DIR, "..", ".env"))

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# ============================================================
#  LLM
# ============================================================
llm = ChatOpenAI(model="qwen-plus", temperature=0.0,
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

# ============================================================
#  RAGAS 四指标
# ============================================================

FAITHFULNESS_PROMPT = """你是一个事实核查员。判断以下 AI 回答中的每条声明是否能从【检索上下文】中找到依据。

【检索上下文】
{context}

【AI 回答】
{answer}

请逐一列出 AI 回答中的事实声明，判断每条是否在上下文中有支撑。最后输出 JSON：
{{"claims": [{{"claim": "声明文本", "supported": true/false, "evidence": "证据"}}], "faithful_count": N, "total_count": M}}"""


ANSWER_RELEVANCY_PROMPT = """根据以下 AI 回答，生成 3 个用户可能提出的反向问题（这些问题应该能被这个回答所覆盖）。

【AI 回答】
{answer}

只输出 3 个问题，每行一个，不要编号。"""


CONTEXT_PRECISION_PROMPT = """判断以下检索片段是否与用户问题相关。输出 JSON 数组，每个元素是 true/false。

【用户问题】
{question}

【检索片段】
{contexts}

输出格式：{{"relevant": [true, false, true, ...]}}"""


async def eval_faithfulness(answer: str, context: str) -> dict:
    prompt = FAITHFULNESS_PROMPT.format(context=context[:3000], answer=answer[:1500])
    resp = await llm.ainvoke([SystemMessage(content=prompt)])
    raw = str(resp.content)
    m = re.search(r'\{[\s\S]*\}', raw)
    if m:
        try:
            data = json.loads(m.group(0))
            return {
                "faithful": data.get("faithful_count", 0),
                "total": data.get("total_count", 1),
                "score": round(data.get("faithful_count", 0) / max(data.get("total_count", 1), 1), 3),
            }
        except json.JSONDecodeError:
            pass
    return {"faithful": 0, "total": 0, "score": 0.0}


async def eval_answer_relevancy(answer: str, original_query: str) -> float:
    prompt = ANSWER_RELEVANCY_PROMPT.format(answer=answer[:1500])
    resp = await llm.ainvoke([SystemMessage(content=prompt)])
    gen_questions = [q.strip().lstrip("0123456789.-) ") for q in str(resp.content).split("\n") if q.strip()]

    if not gen_questions:
        return 0.0

    # Embedding similarity: use the same LLM to score relevance
    scores = []
    for gq in gen_questions[:3]:
        score_prompt = f"用 0-10 打分：以下两个问题在语义上有多相似？\n问题1：{original_query}\n问题2：{gq}\n只输出数字。"
        resp2 = await llm.ainvoke([SystemMessage(content=score_prompt)])
        try:
            s = float(re.search(r'\d+', str(resp2.content)).group(0)) if re.search(r'\d+', str(resp2.content)) else 5
            scores.append(min(s / 10, 1.0))
        except Exception:
            scores.append(0.5)

    return round(sum(scores) / len(scores), 3)


async def eval_context_precision(question: str, contexts: list[str]) -> float:
    ctx_text = "\n---\n".join([f"[{i+1}] {c[:300]}" for i, c in enumerate(contexts[:5])])
    prompt = CONTEXT_PRECISION_PROMPT.format(question=question, contexts=ctx_text)
    resp = await llm.ainvoke([SystemMessage(content=prompt)])
    raw = str(resp.content)
    m = re.search(r'\{[\s\S]*\}', raw)
    if m:
        try:
            data = json.loads(m.group(0))
            rel = data.get("relevant", [])
            return round(sum(1 for r in rel if r) / max(len(rel), 1), 3) if rel else 0.5
        except json.JSONDecodeError:
            pass
    return 0.5


def eval_context_recall(retrieved_docs: list[str], ground_truth_keywords: list[str]) -> float:
    """基于关键词覆盖的简化 Context Recall"""
    all_text = " ".join(retrieved_docs)
    found = sum(1 for kw in ground_truth_keywords if kw in all_text)
    return round(found / max(len(ground_truth_keywords), 1), 3)


# ============================================================
#  测试数据
# ============================================================
SAMPLE_QUERIES = [
    {"q": "糖尿病人的饮食禁忌有哪些？", "gt_kw": ["糙米饭", "低GI", "碳水", "血糖"]},
    {"q": "痛风患者应该避免什么食物？", "gt_kw": ["嘌呤", "海鲜", "内脏", "红肉"]},
    {"q": "孕妇查出高血糖，吃东西要注意啥？", "gt_kw": ["妊娠期", "血糖", "少食多餐", "监测"]},
    {"q": "鸡胸肉每100g的蛋白质含量是多少？", "gt_kw": ["31", "蛋白质", "鸡胸肉"]},
    {"q": "高血压DASH饮食的核心原则是什么？", "gt_kw": ["DASH", "低钠", "高钾", "蔬果"]},
    {"q": "尿酸高了吃什么不好？", "gt_kw": ["嘌呤", "痛风", "尿酸"]},
    {"q": "帮我记录午餐：糙米饭150g,鸡胸肉200g", "gt_kw": ["糙米饭", "鸡胸肉", "记录"]},
    {"q": "太胖了想减肥，每天应该吃多少热量？", "gt_kw": ["热量", "肥胖", "减重", "缺口"]},
    {"q": "老年人肌肉流失怎么通过饮食改善？", "gt_kw": ["蛋白质", "肌肉", "老年"]},
    {"q": "今天吃了燕麦片和鸡蛋，营养均衡吗？", "gt_kw": ["燕麦", "蛋白质", "均衡"]},
]


# ============================================================
#  RAG 模拟
# ============================================================
def load_rag():
    # BM25 patch
    snap_dir = os.path.join(os.path.expanduser("~"), ".cache/huggingface/hub/models--Qdrant--bm25/snapshots")
    if os.path.isdir(snap_dir):
        snaps = sorted(os.listdir(snap_dir), reverse=True)
        if snaps:
            from pathlib import Path
            from fastembed.common.model_management import ModelManagement as _MM
            bm25_path = os.path.join(snap_dir, snaps[0])
            _orig = _MM.download_model
            @classmethod
            def _fix(cls, md, cache_dir=None, local_files_only=False, **kw):
                if "bm25" in str(getattr(md, "model", md)).lower(): return Path(bm25_path)
                return _orig.__func__(cls, md, cache_dir=cache_dir, local_files_only=local_files_only, **kw)
            _MM.download_model = _fix

    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_qdrant import QdrantVectorStore, FastEmbedSparse
    from sentence_transformers import CrossEncoder
    from langchain_text_splitters import MarkdownHeaderTextSplitter

    bge_path = os.path.join(BASE_DIR, "models", "bge-large-zh-v1.5")
    reranker_path = os.path.join(BASE_DIR, "models", "bge-reranker-v2-m3")
    corpus_path = os.path.join(BASE_DIR, "data", "mock_corpus.md")

    embedder = HuggingFaceEmbeddings(model_name=bge_path)
    sparse = FastEmbedSparse(model_name="Qdrant/bm25")
    reranker = CrossEncoder(reranker_path)

    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = f.read()
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("##","Chapter"),("###","Section")])
    docs = splitter.split_text(corpus)
    vs = QdrantVectorStore.from_documents(docs, embedding=embedder, sparse_embedding=sparse,
        location=":memory:", collection_name="eval_ragas", retrieval_mode="hybrid")
    retriever = vs.as_retriever(search_kwargs={"k": 10})
    return retriever, reranker, llm


# ============================================================
#  Main
# ============================================================
async def main():
    print("=" * 60)
    print("RAGAS 风格评估 (LLM-as-judge)")
    print("=" * 60)

    # 1. Load RAG
    print("\n[1/3] 加载 RAG 引擎...")
    retriever, reranker, eval_llm = load_rag()
    print("  OK")

    # 2. Generate answers
    print(f"[2/3] 为 {len(SAMPLE_QUERIES)} 条查询生成回答...")
    qa_pairs = []
    for item in SAMPLE_QUERIES:
        query = item["q"]
        rough = retriever.invoke(query)
        pairs = [[query, d.page_content] for d in rough]
        scores = reranker.predict(pairs)
        scored = sorted(zip(rough, scores), key=lambda x: x[1], reverse=True)
        top3 = [d.page_content for d, _ in scored[:3]]
        context = "\n\n".join(top3)

        # Generate answer
        gen_prompt = f"""你是营养学专家。基于以下检索资料回答用户问题。
检索资料：\n{context[:2000]}\n\n用户问题：{query}
要求：基于资料回答，不编造。不确定的说明不确定。最后加上"以上仅供参考"。"""
        resp = await llm.ainvoke([SystemMessage(content=gen_prompt)])
        answer = str(resp.content)

        qa_pairs.append({"query": query, "answer": answer, "context": context,
            "top3_docs": top3, "gt_kw": item["gt_kw"]})
        print(f"  {query[:30]:30s} → {len(answer)} chars")

    # 3. Evaluate
    print(f"[3/3] 计算 RAGAS 指标...")
    results = []
    for i, qa in enumerate(qa_pairs):
        print(f"  [{i+1}/{len(qa_pairs)}] {qa['query'][:30]}...")
        faith = await eval_faithfulness(qa["answer"], qa["context"])
        relevancy = await eval_answer_relevancy(qa["answer"], qa["query"])
        precision = await eval_context_precision(qa["query"], qa["top3_docs"])
        recall = eval_context_recall(qa["top3_docs"], qa["gt_kw"])
        results.append({
            "query": qa["query"][:40],
            "faithfulness": faith["score"],
            "answer_relevancy": relevancy,
            "context_precision": precision,
            "context_recall": recall,
        })

    # 4. Summary
    print("\n" + "=" * 60)
    print("RAGAS 评估结果")
    print("=" * 60)
    print(f"{'Query':<35s} {'Faith':>6s} {'Rel':>6s} {'CtxP':>6s} {'CtxR':>6s}")
    print("-" * 65)
    for r in results:
        print(f"{r['query']:<35s} {r['faithfulness']:>6.3f} {r['answer_relevancy']:>6.3f} {r['context_precision']:>6.3f} {r['context_recall']:>6.3f}")
    print("-" * 65)
    avg_f = round(sum(r["faithfulness"] for r in results) / len(results), 3)
    avg_r = round(sum(r["answer_relevancy"] for r in results) / len(results), 3)
    avg_p = round(sum(r["context_precision"] for r in results) / len(results), 3)
    avg_cr = round(sum(r["context_recall"] for r in results) / len(results), 3)
    print(f"{'AVERAGE':<35s} {avg_f:>6.3f} {avg_r:>6.3f} {avg_p:>6.3f} {avg_cr:>6.3f}")

    # Save
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "queries": len(SAMPLE_QUERIES),
        "metrics": {
            "avg_faithfulness": avg_f,
            "avg_answer_relevancy": avg_r,
            "avg_context_precision": avg_p,
            "avg_context_recall": avg_cr,
        },
        "details": results,
    }
    with open(os.path.join(BASE_DIR, "ragas_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nReport: ragas_report.json")

    # Resume card
    print("\n=== Resume Numbers ===")
    print(f"Faithfulness:       {avg_f:.3f}")
    print(f"Answer Relevancy:   {avg_r:.3f}")
    print(f"Context Precision:  {avg_p:.3f}")
    print(f"Context Recall:     {avg_cr:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
