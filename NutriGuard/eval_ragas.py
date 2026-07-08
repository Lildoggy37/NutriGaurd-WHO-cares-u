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

FAITHFULNESS_PROMPT = """核查 AI 回答是否基于检索上下文。只输出一行 JSON，不得有任何其他文字、markdown、代码块：

{{"faithful_count": <有依据的声明数>, "total_count": <总声明数>}}

【检索上下文】
{context}

【AI 回答】
{answer}

判断标准：回答中的每条事实声明在上下文中能找到对应证据 → 计入 faithful_count"""


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
    prompt = FAITHFULNESS_PROMPT.format(
        context=context[:4000] if context else "",
        answer=answer[:1200] if answer else "",
    )
    for attempt in range(2):
        resp = await llm.ainvoke([SystemMessage(content=prompt)])
        raw = str(resp.content)
        m = re.search(r'\{[\s\S]*\}', raw)
        if m:
            try:
                data = json.loads(m.group(0))
                fc = data.get("faithful_count", 0)
                tc = data.get("total_count", 1)
                return {"faithful": fc, "total": tc,
                    "score": round(fc / max(tc, 1), 3), "attempts": attempt + 1}
            except json.JSONDecodeError:
                if attempt == 0:
                    prompt += "\n\n上次输出格式错误，请严格只输出一行 JSON。"  # retry hint
    return {"faithful": 0, "total": 0, "score": 0.0, "attempts": 2}


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
#  测试数据 (从 100 条 v2 数据集中分层抽样 50 条)
# ============================================================
def _load_ragas_queries(n=15):
    ds_path = os.path.join(BASE_DIR, "data", "eval_rag_v2.json")
    if not os.path.isfile(ds_path):
        return None
    with open(ds_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    import numpy as np
    np.random.seed(42)
    # 按类别分层抽样
    cats = {}
    for q in data["queries"]:
        cats.setdefault(q["cat"], []).append(q)
    selected = []
    per_cat = max(2, n // len(cats))
    for cat, qs in cats.items():
        chosen = np.random.choice(len(qs), min(per_cat, len(qs)), replace=False)
        for i in chosen:
            q = qs[i]
            selected.append({"q": q["q"], "gt_kw": q["keywords"]})
    np.random.shuffle(selected)
    return selected[:n]

SAMPLE_QUERIES = _load_ragas_queries(24)
if SAMPLE_QUERIES is None:
    SAMPLE_QUERIES = [{"q": "糖尿病人的饮食禁忌有哪些？", "gt_kw": ["糙米饭", "低GI", "碳水", "血糖"]}]


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
    from chunking import chunk_document as _cd2

    bge_path = os.path.join(BASE_DIR, "models", "bge-large-zh-v1.5")
    reranker_path = os.path.join(BASE_DIR, "models", "bge-reranker-v2-m3")
    corpus_path = os.path.join(BASE_DIR, "data", "mock_corpus.md")

    embedder = HuggingFaceEmbeddings(model_name=bge_path)
    sparse = FastEmbedSparse(model_name="Qdrant/bm25")
    reranker = CrossEncoder(reranker_path)

    from qdrant_client import QdrantClient
    disk_path = os.path.join(BASE_DIR, "data", "qdrant_storage")
    # 直接加载已有索引，避免重新嵌入 17,749 chunks（耗时 80 分钟）
    client = QdrantClient(path=disk_path)
    vs = QdrantVectorStore(
        client=client,
        collection_name="nutriguard_collection",
        embedding=embedder,
        sparse_embedding=sparse,
        retrieval_mode="hybrid",
    )
    retriever = vs.as_retriever(search_kwargs={"k": 20})
    print(f"  Qdrant 索引已加载 ({disk_path})")
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

    # Build neighbor map for chunk expansion (2b: prevent fragmentation)
    from chunking import chunk_document
    with open(os.path.join(BASE_DIR, "data", "mock_corpus.md"), "r", encoding="utf-8") as f:
        raw_corpus = f.read()
    ordered_docs = chunk_document(raw_corpus)  # ordered list for neighbor lookup

    def get_neighbor(doc, offset=1):
        """Get adjacent chunk by matching content prefix"""
        for i, d in enumerate(ordered_docs):
            if d.page_content[:50] == doc.page_content[:50]:
                ni = i + offset
                if 0 <= ni < len(ordered_docs):
                    return ordered_docs[ni].page_content
                return ""
        return ""

    # 2. Generate answers
    print(f"[2/3] 为 {len(SAMPLE_QUERIES)} 条查询生成回答...")
    SCORE_THRESHOLD = 0.2  # 2a: 降低阈值避免丢掉低分但相关的文档
    qa_pairs = []
    for item in SAMPLE_QUERIES:
        query = item["q"]
        rough = retriever.invoke(query)
        pairs = [[query, d.page_content] for d in rough]
        scores = reranker.predict(pairs)
        scored = sorted(zip(rough, scores), key=lambda x: x[1], reverse=True)

        # 2a: Score threshold filter
        filtered = [(d, s) for d, s in scored if s >= SCORE_THRESHOLD]
        if len(filtered) < 3:  # at least 3 docs
            filtered = scored[:5]
        else:
            filtered = filtered[:5]

        # 2a.5: LLM relevance filter — 快速判断每个 chunk 是否相关
        rel_filter_prompt = f"""用户问题：{query}\n\n判断以下 5 个文本片段是否与问题相关。每个片段输出 0(无关) 或 1(相关)，用逗号分隔。只输出五个数字，如：1,0,1,1,0"""
        chunk_snippets = "\n---\n".join([d.page_content[:200] for d, _ in scored[:5]])
        try:
            rel_resp = await llm.ainvoke([SystemMessage(content=rel_filter_prompt + "\n\n" + chunk_snippets)])
            rel_scores = [int(x.strip()) for x in str(rel_resp.content).split(",") if x.strip().isdigit()]
            if len(rel_scores) == 5:
                filtered_docs = [(d, s) for (d, s), r in zip(scored[:5], rel_scores) if r == 1]
                if filtered_docs:
                    scored = filtered_docs
                print(f"    LLM filter: {rel_scores} -> kept {len(scored)} docs")
        except Exception:
            pass

        # 2b: Neighbor expansion — 仅对高置信度文档展开邻居，避免噪音
        NEIGHBOR_MIN_SCORE = 0.5
        top_docs = []
        for doc, score in filtered[:5]:
            chunk_text = doc.page_content
            if score >= NEIGHBOR_MIN_SCORE:
                prev = get_neighbor(doc, -1)
                next_n = get_neighbor(doc, 1)
                if prev:
                    chunk_text = f"[上文] {prev[:200]}\n{chunk_text}"
                if next_n:
                    chunk_text = f"{chunk_text}\n[下文] {next_n[:200]}"
            top_docs.append(chunk_text)
        context = "\n\n---\n\n".join(top_docs)

        # Generate answer with stricter prompt
        gen_prompt = f"""基于以下参考资料回答用户问题。严格遵循：

1. 只使用参考资料中明确提到的信息
2. 如果资料不足以回答问题，直接说明"资料中未找到相关信息"，不要猜测
3. 回答控制在 200 字以内，直接回应用户问题，不要延伸

【参考资料】
{context[:4000]}

【用户问题】
{query}

请回答："""
        resp = await llm.ainvoke([SystemMessage(content=gen_prompt)])
        answer = str(resp.content)

        qa_pairs.append({"query": query, "answer": answer, "context": context,
            "top_docs": top_docs, "gt_kw": item["gt_kw"],
            "scores": [s for _, s in filtered]})
        avg_score = sum(s for _, s in filtered) / max(len(filtered), 1)
        print(f"  {query[:30]:30s} → {len(answer)} chars (avg score={avg_score:.2f})")

    # 3. Evaluate
    print(f"[3/3] 计算 RAGAS 指标...")
    results = []
    for i, qa in enumerate(qa_pairs):
        print(f"  [{i+1}/{len(qa_pairs)}] {qa['query'][:30]}...")
        faith = await eval_faithfulness(qa["answer"], qa["context"])
        relevancy = await eval_answer_relevancy(qa["answer"], qa["query"])
        precision = await eval_context_precision(qa["query"], qa["top_docs"])
        recall = eval_context_recall(qa["top_docs"], qa["gt_kw"])
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
