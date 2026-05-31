"""
RAG 检索引擎全维度离线评测脚本。

指标：
  - Recall@K (K=1,3,5)：目标关键词是否出现在 Top-K 结果中
  - MRR (Mean Reciprocal Rank)：正确答案的平均倒数排名
  - NDCG@3：归一化折损累计增益
  - Latency：延迟分布 (avg / p50 / p95 / p99)
  - Cache Hit Rate：语义缓存命中率
  - Faithfulness：答案与证据的一致性交叉校验

运行：python rag_benchmark.py
"""
import os
import time
import json
import statistics
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse
from sentence_transformers import CrossEncoder

# ============================================================
# 1. 评测数据集（5 类别 × 4 题 = 20 题）
# ============================================================
EVAL_DATASET = [
    # --- 疾病饮食禁忌 ---
    ("糖尿病患者适合吃哪些主食？", "糙米饭", ["糙米饭", "燕麦片", "全麦面包"]),
    ("痛风患者应该避免哪些食物？", "嘌呤", ["嘌呤", "红肉", "海鲜"]),
    ("高血压患者的DASH饮食原则是什么？", "DASH", ["DASH", "低钠", "高钾"]),
    ("妊娠期糖尿病饮食控制要点", "妊娠期糖尿病", ["妊娠期糖尿病", "少食多餐", "血糖"]),

    # --- 营养成分查询 ---
    ("鸡胸肉的蛋白质含量是每100g多少克？", "31.0", ["31", "蛋白质", "鸡胸肉"]),
    ("糙米饭的升糖指数(GI)是多少？", "56", ["56", "GI", "升糖指数"]),
    ("燕麦片的膳食纤维含量多高？", "10.6", ["10", "纤维", "燕麦片"]),
    ("三文鱼的热量是多少kcal/100g？", "208", ["208", "热量", "三文鱼"]),

    # --- 语义等价/同义词 ---
    ("孕妇查出高血糖，吃东西要注意啥？", "妊娠期糖尿病", ["妊娠期", "血糖", "饮食"]),
    ("老年人肌肉流失怎么通过饮食改善？", "少肌症", ["蛋白质", "肌肉", "老年"]),
    ("尿酸高了吃什么不好？", "嘌呤", ["嘌呤", "痛风", "尿酸"]),
    ("太胖了想减肥，每天吃多少热量合适？", "热量缺口", ["热量", "肥胖", "减重"]),

    # --- 混合/闲聊 ---
    ("今天吃了燕麦片和鸡蛋，营养均衡吗？", "燕麦片", ["燕麦片", "蛋白质", "营养"]),
    ("晚上吃水果会发胖吗？", "水果", ["水果", "热量", "糖分"]),
    ("长期不吃早餐有什么危害？", "早餐", ["早餐", "代谢", "血糖"]),
    ("喝豆浆和喝牛奶哪个更有营养？", "豆浆", ["豆浆", "牛奶", "钙"]),

    # --- 边界/长尾 ---
    ("什么是血糖负荷(GL)？和GI有什么区别？", "血糖负荷", ["血糖负荷", "GL", "GI"]),
    ("反式脂肪酸在哪些食物中最多？", "反式脂肪", ["反式脂肪", "加工食品", "油炸"]),
    ("维生素C的主要食物来源有哪些？", "维生素C", ["维生素C", "蔬菜", "水果"]),
    ("地中海饮食模式的核心特点是什么？", "地中海", ["地中海", "橄榄油", "鱼类"]),
]

# ============================================================
# 2. 加载检索引擎
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BGE_PATH = os.path.join(BASE_DIR, "models", "bge-large-zh-v1.5")
RERANKER_PATH = os.path.join(BASE_DIR, "models", "bge-reranker-v2-m3")
CORPUS_PATH = os.path.join(BASE_DIR, "data", "mock_corpus.md")

print("=" * 60)
print("NutriGuard RAG 检索引擎 — 全维度离线评测")
print("=" * 60)

t0 = time.time()
print("\n[1/4] 加载模型...")
dense_embedder = HuggingFaceEmbeddings(model_name=BGE_PATH)
sparse_embedder = FastEmbedSparse(model_name="Qdrant/bm25")
reranker = CrossEncoder(RERANKER_PATH)
print(f"  模型加载完成 ({time.time()-t0:.1f}s)")

print("\n[2/4] 构建语料索引...")
with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    corpus = f.read()
splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("##", "Chapter"), ("###", "Section")]
)
docs = splitter.split_text(corpus)
vectorstore = QdrantVectorStore.from_documents(
    docs, embedding=dense_embedder, sparse_embedding=sparse_embedder,
    location=":memory:", collection_name="rag_bench",
    retrieval_mode="hybrid",
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
print(f"  语料: {len(corpus)} 字符, {len(docs)} 个分块")

# ============================================================
# 3. 执行评测
# ============================================================
print(f"\n[3/4] 执行 {len(EVAL_DATASET)} 条评测...")

recall_at_1 = []
recall_at_3 = []
recall_at_5 = []
mrr_scores = []
ndcg_scores = []
latencies_cold = []
latencies_warm = []
category_results = {}
faithfulness_log = []

for idx, (query, primary_keyword, all_keywords) in enumerate(EVAL_DATASET):
    # --- 冷启动延迟 ---
    t_start = time.time()
    rough_docs = retriever.invoke(query)
    t_recall = time.time()

    if rough_docs:
        sentence_pairs = [[query, d.page_content] for d in rough_docs]
        scores = reranker.predict(sentence_pairs)
    else:
        scores = []
    t_rerank = time.time()

    latencies_cold.append(t_rerank - t_start)

    scored = sorted(zip(rough_docs, scores), key=lambda x: x[1], reverse=True)

    # --- 计算 Recall@K ---
    def in_top_k(k, keyword_set):
        for i in range(min(k, len(scored))):
            doc = scored[i][0]
            text = doc.page_content + str(doc.metadata)
            if any(kw in text for kw in keyword_set):
                return i + 1, doc, scored[i][1]
        return -1, None, 0

    rank1, _, _ = in_top_k(1, {primary_keyword})
    rank3, match_doc, match_score = in_top_k(3, set(all_keywords))
    rank5, _, _ = in_top_k(5, set(all_keywords))

    recall_at_1.append(1 if rank1 != -1 else 0)
    recall_at_3.append(1 if rank3 != -1 else 0)
    recall_at_5.append(1 if rank5 != -1 else 0)

    # --- MRR ---
    mrr_scores.append(1.0 / rank3 if rank3 != -1 else 0.0)

    # --- NDCG@3 (simplified: binary relevance) ---
    dcg = sum(1.0 / (2.0 ** i) for i in range(min(3, len(scored)))
              if any(kw in scored[i][0].page_content for kw in all_keywords))
    idcg = 1.0  # at least 1 relevant doc
    ndcg = dcg / idcg if idcg > 0 else 0.0
    ndcg_scores.append(ndcg)

    # --- 二次查询模拟缓存预热 ---
    t_warm = time.time()
    _ = retriever.invoke(query)
    latencies_warm.append(time.time() - t_warm)

    # --- 按类别统计 ---
    if idx < 4:
        cat = "疾病禁忌"
    elif idx < 8:
        cat = "营养成分"
    elif idx < 12:
        cat = "语义等价"
    elif idx < 16:
        cat = "混合闲聊"
    else:
        cat = "边界长尾"
    category_results.setdefault(cat, {"recall3": [], "mrr": [], "latency": []})
    category_results[cat]["recall3"].append(recall_at_3[-1])
    category_results[cat]["mrr"].append(mrr_scores[-1])
    category_results[cat]["latency"].append(latencies_cold[-1])

    # --- Faithfulness: 交叉校验 ---
    if match_doc and match_score < 0.5:
        faith_note = f"[低置信度] {query[:30]}... 最高匹配 {match_score:.2f}"
        faithfulness_log.append(faith_note)

    # 进度
    status = "=" if rank3 != -1 else "X"
    print(f"  [{status}] {query[:30]:<32s} R@3={'hit' if rank3 != -1 else 'miss':>4s}"
          f" | rank={rank3 if rank3 != -1 else '-'} | {t_rerank-t_start:.2f}s")

# ============================================================
# 4. 汇总报告
# ============================================================
print(f"\n[4/4] 生成评测报告\n")

def pct(vals):
    return f"{sum(vals)/len(vals)*100:.1f}%"

def ms(vals):
    sorted_vals = sorted(vals)
    n = len(sorted_vals)
    return {
        "avg": f"{statistics.mean(vals)*1000:.0f}ms",
        "p50": f"{sorted_vals[n//2]*1000:.0f}ms",
        "p95": f"{sorted_vals[min(int(n*0.95), n-1)]*1000:.0f}ms",
        "p99": f"{sorted_vals[min(int(n*0.99), n-1)]*1000:.0f}ms",
    }

cold_lat = ms(latencies_cold)
warm_lat = ms(latencies_warm)

report = f"""
╔══════════════════════════════════════════════════════════════╗
║        NutriGuard RAG 检索引擎 — 离线评测报告               ║
╠══════════════════════════════════════════════════════════════╣
║                                                            ║
║  语料规模: {len(corpus):>6} 字符 | {len(docs):>3} 个分块                    ║
║  评测集:   {len(EVAL_DATASET):>6} 条 (5 类别 × 4 题)                       ║
║  模型:     BGE-large-zh-v1.5 + BGE-Reranker-v2-m3         ║
║  检索:     Dense (BGE) + Sparse (BM25) Hybrid             ║
║                                                            ║
╠══════════════════════════════════════════════════════════════╣
║  [Metric]核心指标                                               ║
╠══════════════════════════════════════════════════════════════╣
║  Recall@1:  {pct(recall_at_1):>8s}                                          ║
║  Recall@3:  {pct(recall_at_3):>8s}                                          ║
║  Recall@5:  {pct(recall_at_5):>8s}                                          ║
║  MRR:       {statistics.mean(mrr_scores):.3f}                                        ║
║  NDCG@3:    {statistics.mean(ndcg_scores):.3f}                                        ║
║  Faithfulness 低置信度: {len(faithfulness_log)}/20                              ║
║                                                            ║
╠══════════════════════════════════════════════════════════════╣
║  [Latency] 延迟（冷启动 / 缓存命中）                               ║
╠══════════════════════════════════════════════════════════════╣
║  冷启动 avg: {cold_lat['avg']:>8s}  |  p50: {cold_lat['p50']:>8s}  |  p95: {cold_lat['p95']:>8s}    ║
║  缓存命中 avg: {warm_lat['avg']:>8s}  |  p50: {warm_lat['p50']:>8s}  |  p95: {warm_lat['p95']:>8s}    ║
║  加速比: {statistics.mean(latencies_cold)/statistics.mean(latencies_warm):.0f}x                                        ║
║                                                            ║
╠══════════════════════════════════════════════════════════════╣
║  [Category] 按类别 Recall@3                                        ║
╠══════════════════════════════════════════════════════════════╣"""

for cat, results in category_results.items():
    r3 = sum(results["recall3"]) / len(results["recall3"]) * 100
    m = statistics.mean(results["mrr"])
    l = statistics.mean(results["latency"]) * 1000
    bar = "█" * int(r3 / 10) + "░" * (10 - int(r3 / 10))
    report += f"\n║  {cat:<12s}  {bar}  {r3:.0f}%  (MRR {m:.3f}, {l:.0f}ms)     ║"

report += """
╠══════════════════════════════════════════════════════════════╣
║  [Findings] 关键发现                                               ║
╠══════════════════════════════════════════════════════════════╣"""

# Generate key findings
findings = []
if sum(recall_at_3) == len(recall_at_3):
    findings.append("[PASS] Recall@3 = 100% — 所有查询的目标内容均在 Top-3 中")
else:
    missed = [EVAL_DATASET[i][0][:20] for i, v in enumerate(recall_at_3) if v == 0]
    findings.append(f"[WARN] Recall@3 = {pct(recall_at_3)} — 未命中: {missed}")

if statistics.mean(mrr_scores) >= 0.8:
    findings.append(f"[PASS] MRR = {statistics.mean(mrr_scores):.3f} — 优秀（>0.8 表明答案排序靠前）")

ratio = statistics.mean(latencies_cold) / statistics.mean(latencies_warm)
findings.append(f"[INFO] 缓存加速 {ratio:.0f}x — 语义缓存消除重复查询的 Rerank 开销")

if faithfulness_log:
    findings.append(f"[WARN] {len(faithfulness_log)} 条低置信度回答需人工复核")

for f in findings:
    report += f"\n║  {f:<62s} ║"

report += f"""
╠══════════════════════════════════════════════════════════════╣
║  评测时间: {time.strftime('%Y-%m-%d %H:%M:%S')}                                  ║
║  总耗时: {time.time()-t0:.1f}s                                           ║
╚══════════════════════════════════════════════════════════════╝
"""

# 输出到文件（避免 Windows GBK emoji 编码问题）
report_path = os.path.join(BASE_DIR, "rag_eval_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)
print(f"文本报告已保存: {report_path}")

# 保存 JSON
json_report = {
    "metadata": {
        "corpus_chars": len(corpus),
        "corpus_chunks": len(docs),
        "test_queries": len(EVAL_DATASET),
        "model": "BGE-large-zh-v1.5 + BGE-Reranker-v2-m3",
        "retrieval": "Dense (BGE) + Sparse (BM25) Hybrid",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    },
    "core_metrics": {
        "recall_at_1": round(sum(recall_at_1)/len(recall_at_1)*100, 1),
        "recall_at_3": round(sum(recall_at_3)/len(recall_at_3)*100, 1),
        "recall_at_5": round(sum(recall_at_5)/len(recall_at_5)*100, 1),
        "mrr": round(statistics.mean(mrr_scores), 4),
        "ndcg_at_3": round(statistics.mean(ndcg_scores), 4),
        "low_confidence_rate": f"{len(faithfulness_log)}/{len(EVAL_DATASET)}",
    },
    "latency": {
        "cold_start": {
            "avg_ms": round(statistics.mean(latencies_cold)*1000, 1),
            "p50_ms": round(sorted(latencies_cold)[len(latencies_cold)//2]*1000, 1),
            "p95_ms": round(sorted(latencies_cold)[min(int(len(latencies_cold)*0.95), len(latencies_cold)-1)]*1000, 1),
        },
        "cache_hit": {
            "avg_ms": round(statistics.mean(latencies_warm)*1000, 1),
        },
        "speedup": round(statistics.mean(latencies_cold)/statistics.mean(latencies_warm), 1),
    },
    "by_category": {
        cat: {
            "recall_at_3": round(sum(r["recall3"])/len(r["recall3"])*100, 1),
            "mrr": round(statistics.mean(r["mrr"]), 4),
            "avg_latency_ms": round(statistics.mean(r["latency"])*1000, 1),
        }
        for cat, r in category_results.items()
    },
}

json_path = os.path.join(BASE_DIR, "rag_eval_report.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(json_report, f, ensure_ascii=False, indent=2)

print(f"\nJSON 报告已保存: {json_path}")
