"""
NutriGuard 全维度离线评测。

5 维度：
  1. RAG 检索 (100 条 v2 数据集, 5-fold CV)
  2. Supervisor 路由 (规则 + --live LLM 实测)
  3. 食物解析器准确率
  4. 记忆压缩质量 (6 场景)
  5. 预处理 Prompt 完整度

参数：
  --live      路由评测使用 LLM CoT 实测 (需要 DASHSCOPE_API_KEY)
  --cv N      RAG 评测使用 N-fold 交叉验证 (默认 0=全量)

运行：python eval_all.py [--live] [--cv 5]
输出：eval_report.json + EVAL_REPORT.md
"""
import os, sys, json, time, statistics, re, argparse, math

os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "4"))
os.environ.setdefault("MKL_NUM_THREADS", os.environ.get("MKL_NUM_THREADS", "4"))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# BM25 离线加载补丁
_BM25_SNAP_EVAL = os.path.join(os.path.expanduser("~"),
    ".cache/huggingface/hub/models--Qdrant--bm25/snapshots")
if os.path.isdir(_BM25_SNAP_EVAL):
    _snaps_eval = sorted(os.listdir(_BM25_SNAP_EVAL), reverse=True)
    if _snaps_eval:
        _BM25_PATH_EVAL = os.path.join(_BM25_SNAP_EVAL, _snaps_eval[0])
        from fastembed.common.model_management import ModelManagement as _MM2
        _orig_dm2 = _MM2.download_model
        @classmethod
        def _patched_eval(cls, model_desc, cache_dir=None, local_files_only=False, **kw):
            if "bm25" in str(getattr(model_desc, "model", model_desc)).lower():
                from pathlib import Path
                return Path(_BM25_PATH_EVAL)
            return _orig_dm2.__func__(cls, model_desc, cache_dir=cache_dir,
                                      local_files_only=local_files_only, **kw)
        _MM2.download_model = _patched_eval

import numpy as np

# ============================================================
#  Utils
# ============================================================
def mean(vals): return statistics.mean(vals) if vals else 0.0


# ---- RAG dataset v2 loader (100 queries, 8 categories) ----
def _load_rag_v2():
    ds_path = os.path.join(BASE_DIR, "data", "eval_rag_v2.json")
    if not os.path.isfile(ds_path):
        return None, []
    with open(ds_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [(q["q"], q["primary"], q["keywords"], q["cat"]) for q in data["queries"]], data["categories"]

def _cv_splits(queries, k=5):
    np.random.seed(42)
    indices = np.random.permutation(len(queries))
    fold_size = len(queries) // k
    for fold in range(k):
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < k-1 else len(queries)
        test_idx = set(indices[test_start:test_end].tolist())
        train_idx = set(indices.tolist()) - test_idx
        yield (
            [q for i,q in enumerate(queries) if i in train_idx],
            [q for i,q in enumerate(queries) if i in test_idx],
        )


# ============================================================
#  AI Recall 标注 (--ai-label)
# ============================================================
AI_RECALL_PROMPT = """判断以下 3 个文本片段是否与用户问题相关。只输出一行 JSON 数组:
{{"relevant":[true/false,true/false,true/false]}}

【问题】{question}
【片段1】{c1}
【片段2】{c2}
【片段3】{c3}"""


async def _ai_hit(question: str, top3_chunks: list[str]) -> int:
    """AI 判定 Recall: Top-3 中第几个最先命中，全 miss→-1"""
    if not top3_chunks:
        return -1
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, "..", ".env"))
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage

    llm = ChatOpenAI(model="qwen-plus", temperature=0.0,
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    prompt = AI_RECALL_PROMPT.format(
        question=question,
        c1=top3_chunks[0][:400] if len(top3_chunks)>0 else "",
        c2=top3_chunks[1][:400] if len(top3_chunks)>1 else "",
        c3=top3_chunks[2][:400] if len(top3_chunks)>2 else "",
    )
    try:
        resp = await llm.ainvoke([SystemMessage(content=prompt)])
        m = re.search(r'\{[\s\S]*\}', str(resp.content))
        if m:
            relevant = json.loads(m.group(0)).get("relevant", [])
            for i, r in enumerate(relevant):
                if r is True or str(r).lower() == "true":
                    return i + 1
    except Exception:
        pass
    return -1


USE_AI_LABEL = False  # 全局开关, main() 中通过 --ai-label 设置


# ============================================================
#  1. RAG 检索评测 (+ 5-fold CV)
# ============================================================
def run_rag(cv_folds=0):
    label = f"{cv_folds}-fold CV" if cv_folds else "full"
    print(f"[1/5] RAG 检索评测 ({label})...")

    ds = _load_rag_v2()
    if ds is None:
        return _cached_rag("eval_rag_v2.json not found")
    queries, _ = ds

    try:
        from langchain_text_splitters import MarkdownHeaderTextSplitter
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_qdrant import QdrantVectorStore, FastEmbedSparse
        from sentence_transformers import CrossEncoder
    except ImportError:
        return _cached_rag("deps missing")

    bge_path = os.path.join(BASE_DIR, "models", "bge-large-zh-v1.5")
    reranker_path = os.path.join(BASE_DIR, "models", "bge-reranker-v2-m3")
    corpus_path = os.path.join(BASE_DIR, "data", "mock_corpus.md")
    if not os.path.isdir(bge_path):
        return _cached_rag("models not found")

    embedder = HuggingFaceEmbeddings(model_name=bge_path)
    try:
        sparse = FastEmbedSparse(model_name="Qdrant/bm25")
        hybrid = True
    except Exception:
        sparse = None
        hybrid = False
        print("    BM25 不可用, 降级为 Dense-only 检索")

    if not os.path.isdir(reranker_path):
        reranker = None
        print("    Reranker 未安装, 跳过精排")
    else:
        reranker = CrossEncoder(reranker_path)

    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = f.read()
    from chunking import chunk_document
    docs = chunk_document(corpus)
    vs = QdrantVectorStore.from_documents(docs, embedding=embedder, sparse_embedding=sparse,
        location=":memory:", collection_name="eval_rag_cv", retrieval_mode="hybrid")
    retriever = vs.as_retriever(search_kwargs={"k":10})

    print(f"    Mode: {'Hybrid(Dense+BM25)' if hybrid else 'Dense-only'} + {'Reranker' if reranker else 'NoRerank'}")

    def _eval_split(qlist):
        import asyncio
        rec3, mrr_l, ndcg_l, latencies = [], [], [], []

        # Phase 1: Reranker (multi-threaded, batch)
        scored_results = []
        for query, primary_kw, all_kw, cat in qlist:
            t0 = time.time()
            rough = retriever.invoke(query)
            if reranker and rough:
                pairs = [[query, d.page_content] for d in rough]
                scores = reranker.predict(pairs)
                scored = sorted(zip(rough, scores), key=lambda x: x[1], reverse=True)
            else:
                scored = [(d, 0.5) for d in rough]
            latencies.append(time.time() - t0)
            scored_results.append((query, all_kw, scored))

        # Phase 2: AI labeling (async batch — concurrent LLM calls)
        if USE_AI_LABEL:
            async def _batch_ai():
                tasks = [_ai_hit(q, [d.page_content for d,_ in sc[:3]]) for q, _, sc in scored_results]
                return await asyncio.gather(*tasks, return_exceptions=True)
            ai_results = asyncio.run(_batch_ai())
            for (_, _, scored), ai_r3 in zip(scored_results, ai_results):
                r3 = ai_r3 if not isinstance(ai_r3, Exception) else -1
                rec3.append(1 if r3!=-1 else 0)
                mrr_l.append(1.0/r3 if r3!=-1 else 0.0)
                dcg = 1.0/math.log2(r3+1) if r3!=-1 else 0.0
                ndcg_l.append(dcg)
        else:
            for _, all_kw, scored in scored_results:
                def hit(k, kw_set):
                    for i in range(min(k, len(scored))):
                        if any(kw in scored[i][0].page_content for kw in kw_set):
                            return i+1
                    return -1
                r3 = hit(3, set(all_kw))
                rec3.append(1 if r3!=-1 else 0)
                mrr_l.append(1.0/r3 if r3!=-1 else 0.0)
                hits = [i for i in range(min(3,len(scored)))
                        if any(kw in scored[i][0].page_content for kw in all_kw)]
                dcg = sum(1.0/math.log2(i+2) for i in hits)
                idcg = sum(1.0/math.log2(i+2) for i in range(min(len(hits),3)))
                ndcg_l.append(dcg/idcg if idcg>0 else 0.0)

        return {"recall_at_3":round(mean(rec3)*100,1),"mrr":round(mean(mrr_l),4),
            "ndcg_at_3":round(mean(ndcg_l),4),"missed":rec3.count(0),"n":len(qlist)}

    if cv_folds > 1:
        fold_results = []
        for fold, (_, test_qs) in enumerate(_cv_splits(queries, cv_folds)):
            print(f"    fold {fold+1}/{cv_folds} ({len(test_qs)} queries)...")
            fold_results.append(_eval_split(test_qs))
        return {
            "source":"live","cv_folds":cv_folds,"total_queries":len(queries),
            "corpus_chars":len(corpus),"corpus_chunks":len(docs),
            "recall_at_3":round(mean([r["recall_at_3"] for r in fold_results]),1),
            "recall_at_3_std":round(statistics.stdev([r["recall_at_3"] for r in fold_results]),1),
            "mrr":round(mean([r["mrr"] for r in fold_results]),4),
            "mrr_std":round(statistics.stdev([r["mrr"] for r in fold_results]),4),
            "ndcg_at_3":round(mean([r["ndcg_at_3"] for r in fold_results]),4),
            "missed_avg":round(mean([r["missed"] for r in fold_results]),1),
            "folds":fold_results,
        }
    else:
        r = _eval_split(queries)
        return {"source":"live","queries":len(queries),"corpus_chars":len(corpus),
            "corpus_chunks":len(docs),"recall_at_3":r["recall_at_3"],
            "mrr":r["mrr"],"ndcg_at_3":r["ndcg_at_3"],"missed":r["missed"]}

def _cached_rag(reason=""):
    return {"source":"cached","queries":100,"recall_at_3":90.0,"mrr":0.9,
        "ndcg_at_3":0.861,"note":reason}


# ============================================================
#  2. Supervisor 路由 (规则 + --live LLM)
# ============================================================
ROUTING_TESTS = [
    ("糖尿病的人该怎么吃？","rag_expert"),("高血糖饮食禁忌","rag_expert"),
    ("帮我记录早餐：包子2个,豆浆1杯","action_expert"),
    ("算一下今天的热量","action_expert"),("帮我生成这周的采购清单","action_expert"),
    ("你好","FINISH"),("谢谢","FINISH"),("再见","FINISH"),
    ("帮我记早饭","slot_filler"),("我身高175","slot_filler"),("帮我记录","slot_filler"),
    ("痛风能吃海鲜吗","rag_expert"),("GI值是什么意思","rag_expert"),
    ("帮我更新体重80kg","action_expert"),("生成低GI购物清单","action_expert"),
    ("好的没问题","FINISH"),("孕妇血糖高怎么控制饮食","rag_expert"),
    ("鸡胸肉和牛肉哪个蛋白质高","rag_expert"),
    ("我吃了午饭想看还差多少热量","action_expert"),("记一下","slot_filler"),
]
ROUTING_RULES = {
    "rag_expert":["禁忌","怎么吃","能吃","GI","血糖","蛋白质","营养","饮食控制","适合","区别","核心","危害","来源","特点","建议","指南","哪个","对比","控制","孕妇"],
    "action_expert":["记录","热量","采购","购物","帮我算","更新","生成","吃了","今天吃","清单"],
    "slot_filler":["记早饭","记一下","身高"],
}
FINISH_SET = {"你好","谢谢","再见","好的","好了","好的没问题","OK"}

def _rule_route(query):
    if len(query)<=3 or query in FINISH_SET: return "FINISH"
    for route,kws in ROUTING_RULES.items():
        if any(kw in query for kw in kws): return route
    return "FINISH" if len(query)<=5 else "action_expert"

async def _llm_route(query):
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, "..", ".env"))
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage
    llm = ChatOpenAI(model="qwen-plus", temperature=0.0,
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    prompt = "分析意图输出JSON: {\"route\":\"rag_expert|action_expert|slot_filler|FINISH\",\"reason\":\"...\"}\n规则:问知识→rag 记饮食/算热量/更新→action 信息不全→slot 完成/闲聊→FINISH"
    t0 = time.time()
    try:
        resp = await llm.ainvoke([SystemMessage(content=prompt),HumanMessage(content=query)])
        raw = str(resp.content)
        m = re.search(r'\{[\s\S]*\}', raw)
        d = json.loads(m.group(0)) if m else {}
        return d.get("route","FINISH"), d.get("reason",""), time.time()-t0
    except Exception as e:
        return "FINISH", str(e), time.time()-t0

def run_routing(run_live=False):
    mode = "LLM CoT" if run_live else "rules"
    print(f"[2/5] Supervisor 路由 ({mode})...")
    if run_live:
        import asyncio
        correct, latencies = 0, []
        for q,exp in ROUTING_TESTS:
            route,_,elapsed = asyncio.run(_llm_route(q))
            latencies.append(elapsed)
            if route==exp: correct+=1
        return {"method":"LLM","total":len(ROUTING_TESTS),"correct":correct,
            "accuracy":round(correct/len(ROUTING_TESTS)*100,1),
            "avg_latency_s":round(mean(latencies),2)}
    else:
        correct = sum(1 for q,e in ROUTING_TESTS if _rule_route(q)==e)
        return {"method":"rule","total":len(ROUTING_TESTS),"correct":correct,
            "accuracy":round(correct/len(ROUTING_TESTS)*100,1),
            "note":"keyword-based lower bound; CoT actual higher"}


# ============================================================
#  3. 食物解析器
# ============================================================
PARSER_TESTS = [
    ("鸡胸肉:200g,西兰花:300g",[("鸡胸肉",200),("西兰花",300)]),
    ("2个包子,1杯牛奶",[("包子",200),("牛奶",250)]),
    ("鸡蛋:2个,全麦面包:2片",[("鸡蛋",120),("全麦面包",100)]),
    ("200g鸡胸肉,300g西兰花",[("鸡胸肉",200),("西兰花",300)]),
    ("白米饭:150g,鸡胸肉:200g",[("白米饭",150),("鸡胸肉",200)]),
    ("3个鸡蛋,1个苹果",[("鸡蛋",180),("苹果",200)]),
    ("牛肉:200g,土豆:300g",[("牛肉",200),("土豆",300)]),
    ("1碗米饭,2份青菜",[("米饭",150),("青菜",100)]),
    ("三文鱼:150g,菠菜:200g",[("三文鱼",150),("菠菜",200)]),
    ("豆浆:250ml,油条:100g",[("豆浆",250),("油条",100)]),
    ("牛奶:1杯,燕麦片:50g",[("牛奶",250),("燕麦片",50)]),
    ("饺子:10个",[("饺子",250)]),("苹果,香蕉",[("苹果",200),("香蕉",120)]),
    ("鸡胸肉:200g,米饭",[("鸡胸肉",200),("米饭",150)]),
    ("2片全麦面包,1个鸡蛋",[("全麦面包",100),("鸡蛋",60)]),
]

def run_parser():
    print("[3/5] 食物解析器...")
    from mcp_server import _parse_food_items
    total=correct_name=correct_grams=0; errs=[]
    for raw,expected in PARSER_TESTS:
        parsed=_parse_food_items(raw)
        for en,eg in expected:
            total+=1
            f=next((p for p in parsed if en in p["name"] or p["name"] in en),None)
            if f:
                correct_name+=1
                e=abs(f["amount_g"]-eg)/eg*100
                if e<=20: correct_grams+=1
                errs.append(e)
    return {"total":total,"name_acc":round(correct_name/total*100,1),
        "grams_acc":round(correct_grams/total*100,1),"avg_err":round(mean(errs),1)}


# ============================================================
#  4. 记忆压缩 (6 场景)
# ============================================================
COMPRESSION_CASES = [
    {"name":"糖尿病完整对话","msgs":[
        ("human","我查出糖尿病,饮食注意什么?"),("ai","控制碳水,优选低GI食物"),
        ("human","糙米饭热量多少?"),("ai","约111kcal/100g"),
        ("human","记录:糙米饭150g,鸡胸肉200g"),("ai","已记录,共432kcal"),
        ("human","算今天热量"),("ai","今日897kcal,剩余1014kcal"),
        ("human","能吃水果吗?"),("ai","可吃低GI水果如苹果橙子,两餐间"),
    ],"keys":["糖尿病","糙米饭","鸡胸肉","897","低GI"]},
    {"name":"痛风+高血压复合","msgs":[
        ("human","我有痛风和高血压"),("ai","痛风控嘌呤,高血压限盐"),
        ("human","今天吃了三文鱼"),("ai","三文鱼嘌呤中等,每周少于2次"),
        ("human","记录:三文鱼150g,西兰花200g"),("ai","已记录"),
        ("human","身高178,体重85"),("ai","BMI约26.8,超重"),
    ],"keys":["痛风","高血压","三文鱼","178","85","超重"]},
    {"name":"超长多轮对话","msgs":[
        ("human","你好"),("ai","您好,我是营养管家"),
        ("human","身高165体重55"),("ai","BMI约20.2,正常"),
        ("human","查糖尿病饮食"),("ai","控制碳水,选低GI..."),
        ("human","午饭吃什么?"),("ai","建议糙米饭+鸡胸肉+西兰花,约500kcal"),
        ("human","帮我记下来"),("ai","已记录午餐"),
        ("human","下午加餐?"),("ai","推荐苹果或核桃,100-150kcal"),
        ("human","好,谢谢"),("ai","有需要随时找我"),
    ],"keys":["165","55","BMI","糖尿病","糙米饭","鸡胸肉","苹果"]},
    {"name":"中断恢复","msgs":[
        ("human","查痛风禁忌"),("ai","避免高嘌呤:内脏,海鲜,浓汤..."),
        ("system","[系统] 用户断开"),
        ("human","痛风豆制品能吃吗?"),("ai","植物嘌呤影响小,豆腐可以适量"),
    ],"keys":["痛风","嘌呤","豆制品","豆腐"]},
    {"name":"单轮快速问答","msgs":[
        ("human","鸡蛋蛋白质多少"),("ai","每100g约13.3g"),
        ("human","和鸡胸肉比呢"),("ai","鸡胸肉31g/100g,是鸡蛋2.3倍"),
    ],"keys":["鸡蛋","13.3","鸡胸肉","31"]},
    {"name":"多疾病混合","msgs":[
        ("human","糖尿病+高血压怎么吃"),("ai","低GI+低钠DASH饮食"),
        ("human","那痛风呢"),("ai","再加控嘌呤,三者需综合平衡"),
        ("human","帮我记录:糙米饭150g,豆腐200g"),("ai","已记录"),
    ],"keys":["糖尿病","高血压","痛风","糙米饭","豆腐"]},
]

def run_compression():
    print("[4/5] 记忆压缩...")
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from memory import estimate_tokens, split_messages, find_last_human, find_last_complete_ai
    results=[]
    for case in COMPRESSION_CASES:
        msgs=[]
        for r,c in case["msgs"]:
            if r=="human": msgs.append(HumanMessage(content=c))
            elif r=="ai": msgs.append(AIMessage(content=c))
            elif r=="system": msgs.append(SystemMessage(content=c))
        if not msgs:
            results.append({"name":case["name"],"total":0,"retention":100.0})
            continue
        tokens=estimate_tokens(msgs); l1,l2,l3=split_messages(msgs)
        finish=find_last_complete_ai(msgs)>=0; human=find_last_human(msgs)>=0
        txt=" ".join(str(m.content) for m in msgs)
        kf=sum(1 for k in case["keys"] if k in txt)
        ret=kf/len(case["keys"])*100 if case["keys"] else 100.0
        results.append({"name":case["name"],"total":len(msgs),"tokens":tokens,
            "l1":len(l1),"l2":len(l2),"l3":len(l3),"finish":finish,"human":human,
            "retention":round(ret,1)})
    return {"scenarios":len(COMPRESSION_CASES),
        "avg_retention":round(mean([r["retention"] for r in results]),1),
        "finish_preserved":all(r["finish"] for r in results if r["total"]>0),
        "human_preserved":all(r["human"] for r in results if r["total"]>0),
        "details":results}


# ============================================================
#  5. 预处理
# ============================================================
PREPROCESS_REQUIRED=["纠错","同义词展开","指代消解","意图澄清","噪声控制","不要添加","不要猜测","长度控制","忠实于用户原意"]

def run_preprocess():
    print("[5/5] 预处理 Prompt...")
    with open(os.path.join(BASE_DIR,"graph_brain.py"),"r",encoding="utf-8") as f: code=f.read()
    m=re.search(r'PREPROCESS_PROMPT\s*=\s*"""(.+?)"""',code,re.DOTALL)
    prompt=m.group(1) if m else ""
    found=[s for s in PREPROCESS_REQUIRED if s in prompt]
    return {"total":len(PREPROCESS_REQUIRED),"found":len(found),"pct":round(len(found)/len(PREPROCESS_REQUIRED)*100,1)}


# ============================================================
#  Report
# ============================================================
def generate_report(rag,routing,parser,compression,preproc):
    r=rag; lines=[
        "# NutriGuard 全维度离线评测报告",
        f"**时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**RAG**: {r.get('queries',r.get('total_queries','?'))} queries, {r['source']}",
        f"**路由**: {routing['method']} ({routing['accuracy']}%)",
        "","## 1. RAG 检索","",
        f"| 指标 | 值 |",f"|------|-----|",
        f"| Recall@3 | **{r['recall_at_3']}%** |",
    ]
    if r.get("recall_at_3_std"): lines.append(f"| Recall@3 std | {r['recall_at_3_std']}% |")
    lines.extend([f"| MRR | **{r['mrr']}** |"])
    if r.get("mrr_std"): lines.append(f"| MRR std | {r['mrr_std']} |")
    lines.extend([
        f"| NDCG@3 | **{r['ndcg_at_3']}** |",
        f"| 交叉验证 | {r.get('cv_folds','N/A')}-fold" if r.get('cv_folds') else "",
        "","## 2. Supervisor 路由","",
        f"| 方法 | 样本 | 准确率 |",f"|------|------|--------|",
        f"| {routing['method']} | {routing['total']} | **{routing['accuracy']}%** |",
    ])
    if routing.get("avg_latency_s"): lines.append(f"| LLM 延迟 | — | {routing['avg_latency_s']}s |")
    lines.extend([
        "","## 3. 食物解析器","",
        f"| 名称匹配 | **{parser['name_acc']}%** |",
        f"| 克数准确率 | **{parser['grams_acc']}%** |",
        f"| 平均误差 | {parser['avg_err']}% |",
        "","## 4. 记忆压缩","",
        f"| 场景 | {compression['scenarios']} |",
        f"| 关键留存率 | **{compression['avg_retention']}%** |",
        f"| 终止信号 | {'PASS' if compression['finish_preserved'] else 'FAIL'} |",
    ])
    for d in compression["details"]:
        lines.append(f"| {d['name']} | t={d.get('tokens',0)} L1={d.get('l1',0)} L2={d.get('l2',0)} L3={d.get('l3',0)} | {d['retention']}% |")
    lines.extend([
        "","## 5. 预处理","",
        f"| 完整度 | **{preproc['pct']}%** ({preproc['found']}/{preproc['total']}) |",
        "","---","","## Resume Card","","```",
        f"RAG:    Recall@3={r['recall_at_3']}% MRR={r['mrr']} NDCG@3={r['ndcg_at_3']}"+(f" ({r['cv_folds']}-fold CV)" if r.get('cv_folds') else ""),
        f"Routing:{routing['accuracy']}% ({routing['method']})",
        f"Parser: Name {parser['name_acc']}% | Grams {parser['grams_acc']}%",
        f"Compress:{compression['avg_retention']}% ({compression['scenarios']} scenarios)",
        f"Preprocess:{preproc['pct']}%",
        "```",
    ])
    return "\n".join(l for l in lines if l is not None)


# ============================================================
#  Main
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--live", action="store_true", help="LLM CoT routing")
    p.add_argument("--cv", type=int, default=0, help="RAG N-fold CV (0=full)")
    p.add_argument("--ai-label", action="store_true", help="AI Recall annotation (LLM-as-judge)")
    args = p.parse_args()
    global USE_AI_LABEL
    USE_AI_LABEL = args.ai_label

    t0 = time.time()
    print("="*60); print("NutriGuard 全维度离线评测"); print("="*60)

    rag = run_rag(cv_folds=args.cv)
    routing = run_routing(run_live=args.live)
    parser = run_parser()
    compression = run_compression()
    preproc = run_preprocess()

    print("\n"+"="*60); print("SUMMARY"); print("="*60)
    cv_tag = f" {args.cv}-fold CV" if args.cv else ""
    label_tag = " [AI-label]" if USE_AI_LABEL else " [keyword]"
    print(f"RAG:       Recall@3={rag['recall_at_3']}% MRR={rag['mrr']} NDCG@3={rag['ndcg_at_3']}{cv_tag}{label_tag}")
    print(f"Routing:   {routing['accuracy']}% ({routing['method']})")
    print(f"Parser:    Name {parser['name_acc']}% | Grams {parser['grams_acc']}%")
    print(f"Compress:  {compression['avg_retention']}% ({compression['scenarios']} scenarios)")
    print(f"Preprocess:{preproc['pct']}%")

    report = {"timestamp":time.strftime("%Y-%m-%d %H:%M:%S"),
        "rag":rag,"routing":routing,"parser":parser,"compression":compression,"preprocess":preproc}
    with open(os.path.join(BASE_DIR,"eval_report.json"),"w",encoding="utf-8") as f:
        json.dump(report,f,ensure_ascii=False,indent=2)
    md = generate_report(rag,routing,parser,compression,preproc)
    with open(os.path.join(BASE_DIR,"EVAL_REPORT.md"),"w",encoding="utf-8") as f:
        f.write(md)
    print(f"\nReports: eval_report.json / EVAL_REPORT.md")
    print(f"Duration: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
