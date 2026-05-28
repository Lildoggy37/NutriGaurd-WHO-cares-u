import sys
import json
import time
import os
import numpy as np
import redis
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from fastmcp import FastMCP
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse
from sentence_transformers import CrossEncoder

from db import init_db, log_meal, get_today_calories, lookup_food, list_foods, get_food_categories

# ============================
#   1. 构建MCP Redis  本地向量模型
# ============================
mcp = FastMCP("NutriGuard_Tools")
# 链接本地的Docker 与 Redis Stack
try:
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
    redis_client.ping()
    print("✓ [MCP Server] Redis 连接成功", file=sys.stderr)
except Exception as e:
    print(f"⚠ [MCP Server] Redis 未连接，缓存功能将禁用: {e}", file=sys.stderr)
    redis_client = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BGE_MODEL_PATH = os.path.join(BASE_DIR, "models", "bge-large-zh-v1.5")
RERANKER_PATH = os.path.join(BASE_DIR, "models", "bge-reranker-v2-m3")
CORPUS_PATH = os.path.join(BASE_DIR, "data", "mock_corpus.md")


# 加载BGE模型，来Cache计算向量
embedder = None
VECTOR_DIM = 1024  # bge-large-zh-v1.5 的维度是 1024

try:
    if os.path.isdir(BGE_MODEL_PATH):
        embedder = HuggingFaceEmbeddings(model_name=BGE_MODEL_PATH)
        print("✓ [MCP Server] BGE 模型加载成功", file=sys.stderr)
    else:
        print(f"⚠ [MCP Server] BGE 模型路径不存在: {BGE_MODEL_PATH}", file=sys.stderr)
        print(f"  请下载模型到该路径或修改 BGE_MODEL_PATH", file=sys.stderr)
except Exception as e:
    print(f"⚠ [MCP Server] BGE 模型加载失败: {e}", file=sys.stderr)

# 初始化 SQLite 数据库
init_db()
print("✓ [MCP Server] SQLite 数据库就绪", file=sys.stderr)


# ============================
#   2. RAG
# ============================
print("初始化Qdrant双路召回与本地知识库......",file=sys.stderr)

# sparse model
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

# 读取本地语料进行分割
with open(CORPUS_PATH,"r",encoding="utf-8") as f:
    markdown_document = f.read()

headers_to_split_on = [("##", "Chapter"), ("###", "Section")]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
docs = markdown_splitter.split_text(markdown_document)

retriever = None
reranker = None

if embedder is None:
    print(" [RAG] BGE 模型未加载，RAG 检索引擎将不可用", file=sys.stderr)
else:
    # 构建 Qdrant 混合索引
    vectorstore = QdrantVectorStore.from_documents(
        docs,
        embedding=embedder,
        sparse_embedding=sparse_embeddings,
        location=":memory:",
        collection_name="nutriguard_collection",
        retrieval_mode="hybrid"
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    if os.path.isdir(RERANKER_PATH):
        print(f" 正在挂载本地 BGE-Reranker 精排模型: {RERANKER_PATH}", file=sys.stderr)
        reranker = CrossEncoder(RERANKER_PATH)
    else:
        print(f" 本地 Reranker 不存在，从 HuggingFace 加载...", file=sys.stderr)
        reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

    print(" RAG 检索引擎全量就绪！", file=sys.stderr)


def perform_rag_search(query: str, top_k: int = 3) -> str:
    """
    内部RAG检索引擎：双路召回 + Reranker 精排。
    若模型未就绪则返回空字符串，由调用方降级处理。
    """
    if retriever is None or reranker is None:
        return ""

    # A 双路召回
    rough_docs = retriever.invoke(query)
    if not rough_docs:
        return ""

    # B Rerank
    sentence_pairs = [[query, doc.page_content] for doc in rough_docs]
    scores = reranker.predict(sentence_pairs)

    scored_docs = list(zip(rough_docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    top_docs = scored_docs[:top_k]

    # 组装为Context
    context_parts = []
    for doc, score in top_docs:
        meta = doc.metadata.get("Section", "通用营养知识")
        context_parts.append(f"【{meta}】(相关度: {score:.2f})\n{doc.page_content}")

    return "\n\n".join(context_parts)


# ============================
#   3. Redis 语义缓存
# ============================
INDEX_NAME = "idx:semantic_cache"

def setup_redis_index():
    """
    在Redis 创建向量索引 if not exist
    """
    if redis_client is None:
        return
    try:
        redis_client.ft(INDEX_NAME).info()
    except Exception:
        try:
            print("🧱 正在 Redis 中初始化 Vector Index...", file=sys.stderr)
            schema = (
                TextField("query_text"),
                TextField("answer"),
                VectorField("query_vector", "FLAT", {
                    "TYPE": "FLOAT32",
                    "DIM": VECTOR_DIM,
                    "DISTANCE_METRIC": "COSINE"
                })
            )
            definition = IndexDefinition(prefix=["cache:"], index_type=IndexType.HASH)
            redis_client.ft(INDEX_NAME).create_index(fields=schema, definition=definition)
        except Exception as e:
            print(f"⚠ [MCP Server] Redis 索引初始化失败（可能需要 Redis Stack 而非普通 Redis）: {e}", file=sys.stderr)

setup_redis_index()

def get_from_cache(query: str, threshold: float = 0.85):
    """
    从Redis查询语义缓存（需要 embedder + redis_client 均可用）
    """
    if embedder is None or redis_client is None:
        return None
    try:
        query_vector = embedder.embed_query(query)
        query_vector_bytes = np.array(query_vector, dtype=np.float32).tobytes()

        q = Query(f"*=>[KNN 1 @query_vector $vec AS score]")\
            .return_fields("query_text", "answer", "score")\
            .sort_by("score")\
            .dialect(2)

        results = redis_client.ft(INDEX_NAME).search(q, query_params={"vec": query_vector_bytes})

        if results.docs:
            distance = float(results.docs[0].score)
            similarity = 1 - distance

            if similarity >= threshold:
                print(f"[Cache 命中] 相似度: {similarity:.4f} | 匹配历史提问: '{results.docs[0].query_text}'", file=sys.stderr)
                return results.docs[0].answer
    except Exception as e:
        print(f"⚠ [Cache 查询异常] {e}", file=sys.stderr)

    return None

def save_to_cache(query: str, answer: str):
    """
    将新问题和答案写入 Redis 缓存，24h过期（需要 embedder + redis_client 均可用）
    """
    if embedder is None or redis_client is None:
        return
    try:
        query_vector = embedder.embed_query(query)
        query_vector_bytes = np.array(query_vector, dtype=np.float32).tobytes()

        cache_key = f"cache:{hash(query)}"
        redis_client.hset(cache_key, mapping={
            "query_text": query,
            "answer": answer,
            "query_vector": query_vector_bytes
        })
        redis_client.expire(cache_key, 86400)
    except Exception as e:
        print(f"⚠ [Cache 写入异常] {e}", file=sys.stderr)


# ============================
#   class1  知识检索类 read-only
# ============================
@mcp.tool()
async def search_diet_guidelines(query:str)->str:
    """
    专门用于查询《中国居民膳食指南》和普适性营养原则。
    """
    print(f" [MCP 检索] 正在查阅膳食指南: {query}", file=sys.stderr)
    context = perform_rag_search(query)
    if not context:
        return "抱歉，本地知识库暂时不可用，请稍后再试。"
    return f"【膳食指南检索结果】\n{context}"

@mcp.tool()
async def check_food_gi(food_name: str) -> str:
    """
    查询特定食物的升糖指数(GI)和血糖负荷(GL)。
    """
    print(f" [MCP 检索] 正在查询 GI 数据库: {food_name}", file=sys.stderr)
    context = perform_rag_search(f"{food_name} 升糖指数 GI 血糖负荷")
    if not context:
        return f"关于「{food_name}」的升糖指数数据暂未收录，建议咨询专业营养师。"
    return f"【{food_name} GI 检索结果】\n{context}"

@mcp.tool()
async def search_medical_taboos(disease_name: str) -> str:
    """
    查询特定慢性病（如糖尿病、痛风）的饮食禁忌。
    """
    
    # 先查 Redis 语义缓存
    start_time = time.time()
    cached_result = get_from_cache(disease_name, threshold=0.85)
    
    if cached_result:
        latency = (time.time() - start_time) * 1000
        print(f"⚡ [Redis 极速返回] 耗时: {latency:.2f} ms", file=sys.stderr)
        return cached_result

    # 未命中缓存，执行真实的双路召回 + Reranker 检索
    print(f"[Cache 未命中] 正在执行深度 RAG 检索: {disease_name}...", file=sys.stderr)
    context = perform_rag_search(disease_name)
    if not context:
        return "抱歉，本地知识库暂时不可用，请稍后再试。"
    real_answer = f"【{disease_name} 深度检索结果】\n{context}"
    
    # 异步写入缓存（为后续相同的提问铺路）
    save_to_cache(disease_name, real_answer)
    
    latency = (time.time() - start_time) * 1000
    print(f"[真实计算返回] 耗时: {latency:.2f} ms", file=sys.stderr)
    return real_answer


# ============================
#   class2  业务行动类 Write N Compute
# ============================

# 常见食物份量映射（用于解析自然语言描述）
PORTION_MAP = {
    "个": 1, "只": 1, "根": 1, "块": 1, "片": 1, "粒": 1, "颗": 1,
    "碗": 1, "盘": 1, "杯": 1, "勺": 1, "份": 1,
}
# 食物名到默认份量的映射
_DEFAULT_GRAMS_MAP = {
    "包子": 100, "馒头": 100, "饺子": 25, "馄饨": 20, "鸡蛋": 60,
    "苹果": 200, "香蕉": 120, "橙子": 200, "米饭": 150, "面条": 200,
    "牛奶": 250, "豆浆": 250, "酸奶": 250, "粥": 300, "汤": 300,
    "面包": 50, "饼干": 20, "蛋糕": 80, "核桃": 10, "花生": 10,
    "豆腐": 200, "玉米": 150, "红薯": 200, "土豆": 150,
    "鸡胸肉": 150, "牛肉": 150, "猪肉": 100, "猪瘦肉": 100,
    "三文鱼": 150, "虾仁": 150, "带鱼": 150,
    "西兰花": 200, "菠菜": 200, "番茄": 150, "黄瓜": 150,
    "胡萝卜": 150, "大白菜": 200, "冬瓜": 200,
}


def _estimate_default_grams(food_name: str) -> float:
    """根据食物名模糊匹配默认份量"""
    if food_name in _DEFAULT_GRAMS_MAP:
        return _DEFAULT_GRAMS_MAP[food_name]
    for key, val in _DEFAULT_GRAMS_MAP.items():
        if key in food_name or food_name in key:
            return val
    return 100.0


def _parse_food_items(raw: str) -> list[dict]:
    """
    解析食物输入字符串为结构化列表。
    支持格式：
      - "食物名:200g" 或 "食物名:3个"
      - "200g鸡胸肉, 300g西兰花"
      - "2个包子, 1杯牛奶"
    未指定份量时使用 _DEFAULT_GRAMS_MAP 估算。
    """
    import re

    results = []
    parts = [p.strip() for p in raw.replace("，", ",").split(",") if p.strip()]

    for part in parts:
        # 尝试 "数量g食物名" 或 "食物名:数量g" 格式
        match = re.match(r"(\d+)\s*g\s*(.+)", part)
        if not match:
            match = re.match(r"(.+)[:：]\s*(\d+)\s*(g|克|ml|个|碗|杯|只|根|块|片|粒|颗|勺|盘|份)?", part)
            if match:
                name = match.group(1).strip()
                amount = float(match.group(2))
                unit = match.group(3) or "g"
                if unit in ("ml", "克", "g"):
                    results.append({"name": name, "amount_g": amount})
                elif unit in PORTION_MAP:
                    default = _estimate_default_grams(name)
                    results.append({"name": name, "amount_g": default * amount})
                else:
                    results.append({"name": name, "amount_g": amount})
                continue
            # 尝试 "数量单位食物名"
            match = re.match(r"(\d+)\s*(个|碗|杯|只|根|块|片|粒|颗|勺|盘|份|g|克|ml)\s*(.+)", part)
            if match:
                amount = float(match.group(1))
                unit = match.group(2)
                name = match.group(3).strip()
                if unit in ("g", "克", "ml"):
                    results.append({"name": name, "amount_g": amount})
                else:
                    default = _estimate_default_grams(name)
                    results.append({"name": name, "amount_g": default * amount})
                continue
            # 纯食物名，用默认份量
            if part:
                default = _estimate_default_grams(part)
                results.append({"name": part, "amount_g": default})
            continue

        if match:
            amount = float(match.group(1))
            name = match.group(2).strip()
            results.append({"name": name, "amount_g": amount})
        else:
            if part:
                default = _estimate_default_grams(part)
                results.append({"name": part, "amount_g": default})

    return results


@mcp.tool()
async def search_food(food_name: str) -> str:
    """
    查询食物营养数据库，获取食物的热量、蛋白质、脂肪、碳水、纤维、GI等数据。
    适合查询具体食物的营养成分，比 RAG 检索更精确。
    """
    food = lookup_food(food_name)
    if not food:
        return f"未找到「{food_name}」的营养数据。可以尝试使用更通用的食物名称。"

    lines = [
        f"【{food['name']}】({food['category']})",
        f"  热量: {food['calories_per_100g']} kcal/100g",
        f"  蛋白质: {food['protein_per_100g']} g/100g",
        f"  脂肪: {food['fat_per_100g']} g/100g",
        f"  碳水化合物: {food['carbs_per_100g']} g/100g",
    ]
    if food["fiber_per_100g"]:
        lines.append(f"  膳食纤维: {food['fiber_per_100g']} g/100g")
    if food["gi"] and food["gi"] > 0:
        gi_level = "低 GI" if food["gi"] <= 55 else ("中 GI" if food["gi"] <= 70 else "高 GI")
        lines.append(f"  升糖指数(GI): {food['gi']} ({gi_level})")

    return "\n".join(lines)


@mcp.tool()
async def log_user_meal(user_id: str, meal_type: str, food_items: str) -> str:
    """
    记录用户的实际饮食（如早餐、午餐）。
    必须参数:
      - user_id: 用户唯一ID
      - meal_type: 餐次类型（早餐/午餐/晚餐/加餐）
      - food_items: 食物描述，例如 '2个包子,1杯牛奶'、'鸡胸肉:200g,西兰花:300g'

    工具会自动匹配食物营养数据库并计算热量。
    """
    print(f" [MCP 行动] 正在写入饮食日志 | 用户:{user_id} | {meal_type}: {food_items}", file=sys.stderr)

    items = _parse_food_items(food_items)
    if not items:
        return "无法解析食物内容，请使用格式如：'鸡胸肉:200g, 西兰花:300g'"

    try:
        meal_id = log_meal(user_id, meal_type, items)
    except Exception as e:
        print(f" [DB 写入失败] {e}", file=sys.stderr)
        return f"记录失败：{e}"

    # 统计本餐热量
    total_cal = 0.0
    detail_lines = []
    for item in items:
        food = lookup_food(item["name"])
        if food and food["calories_per_100g"]:
            cal = round(food["calories_per_100g"] * item["amount_g"] / 100, 1)
            total_cal += cal
            detail_lines.append(f"  {item['name']} {item['amount_g']}g → {cal} kcal")
        else:
            detail_lines.append(f"  {item['name']} {item['amount_g']}g → 未找到营养数据")

    header = f"已记录 {user_id} 的{meal_type}（#{meal_id}），本餐约 {round(total_cal, 1)} kcal"
    return header + "\n" + "\n".join(detail_lines)


@mcp.tool()
async def calculate_daily_calories(user_id: str) -> str:
    """
    计算用户今天已记录的总热量摄入，并与推荐摄入量对比。
    """
    print(f" [MCP 行动] 正在核算今日卡路里 | 用户:{user_id}", file=sys.stderr)

    total, rows = get_today_calories(user_id)

    if not rows:
        return f"用户 {user_id} 今天还没有饮食记录。建议每日摄入约 2000 kcal。"

    # 统计各餐次
    meals_summary = {}
    for r in rows:
        mt = r["meal_type"]
        if mt not in meals_summary:
            meals_summary[mt] = 0.0
        meals_summary[mt] += r["calories"] or 0

    lines = [f"用户 {user_id} 今日饮食汇总：", f"  总摄入: {total} kcal"]
    for mt, cal in meals_summary.items():
        lines.append(f"  {mt}: {round(cal, 1)} kcal")

    recommended = 2000
    remaining = round(recommended - total, 1)
    if remaining > 0:
        lines.append(f"  推荐摄入: {recommended} kcal，剩余配额: {remaining} kcal")
    else:
        lines.append(f"  推荐摄入: {recommended} kcal，已超出: {abs(remaining)} kcal")

    return "\n".join(lines)


@mcp.tool()
async def generate_shopping_list(ingredients: str) -> str:
    """
    根据食谱生成按分类排列的超市采购清单。
    """
    print(f" [MCP 行动] 正在生成采购清单: {ingredients}", file=sys.stderr)

    parts = [p.strip() for p in ingredients.replace("，", ",").split(",") if p.strip()]
    if not parts:
        return "请提供需要采购的食材列表。"

    categories: dict[str, list[str]] = {}
    unknown = []

    for name in parts:
        # 去掉数量前缀
        clean = name.lstrip("0123456789克公斤斤个只根块片粒颗勺碗盘杯份gml ")
        food = lookup_food(clean)
        if food:
            cat = food["category"]
            categories.setdefault(cat, []).append(clean)
        else:
            unknown.append(clean)

    lines = ["【采购清单】"]
    for cat, items in categories.items():
        lines.append(f"\n  [{cat}]")
        for item in items:
            lines.append(f"    - {item}")
    if unknown:
        lines.append(f"\n  [未分类]")
        for item in unknown:
            lines.append(f"    - {item}")

    return "\n".join(lines)


if __name__ == "__main__":
    print("🚀 [NutriGuard MCP] 健康膳食微服务已启动...", file=sys.stderr)
    mcp.run()