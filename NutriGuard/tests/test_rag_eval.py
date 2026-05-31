"""
RAG 离线评测 — pytest 改造版。

原 rag_eval.py 的功能：
  - 加载语料 → 分块 → 构建 Qdrant 双路召回索引 → BGE-Reranker 精排
  - 对 5 条手写测试集计算 Recall@3 和 MRR

改造后：
  - 重量级组件（embedder / vectorstore / reranker）作为 session 级 fixture，所有测试共享
  - 每条测试用例独立断言，失败时精确定位是哪条 query 的问题
  - 新增整体指标阈值断言 + 边界情况测试
"""
import pytest
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse
from sentence_transformers import CrossEncoder


# ============================================================
#  session 级 fixture：一次初始化，所有测试复用
# ============================================================

@pytest.fixture(scope="session")
def docs(corpus_path):
    """加载语料并按 Markdown 标题分块"""
    with open(corpus_path, "r", encoding="utf-8") as f:
        markdown_document = f.read()

    headers_to_split_on = [("##", "Chapter"), ("###", "Section")]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    chunks = splitter.split_text(markdown_document)
    assert len(chunks) > 0, "语料分块结果为空！"
    return chunks


@pytest.fixture(scope="session")
def dense_embeddings(bge_model_path):
    """BGE 稠密向量模型"""
    return HuggingFaceEmbeddings(model_name=bge_model_path)


@pytest.fixture(scope="session")
def retriever(docs, dense_embeddings):
    """构建 Qdrant 双路混合召回 retriever"""
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
    vectorstore = QdrantVectorStore.from_documents(
        docs,
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        location=":memory:",
        collection_name="nutriguard_test_collection",
        retrieval_mode="hybrid",
    )
    return vectorstore.as_retriever(search_kwargs={"k": 10})


@pytest.fixture(scope="session")
def reranker(reranker_model_path):
    """BGE-Reranker 精排模型"""
    return CrossEncoder(reranker_model_path)


# ============================================================
#  核心检索引擎（与 mcp_server.perform_rag_search 逻辑一致）
# ============================================================

def search_top_k(retriever, reranker, query: str, top_k: int = 3):
    """双路召回 + Reranker 精排，返回 [(doc, score), ...]"""
    rough_docs = retriever.invoke(query)
    if not rough_docs:
        return []

    sentence_pairs = [[query, doc.page_content] for doc in rough_docs]
    scores = reranker.predict(sentence_pairs)

    scored = list(zip(rough_docs, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def hit_rank(top_docs, target_keyword: str):
    """返回 target_keyword 在 top_docs 中的排名（1-based），未命中返回 -1"""
    for rank, (doc, _score) in enumerate(top_docs):
        if target_keyword in doc.page_content or target_keyword in str(doc.metadata):
            return rank + 1
    return -1


# ============================================================
#  测试用例
# ============================================================

class TestRagPerQuery:
    """逐条 query 的精准测试 — 哪条挂了都能一眼看到"""

    @pytest.mark.parametrize(
        "query, target_keyword",
        [
            ("糙米饭的升糖指数是多少？", "糙米饭"),
            ("孕妇查出糖尿病，该怎么控制？", "妊娠期糖尿病"),
            ("得了痛风，饮食上要注意啥？", "痛风"),
            ("我今天吃了燕麦片，纤维素高吗？", "燕麦片"),
            ("糖尿病初期的保守治疗手段是什么", "生活方式干预"),
        ],
    )
    def test_single_query_hit(self, retriever, reranker, query, target_keyword):
        """每条 query 的目标关键词必须出现在 Top-3 中"""
        top_docs = search_top_k(retriever, reranker, query, top_k=3)

        assert len(top_docs) > 0, f"检索返回空结果！query='{query}'"

        rank = hit_rank(top_docs, target_keyword)
        assert rank != -1, (
            f"关键词 '{target_keyword}' 未出现在 Top-3 中\n"
            f"query='{query}'\n"
            f"Top-3 内容摘要: {[d.page_content[:80] for d, _ in top_docs]}"
        )


class TestRagOverallMetrics:
    """整体指标断言"""

    def test_recall_at_3_equals_100_percent(self, retriever, reranker, eval_dataset):
        """回归测试：Recall@3 必须保持 100%"""
        hits = 0
        for query, target in eval_dataset:
            top_docs = search_top_k(retriever, reranker, query)
            if hit_rank(top_docs, target) != -1:
                hits += 1

        recall = hits / len(eval_dataset)
        assert recall == 1.0, (
            f"Recall@3 退化！当前: {recall:.0%}，期望: 100%\n"
            f"可能是语料或检索参数发生了变化。"
        )

    def test_mrr_above_threshold(self, retriever, reranker, eval_dataset):
        """MRR 应不低于 0.7"""
        total_mrr = 0.0
        for query, target in eval_dataset:
            top_docs = search_top_k(retriever, reranker, query)
            rank = hit_rank(top_docs, target)
            if rank != -1:
                total_mrr += 1.0 / rank

        mrr = total_mrr / len(eval_dataset)
        assert mrr >= 0.7, f"MRR 过低: {mrr:.4f}，期望 >= 0.7"


class TestRagEdgeCases:
    """边界情况"""

    def test_empty_query_does_not_crash(self, retriever, reranker):
        """空字符串不应抛异常"""
        result = search_top_k(retriever, reranker, "")
        assert isinstance(result, list)

    def test_english_query(self, retriever, reranker):
        """英文查询不应崩溃（即使语料是中文，RAG 也应优雅降级）"""
        result = search_top_k(retriever, reranker, "diabetes diet restrictions")
        assert isinstance(result, list)

    def test_returned_docs_have_metadata(self, retriever, reranker):
        """返回的文档应包含 Section 和 Chapter 元数据"""
        top_docs = search_top_k(retriever, reranker, "糖尿病饮食禁忌")
        if len(top_docs) > 0:
            doc = top_docs[0][0]
            assert hasattr(doc, "metadata"), "文档缺少 metadata 属性"
            # 至少要有 Chapter 或 Section 之一
            has_chapter = "Chapter" in doc.metadata or "Section" in doc.metadata
            assert has_chapter, f"metadata 缺少标题信息: {doc.metadata}"
