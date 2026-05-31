import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import time
from typing import List
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse
from sentence_transformers import CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# ================1.Chunking===================
print("step1:We are chunking now......")
with open("data/mock_corpus.md","r",encoding="utf-8") as f:
    markdown_document = f.read()

# 按照 markdown中标题层级进行划分，保留表格和段落
headers_to_split_on = [
    ("##","Chapter"), # 二级标题
    ("###","Section"), # 三级标题
]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
docs = markdown_splitter.split_text(markdown_document)
print(f"切分完毕！切出了{len(docs)} 个 Chunk！！！")
for i,doc in enumerate(docs[:4]):
    print(f"   预览 Chunk {i+1} 元数据 ：{doc.metadata}")


# ======================2. Use Qdrant to Hybrid Search =======
print("\nstep2: BGE and BN25 model is loading...")
# dense
dense_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5")
# sparse
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

print("Qdrant database is creating...")

# 使用内存模式方便测试
client = QdrantClient(location=":memory:")

vectorstore = QdrantVectorStore.from_documents(
    docs,
    embedding=dense_embeddings,
    sparse_embedding=sparse_embeddings,
    location=":memory:",         # 我们用内存版演示
    collection_name="nutriguard_collection",
    retrieval_mode="hybrid"      # 开启双路召回
)

# 第一次粗捞召回Top10
retriever = vectorstore.as_retriever(search_kwargs={"k":10})
print(" 双路混合召回库建立完毕！")

# ===========================3. BGE-Reranker ============
print("\nstep3: BGE-Reranker-v2-m3 model is loading....")
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
print("Reranker is already!")



# ===========================4. 离线测评 Recall@3 and MRR ============
print("\nstep4:开始离线评测 (RAG Shift-Left Evaluation)...")
# 手写 Ground Truth 测试集
eval_dataset = [
    ("糙米饭的升糖指数是多少？", "糙米饭"), # 考察表格数据的精准提取
    ("孕妇查出糖尿病，该怎么控制？", "妊娠期糖尿病"), # 考察语义同义词替换 (孕妇 -> 妊娠期)
    ("得了痛风，饮食上要注意啥？", "痛风"), # 考察疾病大类召回
    ("我今天吃了燕麦片，纤维素高吗？", "燕麦片"), # 考察闲聊+营养成分提取
    ("糖尿病初期的保守治疗手段是什么", "生活方式干预") # 考察叙事段落里的深度理解
]
def run_evaluation(dataset: List[tuple]):
    total_queries = len(dataset)
    recall_at_3_hits = 0
    mrr_score = 0.0

    for query,target_keyword in dataset:
        print(f"\n 测试问题：{query}")
        
        # a.双路召回Top10
        rough_docs = retriever.invoke(query)
        # b.Reranker重排Top3
        sentence_pairs = [[query, doc.page_content] for doc in rough_docs]
        scores = reranker.predict(sentence_pairs)

        # 绑定得分并排序
        scored_docs = list(zip(rough_docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        top_3_docs = scored_docs[:3]

        # 评测 检查目标关键词是否出现在 Top-3 的正文或 Metadata(如章节名) 中
        hit_rank = -1
        for rank, (doc, score) in enumerate(top_3_docs):
            if target_keyword in doc.page_content or target_keyword in str(doc.metadata):
                hit_rank = rank + 1
                break
                
        if hit_rank != -1:
            recall_at_3_hits += 1
            mrr_score += (1.0 / hit_rank)
            print(f"  ✅ 命中！排在第 {hit_rank} 名 (Rerank得分: {top_3_docs[hit_rank-1][1]:.2f})")
        else:
            print(f"  ❌ 未命中！Top 3 未包含预期线索: '{target_keyword}'")

    # 计算最终指标
    recall_at_3 = recall_at_3_hits / total_queries
    mrr = mrr_score / total_queries

    print("\n" + "="*40)
    print("📊 离线评测最终成绩单")
    print(f"总测试问题数 : {total_queries}")
    print(f"Recall@3 (召回率) : {recall_at_3 * 100:.2f}%")
    print(f"MRR (平均倒数排名): {mrr:.4f}")
    print("="*40)

# 执行评测
run_evaluation(eval_dataset)
