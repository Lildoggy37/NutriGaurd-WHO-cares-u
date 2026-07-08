"""重建 Qdrant 索引并报告统计"""
import os, sys, time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
os.chdir(str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR))

from chunking import chunk_with_section_ids
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse
from db import init_db, upsert_corpus_section

CORPUS_PATH = BASE_DIR / "data" / "mock_corpus.md"
BGE_MODEL_PATH = BASE_DIR / "models" / "bge-large-zh-v1.5"
QDRANT_PATH = BASE_DIR / "data" / "qdrant_storage"

print("=" * 55)
print("  NutriGuard Qdrant 索引重建")
print("=" * 55)

t0 = time.time()

# 1. 读取并分块
print("\n[1/4] 读取语料并分块...")
with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    text = f.read()
print(f"  语料大小: {len(text):,} 字符 ({len(text)/1024/1024:.1f} MB)")

docs, section_hashes = chunk_with_section_ids(text)
print(f"  Chunk 数: {len(docs)}")
print(f"  Section 数: {len(section_hashes)}")

# chunk 大小分布
sizes = [len(d.page_content) for d in docs]
print(f"  Chunk 大小: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)//len(sizes)}")

# 2. 写入 SQLite section 记录
print("\n[2/4] 写入 SQLite section 记录...")
init_db()
for sid, info in section_hashes.items():
    upsert_corpus_section(sid, info["chapter"], info["section"], info["hash"], info["chunk_count"])
print(f"  已写入 {len(section_hashes)} 条 section你  记录")

# 3. 加载 Embedding 模型
print("\n[3/4] 加载 BGE Embedding 模型...")
t1 = time.time()
embedder = HuggingFaceEmbeddings(model_name=str(BGE_MODEL_PATH))
print(f"  BGE 加载完成 ({time.time()-t1:.1f}s)")

# 4. 构建 Qdrant 索引
print("\n[4/4] 构建 Qdrant 混合索引 (Hybrid: Dense + BM25)...")
t2 = time.time()
sparse = FastEmbedSparse(model_name="Qdrant/bm25")

# 确保目录存在
QDRANT_PATH.mkdir(parents=True, exist_ok=True)

vectorstore = QdrantVectorStore.from_documents(
    docs,
    embedding=embedder,
    sparse_embedding=sparse,
    path=str(QDRANT_PATH),  # 使用 path= 而非 location= (避免 Windows 路径被误判为 URL)
    collection_name="nutriguard_collection",
    retrieval_mode="hybrid",
)
print(f"  索引构建完成 ({time.time()-t2:.1f}s)")

# 最终统计
total_time = time.time() - t0
db_size = sum(f.stat().st_size for f in QDRANT_PATH.rglob("*") if f.is_file())
print(f"\n{'='*55}")
print(f"  重建完成!")
print(f"  总耗时: {total_time:.1f}s")
print(f"  Chunk 总数: {len(docs)}")
print(f"  Section 总数: {len(section_hashes)}")
print(f"  Qdrant 存储大小: {db_size/1024/1024:.1f} MB")
print(f"  存储路径: {QDRANT_PATH}")
print(f"{'='*55}")
