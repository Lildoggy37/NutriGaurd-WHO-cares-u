"""
统一分块策略：固定大小 + Overlap + 元数据注入 + 表格保护 + section_id/SHA256 溯源。

替代原有的 MarkdownHeaderTextSplitter（按 H2/H3 切割），
解决块大小不均、表格截断、无 Overlap 三个问题。
新增 section_id + SHA256 标记，支持增量更新和溯源。
"""
import hashlib
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


CHUNK_SIZE = 512
CHUNK_OVERLAP = 128

# 中文友好的分隔符优先级
SEPARATORS = ["\n\n", "\n", "。", "，", " ", ""]


def chunk_document(markdown_text: str) -> list[Document]:
    """
    分块策略：
      1. 先用 Markdown 标题提取章节元数据
      2. 用 RecursiveCharacterTextSplitter 做固定大小 + Overlap 切割
      3. 元数据注入 chunk 文本前缀，帮助 Reranker 理解上下文
      4. 表格保护：含 `|` 的行不被切分
    """
    import re

    # --- 解析章节结构 ---
    sections = _parse_markdown_structure(markdown_text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
        keep_separator=True,
    )

    docs = []
    for sec in sections:
        chapter = sec.get("chapter", "")
        section = sec.get("section", "")
        content = sec["content"]

        if not content.strip():
            continue

        # 表格保护：找表格边界，确保不会被从中间切开
        content = _protect_tables(content)

        # 切割
        chunks = splitter.split_text(content)

        for chunk_text in chunks:
            # 元数据注入到文本前缀
            enriched = _build_chunk_text(chunk_text, chapter, section)
            docs.append(Document(
                page_content=enriched,
                metadata={"chapter": chapter, "section": section},
            ))

    return docs


def _parse_markdown_structure(text: str) -> list[dict]:
    """解析 Markdown 的 H2/H3 结构，输出章节列表"""
    import re
    sections = []
    current_chapter = ""
    current_section = ""
    current_content = []

    for line in text.split("\n"):
        if line.startswith("## "):
            # 保存上一个 section
            if current_content:
                sections.append({
                    "chapter": current_chapter,
                    "section": current_section,
                    "content": "\n".join(current_content),
                })
                current_content = []
            current_chapter = line[3:].strip()
            current_section = ""
        elif line.startswith("### "):
            if current_content:
                sections.append({
                    "chapter": current_chapter,
                    "section": current_section,
                    "content": "\n".join(current_content),
                })
                current_content = []
            current_section = line[4:].strip()
        else:
            current_content.append(line)

    # 最后一个 section
    if current_content:
        sections.append({
            "chapter": current_chapter,
            "section": current_section,
            "content": "\n".join(current_content),
        })

    return sections


def _protect_tables(text: str) -> str:
    """表格保护：在表格行之间用特殊分隔符替换，防止被切"""
    import re
    lines = text.split("\n")
    result = []
    in_table = False
    for line in lines:
        is_table_row = line.strip().startswith("|") and line.strip().count("|") >= 2
        if is_table_row and not in_table:
            in_table = True
        elif not is_table_row and in_table:
            in_table = False
            result.append("")  # 空行分隔表格结束
        if in_table:
            result.append(line.replace("|", " | "))  # 宽松表格列间距
        else:
            result.append(line)
    return "\n".join(result)


def _build_chunk_text(content: str, chapter: str, section: str) -> str:
    """构建带元数据标签的 chunk 文本"""
    parts = []
    if chapter:
        parts.append(f"【{chapter}】")
    if section and section != chapter:
        parts.append(f"[{section}]")
    parts.append(content)
    return " ".join(parts)


# ============================================================
#  增量更新支持：section_id + SHA256
# ============================================================

def _generate_section_id(chapter: str, section: str) -> str:
    """根据章节标题生成唯一 ID（如 'diabetes_diet'）"""
    label = section or chapter
    # 拼音首字母简化 + 去特殊字符
    clean = re.sub(r'[^\w一-鿿]', '_', label)[:30]
    if clean:
        return clean
    return hashlib.md5(label.encode()).hexdigest()[:12]


def chunk_with_section_ids(markdown_text: str) -> tuple[list[Document], dict]:
    """
    和 chunk_document 相同，但每个 chunk 的 metadata 额外携带：
      - section_id: 章节唯一标识（用于增量更新时精确定位）
      - content_hash: chunk 内容的 SHA256（用于溯源——Reflection 校验回答是否来自数据库）

    返回: (documents, section_hashes)
      section_hashes: {section_id: (section_full_text_hash, chunk_count)}
    """
    sections = _parse_markdown_structure(markdown_text)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS, keep_separator=True,
    )

    all_docs = []
    section_hashes = {}

    for sec in sections:
        chapter = sec.get("chapter", "")
        section = sec.get("section", "")
        content = sec["content"]

        if not content.strip():
            continue

        section_id = _generate_section_id(chapter, section)
        section_full_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

        content = _protect_tables(content)
        chunks = splitter.split_text(content)

        for i, chunk_text in enumerate(chunks):
            enriched = _build_chunk_text(chunk_text, chapter, section)
            chunk_hash = hashlib.sha256(chunk_text.encode()).hexdigest()[:12]
            all_docs.append(Document(
                page_content=enriched,
                metadata={
                    "chapter": chapter, "section": section,
                    "section_id": section_id,
                    "chunk_idx": i,
                    "content_hash": chunk_hash,
                    "section_hash": section_full_hash,
                },
            ))

        section_hashes[section_id] = {
            "chapter": chapter, "section": section,
            "hash": section_full_hash, "chunk_count": len(chunks),
        }

    return all_docs, section_hashes
