"""
build_kb.py — 构建知识库向量库

流程：
1. 读取 knowledge/*.md
2. 按二级标题（## ）切分为知识片段（语义切分，每段一个因子/算子/族系）
3. 用本地 embedding 写入 Chroma 持久化向量库

用法：
    python rag_explain/build_kb.py            # 增量重建（先清空再写入）
"""

import re
from pathlib import Path

from retriever import KB_DIR, get_collection, _client, COLLECTION_NAME


def split_markdown(text: str, source: str) -> list[dict]:
    """
    按 '## ' 二级标题切分 markdown，每段为一个知识片段。
    片段过短的（无 ## 的前言）以一级标题作为整体片段。
    """
    chunks = []
    # 文档主标题（# ）作为上下文前缀
    h1_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    doc_title = h1_match.group(1).strip() if h1_match else source

    # 以 '## ' 分段
    parts = re.split(r"(?m)^##\s+", text)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # 跳过仅含一级标题/前言、过短的块
        first_line = part.splitlines()[0].strip()
        body = part.strip()
        if len(body) < 30:
            continue
        # 第一块可能是 H1 前言，没有 section 标题
        title = first_line if not first_line.startswith("#") else doc_title
        chunk_text = f"【{doc_title}】{title}\n{body}"
        chunks.append({"text": chunk_text, "title": title, "source": source})
    return chunks


def main():
    md_files = sorted(KB_DIR.glob("*.md"))
    if not md_files:
        raise SystemExit(f"未找到知识库语料：{KB_DIR}/*.md")

    all_chunks = []
    for f in md_files:
        text = f.read_text(encoding="utf-8")
        chunks = split_markdown(text, source=f.name)
        all_chunks.extend(chunks)
        print(f"  {f.name}: {len(chunks)} 个片段")

    if not all_chunks:
        raise SystemExit("切分后无有效片段，请检查语料格式。")

    # 清空旧集合后重建，避免重复
    client = _client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    coll = get_collection(create=True)
    coll.add(
        ids=[f"chunk_{i}" for i in range(len(all_chunks))],
        documents=[c["text"] for c in all_chunks],
        metadatas=[{"source": c["source"], "title": c["title"]} for c in all_chunks],
    )

    print(f"\n知识库构建完成：共 {len(all_chunks)} 个片段，已写入向量库。")
    print(f"集合 '{COLLECTION_NAME}' 当前条目数：{coll.count()}")


if __name__ == "__main__":
    main()
