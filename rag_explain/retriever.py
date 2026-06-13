"""
retriever.py — 本地向量库检索封装

职责：
1. 统一管理 embedding 模型（sentence-transformers，中文模型 bge-small-zh-v1.5）
2. 提供 get_collection() 获取/创建 Chroma 持久化集合
3. 提供 retrieve() 对查询文本做 top-k 相似检索

说明：
- 在导入任何 transformers 相关库之前设置 USE_TF=0，避免本机 Keras3/TF 冲突。
- embedding 模型首次使用会自动下载（约 100MB），之后离线可用。
"""

import os
from pathlib import Path
from functools import lru_cache

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "model"

# 必须在 import sentence_transformers / transformers 之前设置
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# 将 embedding 模型下载和缓存统一放到仓库的 model/ 目录
MODEL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_ENDPOINT", os.environ.get("RAG_HF_ENDPOINT", "https://hf-mirror.com"))
os.environ.setdefault("HF_HOME", str(MODEL_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(MODEL_DIR / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(MODEL_DIR / "transformers"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(MODEL_DIR / "sentence-transformers"))

import chromadb
from chromadb.utils import embedding_functions

# 知识库与向量库路径
KB_DIR = Path(__file__).resolve().parent / "knowledge"
CHROMA_DIR = Path(__file__).resolve().parent / "chroma_db"
COLLECTION_NAME = "quant_factor_kb"

# 中文小型 embedding 模型；离线、免费
EMBED_MODEL_NAME = os.environ.get("RAG_EMBED_MODEL", "BAAI/bge-small-zh-v1.5")
EMBED_DEVICE = os.environ.get("RAG_EMBED_DEVICE", "cpu")


@lru_cache(maxsize=1)
def _embedding_function():
    """构造 Chroma 用的 sentence-transformers embedding 函数（单例）。"""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL_NAME,
        device=EMBED_DEVICE,
        cache_folder=str(MODEL_DIR / "sentence-transformers"),
    )


@lru_cache(maxsize=1)
def _client():
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection(create: bool = True):
    """获取（或创建）知识库集合。"""
    client = _client()
    if create:
        return client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=_embedding_function(),
            metadata={"hnsw:space": "cosine"},
        )
    return client.get_collection(
        name=COLLECTION_NAME, embedding_function=_embedding_function()
    )


def retrieve(query: str, top_k: int = 4) -> list[dict]:
    """
    对 query 做相似检索，返回 top_k 个知识片段。

    Returns:
        [{"text": str, "source": str, "title": str, "distance": float}, ...]
    """
    try:
        coll = get_collection(create=False)
    except Exception as e:
        raise RuntimeError(
            f"向量库不存在或为空，请先运行 build_kb.py。原始错误: {e}"
        )

    res = coll.query(query_texts=[query], n_results=top_k)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    out = []
    for doc, meta, dist in zip(docs, metas, dists):
        meta = meta or {}
        out.append(
            {
                "text": doc,
                "source": meta.get("source", "unknown"),
                "title": meta.get("title", ""),
                "distance": float(dist),
            }
        )
    return out


if __name__ == "__main__":
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "量价背离反转因子"
    print(f"查询: {q}\n")
    for i, hit in enumerate(retrieve(q, top_k=4), 1):
        print(f"[{i}] 来源={hit['source']} 标题={hit['title']} 距离={hit['distance']:.4f}")
        print(hit["text"][:200].replace("\n", " "), "...\n")
