"""
向量检索（embedding + 余弦相似度）。

第一次运行会把 1015 个块全部向量化，缓存到 build/embeddings.npy。
之后启动直接读缓存，只把用户的问题向量化一次。
"""

import hashlib
import json
import sys
import time

import numpy as np
from openai import OpenAI

from config import (
    BUILD_DIR,
    DASHSCOPE_API_KEY,
    EMBEDDING_BASE_URL,
    EMBEDDING_MODEL,
)

sys.stdout.reconfigure(encoding="utf-8")

# ---------------- 常量配置 ----------------
BATCH_SIZE = 20          # 百炼 embedding 接口单次请求的条数上限是 20
DEFAULT_TOP_K = 5
SNIPPET_CHARS = 90
VECTOR_PATH = BUILD_DIR / "embeddings.npy"
META_PATH = BUILD_DIR / "embeddings_meta.json"


def get_client() -> OpenAI:
    return OpenAI(api_key=DASHSCOPE_API_KEY, base_url=EMBEDDING_BASE_URL)


# ---------------- 向量化 ----------------
def embed_texts(texts: list[str], verbose: bool = True) -> np.ndarray:
    """把一批文本变成向量。返回形状 (文本数, 维度)，float32。"""
    client = get_client()
    vectors: list[list[float]] = []

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        vectors.extend(item.embedding for item in sorted(response.data, key=lambda x: x.index))
        if verbose:
            print(f"  已向量化 {start + len(batch)}/{len(texts)}", end="\r")

    if verbose:
        print()
    return np.array(vectors, dtype=np.float32)


def chunks_fingerprint(raw: bytes) -> str:
    """chunks.json 的指纹。文件一变，指纹就变，缓存自动失效。"""
    return hashlib.sha256(raw).hexdigest()[:16]


def load_or_build_vectors(chunks: list[dict], raw_chunks: bytes, force: bool = False) -> np.ndarray:
    fingerprint = chunks_fingerprint(raw_chunks)
    meta = None
    if META_PATH.exists():
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))

    cache_ok = (
        not force
        and VECTOR_PATH.exists()
        and meta is not None
        and meta.get("model") == EMBEDDING_MODEL
        and meta.get("fingerprint") == fingerprint
        and meta.get("count") == len(chunks)
    )

    if cache_ok:
        vectors = np.load(VECTOR_PATH)
        print(f"读取缓存：{vectors.shape[0]} 块 × {vectors.shape[1]} 维")
        return vectors

    print(f"开始向量化 {len(chunks)} 块，模型 {EMBEDDING_MODEL}，每批 {BATCH_SIZE} 条……")
    started = time.perf_counter()
    vectors = embed_texts([chunk["text"] for chunk in chunks])
    elapsed = time.perf_counter() - started

    np.save(VECTOR_PATH, vectors)
    META_PATH.write_text(
        json.dumps({
            "model": EMBEDDING_MODEL,
            "count": len(chunks),
            "dim": int(vectors.shape[1]),
            "fingerprint": fingerprint,
        }, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )

    print(f"向量维度：{vectors.shape[1]}")
    print(f"用时 {elapsed:.1f} 秒，已保存：{VECTOR_PATH.name}")
    return vectors


# ---------------- 索引 ----------------
class VectorIndex:
    def __init__(self, chunks: list[dict], vectors: np.ndarray):
        if len(chunks) != len(vectors):
            raise ValueError(f"块数({len(chunks)})和向量数({len(vectors)})对不上")

        self.chunks = chunks
        # 先归一化，之后余弦相似度就是点积
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.normalized = vectors / norms

    def search(self, query: str, k: int = DEFAULT_TOP_K, dedupe: bool = True) -> list[dict]:
        query_vec = embed_texts([query], verbose=False)[0]
        norm = np.linalg.norm(query_vec) or 1.0
        query_vec = query_vec / norm

        scores = self.normalized @ query_vec
        order = np.argsort(-scores)

        results: list[dict] = []
        seen: set = set()
        for i in order:
            item = dict(self.chunks[int(i)])
            item["score"] = round(float(scores[i]), 3)

            if dedupe:
                key = (item["kind"], item.get("section_no"), item.get("item_no")) \
                    if item["kind"] == "tip" else item["id"]
                if key in seen:
                    continue
                seen.add(key)

            results.append(item)
            if len(results) >= k:
                break
        return results


def load_index(force_build: bool = False) -> VectorIndex:
    raw = (BUILD_DIR / "chunks.json").read_bytes()
    chunks = json.loads(raw.decode("utf-8"))
    vectors = load_or_build_vectors(chunks, raw, force=force_build)
    return VectorIndex(chunks, vectors)


# ---------------- 展示 ----------------
def show(index: VectorIndex, query: str, k: int = DEFAULT_TOP_K) -> None:
    print(f"\n查询：{query}")
    started = time.perf_counter()
    results = index.search(query, k)
    elapsed = (time.perf_counter() - started) * 1000

    print(f"耗时：{elapsed:.0f} 毫秒（含一次 embedding 请求）")
    if not results:
        print("  （没有结果）")
        return

    for rank, item in enumerate(results, 1):
        position = f"第 {item['section_no']} 节第 {item['item_no']} 条" if item["kind"] == "tip" else item["kind"]
        preview = " ".join(item["text"].split())[:SNIPPET_CHARS]
        print(f"\n  {rank}. [{position} | 第 {item['page']} 页] 相似度 {item['score']:.3f}")
        print(f"     {preview}…")


def main() -> None:
    force_build = "--rebuild" in sys.argv
    args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    index = load_index(force_build=force_build)

    if args:
        show(index, " ".join(args))
        return

    if force_build:
        print("重建完成。")
        return

    print("\n向量检索。输入 q 退出。")
    while True:
        try:
            query = input("\n查询 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query.lower() in {"q", "quit", "exit"}:
            break
        show(index, query)


if __name__ == "__main__":
    main()
