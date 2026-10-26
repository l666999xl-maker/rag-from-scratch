"""
混合检索：BM25 + 向量，用加权 RRF 按名次融合。

不比较两路的分数（51.1 和 0.748 没有可比性），只看各自排第几。
"""

import sys
import time

import bm25_search
import vector_search

sys.stdout.reconfigure(encoding="utf-8")

# ---------------- 常量配置 ----------------
RRF_K = 10               # 名次平滑常数。越小，前几名的话语权越大
BM25_WEIGHT = 1.0        # BM25 这一路的权重
VECTOR_WEIGHT = 3.0      # 向量这一路的权重（见下面注释）
CANDIDATES = 10          # 每一路先取多少条进入融合
DEFAULT_TOP_K = 5
SNIPPET_CHARS = 90

# 为什么向量权重是 BM25 的 3 倍？
# 这份数据上 BM25 的精确词很强，但它会“自信地错”：
# 问“药和水果能不能一起吃”，BM25 第 1 名是无关条目，向量第 1 名才是对的。
# 1 : 3 是在第 4、5 课的 14 条问题上试出来的起点，不是普适值。
# 第 8 课会用更大的评测集重新调。


def chunk_key(item: dict) -> tuple:
    """去重键：一条建议被拆成多块时，它们共享同一个 (节号, 条号)。"""
    if item["kind"] == "tip":
        return ("tip", item["section_no"], item["item_no"])
    return ("chunk", item["id"])


class HybridRetriever:
    def __init__(self, bm25: bm25_search.BM25Index, vector: vector_search.VectorIndex):
        self.bm25 = bm25
        self.vector = vector

    def search(self, query: str, k: int = DEFAULT_TOP_K, candidates: int = CANDIDATES) -> list[dict]:
        bm25_results = self.bm25.search(query, candidates)
        vector_results = self.vector.search(query, candidates)

        scores: dict[tuple, float] = {}
        items: dict[tuple, dict] = {}
        bm25_rank: dict[tuple, int] = {}
        vector_rank: dict[tuple, int] = {}

        for rank, item in enumerate(bm25_results, 1):
            key = chunk_key(item)
            scores[key] = scores.get(key, 0.0) + BM25_WEIGHT / (RRF_K + rank)
            bm25_rank[key] = rank
            items.setdefault(key, item)

        for rank, item in enumerate(vector_results, 1):
            key = chunk_key(item)
            scores[key] = scores.get(key, 0.0) + VECTOR_WEIGHT / (RRF_K + rank)
            vector_rank[key] = rank
            items.setdefault(key, item)

        order = sorted(scores, key=lambda key: scores[key], reverse=True)[:k]

        results: list[dict] = []
        for key in order:
            item = dict(items[key])
            item["score"] = round(scores[key], 4)       # RRF 融合分
            item["bm25_rank"] = bm25_rank.get(key)
            item["vector_rank"] = vector_rank.get(key)
            results.append(item)
        return results


def load_retriever() -> HybridRetriever:
    bm25 = bm25_search.load_index()
    vector = vector_search.load_index()
    return HybridRetriever(bm25, vector)


def show(retriever: HybridRetriever, query: str, k: int = DEFAULT_TOP_K) -> None:
    print(f"\n查询：{query}")
    started = time.perf_counter()
    results = retriever.search(query, k)
    elapsed = (time.perf_counter() - started) * 1000

    print(f"耗时：{elapsed:.0f} 毫秒（BM25 + 一次 embedding 请求）")
    if not results:
        print("  （没有结果）")
        return

    for rank, item in enumerate(results, 1):
        position = f"第 {item['section_no']} 节第 {item['item_no']} 条" if item["kind"] == "tip" else item["kind"]
        bm25_part = f"#{item['bm25_rank']}" if item["bm25_rank"] else "—"
        vector_part = f"#{item['vector_rank']}" if item["vector_rank"] else "—"
        preview = " ".join(item["text"].split())[:SNIPPET_CHARS]
        print(f"\n  {rank}. [{position} | 第 {item['page']} 页] RRF {item['score']:.4f}")
        print(f"     BM25 {bm25_part}   向量 {vector_part}")
        print(f"     {preview}…")


def main() -> None:
    retriever = load_retriever()

    if len(sys.argv) > 1:
        show(retriever, " ".join(sys.argv[1:]))
        return

    print("\n混合检索（BM25 + 向量 + RRF）。输入 q 退出。")
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
        show(retriever, query)


if __name__ == "__main__":
    main()
