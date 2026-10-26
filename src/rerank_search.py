"""
重排（rerank）：先用混合检索取一批候选，再用重排模型按"问题—文档"相关性重新排序。

混合检索看的是"两路各自排第几"，重排模型直接读问题和候选原文，
所以能把顺序纠正过来，代价是每条查询多一次接口调用。
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request

import hybrid_search
from config import DASHSCOPE_API_KEY

sys.stdout.reconfigure(encoding="utf-8")

# ---------------- 常量配置 ----------------
RERANK_URL = os.getenv(
    "RERANK_URL",
    "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
)
RERANK_MODEL = os.getenv("RERANK_MODEL", "qwen3.7-text-rerank")
CANDIDATES = 20        # 混合检索先取多少条进重排
DEFAULT_TOP_K = 5
MAX_DOC_CHARS = 2000   # 单条文档截断长度，防止超长块超模型上限


def rerank(query: str, documents: list[str]) -> list[float]:
    """调重排接口，返回和 documents 顺序一一对应的相关性分数。"""
    body = {
        "model": RERANK_MODEL,
        "input": {
            "query": query,
            "documents": [doc[:MAX_DOC_CHARS] for doc in documents],
        },
        "parameters": {"return_documents": False, "top_n": len(documents)},
    }
    request = urllib.request.Request(
        RERANK_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))

    scores = [0.0] * len(documents)
    for item in data["output"]["results"]:
        scores[item["index"]] = float(item["relevance_score"])
    return scores


class RerankRetriever:
    def __init__(self, base: hybrid_search.HybridRetriever):
        self.base = base

    def search(self, query: str, k: int = DEFAULT_TOP_K, candidates: int = CANDIDATES) -> list[dict]:
        hits = self.base.search(query, candidates)
        if not hits:
            return []

        try:
            scores = rerank(query, [hit["text"] for hit in hits])
        except (urllib.error.URLError, KeyError, ValueError, TimeoutError) as exc:
            # 重排接口挂了不该让整个检索挂掉，退回混合检索的顺序
            print(f"[警告] 重排失败，退回混合检索排序：{type(exc).__name__}")
            return hits[:k]

        order = sorted(range(len(hits)), key=lambda i: scores[i], reverse=True)
        results: list[dict] = []
        for i in order[:k]:
            item = dict(hits[i])
            item["score"] = round(scores[i], 4)     # 重排相关性分数
            item["base_rank"] = i + 1               # 重排前它在混合检索里排第几
            item["base_score"] = hits[i]["score"]
            results.append(item)
        return results


def load_retriever() -> RerankRetriever:
    return RerankRetriever(hybrid_search.load_retriever())


# ---------------- 展示 ----------------
def show(retriever: RerankRetriever, query: str, k: int = DEFAULT_TOP_K) -> None:
    print(f"\n查询：{query}")
    started = time.perf_counter()
    results = retriever.search(query, k)
    elapsed = (time.perf_counter() - started) * 1000
    print(f"耗时：{elapsed:.0f} 毫秒（混合检索 + 一次重排请求）")

    for rank, item in enumerate(results, 1):
        position = f"第 {item['section_no']} 节第 {item['item_no']} 条" if item["kind"] == "tip" else item["kind"]
        move = "" if item["base_rank"] == rank else f"，重排前第 {item['base_rank']} 名"
        preview = " ".join(item["text"].split())[:90]
        print(f"\n  {rank}. [{position} | 第 {item['page']} 页] 相关性 {item['score']:.3f}{move}")
        print(f"     {preview}…")


def main() -> None:
    retriever = load_retriever()

    if len(sys.argv) > 1:
        show(retriever, " ".join(sys.argv[1:]))
        return

    print("\n重排检索。输入 q 退出。")
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