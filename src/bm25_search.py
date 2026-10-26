"""
关键词检索（BM25）。

输入一个问题，从 build/chunks.json 里找出最相关的块。
索引只在启动时建一次，之后每次查询直接用。
"""

import json
import logging
import re
import sys
import time

import jieba
from rank_bm25 import BM25Okapi

from config import BUILD_DIR

sys.stdout.reconfigure(encoding="utf-8")
jieba.setLogLevel(logging.WARNING)   # 第一次运行会建词典缓存，别让日志刷屏

# ---------------- 常量配置 ----------------
DEFAULT_TOP_K = 5
SNIPPET_CHARS = 90

# 只过滤最没有区分度的一批虚词。不要贪多：
# “不”“没”“要”这些词在问句里有实际含义，删掉反而降低召回。
STOPWORDS = {
    "的", "了", "是", "在", "和", "与", "或", "有", "也", "就", "都", "而", "及", "等",
    "这", "那", "你", "我", "他", "她", "它", "我们", "你们", "他们",
    "一个", "可以", "如果", "因为", "所以", "但是", "把", "被", "让", "给",
    "对", "从", "到", "为", "以", "其", "之", "个", "们", "吗", "呢", "啊", "吧",
    "着", "过", "很", "更", "最", "还", "又", "再", "才", "只", "已", "并",
}

MEANINGFUL_RE = re.compile(r"[一-龥a-z0-9]")


# ---------------- 分词 ----------------
def tokenize(text: str) -> list[str]:
    """
    中文用 jieba，英文统一小写，标点全部丢掉。

    用 lcut_for_search 而不是 lcut：它会把长词再切成子词，
    比如“高性价比”同时给出“高性价比”和“性价比”，查询更容易命中。
    """
    tokens: list[str] = []
    for token in jieba.lcut_for_search(text.lower()):
        token = token.strip()
        if not token or token in STOPWORDS:
            continue
        if not MEANINGFUL_RE.search(token):
            continue
        tokens.append(token)
    return tokens


# ---------------- 索引 ----------------
class BM25Index:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.corpus = [tokenize(chunk["text"]) for chunk in chunks]
        self.bm25 = BM25Okapi(self.corpus)

    def search(self, query: str, k: int = DEFAULT_TOP_K, dedupe: bool = True) -> list[dict]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results: list[dict] = []
        seen: set = set()
        for i in order:
            if scores[i] <= 0:      # 一个词都没匹配上，分数是 0
                break

            item = dict(self.chunks[i])
            item["score"] = round(float(scores[i]), 3)

            if dedupe:
                # 一条长建议被切成多块时，只保留分数最高的那一块，别占满 top-k
                key = (item["kind"], item.get("section_no"), item.get("item_no")) \
                    if item["kind"] == "tip" else item["id"]
                if key in seen:
                    continue
                seen.add(key)

            results.append(item)
            if len(results) >= k:
                break
        return results


def load_index() -> BM25Index:
    chunks = json.loads((BUILD_DIR / "chunks.json").read_text(encoding="utf-8"))
    started = time.perf_counter()
    index = BM25Index(chunks)
    elapsed = time.perf_counter() - started

    vocab = len({token for doc in index.corpus for token in doc})
    print(f"索引就绪：{len(chunks)} 块，词表 {vocab} 个词，用时 {elapsed:.2f} 秒")
    return index


# ---------------- 展示 ----------------
def show(index: BM25Index, query: str, k: int = DEFAULT_TOP_K) -> None:
    tokens = tokenize(query)
    print(f"\n查询：{query}")
    print(f"分词：{' / '.join(tokens)}")

    results = index.search(query, k)
    if not results:
        print("  （没有匹配到任何块）")
        return

    for rank, item in enumerate(results, 1):
        position = f"第 {item['section_no']} 节第 {item['item_no']} 条" if item["kind"] == "tip" else item["kind"]
        preview = " ".join(item["text"].split())[:SNIPPET_CHARS]
        print(f"\n  {rank}. [{position} | 第 {item['page']} 页] 分数 {item['score']}")
        print(f"     {preview}…")


def main() -> None:
    index = load_index()

    if len(sys.argv) > 1:
        show(index, " ".join(sys.argv[1:]))
        return

    print("\n关键词检索（BM25）。输入 q 退出。")
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
