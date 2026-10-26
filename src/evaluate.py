"""
第 8 课（升级版）：评测。

测四件事：
1. 检索：Recall@1/@3、MRR@3、nDCG@3，按 dev / holdout 分组
2. 行为：资料内有没有被误拒、资料外有没有正确拒答、部分覆盖有没有按部分回答
3. 引用：编号是否都真实存在
4. 回答质量：LLM 按标准答案打分（可选）

用法：
    python src/evaluate.py                       # 只跑检索，快
    python src/evaluate.py --rerank               # 检索里加上重排那一路
    python src/evaluate.py --answers              # 再跑 50 条问答
    python src/evaluate.py --answers --judge      # 再加 LLM 评分
    python src/evaluate.py --judge-only           # 只重跑评分，复用上次回答
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

from openai import OpenAI

import bm25_search
import hybrid_search
import rag_answer
import rerank_search
import vector_search
from config import BUILD_DIR, CHAT_MODEL, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, ROOT

sys.stdout.reconfigure(encoding="utf-8")

QUESTIONS_PATH = ROOT / "data" / "eval" / "questions.json"
SPLITS = ("dev", "holdout")

JUDGE_SYSTEM = """你是严格的评分员。根据【问题】【标准答案】【模型回答】打分。
2 分：标准答案里的关键结论和数字都答对；可以有多余内容，只要不与标准答案矛盾；
1 分：漏掉至少一个关键结论或数字，或有一处不影响主要结论的小错误；
0 分：主要结论错误、编造关键数字、或答非所问。
只输出 JSON，不要输出其他文字：{"score": 0, "reason": "不超过 30 字"}"""


# ---------------- 工具 ----------------
def tip_key(item: dict) -> tuple | None:
    if item["kind"] == "tip":
        return (item["section_no"], item["item_no"])
    return None


def load_questions(limit: int | None = None) -> list[dict]:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    return questions[:limit] if limit else questions


def percent(hit: int, total: int) -> str:
    return f"{hit}/{total} ({hit / total:.0%})" if total else "0/0"


def first_gold_rank(keys: list, gold: set, k: int) -> int | None:
    for rank, key in enumerate(keys[:k], 1):
        if key in gold:
            return rank
    return None


def ndcg_at_k(keys: list, gold: set, k: int) -> float:
    dcg = sum(1 / math.log2(rank + 1) for rank, key in enumerate(keys[:k], 1) if key in gold)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0


# ---------------- 检索 ----------------
def eval_retrieval(cases: list[dict], methods: list[tuple[str, callable]]) -> dict:
    report = {}
    for name, search in methods:
        r1 = r3 = 0
        mrr = ndcg = 0.0
        details = []
        for case in cases:
            hits = search(case["question"], 3)
            gold = {tuple(tip) for tip in case["gold_tips"]}
            keys = [tip_key(item) for item in hits]
            rank = first_gold_rank(keys, gold, 3)
            hit1 = rank == 1
            hit3 = rank is not None
            r1 += hit1
            r3 += hit3
            mrr += 1 / rank if rank else 0.0
            ndcg += ndcg_at_k(keys, gold, 3)
            details.append({
                "id": case["id"],
                "question": case["question"],
                "gold": [list(tip) for tip in case["gold_tips"]],
                "top3": [list(key) if key else None for key in keys],
                "first_gold_rank": rank,
            })
        n = len(cases)
        report[name] = {
            "recall@1": r1 / n,
            "recall@3": r3 / n,
            "mrr@3": mrr / n,
            "ndcg@3": ndcg / n,
            "n": n,
            "details": details,
        }
    return report


def print_retrieval(retrieval: dict) -> None:
    for split, methods in retrieval.items():
        total = next(iter(methods.values()))["n"]
        print(f"\n  [{split}] 资料内问题 {total} 条")
        print(f"    {'方法':<8}{'R@1':>7}{'R@3':>7}{'MRR':>7}{'nDCG@3':>8}")
        for name, m in methods.items():
            print(f"    {name:<8}{m['recall@1']:>6.0%}{m['recall@3']:>7.0%}"
                  f"{m['mrr@3']:>7.3f}{m['ndcg@3']:>8.3f}")
        for name, m in methods.items():
            misses = [d["id"] for d in m["details"] if d["first_gold_rank"] != 1]
            if misses:
                print(f"    {name} R@1 未中：{'、'.join(misses)}")


# ---------------- 回答行为 ----------------
def eval_behavior(cases: list[dict], retriever, top_k: int) -> list[dict]:
    details = []
    for case in cases:
        result = rag_answer.ask(retriever, case["question"], top_k)
        check = result["check"]
        status = check["status"]
        answer = result["answer"]

        if case["type"] == "in":
            passed = status in {"answered", "partial"}
        elif case["type"] == "out":
            passed = status in {"refused", "blocked"}
        else:  # partial：答成"部分回答"最好，明确拒答也算合格
            passed = status in {"partial", "refused", "blocked"}

        details.append({
            "id": case["id"],
            "type": case["type"],
            "split": case["split"],
            "question": case["question"],
            "status": status,
            "passed": passed,
            "answer": answer,
            "cited": check["cited"],
            "invalid": check["invalid"],
            "gold_tips": case["gold_tips"],
        })
        print(f"  {'✓' if passed else '✗'} [{case['split'][:4]:<4} {case['type']:<7}] {status:<9} {case['question']}")
    return details


def summarize_behavior(details: list[dict]) -> dict:
    summary = {}
    for split in SPLITS:
        rows = [d for d in details if d["split"] == split]
        answered = [d for d in rows if d["status"] in {"answered", "partial"}]
        summary[split] = {
            "in_ok": sum(1 for d in rows if d["type"] == "in" and d["passed"]),
            "in_total": sum(1 for d in rows if d["type"] == "in"),
            "out_ok": sum(1 for d in rows if d["type"] == "out" and d["passed"]),
            "out_total": sum(1 for d in rows if d["type"] == "out"),
            "partial_ok": sum(1 for d in rows if d["type"] == "partial" and d["passed"]),
            "partial_total": sum(1 for d in rows if d["type"] == "partial"),
            "partial_as_partial": sum(1 for d in rows if d["type"] == "partial" and d["status"] == "partial"),
            "citation_ok": sum(1 for d in answered if d["cited"] and not d["invalid"]),
            "answered_total": len(answered),
            "api_blocked": sum(1 for d in rows if d["status"] == "api_blocked"),
        }
    return summary


def print_behavior(summary: dict) -> None:
    for split, s in summary.items():
        print(
            f"\n  [{split}] 资料内未误拒 {percent(s['in_ok'], s['in_total'])}"
            f" | 资料外拒答 {percent(s['out_ok'], s['out_total'])}"
            f" | 部分覆盖 {percent(s['partial_ok'], s['partial_total'])}"
            f"（其中按部分回答 {s['partial_as_partial']} 条）"
            f"\n           引用有效 {percent(s['citation_ok'], s['answered_total'])}"
            f" | 接口拦截 {s['api_blocked']}"
        )


# ---------------- LLM 评分 ----------------
def judge_one(case: dict, answer: str) -> tuple[int | None, str]:
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    prompt = (
        f"【问题】\n{case['question']}\n\n"
        f"【标准答案】\n{case['gold_answer']}\n\n"
        f"【模型回答】\n{answer}"
    )
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    text = response.choices[0].message.content
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None, f"评分输出无法解析：{text[:60]}"
    try:
        data = json.loads(match.group(0))
        return int(data["score"]), str(data.get("reason", ""))
    except (json.JSONDecodeError, KeyError, ValueError):
        return None, f"评分输出无法解析：{text[:60]}"


def eval_judge(cases: list[dict], details: list[dict]) -> list[dict]:
    by_id = {d["id"]: d for d in details}
    scores = []
    for case in cases:
        detail = by_id[case["id"]]
        if case["type"] != "in" or not case["gold_answer"]:
            continue
        if detail["status"] not in {"answered", "partial"}:
            scores.append({"id": case["id"], "split": case["split"], "score": 0, "reason": "资料内问题被拒答"})
            continue

        try:
            score, reason = judge_one(case, detail["answer"])
        except Exception as exc:
            score, reason = None, f"评分接口错误：{type(exc).__name__}"
        scores.append({"id": case["id"], "split": case["split"], "score": score, "reason": reason})
        print(f"  {case['id']:<12} {score if score is not None else '?'}  {reason}")
    return scores


def summarize_judge(scores: list[dict]) -> dict:
    summary = {}
    for split in SPLITS:
        valid = [s["score"] for s in scores if s["split"] == split and s["score"] is not None]
        summary[split] = {
            "n": len(valid),
            "mean": sum(valid) / len(valid) if valid else 0.0,
            "exact": sum(1 for s in valid if s == 2),
            "partial": sum(1 for s in valid if s == 1),
            "wrong": sum(1 for s in valid if s == 0),
        }
    return summary


def print_judge(summary: dict) -> None:
    for split, s in summary.items():
        print(f"  [{split}] 平均 {s['mean']:.2f}/2  完全正确 {s['exact']}  部分正确 {s['partial']}  错误 {s['wrong']}")


# ---------------- 主流程 ----------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", action="store_true", help="跑 50 条问答")
    parser.add_argument("--judge", action="store_true", help="用 LLM 打分（隐含 --answers）")
    parser.add_argument("--judge-only", action="store_true", help="只重跑评分，复用上次报告里的回答")
    parser.add_argument("--rerank", action="store_true", help="检索/问答都改用重排")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条，调试用")
    parser.add_argument("--top", type=int, default=rag_answer.DEFAULT_TOP_K, help="问答取前几条资料")
    parser.add_argument("--out", default=str(BUILD_DIR / "eval_report.json"))
    args = parser.parse_args()

    if args.judge:
        args.answers = True

    cases = load_questions(args.limit)
    report_path = Path(args.out)

    if args.judge_only:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        details = report.get("answers", {}).get("details")
        if not details:
            print("报告里没有回答记录，先跑 --answers")
            return
        print("只重跑 LLM 评分（复用上次的回答）……")
        scores = eval_judge(cases, details)
        report["judge"] = {"details": scores, "by_split": summarize_judge(scores)}
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print()
        print_judge(report["judge"]["by_split"])
        print(f"\n报告已更新：{report_path}")
        return

    print("=" * 68)
    print(f"评测：{len(cases)} 条（dev {sum(1 for c in cases if c['split'] == 'dev')}，"
          f"holdout {sum(1 for c in cases if c['split'] == 'holdout')}）"
          f"｜检索器：{'rerank' if args.rerank else 'hybrid'}")
    print("=" * 68)

    bm25 = bm25_search.load_index()
    vector = vector_search.load_index()
    hybrid = hybrid_search.HybridRetriever(bm25, vector)
    retriever = rerank_search.RerankRetriever(hybrid) if args.rerank else hybrid

    methods = [("BM25", bm25.search), ("向量", vector.search), ("混合", hybrid.search)]
    if args.rerank:
        methods.append(("重排", retriever.search))

    print("\n[1/3] 检索指标")
    retrieval = {}
    for split in SPLITS:
        split_cases = [c for c in cases if c["split"] == split and c["type"] == "in"]
        if split_cases:
            retrieval[split] = eval_retrieval(split_cases, methods)
    print_retrieval(retrieval)

    report = {
        "meta": {
            "n": len(cases),
            "top_k": args.top,
            "retriever": "rerank" if args.rerank else "hybrid",
            "chat_model": CHAT_MODEL,
            "chunks": len(bm25.chunks),
        },
        "retrieval": {
            split: {name: {k: v for k, v in m.items() if k != "details"} for name, m in methods_rep.items()}
            for split, methods_rep in retrieval.items()
        },
        "retrieval_details": {split: {name: m["details"] for name, m in methods_rep.items()}
                              for split, methods_rep in retrieval.items()},
    }

    if args.answers:
        print("\n[2/3] 回答行为")
        details = eval_behavior(cases, retriever, args.top)
        summary = summarize_behavior(details)
        print_behavior(summary)
        report["answers"] = {"by_split": summary, "details": details}

        if args.judge:
            print("\n[3/3] LLM 评分")
            scores = eval_judge(cases, details)
            report["judge"] = {"details": scores, "by_split": summarize_judge(scores)}
            print()
            print_judge(report["judge"]["by_split"])
    else:
        print("\n（跳过回答行为评测，加 --answers 运行）")

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n报告已保存：{report_path}")


if __name__ == "__main__":
    main()