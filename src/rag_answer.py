"""
RAG 问答：混合检索 + DeepSeek 生成 + 引用/拒答校验。

只靠 prompt 约束模型是不够的，所以这里做两层：
1. prompt 明确要求“资料里没有就拒答，并用 [编号] 标注出处”
2. 代码检查回答有没有引用、引用编号是否真实存在；不合规就重试一次，再不合规就降级拒答
"""

import re
import sys

from openai import BadRequestError, OpenAI

import hybrid_search
from config import CHAT_MODEL, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

sys.stdout.reconfigure(encoding="utf-8")

# ---------------- 提示词 ----------------
SYSTEM_PROMPT = """你是《高性价比人生指南》的问答助手。你只能依据【资料】回答，不能使用你自己的知识。

硬性规则：
1. 先判断【资料】里有没有回答问题所需的信息。
2. 如果资料里完全没有，只回复一句：「资料里没有提到这个问题。」不要猜，不要补充常识，不要编数字、法条或出处。
3. 如果资料只回答了一部分：先把资料里有的部分答完，每个结论用 [编号] 标注出处；然后另起一行写「资料里没有提到：」再加上缺的那部分。不要因为缺一部分就整段拒答。
4. 如果资料里有完整答案：用资料里的原话和数字回答，每个关键结论后面用 [编号] 标注出处，例如 [1][3]。
5. 只能引用资料里出现过的编号，不能编造编号。
6. 不要给资料之外的医疗、法律、投资建议。
7. 中文回答，先结论后细节，尽量简短。"""

RETRY_PROMPT = """你上一次的回答没有按规则标注出处。请重新回答：
- 资料里有答案：用 [编号] 标注每个关键结论；
- 资料里没有：只回复「资料里没有提到这个问题。」"""

FALLBACK_ANSWER = "资料里没有找到可以核实的依据，无法回答这个问题。"

CITE_RE = re.compile(r"[\[【]\s*(\d+(?:\s*[,，、]\s*\d+)*)\s*[\]】]")
REFUSAL_RE = re.compile(r"(资料|材料|书中|本文).{0,12}(没有|未|找不到|不涉及|没提)|无法回答|不能回答")

DEFAULT_TOP_K = 5
MAX_RETRIES = 1


# ---------------- 工具函数 ----------------
def get_client() -> OpenAI:
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


def position_of(item: dict) -> str:
    if item["kind"] == "tip":
        return f"第 {item['section_no']} 节第 {item['item_no']} 条"
    if item["kind"] == "intro":
        return f"第 {item['section_no']} 节 · 导语"
    if item["kind"] == "essay":
        return f"附录 · {item['section']}"
    return item["section"]


def build_context(hits: list[dict]) -> str:
    parts = []
    for i, hit in enumerate(hits, 1):
        parts.append(f"[{i}] {position_of(hit)} | 第 {hit['page']} 页\n{hit['text']}")
    return "\n\n".join(parts)


def extract_citations(answer: str) -> list[int]:
    numbers: list[int] = []
    for group in CITE_RE.findall(answer):
        for piece in re.split(r"[,，、]", group):
            piece = piece.strip()
            if piece.isdigit():
                numbers.append(int(piece))
    return list(dict.fromkeys(numbers))


def check_answer(answer: str, hits: list[dict]) -> dict:
    cited = extract_citations(answer)
    valid = [n for n in cited if 1 <= n <= len(hits)]
    invalid = [n for n in cited if not (1 <= n <= len(hits))]
    said_missing = bool(REFUSAL_RE.search(answer))

    if valid and said_missing:
        status = "partial"     # 答了资料里有的部分，也说明缺了什么
    elif valid:
        status = "answered"
    elif said_missing:
        status = "refused"
    else:
        status = "unverified"

    return {
        "cited": cited,
        "valid": valid,
        "invalid": invalid,
        "refused": said_missing and not valid,
        "status": status,
    }


# ---------------- 问答 ----------------
def ask(retriever: hybrid_search.HybridRetriever, query: str, top_k: int = DEFAULT_TOP_K) -> dict:
    hits = retriever.search(query, top_k)

    if not hits:
        return {
            "query": query,
            "answer": FALLBACK_ANSWER,
            "hits": [],
            "check": {"cited": [], "valid": [], "invalid": [], "refused": True, "status": "refused"},
        }

    client = get_client()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"【资料】\n{build_context(hits)}\n\n【问题】\n{query}"},
    ]

    answer = ""
    check: dict = {}
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                temperature=0,
            )
        except BadRequestError as exc:
            # 接口侧的内容审核偶尔会拦下合法的法律/医疗段落，返回可读提示而不是崩掉
            if "data_inspection_failed" in str(exc):
                return {
                    "query": query,
                    "answer": "这条内容被模型接口的内容审核拦截了，换个问法再试。",
                    "hits": hits,
                    "check": {
                        "cited": [], "valid": [], "invalid": [], "refused": False,
                        "status": "api_blocked",
                    },
                }
            raise

        answer = response.choices[0].message.content.strip()
        check = check_answer(answer, hits)

        if check["status"] != "unverified":
            break
        if attempt < MAX_RETRIES:
            messages.append({"role": "assistant", "content": answer})
            messages.append({"role": "user", "content": RETRY_PROMPT})

    if check["status"] == "unverified":
        answer = FALLBACK_ANSWER
        check["status"] = "blocked"

    return {"query": query, "answer": answer, "hits": hits, "check": check}


# ---------------- 展示 ----------------
def show(result: dict, show_context: bool = False, show_refs: bool = False) -> None:
    hits = result["hits"]
    check = result["check"]

    print("=" * 64)
    print(f"问：{result['query']}")
    print("=" * 64)
    print(result["answer"])

    if show_context:
        print("\n" + "-" * 64)
        print("送给模型的资料：")
        print(build_context(hits))

    print("\n" + "-" * 64)
    print(f"检索到 {len(hits)} 条：")
    for i, hit in enumerate(hits, 1):
        title = hit["title"] or hit["section"]
        print(f"  [{i}] {position_of(hit)} | 第 {hit['page']} 页 | {title[:40]}")

    print(f"引用编号：{check['cited'] or '（无）'}")
    print(f"无效编号：{check['invalid'] or '无'}")
    labels = {
        "answered": "通过",
        "partial": "部分回答（已说明缺什么）",
        "refused": "拒答",
        "blocked": "已降级为拒答",
        "api_blocked": "接口拦截",
    }
    print(f"校验结果：{labels.get(check['status'], check['status'])}")

    if show_refs and check["valid"]:
        print("\n参考文献：")
        for number in check["valid"]:
            for ref in hits[number - 1]["refs"][:3]:
                print(f"  [{number}] {ref}")


def main() -> None:
    show_context = "--context" in sys.argv
    show_refs = "--refs" in sys.argv
    top_k = DEFAULT_TOP_K
    if "--top" in sys.argv:
        index = sys.argv.index("--top")
        try:
            top_k = int(sys.argv[index + 1])
            del sys.argv[index:index + 2]
        except (IndexError, ValueError):
            print("用法：--top 5")
            return

    args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    retriever = hybrid_search.load_retriever()

    if args:
        show(ask(retriever, " ".join(args), top_k), show_context, show_refs)
        return

    print("\nRAG 问答。输入 q 退出。")
    while True:
        try:
            query = input("\n问题 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query.lower() in {"q", "quit", "exit"}:
            break
        show(ask(retriever, query, top_k), show_context, show_refs)


if __name__ == "__main__":
    main()
