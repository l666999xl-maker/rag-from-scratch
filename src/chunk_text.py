"""
把 pages.json 切成可检索的块。

这本书的结构不是“一段一段的正文”，而是：
    每节 = 若干条编号建议（1. 2. 3. …）
    每条 = 标题 + • 成本 / • 说人话 / • 收益 / 证据等级 / ‣ 来源 / • 备注

所以切块的正确粒度是“一条建议 = 一块”，不是“500 字 = 一块”。
切分顺序：先还原被 PDF 折行的段落，再按条目标题切，最后才处理超长条目。
"""

import json
import re
import sys

from config import BUILD_DIR

# Windows 控制台默认是 GBK，打印中文会乱码
sys.stdout.reconfigure(encoding="utf-8")

# ---------------- 常量配置 ----------------
MAX_CHUNK_CHARS = 1000   # 单块上限。超过就在段落边界拆开，拆出的每块都重复条目标题
MIN_CHUNK_CHARS = 60     # 过滤掉前言里“许可”这类没有检索价值的碎片
MIN_PART_CHARS = 250     # 拆条时，末尾不足这么多字的分段并回前一块，避免出现半截碎块

HEADER_PREFIX = "高性价比人生指南"   # 每页页眉：高性价比人生指南 <节名>
PAGE_NO_RE = re.compile(r"^\d+\s*/\s*\d+$")          # 页脚：177 / 586
TOC_LINE_RE = re.compile(r"[.…]{6,}")               # 目录行：标题 ........ 页码
TIP_START_RE = re.compile(r"^(\d+)\.\s+(.*)$")       # 条目标题：14. 求职别只投简历…（点后必须有空格）
SECTION_NO_RE = re.compile(r"^(\d+)\.\s")           # 节名：5. 不要浪费钱
BULLET_RE = re.compile(r"^[•‣]")                    # 条目里的小标题和来源行
BARE_LABEL_RE = re.compile(r"^[•‣]\s*[^：:]{1,8}[：:]\s*$")  # 单独一行的“• 收益：”
SENTENCE_END_RE = re.compile(r"[。！？；]$")
SOURCE_RE = re.compile(r"^[•‣]\s*来源")                     # “• 来源：”
URL_RE = re.compile(r"https?://")                          # 带网址的段落才算参考文献
# 附录长文里的小标题：“一、先把问题拆开”“头一周”“时间账：……”
HEADING_RE = re.compile(r"^[一二三四五六七八九十]+、|^第[一二三四五六七八九十]+[章节]")

FRONT_SECTIONS = {"前言"}       # 正文之前要保留的部分
SKIP_SECTIONS = {"各节简介"}    # 目录型的章节摘要，和正文重复，跳过


# ---------------- 小工具 ----------------
def is_ascii_alnum(ch: str) -> bool:
    return bool(ch) and ch.isascii() and ch.isalnum()


def has_open_url(text: str) -> bool:
    """行尾是不是一个没写完的 URL（PDF 折行会把 URL 从中间断开）。"""
    return bool(re.search(r"https?://\S*$", text[-400:]))


def is_reference(paragraph: str) -> bool:
    """
    判断一段是不是“参考文献”。
    本书里 ‣ 和 “• 来源：”既放参考文献，也放证据说明；
    只有带网址的才移出正文（网址会污染向量和 BM25），纯文字的证据说明留在正文里。
    """
    return bool((paragraph.startswith("‣") or SOURCE_RE.match(paragraph)) and URL_RE.search(paragraph))


def join_wrapped(prev: str, line: str) -> str:
    """把 PDF 折行后的两行接回一句，顺便处理英文单词之间被吃掉的那个空格。"""
    if is_ascii_alnum(prev[-1]) and is_ascii_alnum(line[0]) and not has_open_url(prev):
        return prev + " " + line
    return prev + line


def split_long_paragraph(text: str, limit: int) -> list[str]:
    """单段就超过上限时，按句号切开；实在切不动再按字数硬切。"""
    sentences = re.findall(r"[^。！？；]*[。！？；]?", text)
    sentences = [s for s in sentences if s]

    pieces: list[str] = []
    buffer = ""
    for sentence in sentences:
        while len(sentence) > limit:
            if buffer:
                pieces.append(buffer)
                buffer = ""
            pieces.append(sentence[:limit])
            sentence = sentence[limit:]
        if len(buffer) + len(sentence) <= limit:
            buffer += sentence
        else:
            if buffer:
                pieces.append(buffer)
            buffer = sentence
    if buffer:
        pieces.append(buffer)
    return pieces


def pack_paragraphs(paragraphs: list[tuple[int, str]], limit: int) -> list[list[tuple[int, str]]]:
    """把段落按顺序装进不超过 limit 的块里，段落本身绝对不拆（除非它自己就超长）。"""
    groups: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    current_len = 0

    for page, para in paragraphs:
        if len(para) > limit:
            if current:
                groups.append(current)
                current, current_len = [], 0
            for piece in split_long_paragraph(para, limit):
                groups.append([(page, piece)])
            continue

        if current and current_len + len(para) + 1 > limit:
            groups.append(current)
            current, current_len = [], 0

        current.append((page, para))
        current_len += len(para) + 1

    if current:
        groups.append(current)
    return groups


def merge_short_tails(groups: list[list[tuple[int, str]]]) -> list[list[tuple[int, str]]]:
    """拆条留下的短短一截（比如只剩一句备注），并回前一块，别单独成块。"""
    if len(groups) <= 1:
        return groups
    merged = [groups[0]]
    for group in groups[1:]:
        if sum(len(p) + 1 for _, p in group) < MIN_PART_CHARS:
            merged[-1].extend(group)
        else:
            merged.append(group)
    return merged


# ---------------- 清洗与配对 ----------------
def content_lines(page_text: str, section: str) -> list[str]:
    """去掉页眉、页脚、目录行，返回这一页真正的内容行。"""
    lines: list[str] = []
    for raw in page_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(HEADER_PREFIX):
            continue
        if PAGE_NO_RE.match(line):
            continue
        if TOC_LINE_RE.search(line):
            continue
        if line in {"许可"} or line.startswith("正文按 CC BY"):
            continue
        if line == section:          # 节标题单独一行，内容由每块的【节名】代替
            continue
        lines.append(line)
    return lines


def looks_like_heading(line: str) -> bool:
    """附录长文里的小标题：短、不带句末标点，常见形如“一、xxx”“时间账：xxx”。"""
    if BULLET_RE.match(line) or line.startswith("http"):
        return False
    if HEADING_RE.match(line):
        return True
    return len(line) <= 30 and "：" in line and not re.search(r"[。！？；，、]$", line)


def build_paragraphs(lines: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """把折行的内容还原成逻辑段落。bullet 行、小标题、上一段已结束，都表示新段落。"""
    paragraphs: list[tuple[int, str]] = []

    for page, line in lines:
        if not paragraphs:
            paragraphs.append((page, line))
            continue

        prev_page, prev = paragraphs[-1]
        if looks_like_heading(prev):
            paragraphs.append((page, line))
        elif BARE_LABEL_RE.match(prev) and BULLET_RE.match(line):
            # “• 收益：”单独一行，下一行才是内容，接在一起
            paragraphs[-1] = (prev_page, prev + line.lstrip("•‣ ").strip())
        elif BULLET_RE.match(line) or looks_like_heading(line) or SENTENCE_END_RE.search(prev):
            paragraphs.append((page, line))
        else:
            paragraphs[-1] = (prev_page, join_wrapped(prev, line))

    return paragraphs


def assign_sections(pages: list[dict]) -> list[str | None]:
    """
    给每一页标上它属于哪一节。
    有页眉的页直接从页眉读；没页眉的页（新一节的第一页）看前后节名：
    前后不同说明这一页是新一节的开头，归后面那一节。
    """
    sections: list[str | None] = []
    for page in pages:
        first = next((ln.strip() for ln in page["text"].splitlines() if ln.strip()), "")
        if first.startswith(HEADER_PREFIX):
            sections.append(first[len(HEADER_PREFIX):].strip())
        else:
            sections.append(None)

    known = [(i, s) for i, s in enumerate(sections) if s]
    if not known:
        raise RuntimeError("pages.json 里找不到任何页眉，检查第 2 步的提取结果")

    for i, _ in enumerate(sections):
        if sections[i]:
            continue
        prev_s = next((s for j, s in reversed(known) if j < i), None)
        next_s = next((s for j, s in known if j > i), None)
        if next_s and next_s != prev_s:
            sections[i] = next_s
        else:
            sections[i] = prev_s or next_s

    return sections


# ---------------- 正文：按“条”切 ----------------
def extract_tips(pages: list[dict], sections: list[str]) -> tuple[list[dict], list[dict]]:
    """
    正文的切法：一条建议一块；每节开头那段导语也单独成一块。
    怎么认出条目标题：行首是 “数字. ”，而且数字正好是本节的下一条编号。
    这个“编号必须连续”的条件，能排除正文里偶尔出现的普通编号列表。
    """
    tips: list[dict] = []
    intros: list[dict] = []
    expected: dict[str, int] = {}
    current: dict | None = None
    pending_intro: list[tuple[int, str]] = []
    intro_section = ""

    def flush() -> None:
        nonlocal current
        if current:
            tips.append(current)
            current = None

    def flush_intro() -> None:
        nonlocal pending_intro
        if pending_intro:
            intros.append({
                "section": intro_section,
                "section_no": int(SECTION_NO_RE.match(intro_section).group(1)),
                "page_start": pending_intro[0][0],
                "page_end": pending_intro[-1][0],
                "paras": build_paragraphs(pending_intro),
            })
            pending_intro = []

    for page, section in zip(pages, sections):
        if not SECTION_NO_RE.match(section):
            continue

        if current and current["section"] != section:
            flush()

        for line in content_lines(page["text"], section):
            match = TIP_START_RE.match(line)
            number = int(match.group(1)) if match else -1
            is_tip_title = (
                match is not None
                and number == expected.get(section, 1)
                and bool(re.search(r"[一-龥]", match.group(2)))
                and not SENTENCE_END_RE.search(line)
            )

            if is_tip_title:
                if pending_intro:
                    flush_intro()
                flush()
                expected[section] = number + 1
                current = {
                    "section": section,
                    "section_no": int(SECTION_NO_RE.match(section).group(1)),
                    "item_no": number,
                    "page_start": page["page"],
                    "page_end": page["page"],
                    "paras": [(page["page"], line)],
                }
                continue

            if current is None:
                if not pending_intro:
                    intro_section = section
                pending_intro.append((page["page"], line))
                continue

            current["page_end"] = page["page"]
            paras = current["paras"]
            if BARE_LABEL_RE.match(paras[-1][1]) and BULLET_RE.match(line):
                p, text = paras[-1]
                paras[-1] = (p, text + line.lstrip("•‣ ").strip())
            elif BULLET_RE.match(line) or SENTENCE_END_RE.search(paras[-1][1]):
                paras.append((page["page"], line))
            else:
                p, text = paras[-1]
                paras[-1] = (p, join_wrapped(text, line))

    flush()
    flush_intro()

    # 编号连续性自检：哪一节漏了条，会在这里暴露
    for section, next_no in expected.items():
        got = [t["item_no"] for t in tips if t["section"] == section]
        missing = sorted(set(range(1, next_no)) - set(got))
        if missing:
            print(f"[警告] {section} 缺少条号：{missing}")

    return tips, intros


# ---------------- 前言 / 附录：按段落打包 ----------------
def extract_flow_chunks(
    pages: list[dict], sections: list[str], section_filter: set[str] | None, kind: str
) -> list[dict]:
    """前言、术语表、各节简介、附录长文没有“条”，按段落攒到上限为止。"""
    runs: list[tuple[str, list[tuple[int, str]]]] = []
    for page, section in zip(pages, sections):
        if section in SKIP_SECTIONS:
            continue
        if section_filter is None:
            if section in FRONT_SECTIONS or SECTION_NO_RE.match(section):
                continue
        elif section not in section_filter:
            continue
        if not runs or runs[-1][0] != section:
            runs.append((section, []))
        runs[-1][1].extend((page["page"], ln) for ln in content_lines(page["text"], section))

    chunks: list[dict] = []
    for section, lines in runs:
        for group in pack_paragraphs(build_paragraphs(lines), MAX_CHUNK_CHARS):
            text = "\n".join(p for _, p in group)
            if len(text) < MIN_CHUNK_CHARS:
                continue
            chunks.append({
                "kind": kind,
                "section": section,
                "page_start": group[0][0],
                "page_end": group[-1][0],
                "paras": group,
            })
    return chunks


# ---------------- 主流程 ----------------
def main() -> None:
    pages = json.loads((BUILD_DIR / "pages.json").read_text(encoding="utf-8"))
    source = pages[0]["source"]
    sections = assign_sections(pages)

    rows: list[dict] = []

    def push(kind: str, section: str, section_no: int | None, item_no: int | None,
             title: str | None, page_start: int, page_end: int, text: str, suffix: str,
             refs: list[str] | None = None) -> None:
        rows.append({
            "id": f"{source}#p{page_start}#{suffix}",
            "source": source,
            "page": page_start,
            "page_end": page_end,
            "kind": kind,
            "section": section,
            "section_no": section_no,
            "item_no": item_no,
            "title": title,
            "text": text,
            "refs": refs or [],
        })

    # 1. 正文：767 条建议，一条（或它的一个分段）一块
    tips, intros = extract_tips(pages, sections)
    split_tips = 0
    moved_refs = 0
    for tip in tips:
        title = tip["paras"][0][1]
        breadcrumb = f"【{tip['section']}】"
        body = tip["paras"][1:]
        refs = [p for _, p in body if is_reference(p)]
        content = [(pg, p) for pg, p in body if not is_reference(p)]
        moved_refs += len(refs)
        groups = pack_paragraphs(content, MAX_CHUNK_CHARS) if content else [[]]
        groups = merge_short_tails(groups)
        if len(groups) > 1:
            split_tips += 1
        for part, group in enumerate(groups, 1):
            text = breadcrumb + "\n" + title
            if group:
                text += "\n" + "\n".join(p for _, p in group)
            suffix = f"s{tip['section_no']:02d}-{tip['item_no']:02d}"
            if len(groups) > 1:
                suffix += f"-{part}"
            push("tip", tip["section"], tip["section_no"], tip["item_no"], title,
                 group[0][0] if group else tip["page_start"],
                 group[-1][0] if group else tip["page_end"], text, suffix, refs)

    # 1b. 每节开头的导语（“这一节讲三件事……”）
    for intro in intros:
        breadcrumb = f"【{intro['section']}】"
        for part, group in enumerate(pack_paragraphs(intro["paras"], MAX_CHUNK_CHARS), 1):
            text = breadcrumb + "\n" + "\n".join(p for _, p in group)
            suffix = f"intro-s{intro['section_no']:02d}"
            if part > 1:
                suffix += f"-{part}"
            push("intro", intro["section"], intro["section_no"], None, None,
                 group[0][0], group[-1][0], text, suffix)

    # 2. 前言、术语表、各节简介
    for i, chunk in enumerate(extract_flow_chunks(pages, sections, FRONT_SECTIONS, "front"), 1):
        text = f"【{chunk['section']}】\n" + "\n".join(p for _, p in chunk["paras"])
        push("front", chunk["section"], None, None, None,
             chunk["page_start"], chunk["page_end"], text, f"front-{i:02d}")

    # 3. 附录长文；附录的节名就是文章标题
    for i, chunk in enumerate(extract_flow_chunks(pages, sections, None, "essay"), 1):
        text = f"【附录 · {chunk['section']}】\n" + "\n".join(p for _, p in chunk["paras"])
        push("essay", chunk["section"], None, None, None,
             chunk["page_start"], chunk["page_end"], text, f"essay-{i:02d}")

    (BUILD_DIR / "chunks.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )

    # ---------------- 统计与自检 ----------------
    lengths = [len(r["text"]) for r in rows]
    print("=" * 60)
    print(f"切块完成：{len(rows)} 块")
    print(f"  正文条目 {sum(1 for r in rows if r['kind'] == 'tip')} 块"
          f"（{len(tips)} 条建议，其中 {split_tips} 条太长被拆成多块）")
    print(f"  移出正文的参考文献 {moved_refs} 段（留在每块的 refs 字段里，不进检索文本）")
    print(f"  各节导语 {sum(1 for r in rows if r['kind'] == 'intro')} 块")
    print(f"  前言/简介 {sum(1 for r in rows if r['kind'] == 'front')} 块")
    print(f"  附录长文 {sum(1 for r in rows if r['kind'] == 'essay')} 块")
    print(f"长度：最短 {min(lengths)} 字，中位 {sorted(lengths)[len(lengths) // 2]} 字，"
          f"最长 {max(lengths)} 字，平均 {sum(lengths) // len(lengths)} 字")

    leaked_page_no = [
        r["id"] for r in rows
        if any(PAGE_NO_RE.match(ln.strip()) for ln in r["text"].splitlines())
    ]
    leaked_header = [
        r["id"] for r in rows
        if any(ln.strip().startswith(HEADER_PREFIX) for ln in r["text"].splitlines())
    ]
    print(f"自检：残留页码 {len(leaked_page_no)} 块，残留页眉 {len(leaked_header)} 块")
    print("=" * 60)

    print("\n--- 样例 1：一条正常的建议 ---")
    print(rows[0]["text"][:500])
    print("\n--- 样例 2：一条被拆开的建议（如果存在）---")
    sample = next((r for r in rows if re.search(r"s\d{2}-\d{2}-[2-9]$", r["id"])), None)
    print(sample["text"][:500] if sample else "（没有超长条目）")
    print("\n输出：", BUILD_DIR / "chunks.json")


if __name__ == "__main__":
    main()
