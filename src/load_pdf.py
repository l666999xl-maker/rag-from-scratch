import json
import re
from collections import Counter
from pathlib import Path

import pdfplumber

# 从你的配置文件里导入路径
# 注意：如果你在本地直接运行报错，请使用 python -m src.step2_extract 运行
from config import BUILD_DIR, PDF_DIR

# ---------------- 常量配置 ----------------
HEADER_RATIO = 0.5      # 出现在 50% 以上页面的短行，判定为页眉/页脚
HEADER_MAX_LEN = 40     # 页眉页脚通常不会太长，限制在 40 字符以内，防止误删正文
MIN_MEANINGFUL_CHARS = 5 # 只要包含 5 个以上的中文字符或英文字母，就不算空白页


# ---------------- 核心函数 ----------------
def strip_repeated_lines(pages: list[str]) -> list[str]:
    """去除页眉和页脚"""
    if len(pages) < 3:
        return pages

    counter = Counter()
    for text in pages:
        # 使用 set 对单页内去重：如果同一页里同一行出现 3 次，只算作 1 次
        for line in {ln.strip() for ln in text.splitlines() if ln.strip()}:
            counter[line] += 1

    # 计算阈值：比如 40 页的文件，阈值是 20 页
    threshold = max(2, int(len(pages) * HEADER_RATIO))
    
    # 找出高频且较短的重复行
    repeated = {
        line for line, count in counter.items()
        if count >= threshold and len(line) <= HEADER_MAX_LEN
    }
    
    if not repeated:
        return pages

    # 将识别出的页眉页脚从原文本中剔除
    return [
        "\n".join(ln for ln in text.splitlines() if ln.strip() not in repeated)
        for text in pages
    ]


def is_meaningful(text: str) -> bool:
    """
    判断这页是否包含有意义的信息。
    防止把只有页码、纯标点符号、或者解析乱码的页面存入库中。
    """
    stripped = text.strip()
    if not stripped:
        return False
    
    # 用正则提取所有的汉字、英文字母和数字
    # \u4e00-\u9fa5 是常用汉字的 Unicode 范围
    meaningful_chars = re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]', stripped)
    
    # 如果有效字符数量少于阈值，说明大概率是排版垃圾
    if len(meaningful_chars) < MIN_MEANINGFUL_CHARS:
        return False
        
    return True


def extract_pdf(path: Path) -> list[str]:
    """提取单个 PDF 的文本，并自动清洗页眉页脚"""
    with pdfplumber.open(str(path)) as pdf:
        # or "" 是为了防止 extract_text() 返回 None 导致后续报错
        pages = [(page.extract_text() or "") for page in pdf.pages]
    
    # 提取完后，立刻进行页眉页脚清洗
    return strip_repeated_lines(pages)


# ---------------- 主流程 ----------------
def main() -> None:
    BUILD_DIR.mkdir(exist_ok=True)

    rows = []
    print("=" * 60)
    print(f"{'文件':<36}{'页数':>6}{'有效页':>7}")
    print("-" * 60)

    # 遍历 PDF 目录下所有的 PDF 文件
    for pdf in sorted(PDF_DIR.glob("*.pdf")):
        pages = extract_pdf(pdf)
        valid = 0
        
        for page_no, text in enumerate(pages, 1):
            # 使用我们的自定义函数进行判断
            if not is_meaningful(text):
                # 如果你想看看到底跳过了哪些页，把下面这行注释打开
                # print(f"[跳过] {pdf.name} 第 {page_no} 页，内容过短：{repr(text.strip()[:20])}")
                continue
            
            valid += 1
            # 将源文件名、页码、以及干净的文本保存为字典
            rows.append({
                "source": pdf.name,
                "page": page_no,
                "text": text.strip(),
            })
            
        print(f"{pdf.name:<36}{len(pages):>6}{valid:>7}")

    # 将结果写入 JSON 文件
    (BUILD_DIR / "pages.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )

    # 统计信息
    total_chars = sum(len(r["text"]) for r in rows)
    print("-" * 60)
    print(f"共 {len(rows)} 页有效，{total_chars:,} 字")
    print(f"已保存：{BUILD_DIR / 'pages.json'}")


if __name__ == "__main__":
    main()