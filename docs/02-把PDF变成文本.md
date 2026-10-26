# 第 2 课：把 PDF 变成纯文本

## 这一课结束你会得到什么

- `build/pages.json`：6 份 PDF 的每一页文字
- 理解"PDF 不等于文本"这件事，以及页眉页脚为什么必须删

---

## 先理解三件事

### 1. PDF 不是文本文件

`.pdf` 里存的是**排版指令**：在哪个坐标画什么字形。
一个汉字在 PDF 里可能是"用 3 号字体在 (120, 450) 画出字形编号 3821"。

所以要用工具把它还原成文字——`pdfplumber` 就是干这个的。

### 2. 页眉页脚会污染每一块

政策文件每一页顶部都有：

```
深圳市人民政府公报 2023 年第 35 期 3
```

如果你不删，后面切成 233 个块，**每个块里都有这串字**。
结果：搜任何问题，这串字都可能让某些块"看起来相关"，浪费检索名额。

### 3. 有些 PDF 根本提取不出来

如果 PDF 用了自定义字体、又没嵌入字符映射表，提取出来会是：

```
/G45/G84/GA2/G3F/G40
```

这是**文件本身的问题**，不是你的代码问题。这种情况只能 OCR。
本项目的 6 份文件都可以正常提取。

---

## 动手

把 `src/step2_extract.py` 的内容换成下面这段。

### 第一步：导入

```python
import json
from collections import Counter

import pdfplumber

from src.config import BUILD_DIR, PDF_DIR
```

- `Counter` 用来统计"每一行出现了几次"
- `BUILD_DIR` 是中间产物目录

### 第二步：常量

```python
HEADER_RATIO = 0.5      # 出现在一半以上页面的短行，判定为页眉页脚
HEADER_MAX_LEN = 40
MIN_PAGE_CHARS = 30     # 少于这个字数视为空白页
```

**为什么加长度限制**：正文里也可能反复出现某句话，
但通常比较长。只把**短且高频**的行当页眉。

### 第三步：删重复行

```python
def strip_repeated_lines(pages: list[str]) -> list[str]:
    if len(pages) < 3:
        return pages

    counter: Counter[str] = Counter()
    for text in pages:
        # 用 set 去重，避免一页内重复的行被算多次
        for line in {ln.strip() for ln in text.splitlines() if ln.strip()}:
            counter[line] += 1

    threshold = max(2, int(len(pages) * HEADER_RATIO))
    repeated = {
        line for line, count in counter.items()
        if count >= threshold and len(line) <= HEADER_MAX_LEN
    }
    if not repeated:
        return pages

    return [
        "\n".join(ln for ln in text.splitlines() if ln.strip() not in repeated)
        for text in pages
    ]
```

**逐段解释**：

- `counter` 统计"这条线在多少**页**里出现过"。注意用 `set` 先对每页内部去重，
  否则一页里出现 3 次会被算成 3 页。
- `threshold` 是"至少多少页"。41 页的文件，阈值是 20。
- 同时要求 `len(line) <= 40`，避免误删正文。

### 第四步：提取一个 PDF

```python
def extract_pdf(path) -> list[str]:
    with pdfplumber.open(str(path)) as pdf:
        pages = [(page.extract_text() or "") for page in pdf.pages]
    return strip_repeated_lines(pages)
```

`or ""` 是为了处理提取不到文字的空页——`extract_text()` 可能返回 `None`。

### 第五步：主流程

```python
def main() -> None:
    BUILD_DIR.mkdir(exist_ok=True)

    rows = []
    print("=" * 60)
    print(f"{'文件':<36}{'页数':>6}{'有效页':>7}")
    print("-" * 60)

    for pdf in sorted(PDF_DIR.glob("*.pdf")):
        pages = extract_pdf(pdf)
        valid = 0
        for page_no, text in enumerate(pages, 1):
            if len(text.strip()) < MIN_PAGE_CHARS:
                continue
            valid += 1
            rows.append({
                "source": pdf.name,
                "page": page_no,
                "text": text.strip(),
            })
        print(f"{pdf.name:<36}{len(pages):>6}{valid:>7}")

    (BUILD_DIR / "pages.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )

    total_chars = sum(len(r["text"]) for r in rows)
    print("-" * 60)
    print(f"共 {len(rows)} 页有效，{total_chars:,} 字")
    print(f"已保存：{BUILD_DIR / 'pages.json'}")
```

**注意 `enumerate(pages, 1)`**：页码从 1 开始，和人类阅读习惯一致。
后面做引用时要显示"第 3 页"，从 0 开始会很别扭。

---

## 运行

```bash
cd e:/agent/agent_learn/project7-test-rag
python -m src.step2_extract
```

## 预期输出

```
============================================================
文件                                      页数    有效页
------------------------------------------------------------
lyg-appointment-cancel.pdf              43     37
ningyang-visit-notice.pdf                2      2
sysu-student-insurance.pdf              33     31
sz-longhua-cross-region.pdf             63     44
sz-longhua-medical.pdf                  63     43
sz-medical-insurance-2023.pdf           41     41
------------------------------------------------------------
共 198 页有效，72,308 字
已保存：...\build\pages.json
```

数字不完全一样没关系——**只要总数在 190~210 页、7 万字左右就行**。

---

## 如果报错

| 报错 | 原因 | 解决 |
|---|---|---|
| `No module named 'pdfplumber'` | 环境不对 | 用 rag-agent 环境，或 `pip install pdfplumber` |
| `FileNotFoundError: pages.json` | 没跑 step2 就想跑 step3 | 先跑 step2 |
| 输出里中文是乱码 | 终端编码问题 | 在命令前加 `chcp 65001`，或忽略（文件内容是对的） |
| 某份 PDF 提取出来全是符号 | 字体编码问题 | 本项目不会出现；如果出现，跳过这份 |

---

## 动手改一处

把 `HEADER_RATIO` 从 `0.5` 改成 `0.9`，重跑。

**预期**：阈值变高，删掉的行变少。然后打开 `pages.json`，
搜一下 `深圳市人民政府公报`，你会看到页眉又回来了。

再改成 `0.1`，重跑。

**预期**：阈值太低，可能把正文里重复出现的句子也删掉。

**这一改你要体会到**：阈值不是拍脑袋定的，它有明确的取舍——
**删少了污染检索，删多了丢失内容**。

---

## 自测

1. PDF 里存的是什么？为什么不能直接当文本读？
2. 为什么统计页眉时要先对每页内部去重？
3. 为什么要加 `len(line) <= 40` 这个限制？
4. `enumerate(pages, 1)` 和 `enumerate(pages)` 有什么区别？
5. 如果一份 PDF 提取出来是乱码，可能是什么原因？
