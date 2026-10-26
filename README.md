# 高性价比人生指南 · RAG 问答系统

把一本 589 页的中文书变成**可溯源、可拒答、可量化**的问答系统。

全流程手写，不依赖 LangChain——切块、BM25、向量检索、加权 RRF 融合、重排、引用校验、
评测指标（R@1/@3、MRR、nDCG）全部自己实现，所以每一层都能单独替换、单独评测。

## 结果一览

### 检索

50 条评测问题，`dev` 30 条 / `holdout` 20 条；下表是其中的资料内问题（dev 20、holdout 12）。

| 检索方式 | dev R@1 | dev R@3 | dev nDCG@3 | holdout R@1 | holdout nDCG@3 |
|---|---|---|---|---|---|
| BM25 关键词 | 80.0% | 100% | .900 | 91.7% | .969 |
| 向量 | 100% | 100% | 1.000 | 100% | 1.000 |
| 混合（加权 RRF） | 100% | 100% | .996 | 100% | 1.000 |
| **混合 + 重排** | 100% | 100% | **1.000** | 100% | 1.000 |

BM25 建索引约 2 秒、单词查询约 1 毫秒；向量检索约 0.35 秒；重排每次查询多 0.7–1.1 秒。

### 回答行为

| | dev | holdout |
|---|---|---|
| 资料内未误拒 | 20/20 | 12/12 |
| 资料外正确拒答 | 6/6 | 2/2 |
| 部分覆盖（答已知 + 说明缺什么） | 4/4 | 6/6 |
| 引用编号有效 | 22/22 | 18/18 |
| 被模型接口内容审核拦截 | 0 | 0 |

### 回答质量

LLM 按人工标准答案评分（2 分制）：

| | 平均分 | 完全正确 | 部分正确 | 错误 |
|---|---|---|---|---|
| dev | 1.90 | 18 | 2 | 0 |
| holdout | 1.58 | 7 | 5 | 0 |

dev 与 holdout 的差距是**刻意保留的**：调参只看 dev，holdout 到最后一轮才跑，用来暴露过拟合。

## 系统结构

```
HowToLiveBetter.pdf（589 页，34 节，767 条编号建议）
        │  load_pdf.py            pdfplumber 逐页提取
        ▼
   pages.json
        │  chunk_text.py          一条建议一块；MAX_CHUNK_CHARS=1000；参考文献剥离出 text
        ▼
   chunks.json（1017 块）
        │
        ├──► bm25_search.py       jieba lcut_for_search + BM25Okapi + 按条去重      ~1 ms
        ├──► vector_search.py     DashScope 1024 维嵌入 + 缓存（带 chunks 指纹）  ~0.35 s
        │
        └──► hybrid_search.py     加权 RRF 融合（RRF_K=10，BM25 : 向量 = 1 : 3）
                    │
                    ▼
            rerank_search.py      取 20 条候选 → qwen3.7-text-rerank → 失败自动退回混合排序   ~1.1 s
                    │
                    ▼
            rag_answer.py         三层护栏：检索为空直接拒答 / prompt 强制标注出处 / 代码校验引用编号
                    │
                    ▼
            答案 + [编号] 出处（或拒答 / 部分回答）
                    │
                    ▼
            evaluate.py           50 条 × dev/holdout ×（R@1/R@3/MRR/nDCG + 行为 + 引用 + LLM 评分）
```

| 模块 | 职责 |
|---|---|
| `src/config.py` | 读 `.env`，集中配置与目录 |
| `src/load_pdf.py` | PDF → 逐页文本 |
| `src/chunk_text.py` | 逐页文本 → 结构化块（`kind`/`section_no`/`item_no`/`title`/`text`/`refs`）|
| `src/bm25_search.py` | 关键词检索 |
| `src/vector_search.py` | 向量检索（含嵌入缓存与失效检测）|
| `src/hybrid_search.py` | 加权 RRF 混合检索 |
| `src/rerank_search.py` | 候选重排（原生 HTTP 调用，无额外依赖）|
| `src/rag_answer.py` | 生成 + 引用校验 + 拒答/部分回答 |
| `src/evaluate.py` | 分层评测，产出 `build/eval_report.json` |
| `data/eval/questions.json` | 50 条评测集（含标准答案与标准条目）|

## 快速开始

### 环境

Python 3.11。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

在项目根目录建 `.env`：

```ini
DEEPSEEK_API_KEY=...          # 对话模型
DEEPSEEK_BASE_URL=...
DASHSCOPE_API_KEY=...         # 嵌入 + 重排
EMBEDDING_BASE_URL=...
EMBEDDING_MODEL=...
```

### 准备语料

把 PDF 放到 `data/raw/pdf/HowToLiveBetter.pdf`。

> 该 PDF 是正式出版物，**本仓库不包含**（见 `.gitignore`）。要跑通请自备语料；
> 换别的书也能跑，但 `chunk_text.py` 的切块规则是按本书的"标题 + • 成本 / • 说人话 /
> • 收益 / • 证据等级 / ‣ 来源"固定结构写的，换书需要改规则。

### 跑

```powershell
# 建索引（pages.json → chunks.json）
.\.venv\Scripts\python.exe src\load_pdf.py
.\.venv\Scripts\python.exe src\chunk_text.py

# 单路检索
.\.venv\Scripts\python.exe src\bm25_search.py "桶装水怎么挑"
.\.venv\Scripts\python.exe src\vector_search.py "裁员了能拿多少赔偿"   # 首次运行会建嵌入缓存
.\.venv\Scripts\python.exe src\hybrid_search.py "药和水果能一起吃吗"
.\.venv\Scripts\python.exe src\rerank_search.py "身份证丢了怎么办"

# 完整问答（加 --context 打印送给模型的资料，--refs 打印参考文献）
.\.venv\Scripts\python.exe src\rag_answer.py "桶装水怎么挑" --context

# 评测
.\.venv\Scripts\python.exe src\evaluate.py --answers --judge --rerank
```

脚本用 `from config import ...`，所以**必须在项目根目录直接运行**（不能用 `python -m src.xxx`）。

## 评测方法

- **检索指标**：R@1 / R@3 看有没有召回；MRR@3 看第一个命中排多前；nDCG@3 看多个可接受答案的排序质量。
  命中判定按 `(节号, 条号)`，不是按块 id——同一条建议被拆成多块时不算多次命中。
- **回答行为**：用代码判定 `status`（`answered` / `partial` / `refused` / `blocked` / `api_blocked`），
  再看是否与问题的期望类型一致（资料内不能误拒、资料外不能硬答）。
- **引用有效性**：解析回答里的 `[编号]`，校验编号是否落在送给模型的资料范围内——**编造出处会被抓到**。
- **回答质量**：另一路 LLM 按人工标准答案打 0/1/2 分。
- **dev / holdout 分组**：所有检索参数只在 dev 上调，holdout 只在最后跑一次。

## 五个关键设计

1. **切块按语义，不按字数。** 一条建议一块，`MAX_CHUNK_CHARS=1000`（767 条中 138 条超长被拆）。
   参考文献单独放进 `refs` 字段，不混进 `text`——否则参考文献里的数字会干扰 BM25 打分。
2. **加权 RRF 而不是简单加权求和。** 两路检索的分数尺度不可比，RRF 只用排名，天然免归一化。
   权重 BM25 : 向量 = 1 : 3 是在 14 条对照问题上调出来的：语义类问题（"药和水果"、"消炎药 + 酒"）
   朴素 1:1 会翻车。
3. **引用校验放在代码里，不靠 prompt。** prompt 只能"要求"模型标出处；能不能标对由
   `check_answer()` 判定，不合规重试一次，再不合规降级为拒答。护栏必须是可被代码检测的。
4. **拒答、部分回答是三种状态，不是两种。** 资料只覆盖一部分时，先答已知部分，再另起一行写
   「资料里没有提到：缺什么」——比整段拒答更有用，也比硬答安全。
5. **嵌入缓带有 chunks 指纹。** `chunks.json` 一变，缓存自动失效重建，避免"改了切块却用着旧向量"
   这种静默错误。

## 踩过的坑

这两个 bug 都是评测抓出来的，不是看代码看出来的：

- **正则把数字误判成条号。** `TIP_START_RE` 原本是 `^(\d+)\.\s*`，于是正文里的"5.68 倍"
  被判成"第 5 条"开头，块边界全乱。改成 `^(\d+)\.\s+`（点后必须有空格）后，块数 1015 → 1017。
- **评测用例自己写错了。** `h-out-02「怎么写一份求职简历」`原本被标成"资料外"，但书里第 5 节
  第 14 条讲的就是求职方法——**是用例错了，不是系统错了**。改成 `h-partial-06` 重算后，
  未通过用例从 1 条变成 0 条。教训：指标不好看时，先怀疑用例。

## 已知限制

1. **语料小且条目具体**，所以重排的收益有限（nDCG .996 → 1.000）。换到十万块量级的语料上结论可能完全不同。
2. **holdout 只有 20 条**，一条对错就是 8 个百分点，holdout 的分数波动很大。
3. **LLM 评分有噪声**，`h-in-10` 被判"表述有偏差"，可能真是模型不精确，也可能是评分员太挑剔。
4. **重排每次多 0.7–1.1 秒、多一次计费**，高频场景要算成本。
5. **评测集有来源偏差**：多数问题是从条目标题改写的，天然偏向"能被检索到"。
6. **切块强依赖本书的固定结构**，换语料要重写规则。