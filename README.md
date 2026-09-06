# 社交平台用户争议中的新闻传播教育合法性构建及价值重塑（算法透明与复现包）

本仓库由研究项目的 terminal v2.3 算法/代码包清洗而来，面向论文审稿、算法透明与结果审计。

## 公开范围

- 内容分析 Codebook 与关系编码 Codebook
- LLM 编码 Prompt 模板
- 主题模型、框架迁移、统计模型、关系编码、争议网络和能力可见性复算代码
- 可靠性、模型系数、稳健性、框架迁移矩阵、争议网络等**聚合结果**
- FASTopic dominant-topic 占比的定义、审计表和重算脚本
- 正式 GEE 模型规范以及历史 cluster-robust GLM 的区分说明
- 仅包含数据库字段结构的 public schema

## 明确不公开

本仓库**不包含任何原始平台数据或可回溯到单个用户/单条内容的行级数据**，包括：

- 原始/冻结 SQLite 数据库
- 笔记与评论原文、标题、URL、平台 ID、用户 ID / hash、IP 属地、精确时间戳
- LLM 全量标签 JSONL、语料 CSV、embedding 文件
- note-level 框架投影、reply-pair 明细、能力提及明细、人工审计样本
- 匿名化密钥、API 密钥、cookie/token、抓取日志

详见 `DATA_AVAILABILITY.md` 与 `SANITIZATION_REPORT.md`。

## 目录

```text
docs/                 方法、Codebook、Prompt、可靠性与证据映射
src/                  隐私清洗后的分析与复算代码
results_aggregate/    不含行级内容的聚合结果
requirements.txt      项目依赖
.env.example          私有数据根目录/API凭证的本地配置模板
```

## 本地运行

公开代码不会附带数据。需要复算时，在本地准备符合 `docs/data_schema_public.md` 的私有数据，并设置：

```bash
export NEWS_EDU_ROOT=/path/to/private/project_root
```

需要运行 DeepSeek 编码脚本时另设置：

```bash
export DEEPSEEK_API_KEY=YOUR_KEY
```

**不要将任何真实密钥提交到 Git。**

### Formal model policy

The manuscript-facing binary models use binomial GEE with an exchangeable
working correlation and `note_id` as the clustering unit. Count outcomes remain
negative-binomial models and are not relabeled as GEE. The formal GEE runner is
`src/reproducible/final_analysis/run_formal_gee_models.py`.

The files currently named `*_cluster_glm.csv` are historical aggregate outputs
from an earlier binary GLM pipeline. They are not GEE estimates. See
`docs/methods/final_model_policy.md` before using coefficient files in a paper.

### FASTopic percentages

The manuscript values 30.2% and 24.9% are dominant-topic shares:
`argmax(theta, axis=1)` followed by a count divided by 21,779 comments. The
archived audit table is
`results_aggregate/topics/fastopic_k10_dominant_topic_shares_legacy.csv`, and
the method is documented in `docs/methods/fastopic_dominant_shares.md`. Because
the original stage-5 retraining did not record a random seed, the package also
documents the distinction between the archived manuscript run and a new
deterministic recomputation in
`results_aggregate/topics/FASTopic_REPRODUCIBILITY_NOTE.md`.

## 版本

- 研究整合版本：terminal v2.3
- 清洗包：GitHub public-safe v1
- 原始数据：受控保存，不随仓库公开
- 本公开包版本：v2.3 transparency supplement 2026-09-05

## 方法透明边界

代码和聚合结果可公开审计，但由于平台用户生成内容的隐私与再识别风险，本仓库不承诺从原始文本层面的完全公开复现。读者可以审计模型定义、参数、编码规则、聚合结果及复算逻辑。

在正式公开前，请阅读 `RELEASE_CHECKLIST.md`。
