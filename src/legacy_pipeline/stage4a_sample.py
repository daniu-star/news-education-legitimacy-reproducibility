# -*- coding: utf-8 -*-
"""
Stage 4a: 专家编码样本分层抽样
- 从21,998条语义有效评论中抽取400条
- 分层: comment_level×source_type + 关键词超采(AI/就业/劝退/反讽/作者回复)
- 输出: 结构化编码材料CSV + JSONL
"""
import os, sys, json, random, re
import sqlite3, pandas as pd
from collections import Counter

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(BASE, "analysis", "02_views", "analysis.db")
OUT_DIR = os.path.join(BASE, "报告", "04_LLM编码", "data")
os.makedirs(OUT_DIR, exist_ok=True)

random.seed(42)
np_rng = __import__('numpy').random.RandomState(42)

conn = sqlite3.connect(DB)

# ---- 1. 读取全部语义有效评论with语境 ----
comments = pd.read_sql("""
    SELECT c.comment_id, c.note_id, c.comment_level, c.is_note_author, c.source_type,
           c.content_clean, c.like_count_num, c.short_text_flag,
           n.note_title, p.content_clean AS parent_content
    FROM v_comment_base c
    LEFT JOIN (
        SELECT comment_id, content_clean FROM v_comment_base
    ) p ON c.parent_comment_id = p.comment_id
    LEFT JOIN (
        SELECT note_id, MAX(title) AS note_title FROM v_note_base GROUP BY note_id
    ) n ON c.note_id = n.note_id
    WHERE c.semantic_eligible = 1 AND c.is_orphan = 0
""", conn)
conn.close()
print(f"语义有效评论: {len(comments)}")

# ---- 2. 分层标记 ----
comments['has_ai'] = comments['content_clean'].str.contains('AI|ai|人工智能|ChatGPT|替代|写稿|智能', regex=True, na=False).astype(int)
comments['has_employment'] = comments['content_clean'].str.contains('就业|工作|实习|岗位|工资|收入|考公|考研|大厂|毕业|招聘', regex=True, na=False).astype(int)
comments['has_dissuade'] = comments['content_clean'].str.contains('劝退|别学|后悔|已死|垃圾|没用|不值得|张雪峰', regex=True, na=False).astype(int)
comments['short'] = (comments['short_text_flag'] == 1).astype(int)

# 分层key
comments['stratum'] = comments.apply(
    lambda r: f"L{r['comment_level']}_{r['source_type']}", axis=1)

strata = comments['stratum'].value_counts().to_dict()
print("分层基数:", strata)

# ---- 3. 抽样 ----
# 目标: 共400条
# 基础层: 每层80条 (L1_api + L1_browser + L2_api + L2_browser = 320)
# 专项超采: AI 30条 + 就业 20条 + 劝退 10条 + 作者回复 20条 = 80条
sampled = []

# 基础抽样
for stratum_name, n_pop in strata.items():
    pool = comments[comments['stratum'] == stratum_name]
    n = min(80, len(pool))
    idx = np_rng.choice(pool.index, n, replace=False)
    sampled.append(pool.loc[idx])

base = pd.concat(sampled).drop_duplicates('comment_id')
print(f"基础抽样(4×80): {len(base)}")

# 专项超采（从剩余池中抽取）
remaining = comments[~comments['comment_id'].isin(base['comment_id'])]

ai_pool = remaining[remaining['has_ai'] == 1]
emp_pool = remaining[remaining['has_employment'] == 1]
diss_pool = remaining[remaining['has_dissuade'] == 1]
author_pool = remaining[remaining['is_note_author'] == 1]

ai_sample = ai_pool.iloc[np_rng.choice(len(ai_pool), min(30, len(ai_pool)), replace=False)]
emp_sample = emp_pool.iloc[np_rng.choice(len(emp_pool), min(20, len(emp_pool)), replace=False)]
diss_sample = diss_pool.iloc[np_rng.choice(len(diss_pool), min(10, len(diss_pool)), replace=False)]
author_sample = author_pool.iloc[np_rng.choice(len(author_pool), min(20, len(author_pool)), replace=False)]

special = pd.concat([ai_sample, emp_sample, diss_sample, author_sample]).drop_duplicates('comment_id')
print(f"专项超采: AI={len(ai_sample)} 就业={len(emp_sample)} 劝退={len(diss_sample)} 作者={len(author_sample)}")

# 合并
full_sample = pd.concat([base, special]).drop_duplicates('comment_id')
print(f"\n总抽样: {len(full_sample)}")

# 覆盖检查
print(f"  含AI: {full_sample['has_ai'].sum()}")
print(f"  含就业: {full_sample['has_employment'].sum()}")
print(f"  含劝退: {full_sample['has_dissuade'].sum()}")
print(f"  作者回复: {full_sample['is_note_author'].sum()}")
print(f"  Level分布: {full_sample['comment_level'].value_counts().to_dict()}")
print(f"  Source分布: {full_sample['source_type'].value_counts().to_dict()}")

# ---- 4. 导出 ----
# CSV: 供人工/LLM阅读
export_cols = ['comment_id', 'note_title', 'parent_content', 'content_clean',
               'comment_level', 'is_note_author', 'source_type', 'like_count_num']
sample_export = full_sample[export_cols].copy()
sample_export.columns = ['comment_id', 'note_title', 'parent_comment', 'current_comment',
                          'comment_level', 'is_note_author', 'source', 'like_count']
sample_export.to_csv(os.path.join(OUT_DIR, 'expert_sample_400.csv'), index=False, encoding='utf-8-sig')

# JSONL: 供LLM编码管线（每条一行JSON）
with open(os.path.join(OUT_DIR, 'expert_sample_400.jsonl'), 'w', encoding='utf-8') as f:
    for _, r in full_sample.iterrows():
        obj = {
            "comment_id": r['comment_id'],
            "note_title": str(r['note_title'] or ''),
            "parent_comment": str(r['parent_content'] or ''),
            "current_comment": str(r['content_clean'] or ''),
            "comment_level": int(r['comment_level']),
            "is_note_author": int(r['is_note_author']),
        }
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')

print(f"\n导出: {OUT_DIR}/expert_sample_400.csv")
print(f"导出: {OUT_DIR}/expert_sample_400.jsonl")
print("Stage 4a 完成")
