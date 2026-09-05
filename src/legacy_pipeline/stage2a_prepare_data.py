# -*- coding: utf-8 -*-
"""
Stage 2a: 主题发现数据准备
- 从分析库提取语料A（语义有效评论）
- 构建三个输入变体: 纯评论 / 标题+评论 / 标题+标签
- 输出可复现的数据文件供embedding和主题模型使用
"""
import json
import os
import sqlite3

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS_DB = os.path.join(BASE, "analysis", "02_views", "analysis.db")
OUT_DIR = os.path.join(BASE, "analysis", "04_embeddings")
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("Stage 2a: 主题发现数据准备")
print("=" * 60)

conn = sqlite3.connect(ANALYSIS_DB)

# 1. 语义有效评论（语料A）
comments = pd.read_sql("""
    SELECT comment_id, note_id, content_clean, comment_level, like_count_num, source_type
    FROM v_comment_base
    WHERE semantic_eligible = 1 AND is_orphan = 0
""", conn)
print(f"语义有效评论: {len(comments)}")

# 2. 笔记标题（直接查 notes_anon, 含增强字段）
notes = pd.read_sql("""
    SELECT n.note_id, n.title, nt.tags, n.description, n.description_synthetic,
           n.source_type, n.author_hash_anon AS author_hash_secure
    FROM notes_anon n
    LEFT JOIN (SELECT note_id, GROUP_CONCAT(tag_name, '|') AS tags FROM note_tags GROUP BY note_id) nt
    ON n.note_id = nt.note_id
""", conn)
print(f"笔记: {len(notes)}")
print(f"  有标题: {notes['title'].notna().sum()}")
print(f"  有原始正文: {(notes['description'].notna() & (notes['description'] != '')).sum()}")
print(f"  有合成正文: {notes['description_synthetic'].notna().sum()}")

# 3. 构建输入变体
# 变体1: 纯评论
comments['text_v1_comment_only'] = comments['content_clean']

# 变体2: 标题+评论
title_map = dict(zip(notes['note_id'], notes['title']))
comments['note_title'] = comments['note_id'].map(title_map)
def ctx_text(row):
    t = row['note_title'] if pd.notna(row['note_title']) else ''
    c = row['content_clean']
    if t:
        return f"{t}。{c}"
    return c
comments['text_v2_title_comment'] = comments.apply(ctx_text, axis=1)

# 变体3: 标题+标签（笔记级）
notes['text_v3_title_tag'] = notes.apply(
    lambda r: f"{r['title'] if pd.notna(r['title']) else ''}。{r['tags'] if pd.notna(r['tags']) else ''}",
    axis=1
)

# 4. 语料导出
comments[['comment_id', 'note_id', 'text_v1_comment_only', 'text_v2_title_comment',
          'comment_level', 'like_count_num', 'source_type']].to_csv(
    os.path.join(OUT_DIR, 'corpus_A_comments.csv'), index=False, encoding='utf-8-sig')

notes[['note_id', 'text_v3_title_tag', 'title', 'tags', 'source_type']].to_csv(
    os.path.join(OUT_DIR, 'corpus_A_notes.csv'), index=False, encoding='utf-8-sig')

# 5. 分来源语料（M4/M5）
api_comments = comments[comments['source_type'] == 'api']
browser_comments = comments[comments['source_type'] == 'browser']
print(f"API评论: {len(api_comments)}, Browser评论: {len(browser_comments)}")

# 6. 保存meta
meta = {
    "total_semantic_comments": len(comments),
    "api_comments": len(api_comments),
    "browser_comments": len(browser_comments),
    "total_notes": len(notes),
    "notes_with_title": int(notes['title'].notna().sum()),
    "variants": {
        "v1_comment_only": "纯评论",
        "v2_title_comment": "标题+评论",
        "v3_title_tag": "标题+标签(笔记级)",
    },
}
with open(os.path.join(OUT_DIR, 'prepare_meta.json'), 'w', encoding='utf-8') as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("\n数据准备完成:")
print(f"  语料A评论: {OUT_DIR}/corpus_A_comments.csv")
print(f"  语料A笔记: {OUT_DIR}/corpus_A_notes.csv")
print(f"  meta: {OUT_DIR}/prepare_meta.json")
conn.close()
