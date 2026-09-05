# -*- coding: utf-8 -*-
"""
Stage 1: 数据治理
- 创建8个SQL分析视图（在独立分析库上，不改写原始库）
- 匿名化（HMAC-SHA256 二次匿名化 user id）
- 数据质量统计（字段覆盖、文本有效性、回复关系、来源差异、集中度）
- 三语料导出（语料A全量发现 / 语料B完整关系 / 语料C跨来源验证）
"""
import hashlib
import hmac
import json
import os
import sqlite3
import sys
import datetime
import secrets
import re

import pandas as pd
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FROZEN_DB = os.path.join(BASE, "analysis", "00_raw", "crawler_frozen.db")
ANALYSIS_DB = os.path.join(BASE, "analysis", "02_views", "analysis.db")
CORPORA_DIR = os.path.join(BASE, "analysis", "03_corpora")
REPORT_DIR = os.path.join(BASE, "报告", "01_数据治理")
DATA_DIR = os.path.join(REPORT_DIR, "data")
SECRET_FILE = os.path.join(BASE, "analysis", "01_schema", "anonymization_secret.txt")

os.makedirs(os.path.dirname(ANALYSIS_DB), exist_ok=True)
os.makedirs(CORPORA_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

print("=" * 60)
print("Stage 1: 数据治理")
print("=" * 60)

# ============================================================
# 0. 匿名化密钥
# ============================================================
if os.path.exists(SECRET_FILE):
    with open(SECRET_FILE, "rb") as f:
        secret = f.read().strip()
else:
    secret = secrets.token_bytes(32)
    with open(SECRET_FILE, "wb") as f:
        f.write(secret)
    print(f"[匿名化] 新密钥已生成并保存至: {SECRET_FILE}")

def anonymize_id(uid):
    """HMAC-SHA256 二次匿名化，保留前16位hex，不可逆"""
    if not uid:
        return None
    return hmac.new(secret, str(uid).encode(), hashlib.sha256).hexdigest()[:16]

# ============================================================
# 1. 连接冻结库，读取全部数据
# ============================================================
src = sqlite3.connect(FROZEN_DB)
src.row_factory = sqlite3.Row

# 读取核心表
notes_df = pd.read_sql("SELECT * FROM notes", src)
comments_df = pd.read_sql("SELECT * FROM comments", src)
search_hits_df = pd.read_sql("SELECT * FROM search_hits", src)
search_queries_df = pd.read_sql("SELECT * FROM search_queries", src)
note_tags_df = pd.read_sql("SELECT * FROM note_tags", src)
note_metrics_df = pd.read_sql("SELECT * FROM note_metric_snapshots", src)
manifests_df = pd.read_sql("SELECT * FROM comment_collection_manifests", src)
crawl_runs_df = pd.read_sql("SELECT * FROM crawl_runs", src)

# ============================================================
# 1b. 私有增强步骤（公开仓库禁用）
# ============================================================
# 原工程曾包含基于私有原始文件的作者恢复与标题扩展。
# 为避免再识别风险，公开版本不提供作者恢复逻辑或相关中间文件。
notes_df['author_recovered'] = None
notes_df['author_confidence'] = None
notes_df['description_synthetic'] = None
notes_df['has_original_desc'] = notes_df['description'].notna() & (notes_df['description'] != '')
notes_df['has_any_desc'] = notes_df['has_original_desc']

# ============================================================
# 2. 文本清洗与状态标记（在分析层，不改写原始数据）
# ============================================================
def clean_text(t):
    if t is None:
        return ""
    return str(t).strip()

def classify_text(t):
    """返回 (semantic_eligible, nonverbal_only, short_text_flag)"""
    s = clean_text(t)
    if not s:
        return (0, 1, 0)
    # 去除非中文字符、字母、数字后判断是否纯符号
    stripped = re.sub(r'[一-鿿A-Za-z0-9]', '', s)
    if not stripped.strip():
        return (0, 1, 0)  # 纯表情/符号
    if len(s) <= 5:
        return (1, 0, 1)  # 极短但有效
    return (1, 0, 0)

# 评论清洗
comments_df['content_raw'] = comments_df['content']
comments_df['content_clean'] = comments_df['content'].apply(clean_text)
text_class = comments_df['content_clean'].apply(classify_text)
comments_df['semantic_eligible'] = [x[0] for x in text_class]
comments_df['nonverbal_only'] = [x[1] for x in text_class]
comments_df['short_text_flag'] = [x[2] for x in text_class]
comments_df['text_length'] = comments_df['content_clean'].str.len()

# 来源标记
comments_df['source_type'] = comments_df['schema_version'].apply(
    lambda x: 'browser' if 'browser' in str(x) else 'api'
)
notes_df['source_type'] = notes_df['schema_version'].apply(
    lambda x: 'browser' if 'browser' in str(x) else 'api'
)

# 关系资格：回复评论（comment_level>=2）且 parent_comment_id 存在
comments_df['relation_eligible'] = (
    (comments_df['comment_level'] >= 2) &
    (comments_df['parent_comment_id'].notna())
).astype(int)

# 匿名化 user id
for col in ['author_hash', 'target_user_hash']:
    if col in comments_df.columns:
        comments_df[f'{col}_anon'] = comments_df[col].apply(anonymize_id)
    if col in notes_df.columns:
        notes_df[f'{col}_anon'] = notes_df[col].apply(anonymize_id)

# 判断评论是否为笔记作者（通过比对笔记 author_hash 与评论 author_hash）
# 注意：1226条浏览器笔记缺 author_hash，作者身份无法识别 → 标记 author_identifiable=0
author_merge = comments_df[['comment_id', 'note_id', 'author_hash']].merge(
    notes_df[['note_id', 'author_hash']].rename(columns={'author_hash': 'note_author_hash'}),
    on='note_id', how='left'
)
is_author_flag = (
    author_merge['note_author_hash'].notna() &
    author_merge['author_hash'].notna() &
    (author_merge['author_hash'] == author_merge['note_author_hash'])
).astype(int)
comments_df['is_note_author'] = comments_df['comment_id'].map(
    dict(zip(author_merge['comment_id'], is_author_flag))
).fillna(0).astype(int)
# 作者身份可识别标记：笔记有 author_hash 且评论有 author_hash
author_merge['author_identifiable'] = (
    author_merge['note_author_hash'].notna() & author_merge['author_hash'].notna()
).astype(int)
comments_df['author_identifiable'] = comments_df['comment_id'].map(
    dict(zip(author_merge['comment_id'], author_merge['author_identifiable']))
).fillna(0).astype(int)
# 孤儿评论标记：note_id 不在 notes 表中
known_note_ids = set(notes_df['note_id'])
comments_df['is_orphan'] = comments_df['note_id'].apply(
    lambda nid: 0 if nid in known_note_ids else 1
)
print(f"[作者身份] 笔记作者评论数(可识别范围内): {comments_df['is_note_author'].sum()}")
print(f"[作者身份] 作者身份可识别评论数: {comments_df['author_identifiable'].sum()}")
print(f"[孤儿评论] 孤儿评论数: {comments_df['is_orphan'].sum()}")

# ============================================================
# 3. 创建分析库视图（在独立 analysis.db 上）
# ============================================================
ana = sqlite3.connect(ANALYSIS_DB)

# 将匿名化后的数据写入分析库（物化视图）
comments_df.to_sql('comments_anon', ana, if_exists='replace', index=False)
notes_df.to_sql('notes_anon', ana, if_exists='replace', index=False)
search_queries_df.to_sql('search_queries', ana, if_exists='replace', index=False)
search_hits_df.to_sql('search_hits', ana, if_exists='replace', index=False)
crawl_runs_df.to_sql('crawl_runs', ana, if_exists='replace', index=False)

# 视图1: v_search_lineage
ana.execute("""
CREATE VIEW IF NOT EXISTS v_search_lineage AS
SELECT
    sh.query_id,
    sq.query_text,
    sq.query_group,
    sh.crawl_run_id,
    cr.sort_type,
    sh.note_id,
    sh.result_page,
    sh.rank_in_page,
    sh.global_rank,
    sh.captured_at
FROM search_hits sh
JOIN search_queries sq ON sh.query_id = sq.query_id
LEFT JOIN (
    SELECT crawl_run_id, MAX(sort_type) AS sort_type FROM crawl_runs GROUP BY crawl_run_id
) cr ON sh.crawl_run_id = cr.crawl_run_id
""")
print("[视图] v_search_lineage 创建")

# 视图2: v_note_base
note_tags_df.to_sql('note_tags', ana, if_exists='replace', index=False)
note_metrics_df.to_sql('note_metric_snapshots', ana, if_exists='replace', index=False)
ana.execute("""
CREATE VIEW IF NOT EXISTS v_note_base AS
SELECT
    n.note_id,
    n.title,
    n.description,
    nt.tags,
    n.note_type,
    n.author_hash_anon AS author_hash_secure,
    n.publish_ts_ms,
    n.detail_status,
    n.schema_version,
    n.source_type,
    nm.liked_count_num,
    nm.collected_count_num,
    nm.comment_count_num,
    nm.share_count_num,
    sq.query_count,
    sq.query_groups,
    sq.best_rank
FROM notes_anon n
LEFT JOIN (
    SELECT note_id, GROUP_CONCAT(tag_name, '|') AS tags
    FROM note_tags GROUP BY note_id
) nt ON n.note_id = nt.note_id
LEFT JOIN (
    SELECT note_id,
        liked_count_num, collected_count_num, comment_count_num, share_count_num,
        ROW_NUMBER() OVER (PARTITION BY note_id ORDER BY captured_at DESC) AS rn
    FROM note_metric_snapshots
) nm ON n.note_id = nm.note_id AND nm.rn = 1
LEFT JOIN (
    SELECT sh.note_id, COUNT(DISTINCT sh.query_id) AS query_count,
        (SELECT GROUP_CONCAT(DISTINCT qg.query_group) FROM (
            SELECT sq2.query_group FROM search_hits sh2
            JOIN search_queries sq2 ON sh2.query_id = sq2.query_id
            WHERE sh2.note_id = sh.note_id
        ) qg) AS query_groups,
        MIN(COALESCE(sh.global_rank, 999)) AS best_rank
    FROM search_hits sh
    GROUP BY sh.note_id
) sq ON n.note_id = sq.note_id
""")
print("[视图] v_note_base 创建")

# 视图3: v_comment_base
ana.execute("""
CREATE VIEW IF NOT EXISTS v_comment_base AS
SELECT
    comment_id,
    note_id,
    root_comment_id,
    parent_comment_id,
    comment_level,
    author_hash_anon AS author_hash_secure,
    target_user_hash_anon AS target_user_hash_secure,
    content_raw,
    content_clean,
    create_ts_ms,
    like_count_num,
    sub_comment_count_num,
    schema_version,
    source_type,
    is_note_author,
    author_identifiable,
    is_orphan,
    semantic_eligible,
    nonverbal_only,
    short_text_flag,
    relation_eligible
FROM comments_anon
""")
print("[视图] v_comment_base 创建")

# 视图4: v_comment_context
ana.execute("""
CREATE VIEW IF NOT EXISTS v_comment_context AS
SELECT
    c.comment_id,
    c.note_id,
    n.title AS note_title,
    CASE WHEN n.description IS NOT NULL AND n.description != '' THEN 1 ELSE 0 END AS note_description_available,
    p.content_clean AS parent_comment_text,
    c.content_clean AS current_comment_text,
    c.comment_level,
    c.is_note_author,
    c.schema_version
FROM comments_anon c
LEFT JOIN notes_anon n ON c.note_id = n.note_id
LEFT JOIN comments_anon p ON c.parent_comment_id = p.comment_id
""")
print("[视图] v_comment_context 创建")

# 视图5: v_reply_pairs
# 主关系：一级评论—回复线程（root_comment_id 完整，10,801对）
# 精确父子关系（parent_comment_id）单独标记 parent_relation_available
ana.execute("""
CREATE VIEW IF NOT EXISTS v_reply_pairs AS
SELECT
    c.note_id,
    c.root_comment_id,
    c.root_comment_id AS source_comment_id,
    c.comment_id AS reply_comment_id,
    r.author_hash_anon AS source_user,
    c.author_hash_anon AS reply_user,
    c.is_note_author AS reply_is_note_author,
    r.content_clean AS source_text,
    c.content_clean AS reply_text,
    COALESCE(r.like_count_num, 0) AS source_like_count,
    COALESCE(c.like_count_num, 0) AS reply_like_count,
    COALESCE(r.sub_comment_count_num, 0) AS thread_reply_count,
    CASE WHEN c.parent_comment_id IS NOT NULL THEN 1 ELSE 0 END AS has_parent_marker
FROM comments_anon c
JOIN comments_anon r ON c.root_comment_id = r.comment_id AND r.comment_level = 1
WHERE c.comment_level >= 2 AND c.is_orphan = 0
""")
print("[视图] v_reply_pairs 创建")

# 视图6: v_user_participation
ana.execute("""
CREATE VIEW IF NOT EXISTS v_user_participation AS
SELECT
    author_hash_anon AS user_hash_secure,
    COUNT(CASE WHEN comment_level = 1 THEN 1 END) AS root_comment_count,
    COUNT(CASE WHEN comment_level >= 2 THEN 1 END) AS reply_count,
    COUNT(DISTINCT note_id) AS note_count,
    MIN(create_ts_ms) AS first_seen,
    MAX(create_ts_ms) AS last_seen
FROM comments_anon
GROUP BY author_hash_anon
""")
print("[视图] v_user_participation 创建")

# 视图7: v_note_metrics
ana.execute("""
CREATE VIEW IF NOT EXISTS v_note_metrics AS
SELECT
    note_id,
    COUNT(*) AS total_comments,
    SUM(CASE WHEN comment_level = 1 THEN 1 ELSE 0 END) AS root_comments,
    SUM(CASE WHEN comment_level >= 2 THEN 1 ELSE 0 END) AS reply_comments,
    SUM(CASE WHEN is_note_author = 1 THEN 1 ELSE 0 END) AS author_comments,
    AVG(like_count_num) AS avg_comment_likes,
    MAX(like_count_num) AS max_comment_likes,
    COUNT(DISTINCT author_hash_anon) AS unique_commenters
FROM comments_anon
GROUP BY note_id
""")
print("[视图] v_note_metrics 创建")

# 视图8: v_corpus_source
ana.execute("""
CREATE VIEW IF NOT EXISTS v_corpus_source AS
SELECT
    source_type,
    COUNT(*) AS comments,
    COUNT(DISTINCT note_id) AS notes,
    COUNT(DISTINCT author_hash_anon) AS users,
    SUM(CASE WHEN semantic_eligible = 1 THEN 1 ELSE 0 END) AS semantic_eligible
FROM comments_anon
GROUP BY source_type
""")
print("[视图] v_corpus_source 创建")

ana.commit()

# 验证视图
print("\n[视图验证]")
for v in ['v_search_lineage', 'v_note_base', 'v_comment_base', 'v_comment_context',
          'v_reply_pairs', 'v_user_participation', 'v_note_metrics', 'v_corpus_source']:
    try:
        cnt = ana.execute(f"SELECT COUNT(*) FROM {v}").fetchone()[0]
        print(f"  {v}: {cnt} 行")
    except Exception as e:
        print(f"  {v}: 错误 {e}")

# ============================================================
# 4. 数据质量统计
# ============================================================
quality = {}

# 4.1 总体规模
quality['total'] = {
    'notes': len(notes_df),
    'comments': len(comments_df),
    'unique_notes_in_comments': comments_df['note_id'].nunique(),
    'unique_users': comments_df['author_hash_anon'].nunique(),
    'search_hits': len(search_hits_df),
    'unique_hit_notes': search_hits_df['note_id'].nunique(),
    'reply_comments': len(comments_df[comments_df['comment_level'] >= 2]),
    'root_comments': len(comments_df[comments_df['comment_level'] == 1]),
}

# 4.2 笔记字段覆盖
quality['note_field_coverage'] = {}
for col in ['title', 'description', 'note_type', 'author_hash', 'publish_ts_ms']:
    n = notes_df[col].notna().sum()
    quality['note_field_coverage'][col] = {'filled': int(n), 'pct': round(n / len(notes_df), 4)}

# 4.2b 作者恢复与正文增强
quality['data_augmentation'] = {
    'author_recovered_bronze': int((notes_df['author_recovered'] == 'bronze_direct').sum()),
    'author_recovered_inference': int((notes_df['author_recovered'] == 'comment_inference_R4').sum()),
    'author_total_after_recovery': int(notes_df['author_hash'].notna().sum()),
    'author_unknown_remaining': int(notes_df['author_hash'].isna().sum()),
    'desc_original': int(notes_df['has_original_desc'].sum()),
    'desc_synthetic': int(notes_df['description_synthetic'].notna().sum()),
    'desc_total_after': int(notes_df['has_any_desc'].sum()),
    'desc_missing_remaining': int((~notes_df['has_any_desc']).sum()),
}

# 4.3 评论字段覆盖
quality['comment_field_coverage'] = {}
for col in ['content', 'like_count_num', 'create_ts_ms', 'ip_location', 'author_hash', 'parent_comment_id']:
    if col in comments_df.columns:
        n = comments_df[col].notna().sum()
        quality['comment_field_coverage'][col] = {'filled': int(n), 'pct': round(n / len(comments_df), 4)}

# 4.4 文本有效性
quality['text_validity'] = comments_df.groupby(['semantic_eligible', 'nonverbal_only', 'short_text_flag']).size().reset_index(name='count').to_dict('records')

# 4.5 来源差异
quality['source_distribution'] = comments_df.groupby('source_type').size().to_dict()
quality['note_source_distribution'] = notes_df.groupby('source_type').size().to_dict()

# 4.6 评论集中度
note_comment_counts = comments_df.groupby('note_id').size().sort_values(ascending=False)
total_comments = len(comments_df)
quality['concentration'] = {
    'top1_note_comments': int(note_comment_counts.iloc[0]),
    'top1_note_pct': round(note_comment_counts.iloc[0] / total_comments, 4),
    'top5_notes_comments': int(note_comment_counts.head(5).sum()),
    'top5_notes_pct': round(note_comment_counts.head(5).sum() / total_comments, 4),
    'top1pct_notes_comments': int(note_comment_counts.head(int(len(note_comment_counts) * 0.01)).sum()) if len(note_comment_counts) >= 100 else None,
    'num_notes': len(note_comment_counts),
    'notes_with_comments': int((note_comment_counts > 0).sum()),
}

# 4.7 点赞分布
like_vals = comments_df['like_count_num'].dropna()
quality['like_distribution'] = {
    'mean': round(like_vals.mean(), 2),
    'median': round(like_vals.median(), 2),
    'std': round(like_vals.std(), 2),
    'max': int(like_vals.max()),
    'zero_pct': round((like_vals == 0).mean(), 4),
    'nonzero_count': int((like_vals > 0).sum()),
}

# 4.8 回复关系（基于 root_comment_id 线程关系）
reply_pairs_count = ana.execute("SELECT COUNT(*) FROM v_reply_pairs").fetchone()[0]
author_replies = ana.execute("SELECT COUNT(*) FROM v_reply_pairs WHERE reply_is_note_author = 1").fetchone()[0]
precise_parent = ana.execute("SELECT COUNT(*) FROM v_reply_pairs WHERE has_parent_marker = 1").fetchone()[0]
author_replies_identifiable = ana.execute("""
    SELECT COUNT(*) FROM v_reply_pairs rp
    JOIN comments_anon c ON rp.reply_comment_id = c.comment_id
    WHERE rp.reply_is_note_author = 1 AND c.author_identifiable = 1
""").fetchone()[0]
quality['reply_relations'] = {
    'thread_reply_pairs': int(reply_pairs_count),          # 一级评论—回复线程关系
    'precise_parent_pairs': int(precise_parent),           # 精确父子回复关系
    'author_replies_identifiable': int(author_replies_identifiable),  # 可识别范围内的作者回复
    'author_reply_pct_identifiable': round(author_replies_identifiable / reply_pairs_count, 4) if reply_pairs_count else None,
}

# 4.9 作者身份可识别性（浏览器笔记缺author_hash的局限）
quality['author_identifiability'] = {
    'comments_identifiable': int(comments_df['author_identifiable'].sum()),
    'comments_not_identifiable': int((comments_df['author_identifiable'] == 0).sum()),
    'notes_without_author_hash': int(notes_df['author_hash'].isna().sum()),
    'author_comments_total': int(comments_df['is_note_author'].sum()),
    'author_comments_level1': int(comments_df[(comments_df['is_note_author'] == 1) & (comments_df['comment_level'] == 1)].shape[0]),
    'author_comments_replies': int(comments_df[(comments_df['is_note_author'] == 1) & (comments_df['comment_level'] >= 2)].shape[0]),
    'orphan_comments': int(comments_df['is_orphan'].sum()),
    'orphan_notes': int(comments_df[comments_df['is_orphan'] == 1]['note_id'].nunique()),
}

# 4.10 时间范围
quality['time_range'] = {
    'earliest_comment': int(comments_df['create_ts_ms'].min()) if comments_df['create_ts_ms'].notna().any() else None,
    'latest_comment': int(comments_df['create_ts_ms'].max()) if comments_df['create_ts_ms'].notna().any() else None,
}

# 4.11 各模型可用样本量预估
sem_eligible = comments_df['semantic_eligible'].sum()
quality['model_samples'] = {
    'semantic_eligible_comments': int(sem_eligible),
    'context_comments': int(comments_df['semantic_eligible'].sum()),  # 语境编码可用
    'thread_reply_pairs': int(reply_pairs_count),
    'precise_parent_pairs': int(precise_parent),
    'notes_with_title': int(notes_df['title'].notna().sum()),
    'notes_with_desc': int(notes_df['description'].notna().sum()),
    'notes_with_any_desc': int(notes_df['has_any_desc'].sum()),
}

with open(os.path.join(DATA_DIR, 'data_quality.json'), 'w', encoding='utf-8') as f:
    json.dump(quality, f, ensure_ascii=False, indent=2)

# ============================================================
# 5. 三语料导出
# ============================================================
# 语料A：全量发现语料
corpus_a = comments_df[comments_df['semantic_eligible'] == 1][
    ['comment_id', 'note_id', 'content_clean', 'comment_level', 'like_count_num', 'source_type']
]
corpus_a['text_length'] = corpus_a['content_clean'].str.len()
corpus_a.to_csv(os.path.join(CORPORA_DIR, 'corpus_A_full_discovery.csv'), index=False, encoding='utf-8-sig')

# 语料B：完整关系语料（回复对 + 笔记语境）
corpus_b = pd.read_sql("SELECT * FROM v_reply_pairs", ana)
corpus_b.to_csv(os.path.join(CORPORA_DIR, 'corpus_B_relation.csv'), index=False, encoding='utf-8-sig')

# 语料C：跨来源验证语料（API vs Browser）
corpus_c = comments_df[
    ['comment_id', 'note_id', 'content_clean', 'comment_level', 'source_type', 'like_count_num']
].copy()
corpus_c.to_csv(os.path.join(CORPORA_DIR, 'corpus_C_source.csv'), index=False, encoding='utf-8-sig')

# 语料hash
corpora_hash = {}
for name in ['corpus_A_full_discovery.csv', 'corpus_B_relation.csv', 'corpus_C_source.csv']:
    p = os.path.join(CORPORA_DIR, name)
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    corpora_hash[name] = h.hexdigest()[:16]

# ============================================================
# 6. 输出汇总
# ============================================================
print("\n" + "=" * 60)
print("数据质量核心指标")
print("=" * 60)
print(json.dumps(quality, ensure_ascii=False, indent=2))

print("\n[三语料]")
print(f"  语料A（全量发现）: {len(corpus_a)} 行")
print(f"  语料B（完整关系）: {len(corpus_b)} 行")
print(f"  语料C（跨来源）:   {len(corpus_c)} 行")
print(f"  语料哈希: {corpora_hash}")

print("\nStage 1 数据治理完成")
