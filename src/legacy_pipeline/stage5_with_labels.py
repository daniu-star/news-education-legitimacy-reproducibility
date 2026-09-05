# -*- coding: utf-8 -*-
"""
Stage 5 v2: 基于LLM编码标签的统计建模
用21,819条实际标签替代主题概率代理，重新分析四大发现
"""
import os, sys, json, warnings
import numpy as np, pandas as pd
from collections import Counter, defaultdict
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "报告", "05_统计建模")
os.makedirs(OUT, exist_ok=True)

# ---- 加载标签和数据 ----
print("加载标签...")
labels = []
with open(os.path.join(BASE, "analysis", "07_labels", "all_encoded_complete.jsonl"), encoding='utf-8') as f:
    for line in f:
        labels.append(json.loads(line))
print(f"标签: {len(labels)}条")

# 连接原始数据（评论元数据、点赞等）
import sqlite3
conn = sqlite3.connect(os.path.join(BASE, "analysis", "02_views", "analysis.db"))
comments_meta = pd.read_sql("""
    SELECT comment_id, note_id, comment_level, like_count_num, is_note_author, source_type
    FROM v_comment_base WHERE semantic_eligible=1 AND is_orphan=0
""", conn)
conn.close()

# 构建分析数据集
df_labels = pd.DataFrame(labels)
df_labels.rename(columns={'id': 'comment_id'}, inplace=True)
df = df_labels.merge(comments_meta, on='comment_id', how='inner')
print(f"合并后: {len(df)}条")

# 展平多标签（取主标签=第一个）
df['eval_obj_main'] = df['evaluation_object'].apply(lambda x: x[0] if isinstance(x,list) and len(x)>0 else '未明确')
df['evidence_main'] = df['evidence_basis'].apply(lambda x: x[0] if isinstance(x,list) and len(x)>0 else '无依据断言')
df['has_like'] = (df['like_count_num'] > 0).astype(int)
df['log_likes'] = np.log1p(df['like_count_num'].clip(0))

# ==========================================================
# 发现一: 评价依据 → 立场 交叉表 + 卡方
# ==========================================================
print("\n" + "=" * 60)
print("发现一: 评价依据 vs 立场")
print("=" * 60)

# 过滤掉"无法判断"
df_valid = df[df['stance'] != '无法判断'].copy()
ct = pd.crosstab(df_valid['evidence_main'], df_valid['stance'], normalize='index') * 100
print("\n评价依据×立场 (%):")
print(ct.round(1).to_string())

# 卡方检验
ct_abs = pd.crosstab(df_valid['evidence_main'], df_valid['stance'])
chi2, p, dof, _ = stats.chi2_contingency(ct_abs)
print(f"\n卡方: chi2={chi2:.0f}, p={p:.2e}")

# 关键对比: 个人经验 vs 专业知识 的'否定'率
for evidence in ['个人学习经历', '实习就业经历', '专业知识与理论', '公共价值', '无依据断言']:
    sub = df_valid[df_valid['evidence_main'] == evidence]
    if len(sub) > 50:
        neg_rate = (sub['stance'] == '否定').mean()
        pos_rate = (sub['stance'] == '认可').mean()
        print(f"  {evidence}: n={len(sub)}, 否定率={neg_rate:.1%}, 认可率={pos_rate:.1%}")

# ==========================================================
# 发现二: 专业能力边界 — 能力类型 vs 能力属性
# ==========================================================
print("\n" + "=" * 60)
print("发现二: 能力类型 → 能力属性")
print("=" * 60)

# 展开能力提及（每能力一行的长表）
ability_rows = []
for _, r in df.iterrows():
    atypes = r.get('ability_type', [])
    aprops = r.get('ability_property', [])
    if not isinstance(atypes, list) or not atypes:
        continue
    for i, at in enumerate(atypes):
        ap = aprops[i] if isinstance(aprops, list) and i < len(aprops) else '未明确'
        ability_rows.append({'comment_id': r['comment_id'], 'ability_type': at, 'ability_property': ap})

df_ability = pd.DataFrame(ability_rows)
print(f"\n能力提及: {len(df_ability)}条")

if len(df_ability) > 0:
    ct_ab = pd.crosstab(df_ability['ability_type'], df_ability['ability_property'])
    print("\n能力类型×属性:")
    print(ct_ab.to_string())
else:
    print("\n能力标签为空(编码时ability_type/property未生产)")

# "专业性去租"差异: 基础生产能力的替代/通用化比例
if len(df_ability) > 0:
    base_prod = df_ability[df_ability['ability_type'] == '基础内容生产']
    core = df_ability[df_ability['ability_type'].isin(['事实核查与证据','采访与信源关系','社会解释与议题发现','伦理责任与公共判断'])]
    diff = {}
    for label in ['可由AI替代', '大学教育通用']:
        diff[label] = {
            '基础生产': base_prod['ability_property'].value_counts(normalize=True).get(label, 0) if len(base_prod) else 0,
            '核心能力': core['ability_property'].value_counts(normalize=True).get(label, 0) if len(core) else 0,
        }
    print("\n'专业性去租'差异:")
    for label, vals in diff.items():
        delta = vals['基础生产'] - vals['核心能力']
        print(f"  {label}: 基础={vals['基础生产']:.1%} vs 核心={vals['核心能力']:.1%} (Δ={delta:+.1%})")
else:
    print("\n能力分析跳过(ability字段为空——需另行编码)")
    diff = {}

# ==========================================================
# 发现三: 平台共鸣 — 评价依据/立场 → 点赞
# ==========================================================
print("\n" + "=" * 60)
print("发现三: 平台共鸣 (评价依据→点赞)")
print("=" * 60)

# 分组统计
for col in ['evidence_main', 'stance']:
    print(f"\n{col} × 点赞:")
    grouped = df.groupby(col).agg(
        n=('has_like', 'count'),
        like_rate=('has_like', 'mean'),
        like_mean=('like_count_num', 'mean'),
        like_median=('like_count_num', 'median'),
    ).sort_values('like_rate', ascending=False)
    for idx, r in grouped.iterrows():
        if r['n'] > 30:
            print(f"  {idx}: n={int(r['n'])}, 赞率={r['like_rate']:.1%}, 均赞={r['like_mean']:.1f}")

# 作者身份
print("\n作者身份×点赞:")
for auth in [0, 1]:
    sub = df[df['is_note_author'] == auth]
    print(f"  author={auth}: n={len(sub)}, 赞率={(sub['like_count_num']>0).mean():.1%}, 均赞={sub['like_count_num'].mean():.1f}")

# ==========================================================
# 发现四: 评论集中度 + 框架偏移 (保持不变，基于已有数据)
# ==========================================================
print("\n" + "=" * 60)
print("发现四: 集中度 + 来源差异")
print("=" * 60)

# 主题概率分组 × 实际标签立场
# 加载主题分布
emb_v1 = np.load(os.path.join(BASE, "analysis", "04_embeddings", "emb_comment_v1.npy"))
import torch, re, jieba
from fastopic import FASTopic
from topmost.preprocess.preprocess import Preprocess

STOPWORDS = set('新闻 一个 什么 怎么 为什么 这样 那些 这些 因为 所以 但是 就是 还是 可以 觉得 感觉 真的 非常 很多 现在 没有 不是 自己 如果 那么 他们 我们 你们 这个 那个 时候 其实 然后 还有 的话 一下 这里 那里 一点'.split())
def chinese_tokenize(text):
    text = re.sub(r'\[[^\]]*\]', '', str(text))
    return [w for w in jieba.cut(text) if len(w.strip())>=2 and w.strip() not in STOPWORDS]
PREPROCESS = Preprocess(tokenizer=chinese_tokenize, min_length=2, max_doc_freq=0.5, verbose=False)

comments_corpus = pd.read_csv(os.path.join(BASE, "analysis", "04_embeddings", "corpus_A_comments.csv"))
texts = comments_corpus['text_v1_comment_only'].fillna('').tolist()

print("重训K=10...")
import time
model = FASTopic(num_topics=10, preprocess=PREPROCESS, device="cpu")
t0 = time.time()
model.fit(docs=texts, preset_doc_embeddings=emb_v1, epochs=100)
print(f"训练: {time.time()-t0:.0f}s")

theta = model.transform(doc_embeddings=torch.as_tensor(emb_v1))
theta_np = theta.numpy() if hasattr(theta,'numpy') else np.array(theta)
tw_list = [str(w).split() for w in model.get_top_words(num_top_words=20)]

# 实质主题
edu_kw = {'专业','就业','实习','工作','新传','新闻','媒体','记者','报道','文科','本科','毕业','考研','考公','运营','大厂','学校','岗位','ai','采访','论文','研究','理解','标题','雪峰','已死','理想','劝退','选择','工资','收入','行业','社会','中国','能力','内容','流量'}
sub_ids = []
for i in range(10):
    hit = sum(1 for w in tw_list[i][:15] if w in edu_kw)
    if hit >= 2: sub_ids.append(i)

sub_prob = theta_np[:, sub_ids].sum(axis=1)
comments_corpus['sub_prob'] = sub_prob
comments_corpus['sub_prob_q'] = pd.qcut(sub_prob, 4, labels=['Q1','Q2','Q3','Q4'])

# 合并标签
df_aug = df.merge(comments_corpus[['comment_id','sub_prob_q']], on='comment_id', how='inner')

print("\n实质主题Q×立场:")
for q in ['Q1','Q2','Q3','Q4']:
    sub = df_aug[df_aug['sub_prob_q'] == q]
    stances = sub['stance'].value_counts(normalize=True)
    print(f"  {q}(n={len(sub)}): 否定={stances.get('否定',0):.1%} 认可={stances.get('认可',0):.1%} 无法判断={stances.get('无法判断',0):.1%}")

# ==========================================================
# 保存结果
# ==========================================================
results = {
    "n_total": len(df),
    "rq1": {
        "evidence_stance_chi2": float(chi2),
        "evidence_stance_p": float(p),
        "top_evidence": df['evidence_main'].value_counts().head(5).to_dict(),
        "personal_experience_neg_rate": float((df[df['evidence_main']=='个人学习经历']['stance']=='否定').mean()),
        "professional_knowledge_neg_rate": float((df[df['evidence_main']=='专业知识与理论']['stance']=='否定').mean()),
    },
    "rq2": {
        "ability_mentions": len(df_ability),
        "note": "ability字段为空——需另行编码",
    },
    "rq3": {
        "ability_mentions": len(df_ability),
        "note": "ability字段为空——编码时未生产，需另行编码" if len(df_ability)==0 else "",
        "de_renting_delta": {k: v['基础生产']-v['核心能力'] for k, v in diff.items()} if diff else {},
    },
    "rq4": {
        "author_like_rate": float((df[df['is_note_author']==1]['like_count_num']>0).mean()),
        "user_like_rate": float((df[df['is_note_author']==0]['like_count_num']>0).mean()),
    },
}
with open(os.path.join(OUT, "stage5_results_v2.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n结果保存: {OUT}/stage5_results_v2.json")
print("Stage 5 v2 完成")
