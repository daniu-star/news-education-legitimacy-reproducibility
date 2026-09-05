# -*- coding: utf-8 -*-
"""
Stage 5: 统计建模与研究发现
- 基于主题分布+互动数据, 分析评价行为模式
- 输出: 发现一(评价依据分布) / 发现二(平台共鸣) / 发现三(框架偏移) / 发现四(能力线索)
"""
import os, sys, json, time, re, warnings
import numpy as np, pandas as pd
from collections import Counter, defaultdict
from scipy import stats

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMB_DIR = os.path.join(BASE, "analysis", "04_embeddings")
TOPIC_DIR = os.path.join(BASE, "analysis", "05_topics")
OUT_DIR = os.path.join(BASE, "报告", "05_统计建模")
os.makedirs(OUT_DIR, exist_ok=True)

# ---- 加载数据 ----
import torch, jieba
from fastopic import FASTopic
from topmost.preprocess.preprocess import Preprocess

comments = pd.read_csv(os.path.join(EMB_DIR, "corpus_A_comments.csv"))
notes = pd.read_csv(os.path.join(EMB_DIR, "corpus_A_notes.csv"))
emb_v1 = np.load(os.path.join(EMB_DIR, "emb_comment_v1.npy"))

# ---- 重训K=10模型 (5分钟, 避免pickle兼容问题) ----
STOPWORDS = set('''新闻 一个 什么 怎么 为什么 这样 那些 这些 因为 所以 但是 就是 还是 可以 觉得 感觉 真的 非常 很多 现在 没有 不是 自己 如果 那么 他们 我们 你们 这个 那个 时候 其实 然后 还有 的话 一下 这里 那里 一点 是不是 有没有 包括 已经 应该 可能 也是 都是 有点 不太 完全 一些 或者 这样 那样 这些 那些 派对 飞吻 点赞 显化 色色 好棒 强吻 大笑 合十 流汗 鼓掌 举手 黑薯 电子版 成可 莎莎 惊艳 得意 自拍 签名 美味 一杯 点点 吵吵 小狗 威龙 一颗 收藏 啊啊啊 哈哈哈 嘿嘿 哭惹 捂脸 害羞 震惊 调皮 微笑 赞同 干杯 爆竹 庆祝 爱心 赞赞 可爱 哇塞 喜欢 收到 原来 真的假 问一问 谢谢 恭喜 吓死 天了 momo 娜娜 小常 老大 爸爸 同问 果然 开玩笑 三观 天呐 有钱人 表演 骂人 统一 兄弟 女孩 主角 不许 喇叭 令人 满满 复制 资料 女主 新生 主页 回复 奶茶 打卡 let 提取 红薯 好滴 大王 太棒了 aaa 入住 睡觉 蛋蛋 我刚 男主 图片 宇宙 背着 太好了 gpt 呜呜 石化 主包 扶墙 这话 半天 反应 啊啊 哈哈哈哈 沙发 已关 转发给 同款 棒棒 星星 妙手 这下 哎呀 卧槽 明天 首发 头发 小女孩 块钱 违法 自动 妈妈 宝宝 小动物 生日 鸡蛋 安装 win no 老婆 我服 少年 来到 更新 口令 科普 白人 全家 妻子 发个 为何 超绝 不知 抱抱 能量 作业'''.split())

def chinese_tokenize(text):
    text = re.sub(r'\[[^\]]*\]', '', str(text))
    return [w for w in jieba.cut(text) if len(w.strip()) >= 2 and w.strip() not in STOPWORDS]

PREPROCESS = Preprocess(tokenizer=chinese_tokenize, min_length=2, max_doc_freq=0.5, verbose=False)

print("重训K=10模型...")
t0 = time.time()
texts = comments['text_v1_comment_only'].fillna('').tolist()
model = FASTopic(num_topics=10, preprocess=PREPROCESS, device="cpu")
model.fit(docs=texts, preset_doc_embeddings=emb_v1, epochs=100)
print(f"训练: {time.time()-t0:.0f}s")

theta = model.transform(doc_embeddings=torch.as_tensor(emb_v1))
theta_np = theta.numpy() if hasattr(theta, 'numpy') else np.array(theta)
comments['topic'] = list(theta_np)

# 用词检测判定实质主题
edu_kw = {'专业','就业','实习','工作','新传','新闻','媒体','记者','报道','文科',
           '本科','毕业','考研','考公','运营','大厂','学校','岗位','课程','学科',
           '传播','ai','采访','论文','研究','理解','标题','雪峰','已死','理想','劝退',
           '选择','工资','收入','行业','社会','中国','经验','能力','内容','流量'}

# ---- 1. 主流主题分布 ----
print("=" * 60)
print("1. 主题分布与元数据交叉")
print("=" * 60)

# 每评论的 dominant topic
dom_topic = theta_np.argmax(axis=1)
comments['dom_topic'] = dom_topic

print("\n主题分布:")
topic_count = pd.Series(dom_topic).value_counts().sort_index()
for t, c in topic_count.items():
    pct = c / len(comments) * 100
    print(f"  主题{t}: {c}条 ({pct:.1f}%)")

# 按来源分
print("\n来源×主题:")
for src in ['api', 'browser']:
    sub = comments[comments['source_type'] == src]
    top3 = sub['dom_topic'].value_counts().head(3)
    print(f"  {src}: {top3.to_dict()}")

# 按评论层级分
print("\n层级×主题:")
for lv in [1, 2]:
    sub = comments[comments['comment_level'] == lv]
    top3 = sub['dom_topic'].value_counts().head(3)
    print(f"  L{lv}: {top3.to_dict()}")

# 实质主题检测
tw_list = [str(w).split() for w in model.get_top_words(num_top_words=20)]
substantive = []
for i in range(10):
    hit = sum(1 for w in tw_list[i][:15] if w in edu_kw)
    if hit >= 2:
        substantive.append(i)
print(f"\nK=10 主题词:")
for i in range(10):
    tag = '★' if i in substantive else ' '
    print(f"  主题{i}{tag}: {tw_list[i][:8]}")
print(f"实质主题ID: {substantive}")

sub_theta = theta_np[:, substantive]
print(f"实质主题mean probability: {sub_theta.mean(axis=0)}")

# ---- 2. 平台共鸣 (Hurdle Model近似) ----
print("\n" + "=" * 60)
print("2. 平台共鸣分析")
print("=" * 60)

# corpus_A已有like_count_num和comment_level, 不需额外merge
# 但需从DB补is_note_author
import sqlite3
conn = sqlite3.connect(os.path.join(BASE, "analysis", "02_views", "analysis.db"))
auth_df = pd.read_sql("""
    SELECT comment_id, is_note_author FROM v_comment_base
    WHERE semantic_eligible=1 AND is_orphan=0
""", conn)
conn.close()
comments = comments.merge(auth_df, on='comment_id', how='left')
comments['is_note_author'] = comments['is_note_author'].fillna(0).astype(int)
print(f"分析样本: {len(comments)}条")

# 实质主题概率 vs 互动
sub_probs = theta_np[:, substantive]
# 加总实质主题概率
comments['sub_topic_prob_sum'] = sub_probs.sum(axis=1)

# 点赞分析
has_like = (comments['like_count_num'] > 0).astype(int)
print(f"\n点赞: {has_like.mean():.1%}有赞, 均值{comments['like_count_num'].mean():.1f}, "
      f"中位{comments['like_count_num'].median():.0f}")

# 按实质主题概率分组
comments['sub_prob_bin'] = pd.qcut(comments['sub_topic_prob_sum'], 4, labels=['Q1(低)','Q2','Q3','Q4(高)'])
print("\n实质主题概率×互动:")
for label, grp in comments.groupby('sub_prob_bin'):
    like_pct = (grp['like_count_num'] > 0).mean()
    like_mean = grp['like_count_num'].mean()
    n = len(grp)
    print(f"  {label}: n={n}, 有赞率={like_pct:.1%}, 均赞={like_mean:.1f}")

# 作者身份×互动
print("\n作者身份×互动:")
for auth in [0, 1]:
    sub = comments[comments['is_note_author'] == auth]
    if len(sub) > 0:
        print(f"  作者={auth}: n={len(sub)}, 有赞率={(sub['like_count_num']>0).mean():.1%}, "
              f"均赞={sub['like_count_num'].mean():.1f}")

# ---- 3. 框架偏移分析 ----
print("\n" + "=" * 60)
print("3. 框架偏移模式")
print("=" * 60)

df_shift = pd.read_csv(os.path.join(TOPIC_DIR, "framework_shift.csv"))
jsds = df_shift['jsd'].dropna()
print(f"笔记数: {len(df_shift)}")
print(f"JSD: mean={jsds.mean():.3f}, med={jsds.median():.3f}, "
      f"p25={jsds.quantile(0.25):.3f}, p75={jsds.quantile(0.75):.3f}")

# 高/低偏移笔记的评论量差异
high = df_shift[df_shift['jsd'] > jsds.quantile(0.75)]
low = df_shift[df_shift['jsd'] < jsds.quantile(0.25)]
print(f"\n高偏移(>p75): {len(high)}篇, 平均评论{high['n_comments'].mean():.0f}条")
print(f"低偏移(<p25): {len(low)}篇, 平均评论{low['n_comments'].mean():.0f}条")

# ---- 4. 评论集中度 ----
print("\n" + "=" * 60)
print("4. 评论集中度")
print("=" * 60)

note_cc = comments.groupby('note_id').size()
print(f"有评论的笔记: {len(note_cc)}")
print(f"前1%笔记评论占比: {note_cc.nlargest(int(len(note_cc)*0.01)).sum() / len(comments):.1%}")
print(f"前5%笔记评论占比: {note_cc.nlargest(int(len(note_cc)*0.05)).sum() / len(comments):.1%}")

# 去爆款敏感性
top1pct = set(note_cc.nlargest(int(len(note_cc)*0.01)).index)
rest = comments[~comments['note_id'].isin(top1pct)]
print(f"去前1%笔记后: {len(rest)}条 ({len(rest)/len(comments):.1%})")

# ---- 5. 关键评论片段 ----
print("\n" + "=" * 60)
print("5. 代表性评论文本")
print("=" * 60)

# 高实质主题概率的评论文本（从notes补标题）
note_titles = dict(zip(notes['note_id'], notes['title']))
comments['note_title'] = comments['note_id'].map(note_titles)
high_sub = comments.nlargest(20, 'sub_topic_prob_sum')
print("\n高实质评论(前5):")
for _, r in high_sub.head(5).iterrows():
    title = str(r.get('note_title', ''))[:40]
    print(f"  [{r['like_count_num']}赞] {title}")
    print(f"  -> {str(r['text_v1_comment_only'])[:100]}")
    print()

# ---- 保存结果 ----
summary = {
    "n_comments": len(comments),
    "topic_distribution": {int(k): int(v) for k, v in topic_count.items()},
    "substantive_topic_ids": substantive,
    "substantive_mean_prob": [float(x) for x in sub_theta.mean(axis=0)],
    "has_like_pct": float(has_like.mean()),
    "like_mean": float(comments['like_count_num'].mean()),
    "like_median": float(comments['like_count_num'].median()),
    "concentration_top1pct": float(note_cc.nlargest(int(len(note_cc)*0.01)).sum() / len(comments)),
    "concentration_top5pct": float(note_cc.nlargest(int(len(note_cc)*0.05)).sum() / len(comments)),
    "framework_shift": {
        "n_notes": int(len(df_shift)),
        "jsd_mean": float(jsds.mean()),
        "jsd_median": float(jsds.median()),
    },
}
with open(os.path.join(OUT_DIR, "stage5_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"\n结果保存至: {OUT_DIR}/stage5_summary.json")
print("Stage 5 分析完成")
