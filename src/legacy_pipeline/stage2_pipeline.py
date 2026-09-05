# -*- coding: utf-8 -*-
"""
Stage 2 一体化流水线 (K=10)
1. V1 K=10最终模型 (epochs=100, 保存)
2. V2 (标题+评论) K=10对照
3. BERTopic对照
4. 框架偏移 (V1 K=10模型 + 笔记标题embedding)
"""
import os, sys, json, time, re, pickle, warnings
import numpy as np, pandas as pd, torch, jieba
from collections import Counter, defaultdict
from scipy.spatial.distance import jensenshannon
from fastopic import FASTopic
from topmost.preprocess.preprocess import Preprocess
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMB_DIR = os.path.join(BASE, "analysis", "04_embeddings")
OUT_DIR = os.path.join(BASE, "analysis", "05_topics")
os.makedirs(OUT_DIR, exist_ok=True)

K = 10
EPOCHS = 100
SEED = 17

# ==== 中文预处理（与网格一致）====
STOPWORDS = set('''
新闻 一个 什么 怎么 为什么 这样 那些 这些 因为 所以
但是 就是 还是 可以 觉得 感觉 真的 非常 很多 现在
没有 不是 自己 如果 那么 他们 我们 你们 这个 那个
时候 其实 然后 还有 的话 一下 这里 那里 一点
是不是 有没有 包括 已经 应该 可能 也是 都是
有点 不太 完全 一些 或者 这样 那样 这些 那些
派对 飞吻 点赞 显化 色色 好棒 强吻 大笑 合十
流汗 鼓掌 举手 黑薯 电子版 成可 莎莎 惊艳 得意
自拍 签名 美味 一杯 点点 吵吵 小狗 威龙 一颗
收藏 啊啊啊 哈哈哈 嘿嘿 哭惹 捂脸 害羞 震惊 调皮
微笑 赞同 干杯 爆竹 庆祝 爱心 赞赞 可爱 哇塞
喜欢 收到 原来 真的假 问一问 谢谢 恭喜 吓死 天了
momo 娜娜 小常 老大 爸爸 同问 果然 开玩笑 三观
天呐 有钱人 表演 骂人 统一 兄弟 女孩 主角 不许
喇叭 令人 满满 复制 资料 女主 新生 主页 回复
奶茶 打卡 let 提取 红薯 好滴 大王 太棒了 aaa
入住 睡觉 蛋蛋 我刚 男主 图片 宇宙 背着 太好了
gpt 呜呜 石化 主包 扶墙 这话 半天 反应 啊啊
哈哈哈哈 沙发 已关 转发给 同款 棒棒 星星 妙手 这下
哎呀 卧槽 明天 首发 头发 小女孩 块钱 违法 自动
妈妈 宝宝 小动物 生日 鸡蛋 安装 win no 老婆
我服 少年 来到 更新 口令 科普 白人 全家 妻子
发个 为何 超绝 不知 抱抱 能量 作业 宝宝
'''.split())


def chinese_tokenize(text):
    text = re.sub(r'\[[^\]]*\]', '', str(text))
    return [w for w in jieba.cut(text) if len(w.strip()) >= 2 and w.strip() not in STOPWORDS]


PREPROCESS = Preprocess(tokenizer=chinese_tokenize, stopwords=None,
                        min_length=2, max_doc_freq=0.5, verbose=False)

# ==== 加载数据 ====
comments = pd.read_csv(os.path.join(EMB_DIR, "corpus_A_comments.csv"))
texts_v1 = comments['text_v1_comment_only'].fillna('').tolist()
texts_v2 = comments['text_v2_title_comment'].fillna('').tolist()
emb_v1 = np.load(os.path.join(EMB_DIR, "emb_comment_v1.npy"))
emb_v2 = np.load(os.path.join(EMB_DIR, "emb_comment_v2.npy"))

notes = pd.read_csv(os.path.join(EMB_DIR, "corpus_A_notes.csv"))
print(f"评论: {len(texts_v1)}, 笔记: {len(notes)}")

# ==========================================================
# 阶段1: V1 K=10 最终模型
# ==========================================================
print("\n" + "=" * 60)
print(f"1. V1 K={K} 最终模型 (epochs={EPOCHS})")
print("=" * 60)
t0 = time.time()
model_v1 = FASTopic(num_topics=K, preprocess=PREPROCESS, device="cpu")
model_v1.fit(docs=texts_v1, preset_doc_embeddings=emb_v1, epochs=EPOCHS)
dt1 = time.time() - t0
print(f"训练: {dt1:.0f}s")

tw_raw = model_v1.get_top_words(num_top_words=20)
top_words = [str(w).split() for w in tw_raw]
weights = [float(w) for w in model_v1.get_topic_weights()]

print("\nK=10 主题词:")
for i, tw in enumerate(top_words):
    # 实质主题判定
    edu_kw = {'专业','就业','实习','工作','新传','新闻','媒体','记者','报道','文科',
               '本科','毕业','考研','考公','运营','大厂','学校','岗位','课程','学科',
               '传播','ai','采访','论文','研究','理解','信息','标题','雪峰','已死',
               '理想','劝退','选择','工资','收入','行业','社会','中国','经验','能力'}
    hit = sum(1 for w in tw[:15] if w in edu_kw)
    tag = '★实质' if hit >= 2 else '  噪声'
    print(f"  主题{i}(w={weights[i]:.3f}, {tag}): {tw[:8]}")

# 提取实质主题ID
substantive_ids = [i for i, tw in enumerate(top_words)
                   if sum(1 for w in tw[:15] if w in {'专业','就业','实习','工作','新传','新闻',
                       '媒体','记者','报道','文科','本科','毕业','考研','考公','运营','大厂',
                       '学校','岗位','课程','学科','传播','ai','采访','论文','研究','理解',
                       '标题','雪峰','已死','理想','劝退','选择','工资','收入','行业','社会'}) >= 2]
print(f"\n实质主题ID: {substantive_ids}")

# 保存模型和结果
model_out = {
    "K": K, "epochs": EPOCHS, "seed": SEED, "time_s": dt1,
    "top_words": top_words, "weights": weights,
    "substantive_topic_ids": substantive_ids,
}
with open(os.path.join(OUT_DIR, "fastopic_K10_final.json"), "w", encoding="utf-8") as f:
    json.dump(model_out, f, ensure_ascii=False, indent=2)
with open(os.path.join(OUT_DIR, "fastopic_K10_model.pkl"), "wb") as f:
    pickle.dump(model_v1, f)
print("模型已保存")

# ==========================================================
# 阶段2: V2 (标题+评论) K=10 对照
# ==========================================================
print("\n" + "=" * 60)
print(f"2. V2 (标题+评论) K={K} 对照")
print("=" * 60)
t0 = time.time()
model_v2 = FASTopic(num_topics=K, preprocess=PREPROCESS, device="cpu")
model_v2.fit(docs=texts_v2, preset_doc_embeddings=emb_v2, epochs=EPOCHS)
dt2 = time.time() - t0
print(f"训练: {dt2:.0f}s")

tw_v2 = [str(w).split() for w in model_v2.get_top_words(num_top_words=20)]
w_v2 = [float(w) for w in model_v2.get_topic_weights()]
print("\nV2 主题词:")
for i, tw in enumerate(tw_v2):
    edu_kw = {'专业','就业','实习','工作','新传','新闻','媒体','记者','报道','文科',
               '本科','毕业','考研','考公','运营','大厂','学校','岗位','课程','学科',
               '传播','ai','采访','论文','研究','理解','信息','标题','雪峰','已死',
               '理想','劝退','选择','工资','收入','行业','社会','中国','经验','能力'}
    hit = sum(1 for w in tw[:15] if w in edu_kw)
    tag = '★实质' if hit >= 2 else '  噪声'
    print(f"  主题{i}(w={w_v2[i]:.3f}, {tag}): {tw[:8]}")

# V1 vs V2 共同主题检测
v1_sub_ids = set(substantive_ids)
v2_sub_words = set()
for i, tw in enumerate(tw_v2):
    if sum(1 for w in tw[:15] if w in {'专业','就业','实习','工作','新传','新闻','媒体','记者',
        '报道','文科','本科','毕业','考研','考公','运营','大厂','学校','岗位','课程','学科',
        '传播','ai','采访','论文','研究','理解','标题','雪峰','已死','理想','劝退','选择',
        '工资','收入','行业','社会','中国'}) >= 2:
        v2_sub_words.update(tw[:15])

# Jaccard计算共同词
v1_sub_words_set = set()
for i in v1_sub_ids:
    v1_sub_words_set.update(top_words[i][:15])
common = len(v1_sub_words_set & v2_sub_words) / max(1, len(v1_sub_words_set | v2_sub_words))
print(f"\nV1 vs V2 实质主题词Jaccard: {common:.3f}")

v2_result = {
    "K": K, "epochs": EPOCHS, "time_s": dt2,
    "top_words": tw_v2, "weights": w_v2,
    "v1_v2_jaccard": round(common, 4),
}
with open(os.path.join(OUT_DIR, "fastopic_K10_v2_contrast.json"), "w", encoding="utf-8") as f:
    json.dump(v2_result, f, ensure_ascii=False, indent=2)

# ==========================================================
# 阶段3: BERTopic 对照
# ==========================================================
print("\n" + "=" * 60)
print("3. BERTopic 对照")
print("=" * 60)

vectorizer = CountVectorizer(tokenizer=chinese_tokenize, token_pattern=None,
                             min_df=10, max_df=0.5)
bertopic_model = BERTopic(min_topic_size=30, vectorizer_model=vectorizer,
                          calculate_probabilities=False, verbose=False)
t0 = time.time()
bt_topics, _ = bertopic_model.fit_transform(texts_v1, emb_v1)
dt3 = time.time() - t0
print(f"训练: {dt3:.0f}s")

bt_info = bertopic_model.get_topic_info()
bt_topic_words = {}
for t in bt_info['Topic'].unique():
    if t == -1:
        continue
    words = bertopic_model.get_topic(t)
    bt_topic_words[int(t)] = [w[0] for w in words[:20]]

n_bt = len(bt_topic_words)
print(f"BERTopic发现主题: {n_bt}")
print("BERTopic 主题词:")
for t in sorted(bt_topic_words.keys())[:12]:
    tw = bt_topic_words[t]
    edu_kw = {'专业','就业','实习','工作','新传','新闻','媒体','记者','报道','文科',
               '本科','毕业','考研','考公','运营','大厂','学校','岗位','课程','学科',
               '传播','ai','采访','论文','研究','理解','信息','标题','雪峰','已死',
               '理想','劝退','选择','工资','收入','行业','社会','中国','经验','能力'}
    hit = sum(1 for w in tw[:15] if w in edu_kw)
    tag = '★实质' if hit >= 3 else ''
    print(f"  主题{t}({tag}): {tw[:8]}")

bt_result = {
    "n_topics": n_bt, "time_s": dt3,
    "topic_words": bt_topic_words,
}
with open(os.path.join(OUT_DIR, "bertopic_contrast.json"), "w", encoding="utf-8") as f:
    json.dump(bt_result, f, ensure_ascii=False, indent=2)

# ==========================================================
# 阶段4: 框架偏移 (V1 K=10 模型 + 笔记标题)
# ==========================================================
print("\n" + "=" * 60)
print("4. 笔记—评论框架偏移 (JSD)")
print("=" * 60)

# 评论θ
theta_c = model_v1.transform(doc_embeddings=torch.as_tensor(emb_v1))
theta_c_np = theta_c.numpy() if hasattr(theta_c, 'numpy') else np.array(theta_c)
comments['theta'] = list(theta_c_np)

# 笔记标题embedding
from sentence_transformers import SentenceTransformer
st_model = SentenceTransformer("BAAI/bge-base-zh-v1.5", device="cpu")
note_texts = notes['title'].fillna('').tolist()
t0 = time.time()
emb_notes = st_model.encode(note_texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True)
print(f"笔记embedding: {time.time()-t0:.0f}s")

theta_n = model_v1.transform(doc_embeddings=torch.as_tensor(emb_notes))
theta_n_np = theta_n.numpy() if hasattr(theta_n, 'numpy') else np.array(theta_n)

# JSD
note_to_coms = comments.groupby('note_id')['theta'].apply(list)
results = []
for i, note_id in enumerate(notes['note_id']):
    if note_id not in note_to_coms:
        continue
    cts = note_to_coms[note_id]
    if not cts:
        continue
    theta_note = theta_n_np[i]
    theta_comment_avg = np.mean(np.array(cts), axis=0)
    jsd = float(jensenshannon(theta_note, theta_comment_avg))
    results.append({"note_id": note_id, "title": notes['title'].iloc[i],
                    "jsd": jsd, "n_comments": len(cts)})

df = pd.DataFrame(results).dropna()
jsds = df['jsd']
print(f"有效笔记: {len(df)}")
print(f"JSD: mean={jsds.mean():.4f}, med={jsds.median():.4f}, p25={jsds.quantile(0.25):.4f}, p75={jsds.quantile(0.75):.4f}")
print(f"高偏移(>p75): {(jsds > jsds.quantile(0.75)).sum()}篇")

df.to_csv(os.path.join(OUT_DIR, "framework_shift.csv"), index=False, encoding="utf-8-sig")
with open(os.path.join(OUT_DIR, "framework_shift_summary.json"), "w", encoding="utf-8") as f:
    json.dump({"n_topics": K, "n_notes": len(df),
               "jsd_mean": round(jsds.mean(), 5), "jsd_median": round(jsds.median(), 5),
               "jsd_std": round(jsds.std(), 5),
               "jsd_p25": round(jsds.quantile(0.25), 5),
               "jsd_p75": round(jsds.quantile(0.75), 5)}, f, ensure_ascii=False, indent=2)

print("\n高偏移笔记样本(前5):")
for _, r in df.nlargest(5, 'jsd').iterrows():
    print(f"  [{r['jsd']:.4f}, {r['n_comments']}评论] {str(r['title'])[:50]}")

print("\n" + "=" * 60)
print("Stage 2 一体化流水线完成")
print(f"输出目录: {OUT_DIR}/")
print("=" * 60)
