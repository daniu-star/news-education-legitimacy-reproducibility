# -*- coding: utf-8 -*-
"""
Stage 2 补全: 三个缺失项
1. K=10 稳定性验证 (5 seeds, epochs=100)
2. BGE-M3 跨模型验证 (2000条子样本)
3. V3 笔记标题+标签 主题模型 (K=10)
"""
import os, sys, json, time, re, warnings
from collections import Counter
import numpy as np, pandas as pd, torch, jieba
from fastopic import FASTopic
from topmost.preprocess.preprocess import Preprocess

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMB_DIR = os.path.join(BASE, "analysis", "04_embeddings")
OUT_DIR = os.path.join(BASE, "analysis", "05_topics")
K = 10
EPOCHS = 100
SEEDS = [17, 29, 43, 59, 83]

# ---- 共享预处理 ----
STOPWORDS = set('''新闻 一个 什么 怎么 为什么 这样 那些 这些 因为 所以 但是 就是 还是 可以 觉得 感觉 真的 非常 很多 现在 没有 不是 自己 如果 那么 他们 我们 你们 这个 那个 时候 其实 然后 还有 的话 一下 这里 那里 一点 是不是 有没有 包括 已经 应该 可能 也是 都是 有点 不太 完全 一些 或者 这样 那样 这些 那些 派对 飞吻 点赞 显化 色色 好棒 强吻 大笑 合十 流汗 鼓掌 举手 黑薯 电子版 成可 莎莎 惊艳 得意 自拍 签名 美味 一杯 点点 吵吵 小狗 威龙 一颗 收藏 啊啊啊 哈哈哈 嘿嘿 哭惹 捂脸 害羞 震惊 调皮 微笑 赞同 干杯 爆竹 庆祝 爱心 赞赞 可爱 哇塞 喜欢 收到 原来 真的假 问一问 谢谢 恭喜 吓死 天了 momo 娜娜 小常 老大 爸爸 同问 果然 开玩笑 三观 天呐 有钱人 表演 骂人 统一 兄弟 女孩 主角 不许 喇叭 令人 满满 复制 资料 女主 新生 主页 回复 奶茶 打卡 let 提取 红薯 好滴 大王 太棒了 aaa 入住 睡觉 蛋蛋 我刚 男主 图片 宇宙 背着 太好了 gpt 呜呜 石化 主包 扶墙 这话 半天 反应 啊啊 哈哈哈哈 沙发 已关 转发给 同款 棒棒 星星 妙手 这下 哎呀 卧槽 明天 首发 头发 小女孩 块钱 违法 自动 妈妈 宝宝 小动物 生日 鸡蛋 安装 win no 老婆 我服 少年 来到 更新 口令 科普 白人 全家 妻子 发个 为何 超绝 不知 抱抱 能量 作业'''.split())


def chinese_tokenize(text):
    text = re.sub(r'\[[^\]]*\]', '', str(text))
    return [w for w in jieba.cut(text) if len(w.strip()) >= 2 and w.strip() not in STOPWORDS]


PREPROCESS = Preprocess(tokenizer=chinese_tokenize, min_length=2, max_doc_freq=0.5, verbose=False)

# 实质主题判定词
edu_kw = {'专业','就业','实习','工作','新传','新闻','媒体','记者','报道','文科',
           '本科','毕业','考研','考公','运营','大厂','学校','岗位','课程','学科',
           '传播','ai','采访','论文','研究','理解','信息','标题','雪峰','已死',
           '理想','劝退','选择','工资','收入','行业','社会','中国','经验','能力','内容','流量'}

# ---- 1. K=10 稳定性(5 seeds) ----
print("=" * 60)
print("1. K=10 稳定性验证 (5 seeds)")
print("=" * 60)

comments = pd.read_csv(os.path.join(EMB_DIR, "corpus_A_comments.csv"))
texts_v1 = comments['text_v1_comment_only'].fillna('').tolist()
emb_v1 = np.load(os.path.join(EMB_DIR, "emb_comment_v1.npy"))

stability_results = {}
for seed in SEEDS:
    t0 = time.time()
    model = FASTopic(num_topics=K, preprocess=PREPROCESS, device="cpu")
    model.fit(docs=texts_v1, preset_doc_embeddings=emb_v1, epochs=EPOCHS)
    dt = time.time() - t0
    tw = [str(w).split() for w in model.get_top_words(num_top_words=20)]
    wts = [float(w) for w in model.get_topic_weights()]
    sub_ids = []
    for i in range(K):
        hit = sum(1 for w in tw[i][:15] if w in edu_kw)
        if hit >= 2:
            sub_ids.append(i)
    stability_results[f"s{seed}"] = {
        "seed": seed, "time_s": dt, "top_words": tw, "weights": wts,
        "substantive_ids": sub_ids, "n_substantive": len(sub_ids)
    }
    print(f"  seed={seed}: {dt:.0f}s, 实质={sub_ids}, top_topic_head={tw[sub_ids[0]][:5] if sub_ids else 'N/A'}")

# 跨种子一致性
n_seeds = len(stability_results)
sub_count = [v['n_substantive'] for v in stability_results.values()]
print(f"\n实质主题数: mean={np.mean(sub_count):.1f}, range=[{min(sub_count)},{max(sub_count)}]")

# 共同词检测
common_words = Counter()
for v in stability_results.values():
    for sid in v['substantive_ids']:
        common_words.update(v['top_words'][sid][:15])
print(f"跨种子高频实质词(≥3种子): {[w for w,c in common_words.most_common(15) if c>=3]}")

with open(os.path.join(OUT_DIR, "k10_stability_validation.json"), "w", encoding="utf-8") as f:
    json.dump(stability_results, f, ensure_ascii=False, indent=2)

# ---- 2. BGE-M3 跨模型验证 ----
print("\n" + "=" * 60)
print("2. BGE-M3 跨模型验证 (2000子样本)")
print("=" * 60)

from sentence_transformers import SentenceTransformer

n_sample = 2000
rng = np.random.RandomState(42)
sample_idx = rng.choice(len(texts_v1), n_sample, replace=False)
sample_texts = [texts_v1[i] for i in sample_idx]

print("生成 BGE-M3 embedding...")
model_m3 = SentenceTransformer("BAAI/bge-m3", device="cpu")
emb_m3 = model_m3.encode(sample_texts, batch_size=8, normalize_embeddings=True, convert_to_numpy=True)
print(f"  M3 embedding: {emb_m3.shape}")

t0 = time.time()
model_m3_topic = FASTopic(num_topics=K, preprocess=PREPROCESS, device="cpu")
model_m3_topic.fit(docs=sample_texts, preset_doc_embeddings=emb_m3, epochs=EPOCHS)
dt_m3 = time.time() - t0
tw_m3 = [str(w).split() for w in model_m3_topic.get_top_words(num_top_words=20)]

# BGE-base 对照
emb_base = emb_v1[sample_idx]
model_base_topic = FASTopic(num_topics=K, preprocess=PREPROCESS, device="cpu")
model_base_topic.fit(docs=sample_texts, preset_doc_embeddings=emb_base, epochs=EPOCHS)
tw_base = [str(w).split() for w in model_base_topic.get_top_words(num_top_words=20)]

# 主题匹配
matched = 0
matched_pairs = []
for i, tw_m in enumerate(tw_m3):
    set_m = set(tw_m[:15])
    best_j, best_jac = -1, 0
    for j, tw_b in enumerate(tw_base):
        set_b = set(tw_b[:15])
        jac = len(set_m & set_b) / max(1, len(set_m | set_b))
        if jac > best_jac:
            best_jac, best_j = jac, j
    if best_jac > 0.15:
        matched += 1
        matched_pairs.append({"m3_topic": i, "base_topic": best_j, "jaccard": round(best_jac, 3)})

print(f"BGE-M3训练: {dt_m3:.0f}s")
print(f"主题匹配: {matched}/{K} (Jaccard>0.15)")
for p in matched_pairs:
    print(f"  M3主题{p['m3_topic']} ↔ Base主题{p['base_topic']}: J={p['jaccard']}")
print(f"\nM3主题词:")
for i, tw in enumerate(tw_m3):
    hit = sum(1 for w in tw[:15] if w in edu_kw)
    tag = '★' if hit >= 2 else ' '
    print(f"  M3主题{i}{tag}: {tw[:8]}")
print(f"\nBase主题词(子样本):")
for i, tw in enumerate(tw_base):
    hit = sum(1 for w in tw[:15] if w in edu_kw)
    tag = '★' if hit >= 2 else ' '
    print(f"  Base主题{i}{tag}: {tw[:8]}")

with open(os.path.join(OUT_DIR, "m3_validation.json"), "w", encoding="utf-8") as f:
    json.dump({
        "n_sample": n_sample, "matched_rate": matched / K,
        "pairs": matched_pairs, "m3_top_words": tw_m3, "base_top_words": tw_base,
    }, f, ensure_ascii=False, indent=2)

# ---- 3. V3 笔记标题+标签 K=10 ----
print("\n" + "=" * 60)
print("3. V3 笔记标题+标签 主题模型")
print("=" * 60)

notes = pd.read_csv(os.path.join(EMB_DIR, "corpus_A_notes.csv"))
texts_v3 = notes['text_v3_title_tag'].fillna('').tolist()
emb_v3 = np.load(os.path.join(EMB_DIR, "emb_note_v3.npy"))

t0 = time.time()
model_v3 = FASTopic(num_topics=K, preprocess=PREPROCESS, device="cpu")
model_v3.fit(docs=texts_v3, preset_doc_embeddings=emb_v3, epochs=EPOCHS)
dt_v3 = time.time() - t0
tw_v3 = [str(w).split() for w in model_v3.get_top_words(num_top_words=20)]
w_v3 = [float(w) for w in model_v3.get_topic_weights()]

print(f"V3训练: {dt_v3:.0f}s")
for i in range(K):
    hit = sum(1 for w in tw_v3[i][:15] if w in edu_kw)
    tag = '★实质' if hit >= 2 else '  噪声'
    print(f"  主题{i}(w={w_v3[i]:.3f}, {tag}): {tw_v3[i][:8]}")

v3_result = {"K": K, "epochs": EPOCHS, "time_s": dt_v3, "top_words": tw_v3, "weights": w_v3}
with open(os.path.join(OUT_DIR, "fastopic_K10_v3_notes.json"), "w", encoding="utf-8") as f:
    json.dump(v3_result, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 60)
print("Stage 2 补全完成")
print("=" * 60)
