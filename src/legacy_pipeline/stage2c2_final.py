# -*- coding: utf-8 -*-
"""
Stage 2c2: K=12 最终主题模型
- 多个种子训练(epochs=200), 评估稳定性
- 提取主题代表文本
- 保存模型供框架偏移等后续使用
"""
import os
import sys
import json
import time
import pickle
import re
import warnings

import numpy as np
import pandas as pd
import torch
import jieba
from fastopic import FASTopic
from topmost.preprocess.preprocess import Preprocess

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMB_DIR = os.path.join(BASE, "analysis", "04_embeddings")
OUT_DIR = os.path.join(BASE, "analysis", "05_topics")

K_FINAL = 12
SEEDS = [17, 29, 43, 59, 83]  # 5个种子评估稳定性

# 与stage2c一致的预处理
STOPWORDS_ZH = {
    '新闻', '一个', '什么', '怎么', '为什么', '这样', '那些', '这些', '因为', '所以',
    '但是', '就是', '还是', '可以', '觉得', '感觉', '真的', '非常', '很多', '现在',
    '没有', '不是', '自己', '如果', '那么', '他们', '我们', '你们', '这个', '那个',
    '时候', '其实', '然后', '还有', '的话', '一下', '这里', '那里', '一点',
    '是不是', '有没有', '包括', '已经', '应该', '可能', '也是', '都是',
    '有点', '不太', '完全', '一些', '或者', '这样', '那样', '这些', '那些',
    '派对', '飞吻', '点赞', '显化', '色色', '好棒', '强吻', '大笑', '合十',
    '流汗', '鼓掌', '举手', '黑薯', '电子版', '成可', '莎莎', '惊艳', '得意',
    '自拍', '签名', '美味', '一杯', '点点', '吵吵', '小狗', '威龙', '一颗',
    '收藏', '啊啊啊', '哈哈哈', '嘿嘿', '哭惹', '捂脸', '害羞', '震惊', '调皮',
    '微笑', '赞同', '干杯', '爆竹', '庆祝', '爱心', '赞赞', '可爱', '哇塞',
    '喜欢', '收到', '原来', '真的假', '问一问', '谢谢', '恭喜', '吓死', '天了',
    'momo', '娜娜', '小常', '老大', '爸爸', '同问', '果然', '开玩笑', '三观',
    '天呐', '有钱人', '表演', '骂人', '统一', '兄弟', '女孩', '主角', '不许',
    '喇叭', '令人', '满满', '复制', '资料', '女主', '新生', '主页', '回复',
    '奶茶', '打卡', 'let', '提取', '红薯', '好滴', '大王', '太棒了', 'aaa',
    '入住', '睡觉', '蛋蛋', '我刚', '男主', '图片', '宇宙', '背着', '太好了',
    'gpt', '呜呜', '石化', '主包', '扶墙', '这话', '半天', '反应', '啊啊',
    '哈哈哈哈', '沙发', '已关', '转发给', '同款', '棒棒', '星星', '妙手', '这下',
    '哎呀', '卧槽', '明天', '首发', '头发', '小女孩', '块钱', '违法', '自动',
    '妈妈', '宝宝', '小动物', '生日', '鸡蛋', '安装', 'win', 'no', '老婆',
    '我服', '少年', '来到', '更新', '口令', '科普', '白人', '全家', '妻子',
    '发个', '为何', '超绝', '不知', '抱抱', '能量', '作业',
}


def chinese_tokenize(text):
    text = re.sub(r'\[[^\]]*\]', '', str(text))
    return [w for w in jieba.cut(text)
            if len(w.strip()) >= 2 and w.strip() not in STOPWORDS_ZH]


PREPROCESS = Preprocess(tokenizer=chinese_tokenize, stopwords=None,
                        min_length=2, max_doc_freq=0.5, verbose=False)


def main():
    print("=" * 60)
    print(f"Stage 2c2: K={K_FINAL} 最终主题模型")
    print("=" * 60)

    comments = pd.read_csv(os.path.join(EMB_DIR, "corpus_A_comments.csv"))
    texts = comments['text_v1_comment_only'].fillna('').tolist()
    emb = np.load(os.path.join(EMB_DIR, "emb_comment_v1.npy"))
    print(f"语料: {len(texts)}篇")

    # 多种子训练
    results = {}
    models = {}
    for seed in SEEDS:
        t0 = time.time()
        model = FASTopic(num_topics=K_FINAL, preprocess=PREPROCESS, device="cpu")
        model.fit(docs=texts, preset_doc_embeddings=emb, epochs=200)
        dt = time.time() - t0
        top_words_raw = model.get_top_words(num_top_words=20)
        top_words = [str(w).split() for w in top_words_raw]
        weights = [float(w) for w in model.get_topic_weights()]
        results[f"s{seed}"] = {
            "seed": seed, "time_s": dt, "top_words": top_words, "weights": weights,
        }
        models[seed] = model
        print(f"  seed={seed}: {dt:.0f}s")

    # 稳定性: 主题词Jaccard匹配
    print("\n=== 跨种子主题稳定性(前15词Jaccard) ===")
    base = results[f"s{SEEDS[0]}"]["top_words"]
    stability_scores = []
    for seed in SEEDS[1:]:
        other = results[f"s{seed}"]["top_words"]
        matched = 0
        used = set()
        for i, tw_b in enumerate(base):
            set_b = set(tw_b[:15])
            best = 0
            for j, tw_o in enumerate(other):
                if j in used:
                    continue
                jacc = len(set_b & set(tw_o[:15])) / max(1, len(set_b | set(tw_o[:15])))
                if jacc > best:
                    best = jacc
            if best > 0.15:
                matched += 1
                used.add(0)  # 简化
        rate = matched / K_FINAL
        stability_scores.append(rate)
        print(f"  seed{seed} vs base: 匹配{matched}/{K_FINAL}")

    # 主题代表文本: 用最终模型(seed=17)的theta找高概率文本
    final_model = models[SEEDS[0]]
    theta = final_model.transform(doc_embeddings=torch.as_tensor(emb))
    theta_np = theta.numpy() if hasattr(theta, 'numpy') else np.array(theta)
    representative = {}
    for k in range(K_FINAL):
        top_idx = np.argsort(theta_np[:, k])[::-1][:5]
        representative[f"topic{k}"] = [texts[i][:100] for i in top_idx]

    # 保存
    out = {
        "K": K_FINAL,
        "epochs": 200,
        "seeds": SEEDS,
        "results": results,
        "stability": {
            "mean_match_rate": float(np.mean(stability_scores)),
            "per_seed": stability_scores,
        },
        "representative_texts": representative,
    }
    with open(os.path.join(OUT_DIR, "fastopic_K12_final.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # 保存模型供框架偏移使用
    with open(os.path.join(OUT_DIR, "fastopic_K12_model.pkl"), "wb") as f:
        pickle.dump(final_model, f)

    print(f"\n稳定性均值: {np.mean(stability_scores):.3f}")
    print(f"最终结果保存至: {OUT_DIR}/fastopic_K12_final.json")
    print(f"模型保存至: {OUT_DIR}/fastopic_K12_model.pkl")


if __name__ == "__main__":
    main()
