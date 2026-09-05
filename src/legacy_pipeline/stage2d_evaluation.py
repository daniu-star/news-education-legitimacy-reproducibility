# -*- coding: utf-8 -*-
"""
Stage 2d: 主题质量评价
- NPMI: 主题词一致性 (每个主题单独计算再平均)
- 主题多样性 TD
- 主题稳定性: 跨种子主题匹配(余弦相似度)
"""
import os
import json
import math
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMB_DIR = os.path.join(BASE, "analysis", "04_embeddings")
OUT_DIR = os.path.join(BASE, "analysis", "05_topics")


def tokenize(text):
    return re.findall(r'[一-鿿]{2,}', str(text))


def compute_npmi_single(topic_words, n_docs, word_doc_freq, word_pair_freq):
    """单个主题的NPMI"""
    npmi_scores = []
    M = len(topic_words)
    if M < 2:
        return 0.0
    for a in range(M):
        for b in range(a+1, M):
            wa, wb = topic_words[a], topic_words[b]
            if wa == wb:
                continue
            p_wa = word_doc_freq.get(wa, 0) / n_docs
            p_wb = word_doc_freq.get(wb, 0) / n_docs
            pair_key = tuple(sorted([wa, wb]))
            p_ab = word_pair_freq.get(pair_key, 0) / n_docs
            if p_ab <= 0 or p_wa <= 0 or p_wb <= 0:
                continue
            pmi = math.log(p_ab / (p_wa * p_wb))
            npmi = pmi / (-math.log(p_ab))
            npmi_scores.append(npmi)
    return float(np.mean(npmi_scores)) if npmi_scores else 0.0


def topic_diversity(all_topic_words, M=30):
    """主题多样性: 唯一top词比例 (all_topic_words: 主题数×词数)"""
    all_words = set()
    for tw in all_topic_words:
        all_words.update(tw[:M])
    unique = len(all_words)
    total = len(all_topic_words) * M
    return unique / total if total else 0.0


def main():
    print("=" * 60)
    print("Stage 2d: 主题质量评价")
    print("=" * 60)

    # 词频统计
    comments = pd.read_csv(os.path.join(EMB_DIR, "corpus_A_comments.csv"))
    all_texts = comments['text_v1_comment_only'].fillna('').tolist()
    n_docs = len(all_texts)
    print(f"词频统计 ({n_docs}篇)...")
    word_doc_freq = Counter()
    word_pair_freq = Counter()
    for i, text in enumerate(all_texts):
        words = list(set(tokenize(text)))
        for w in words:
            word_doc_freq[w] += 1
        for a in range(len(words)):
            for b in range(a+1, len(words)):
                key = tuple(sorted([words[a], words[b]]))
                word_pair_freq[key] += 1
        if (i+1) % 5000 == 0:
            print(f"  {i+1}/{n_docs}")

    # 读取网格结果
    results_file = os.path.join(OUT_DIR, "fastopic_v1_results.json")
    if not os.path.exists(results_file):
        print("未找到FASTopic结果")
        return
    with open(results_file, encoding="utf-8") as f:
        results = json.load(f)

    by_k = defaultdict(list)
    for key, r in results.items():
        if "error" in r:
            continue
        by_k[r["K"]].append(r)

    print("\n=== 各K的评价指标 ===")
    k_eval = {}
    for K in sorted(by_k.keys()):
        items = by_k[K]
        npmis, tds = [], []
        all_topic_words_for_td = []
        for r in items:
            topic_words = r["top_words"]  # [K个主题, 每主题词列表]
            all_topic_words_for_td.extend(topic_words)
            # 每主题NPMI, 再平均
            topic_npmis = [compute_npmi_single(tw, n_docs, word_doc_freq, word_pair_freq)
                           for tw in topic_words]
            npmis.append(float(np.mean(topic_npmis)))
        td = topic_diversity(all_topic_words_for_td)
        k_eval[K] = {
            "n_runs": len(items),
            "avg_npmi": float(np.mean(npmis)) if npmis else None,
            "npmi_std": float(np.std(npmis)) if npmis else None,
            "topic_diversity": td,
        }
        print(f"K={K}: runs={len(items)}, NPMI={np.mean(npmis):.4f}±{np.std(npmis):.4f}, TD={td:.3f}")

    with open(os.path.join(OUT_DIR, "topic_evaluation.json"), "w", encoding="utf-8") as f:
        json.dump(k_eval, f, ensure_ascii=False, indent=2)
    print("\n评价结果保存至 analysis/05_topics/topic_evaluation.json")


if __name__ == "__main__":
    main()
