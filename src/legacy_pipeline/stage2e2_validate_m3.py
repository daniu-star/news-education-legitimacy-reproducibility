# -*- coding: utf-8 -*-
"""
Stage 2e2: BGE-M3 跨模型一致性验证
- 用BGE-M3(568M)对2,000条子样本生成embedding
- 训练FASTopic, 与bge-base-zh-v1.5的主题结构对比
- 验证模型选择不影响核心主题结构
"""
import os
import sys
import json
import time
import warnings

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from fastopic import FASTopic

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMB_DIR = os.path.join(BASE, "analysis", "04_embeddings")
OUT_DIR = os.path.join(BASE, "analysis", "05_topics")

N_SAMPLE = 2000
K = 15


def main():
    print("=" * 60)
    print("Stage 2e2: BGE-M3 跨模型一致性验证")
    print("=" * 60)

    comments = pd.read_csv(os.path.join(EMB_DIR, "corpus_A_comments.csv"))
    texts = comments['text_v1_comment_only'].fillna('').tolist()
    sample_idx = np.random.RandomState(42).choice(len(texts), N_SAMPLE, replace=False)
    sample_texts = [texts[i] for i in sample_idx]
    print(f"子样本: {N_SAMPLE}条")

    # BGE-M3 embedding
    print("用BGE-M3生成子样本embedding...")
    model_m3 = SentenceTransformer("BAAI/bge-m3", device="cpu")
    t0 = time.time()
    emb_m3 = model_m3.encode(sample_texts, batch_size=8, normalize_embeddings=True,
                             convert_to_numpy=True)
    print(f"BGE-M3 embedding: {time.time()-t0:.1f}s")

    # 训练FASTopic
    print(f"训练FASTopic (K={K}) on BGE-M3 embeddings...")
    model = FASTopic(num_topics=K, device="cpu")
    t0 = time.time()
    model.fit(docs=sample_texts, preset_doc_embeddings=emb_m3, epochs=200)
    print(f"训练耗时: {time.time()-t0:.1f}s")
    top_words_m3 = model.get_top_words(num_top_words=30)
    top_words_m3 = [list(map(str, w)) for w in top_words_m3]

    # 对比: bge-base-zh-v1.5 在相同子样本上的主题
    emb_base = np.load(os.path.join(EMB_DIR, "emb_comment_v1.npy"))[sample_idx]
    model2 = FASTopic(num_topics=K, device="cpu")
    model2.fit(docs=sample_texts, preset_doc_embeddings=emb_base, epochs=200)
    top_words_base = model2.get_top_words(num_top_words=30)
    top_words_base = [list(map(str, w)) for w in top_words_base]

    # 主题对应: 计算主题词集合的Jaccard相似度
    print("\n=== 主题对应(Jaccard) ===")
    n_matched = 0
    matched = []
    used = set()
    for i, tw_m3 in enumerate(top_words_m3):
        set_m3 = set(tw_m3[:20])
        best_j = 0
        best_j_idx = -1
        for j, tw_b in enumerate(top_words_base):
            if j in used:
                continue
            set_b = set(tw_b[:20])
            jacc = len(set_m3 & set_b) / max(1, len(set_m3 | set_b))
            if jacc > best_j:
                best_j = jacc
                best_j_idx = j
        if best_j > 0.1 and best_j_idx >= 0:
            n_matched += 1
            used.add(best_j_idx)
            matched.append({"m3_topic": i, "base_topic": best_j_idx, "jaccard": round(best_j, 3)})
            print(f"  主题{i} ↔ 主题{best_j_idx}: Jaccard={best_j:.3f}")

    print(f"\n匹配主题数: {n_matched}/{K}")
    result = {
        "n_sample": N_SAMPLE,
        "n_topics": K,
        "matched_topics": n_matched,
        "match_rate": n_matched / K,
        "topic_matches": matched,
        "m3_top_words": top_words_m3,
        "base_top_words": top_words_base,
    }
    with open(os.path.join(OUT_DIR, "m3_validation.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n验证结果保存至: {OUT_DIR}/m3_validation.json")


if __name__ == "__main__":
    main()
