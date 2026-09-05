# -*- coding: utf-8 -*-
"""
Stage 2e: 对照主题模型验证（BERTopic）
- 使用BGE-M3 embedding + BERTopic (中文分词)
- 验证FASTopic主题结构的跨模型稳健性
"""
import os
import sys
import json
import time
import re
import warnings

import numpy as np
import pandas as pd
import jieba
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMB_DIR = os.path.join(BASE, "analysis", "04_embeddings")
OUT_DIR = os.path.join(BASE, "analysis", "05_topics")


def chinese_tokenizer(text):
    text = re.sub(r'\[[^\]]*\]', '', str(text))
    return [w for w in jieba.cut(text) if len(w.strip()) >= 2]


def main():
    print("=" * 60)
    print("Stage 2e: BERTopic 对照验证")
    print("=" * 60)

    emb = np.load(os.path.join(EMB_DIR, "emb_comment_v1.npy"))
    comments = pd.read_csv(os.path.join(EMB_DIR, "corpus_A_comments.csv"))
    texts = comments['text_v1_comment_only'].fillna('').tolist()
    print(f"语料: {len(texts)}条")

    # 自定义中文CountVectorizer
    vectorizer = CountVectorizer(tokenizer=chinese_tokenizer, token_pattern=None,
                                 min_df=10, max_df=0.5)
    model = BERTopic(
        calculate_probabilities=False,
        min_topic_size=30,
        vectorizer_model=vectorizer,
        verbose=False,
    )
    t0 = time.time()
    topics, probs = model.fit_transform(texts, emb)
    elapsed = time.time() - t0
    print(f"BERTopic训练: {elapsed:.0f}s")

    topic_info = model.get_topic_info()
    topic_words = {}
    for t in topic_info['Topic'].unique():
        if t == -1:
            continue
        words = model.get_topic(t)
        topic_words[int(t)] = [w[0] for w in words[:20]]

    n_topics = len(topic_words)
    print(f"发现主题数: {n_topics} (含离群topic -1)")

    # 保存
    result = {
        "n_topics_found": n_topics,
        "time_s": elapsed,
        "topic_words": topic_words,
        "n_docs_in_topics": {int(t): int(c) for t, c in zip(topic_info['Topic'], topic_info['Count'])
                             if int(t) != -1},
    }
    with open(os.path.join(OUT_DIR, "bertopic_contrast.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 打印主题词概览
    print("\nBERTopic主题词:")
    for t in sorted(topic_words.keys())[:15]:
        print(f"  主题{t}: {topic_words[t][:8]}")

    print(f"\n对照结果保存至: {OUT_DIR}/bertopic_contrast.json")


if __name__ == "__main__":
    main()
