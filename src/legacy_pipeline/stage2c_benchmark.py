# -*- coding: utf-8 -*-
"""
Stage 2c-benchmark: FASTopic CPU训练耗时基准测试
- 用子样本测量单次训练耗时, 估算全量网格搜索时间
- 决定网格规模
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from fastopic import FASTopic

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMB_DIR = os.path.join(BASE, "analysis", "04_embeddings")


def bench(n_docs, K, epochs=200):
    """测量n_docs文档的训练耗时"""
    emb = np.load(os.path.join(EMB_DIR, "emb_comment_v1.npy"))
    comments = pd.read_csv(os.path.join(EMB_DIR, "corpus_A_comments.csv"))
    texts = comments['text_v1_comment_only'].fillna('').tolist()[:n_docs]
    emb_sub = emb[:n_docs]
    print(f"\n测试: {n_docs}篇文档, K={K}, epochs={epochs}")
    model = FASTopic(num_topics=K, device="cpu")
    t0 = time.time()
    model.fit(docs=texts, preset_doc_embeddings=emb_sub, epochs=epochs)
    dt = time.time() - t0
    print(f"耗时: {dt:.1f}s ({dt/epochs:.2f}s/epoch)")
    # 外推全量(21779)
    if n_docs > 0:
        extrap = dt / n_docs * 21779
        print(f"外推全量21779篇: {extrap:.0f}s = {extrap/60:.1f}分钟")
    return dt


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    bench(n, K)
