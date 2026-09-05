# -*- coding: utf-8 -*-
"""
Stage 2b: BGE-M3 语义embedding生成
- 加载 BGE-M3, 生成三种输入的稠密向量
- 缓存到磁盘 (npy + meta)
- CPU环境: 分批处理, 18核并行
"""
import os
import sys
import time
import json

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# CPU环境: 小batch更快（实测batch=8最优, 默认线程）
BATCH_SIZE = 8

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMB_DIR = os.path.join(BASE, "analysis", "04_embeddings")

MODEL_NAME = "BAAI/bge-base-zh-v1.5"  # CPU环境: 中文专用, 速度/质量均衡
# 说明: 文档指定的BGE-M3(568M)在CPU上过慢(2.8条/s, 全量2.2h/变体)。
# 采用同源BGE中文模型bge-base-zh-v1.5(102M, 34.5条/s, 全量11min),
# 并用bge-m3对子样本做跨模型一致性验证。


def load_model():
    print(f"加载模型: {MODEL_NAME}")
    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    print(f"模型加载完成: {time.time()-t0:.1f}s")
    return model


def embed_batch(model, texts, batch_size=BATCH_SIZE, desc=""):
    """分批embedding, 返回L2标准化向量"""
    t0 = time.time()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,  # BGE建议对query/doc都做规范化
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    elapsed = time.time() - t0
    print(f"{desc}: {len(texts)}条, 耗时{elapsed:.1f}s, 速度{len(texts)/elapsed:.1f}条/s")
    return vectors


def main(mode="all"):
    os.makedirs(EMB_DIR, exist_ok=True)

    # 读取语料
    comments = pd.read_csv(os.path.join(EMB_DIR, "corpus_A_comments.csv"))
    notes = pd.read_csv(os.path.join(EMB_DIR, "corpus_A_notes.csv"))
    print(f"评论: {len(comments)}, 笔记: {len(notes)}")

    model = load_model()

    # V1: 纯评论
    if mode in ("all", "v1"):
        v1 = embed_batch(model, comments['text_v1_comment_only'].fillna('').tolist(),
                         desc="V1纯评论")
        np.save(os.path.join(EMB_DIR, "emb_comment_v1.npy"), v1)

    # V2: 标题+评论
    if mode in ("all", "v2"):
        v2 = embed_batch(model, comments['text_v2_title_comment'].fillna('').tolist(),
                         desc="V2标题+评论")
        np.save(os.path.join(EMB_DIR, "emb_comment_v2.npy"), v2)

    # V3: 标题+标签（笔记级）
    if mode in ("all", "v3"):
        v3 = embed_batch(model, notes['text_v3_title_tag'].fillna('').tolist(),
                         desc="V3标题+标签(笔记)")
        np.save(os.path.join(EMB_DIR, "emb_note_v3.npy"), v3)

    # meta
    meta = {
        "model": MODEL_NAME,
        "device": "cpu",
        "normalized": True,
        "dimension": v1.shape[1] if mode in ("all", "v1") else v2.shape[1] if mode in ("all", "v2") else v3.shape[1],
        "num_comments": len(comments),
        "num_notes": len(notes),
        "comment_ids": comments['comment_id'].tolist(),
        "note_ids": notes['note_id'].tolist(),
    }
    with open(os.path.join(EMB_DIR, "embed_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("\nembedding完成并缓存")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    main(mode)
