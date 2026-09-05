# -*- coding: utf-8 -*-
"""
Stage 2f: 笔记—评论框架偏移分析
- 加载K=12最终模型(pickle)
- 笔记主题分布: 笔记标题embedding → transform
- 评论区聚合分布: 该笔记下评论的θ加权平均
- JSD: 衡量笔记设置与评论区讨论的框架偏移
"""
import os
import sys
import json
import pickle
import time
import warnings

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import jensenshannon

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMB_DIR = os.path.join(BASE, "analysis", "04_embeddings")
OUT_DIR = os.path.join(BASE, "analysis", "05_topics")

K = 12


def jensen_shannon(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.sum() == 0 or q.sum() == 0:
        return np.nan
    p = p / p.sum()
    q = q / q.sum()
    return float(jensenshannon(p, q))


def main():
    print("=" * 60)
    print("Stage 2f: 笔记—评论框架偏移")
    print("=" * 60)

    # 1. 加载模型
    model_path = os.path.join(OUT_DIR, "fastopic_K12_model.pkl")
    if not os.path.exists(model_path):
        print("未找到K=12模型, 请先运行stage2c2_final.py")
        return
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"模型加载: K={model.num_topics if hasattr(model,'num_topics') else K}")

    # 2. 评论θ
    comments = pd.read_csv(os.path.join(EMB_DIR, "corpus_A_comments.csv"))
    notes = pd.read_csv(os.path.join(EMB_DIR, "corpus_A_notes.csv"))
    emb_v1 = np.load(os.path.join(EMB_DIR, "emb_comment_v1.npy"))
    theta_comments = model.transform(doc_embeddings=torch.as_tensor(emb_v1))
    theta_np = theta_comments.numpy() if hasattr(theta_comments, 'numpy') else np.array(theta_comments)
    comments['theta'] = list(theta_np)
    print(f"评论θ: {theta_np.shape}")

    # 3. 笔记标题embedding
    from sentence_transformers import SentenceTransformer
    st_model = SentenceTransformer("BAAI/bge-base-zh-v1.5", device="cpu")
    note_texts = notes['title'].fillna('').tolist()
    print("生成笔记标题embedding...")
    emb_notes = st_model.encode(note_texts, batch_size=32, normalize_embeddings=True,
                                convert_to_numpy=True)
    theta_notes = model.transform(doc_embeddings=torch.as_tensor(emb_notes))
    theta_notes_np = theta_notes.numpy() if hasattr(theta_notes, 'numpy') else np.array(theta_notes)
    print(f"笔记标题θ: {theta_notes_np.shape}")

    # 4. 框架偏移
    note_to_comments = comments.groupby('note_id')['theta'].apply(list)
    results = []
    for i, note_id in enumerate(notes['note_id']):
        if note_id not in note_to_comments:
            continue
        comment_thetas = note_to_comments[note_id]
        if not comment_thetas:
            continue
        theta_note = theta_notes_np[i]
        theta_comment_avg = np.mean(np.array(comment_thetas), axis=0)
        jsd = jensen_shannon(theta_note, theta_comment_avg)
        results.append({
            "note_id": note_id,
            "title": notes['title'].iloc[i],
            "jsd": jsd,
            "n_comments": len(comment_thetas),
            "note_theta": theta_note.tolist(),
            "comment_theta_avg": theta_comment_avg.tolist(),
        })

    df = pd.DataFrame(results).dropna(subset=['jsd'])
    print(f"\n有效笔记(有评论): {len(df)}")

    jsds = df['jsd']
    print(f"JSD: mean={jsds.mean():.4f}, median={jsds.median():.4f}, p25={jsds.quantile(0.25):.4f}, p75={jsds.quantile(0.75):.4f}")
    print(f"高偏移(>p75): {(jsds > jsds.quantile(0.75)).sum()}篇, 低偏移(<p25): {(jsds < jsds.quantile(0.25)).sum()}篇")

    # 5. 保存
    df.to_csv(os.path.join(OUT_DIR, "framework_shift.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(OUT_DIR, "framework_shift_summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "n_topics": K,
            "n_notes": int(len(df)),
            "jsd_mean": float(jsds.mean()),
            "jsd_median": float(jsds.median()),
            "jsd_std": float(jsds.std()),
            "jsd_p25": float(jsds.quantile(0.25)),
            "jsd_p75": float(jsds.quantile(0.75)),
        }, f, ensure_ascii=False, indent=2)

    print(f"\n框架偏移结果保存至: {OUT_DIR}/framework_shift.csv")
    print("\n高偏移笔记样本(前8):")
    for _, r in df.nlargest(8, 'jsd').iterrows():
        print(f"  [{r['jsd']:.3f}, {r['n_comments']}评论] {str(r['title'])[:40]}")


if __name__ == "__main__":
    main()
