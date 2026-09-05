# -*- coding: utf-8 -*-
"""
Stage 2c: FASTopic 主题网格搜索
- 输入: BGE-M3 embedding (V1纯评论 / V2标题+评论 / V3标题+标签)
- 网格: K={8,10,12,15,18,20,24}, 每K多个随机种子
- 输出: 主题词、文档主题分布、评价指标(NPMI/多样性/稳定性)
"""
import os
import sys
import time
import json
import warnings

import re

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
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [17, 29, 43, 59, 71, 83, 97, 109, 127, 149]
K_GRID = [8, 10, 12, 15, 18, 20, 24]

# 中文preprocess: jieba分词, 过滤停用词与单字
# 含小红书表情/互动词（[派对R][飞吻R]等被提取的文字形式, momo为默认用户名）
STOPWORDS_ZH = {
    # 通用停用词
    '新闻', '一个', '什么', '怎么', '为什么', '这样', '那些', '这些', '因为', '所以',
    '但是', '就是', '还是', '可以', '觉得', '感觉', '真的', '非常', '很多', '现在',
    '没有', '不是', '自己', '如果', '那么', '他们', '我们', '你们', '这个', '那个',
    '时候', '其实', '然后', '还有', '的话', '一下', '这里', '那里', '一点',
    '是不是', '有没有', '包括', '已经', '应该', '可能', '也是', '都是',
    '有点', '不太', '完全', '一些', '或者', '这样', '那样', '这些', '那些',
    # 小红书表情/互动词
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
    # 先移除[表情]如[派对R][笑哭R]
    text = re.sub(r'\[[^\]]*\]', '', str(text))
    return [w for w in jieba.cut(text)
            if len(w.strip()) >= 2 and w.strip() not in STOPWORDS_ZH]


PREPROCESS = Preprocess(tokenizer=chinese_tokenize, stopwords=None,
                        min_length=2, max_doc_freq=0.5, verbose=False)


def load_data(variant="v1"):
    """加载embedding和对应文本"""
    if variant == "v1":
        emb = np.load(os.path.join(EMB_DIR, "emb_comment_v1.npy"))
        docs_df = pd.read_csv(os.path.join(EMB_DIR, "corpus_A_comments.csv"))
        texts = docs_df['text_v1_comment_only'].fillna('').tolist()
    elif variant == "v2":
        emb = np.load(os.path.join(EMB_DIR, "emb_comment_v2.npy"))
        docs_df = pd.read_csv(os.path.join(EMB_DIR, "corpus_A_comments.csv"))
        texts = docs_df['text_v2_title_comment'].fillna('').tolist()
    elif variant == "v3":
        emb = np.load(os.path.join(EMB_DIR, "emb_note_v3.npy"))
        docs_df = pd.read_csv(os.path.join(EMB_DIR, "corpus_A_notes.csv"))
        texts = docs_df['text_v3_title_tag'].fillna('').tolist()
    else:
        raise ValueError(variant)
    return texts, emb, docs_df


def run_fastopic(texts, emb, K, seed, epochs=100, device="cpu"):
    """运行单次FASTopic训练"""
    model = FASTopic(num_topics=K, preprocess=PREPROCESS, device=device)
    model.fit(docs=texts, preset_doc_embeddings=emb,
              epochs=epochs)
    # 主题词 (get_top_words返回List[str], 每主题一个空格分隔词串)
    top_words_raw = model.get_top_words(num_top_words=30)
    top_words = [str(w).split() for w in top_words_raw]
    # 文档主题分布 (n_docs, K) — 需转torch.Tensor
    theta = model.transform(doc_embeddings=torch.as_tensor(emb))
    # 主题权重
    weights = model.get_topic_weights()
    return {
        "model": model,
        "top_words": top_words,
        "theta": theta,
        "weights": weights,
    }


def main(variant="v1", k_list=None, seeds=None, test_mode=False, epochs=100):
    if k_list is None:
        k_list = K_GRID
    if seeds is None:
        seeds = SEEDS

    texts, emb, docs_df = load_data(variant)
    print(f"变体{variant}: {len(texts)}篇, embedding shape: {emb.shape}")
    print(f"网格: K={k_list}, 种子={seeds}, epochs={epochs}")

    results = {}
    for K in k_list:
        for seed in seeds:
            key = f"K{K}_s{seed}"
            t0 = time.time()
            try:
                res = run_fastopic(texts, emb, K, seed, epochs=epochs)
                elapsed = time.time() - t0
                results[key] = {
                    "K": K,
                    "seed": seed,
                    "time_s": elapsed,
                    "top_words": [list(map(str, words[:30])) for words in res["top_words"]],
                    "weights": [float(w) for w in res["weights"]],
                }
                print(f"  {key}: {elapsed:.1f}s")
                # 不保存theta(大), 只保存指标
            except Exception as e:
                print(f"  {key}: ERROR {e}")
                results[key] = {"K": K, "seed": seed, "error": str(e)}
            if test_mode and len(results) >= 2:
                break

    # 保存结果
    out = os.path.join(OUT_DIR, f"fastopic_{variant}_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果保存至: {out}")
    return results


if __name__ == "__main__":
    variant = sys.argv[1] if len(sys.argv) > 1 else "v1"
    if len(sys.argv) > 2:
        k_list = [int(x) for x in sys.argv[2].split(",")]
    else:
        k_list = None
    seeds = None
    if "--seeds" in sys.argv:
        idx = sys.argv.index("--seeds")
        seeds = [int(x) for x in sys.argv[idx+1].split(",")]
    epochs = 100
    if "--epochs" in sys.argv:
        idx = sys.argv.index("--epochs")
        epochs = int(sys.argv[idx+1])
    test = "--test" in sys.argv
    main(variant=variant, k_list=k_list, seeds=seeds, test_mode=test, epochs=epochs)
