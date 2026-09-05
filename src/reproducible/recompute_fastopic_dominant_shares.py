"""Recompute the FASTopic dominant-topic share table.

The manuscript percentages are dominant-topic shares, not FASTopic topic weights:
for each document, take argmax(theta), count the assigned topic, and divide by the
number of semantic-eligible comments.

The script needs the private embedding and corpus files. It never writes row-level
text, embeddings, theta, or document IDs to the public result directory.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import jieba
import numpy as np
import pandas as pd
import torch
from fastopic import FASTopic
from topmost.preprocess.preprocess import Preprocess


LEGACY_COUNTS = {8: 5418, 9: 6568}
EMPLOYMENT_WORDS = {"就业", "实习", "岗位", "工作", "专业", "职业", "运营", "行业", "工资", "考公"}
AI_WORDS = {"AI", "ai", "人工智能", "替代", "取代", "ChatGPT", "chatgpt", "豆包", "机器"}
STOPWORDS = set("""新闻 一个 什么 怎么 为什么 这样 那些 这些 因为 所以 但是 就是 还是 可以 觉得 感觉 真的 非常 很多 现在 没有 不是 自己 如果 那么 他们 我们 你们 这个 那个 时候 其实 然后 还有 的话 一下 这里 那里 一点 是不是 有没有 包括 已经 应该 可能 也是 都是 有点 不太 完全 一些 或者 派对 飞吻 点赞 显化 色色 好棒 强吻 大笑 合十 流汗 鼓掌 举手 黑薯 电子版 成可 莎莎 惊艳 得意 自拍 签名 美味 一杯 点点 吵吵 小狗 威龙 一颗 收藏 啊啊啊 哈哈哈 嘿嘿 哭惹 捂脸 害羞 震惊 调皮 微笑 赞同 干杯 爆竹 庆祝 爱心 赞赞 可爱 哇塞 喜欢 收到 原来 真的假 问一问 谢谢 恭喜 吓死 天了 momo 娜娜 小常 老大 爸爸 同问 果然 开玩笑 三观 天呐 有钱人 表演 骂人 统一 兄弟 女孩 主角 不许 喇叭 令人 满满 复制 资料 女主 新生 主页 回复 奶茶 打卡 let 提取 红薯 好滴 大王 太棒了 aaa 入住 睡觉 蛋蛋 我刚 男主 图片 宇宙 背着 太好了 gpt 呜呜 石化 主包 扶墙 这话 半天 反应 啊啊 哈哈哈哈 沙发 已关 转发给 同款 棒棒 星星 妙手 这下 哎呀 卧槽 明天 首发 头发 小女孩 块钱 违法 自动 妈妈 宝宝 小动物 生日 鸡蛋 安装 win no 老婆 我服 少年 来到 更新 口令 科普 白人 全家 妻子 发个 为何 超绝 不知 抱抱 能量 作业""".split())


def tokenize(text):
    import re
    text = re.sub(r"\[[^\]]*\]", "", str(text))
    return [word for word in jieba.cut(text) if len(word.strip()) >= 2 and word.strip() not in STOPWORDS]


def locate(root: Path, filename: str) -> Path:
    candidates = [root / "04_embeddings" / filename, root / "analysis" / "04_embeddings" / filename]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(filename)


def topic_label(words):
    words = set(words)
    employment_score = len(words & EMPLOYMENT_WORDS)
    ai_score = len(words & AI_WORDS)
    if employment_score >= 2 and employment_score > ai_score:
        return "就业与职业潜力", 1
    if ai_score >= 2 and ai_score > employment_score:
        return "AI技术的替代性焦虑", 1
    return "噪音或未命名主题", 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    corpus = pd.read_csv(locate(args.root, "corpus_A_comments.csv"))
    embeddings = np.load(locate(args.root, "emb_comment_v1.npy"))
    texts = corpus["text_v1_comment_only"].fillna("").tolist()
    if len(texts) != len(embeddings):
        raise ValueError("Corpus and embedding row counts differ")

    preprocess = Preprocess(tokenizer=tokenize, stopwords=None, min_length=2, max_doc_freq=0.5, verbose=False)
    model = FASTopic(num_topics=10, preprocess=preprocess, device="cpu")
    model.fit(docs=texts, preset_doc_embeddings=embeddings, epochs=args.epochs)
    theta = model.transform(doc_embeddings=torch.as_tensor(embeddings))
    theta = theta.detach().cpu().numpy() if hasattr(theta, "detach") else np.asarray(theta)
    dominant = theta.argmax(axis=1)
    counts = pd.Series(dominant).value_counts().reindex(range(10), fill_value=0).sort_index()
    top_words = [str(words).split() for words in model.get_top_words(num_top_words=20)]

    rows = []
    for topic_id, count in counts.items():
        label, substantive = topic_label(top_words[int(topic_id)])
        rows.append({
            "topic_id": int(topic_id),
            "topic_label": label,
            "top_words": " ".join(top_words[int(topic_id)]),
            "n_documents": int(count),
            "share": float(count / len(dominant)),
            "share_pct": float(count / len(dominant) * 100),
            "is_substantive": substantive,
            "n_total": int(len(dominant)),
            "model": "FASTopic",
            "k": 10,
            "epochs": args.epochs,
            "seed": args.seed,
            "assignment_rule": "argmax(theta, axis=1)",
        })
    table = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    metadata = {
        "input_n": int(len(dominant)),
        "embedding_shape": list(embeddings.shape),
        "seed": args.seed,
        "epochs": args.epochs,
        "legacy_reference_counts": LEGACY_COUNTS,
        "recomputed_counts": {str(k): int(v) for k, v in counts.items()},
        "topic_labels": {str(i): topic_label(top_words[i])[0] for i in range(10)},
        "legacy_reference_definition": "analysis/报告/05_统计建模/stage5_summary.json",
        "note": "The legacy stage5 retraining did not explicitly record a random seed; compare counts before claiming byte-level reproduction.",
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(table[table.is_substantive == 1][["topic_id", "topic_label", "n_documents", "share_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
