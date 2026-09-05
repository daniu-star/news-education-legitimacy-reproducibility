# -*- coding: utf-8 -*-
"""Stage 4 补完: DeepSeek API 并发补全 ability 编码"""
import os
import json, os, re, sys, time, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import openai

BASE = Path(__file__).resolve().parent.parent
os.chdir(BASE)

API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not API_KEY:
    raise RuntimeError("DEEPSEEK_API_KEY is not set; configure it in the environment before running this script.")
BATCH_SIZE = 60
WORKERS = 8  # API I/O-bound, 8并发没问题

CODEBOOK_PROMPT = """你是内容编码员。对以下评论逐条编码ability_type和ability_property。仅输出JSON数组，无其他文字。

ability_type(数组≤3): 基础内容生产/信息搜集整理/采访与信源关系/事实核查与证据/社会解释与议题发现/伦理责任与公共判断/平台运营与商业传播/其他
规则: 评论中提及或暗示的能力。
  "写稿拍视频"→[基础内容生产]
  "找资料找数据"→[信息搜集整理]
  "做采访找人聊"→[采访与信源关系]
  "核实真伪查证"→[事实核查与证据]
  "分析社会问题洞察"→[社会解释与议题发现]
  "伦理判断责任心"→[伦理责任与公共判断]
  "做新媒体运营抖音流量"→[平台运营与商业传播]
  无能力提及→空数组[]

ability_property(数组≤2): 新闻专业独特/大学教育通用/可跨职业迁移/可由AI替代/AI辅助强化/AI强化需求/未明确
规则: 评论如何定位该能力。
  "AI写稿比人快会替代"→[可由AI替代]
  "任何专业都会写谁都能学"→[大学教育通用]
  "只有新闻专业才教这个"→[新闻专业独特]
  "转行也能用"→[可跨职业迁移]
  "AI是辅助工具不是替代"→[AI辅助强化]
  "AI越多越需要人核实"→[AI强化需求]
  无明确属性→空数组[]

格式: [{"id":"cid","ability_type":[],"ability_property":[]}]
只输出JSON数组，一条评论一个对象。"""

lock = threading.Lock()


def encode_batch(batch_df, batch_id):
    """调用DeepSeek API编码一批"""
    lines = [CODEBOOK_PROMPT]
    for _, r in batch_df.iterrows():
        lines.append(f"ID:{r['comment_id']} | [{r['note_title'][:60]}] | [{r['current_comment'][:150]}]")
    prompt = "\n".join(lines)

    client = openai.OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com/v1")

    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=4096,
                timeout=120,
            )
            text = resp.choices[0].message.content
            m = re.search(r'\[\s*\{.*?\}\s*\]', text, re.DOTALL)
            if m:
                labels = json.loads(m.group(0))
                if len(labels) == len(batch_df):
                    return labels
            if attempt < 3:
                time.sleep(2)
        except Exception as e:
            if attempt < 3:
                time.sleep(3)
    return None


def process_batch(bi, df):
    """处理单个批次 (线程安全写入)"""
    done_file = Path("analysis/07_labels/ability_batches") / f"{bi:05d}.done"
    fail_file = Path("analysis/07_labels/ability_batches") / f"{bi:05d}.fail"

    if done_file.exists():
        return bi, "skip"

    start = bi * BATCH_SIZE
    stop = min((bi + 1) * BATCH_SIZE, len(df))
    batch = df.iloc[start:stop]

    labels = encode_batch(batch, bi)
    if labels:
        batch_file = Path("analysis/07_labels") / f"ability_{bi:05d}.jsonl"
        with lock:
            with open(batch_file, "w", encoding="utf-8") as f:
                for lb in labels:
                    f.write(json.dumps(lb, ensure_ascii=False) + "\n")
            if fail_file.exists():
                fail_file.unlink()
            done_file.touch()
        return bi, "ok"
    else:
        with lock:
            fail_file.touch()
        return bi, "fail"


def main():
    df = pd.read_csv(BASE / "analysis" / "04_embeddings" / "corpus_A_comments.csv")
    df['current_comment'] = df.get('text_v1_comment_only', '')
    for c in ['current_comment', 'note_title']:
        if c not in df.columns: df[c] = ''
        df[c] = df[c].fillna('')

    # 只处理未完成的批次
    total_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE
    ab_dir = Path("analysis/07_labels/ability_batches")
    ab_dir.mkdir(parents=True, exist_ok=True)

    remaining = [bi for bi in range(total_batches) if not (ab_dir / f"{bi:05d}.done").exists()]
    print(f"总批: {total_batches}, 待处理: {len(remaining)}, Workers: {WORKERS}")
    if not remaining:
        print("全部完成!")
        return

    ok = fail = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(process_batch, bi, df): bi for bi in remaining}
        for future in as_completed(futures):
            bi, status = future.result()
            if status == "ok": ok += 1
            elif status == "fail": fail += 1
            done_total = ok + fail
            print(f"\r[{time.strftime('%H:%M:%S')}] {done_total}/{len(remaining)} OK={ok} FAIL={fail}", end="", flush=True)

    print(f"\nDONE OK={ok} FAIL={fail}")


if __name__ == "__main__":
    main()
