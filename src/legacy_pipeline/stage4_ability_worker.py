# -*- coding: utf-8 -*-
"""Stage 4补: 补全ability_type/property编码"""
import json, os, re, sys, time, subprocess
import pandas as pd
from pathlib import Path

# 无缓冲输出
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

BASE = Path(__file__).resolve().parent.parent
os.chdir(BASE)

BATCH_SIZE = 60  # 可以更大, 因为prompt更短
RETRIES = 3
TIMEOUT = 600
CODEBOOK = """你是内容编码员。对以下评论逐条编码ability_type和ability_property。仅输出JSON数组。

ability_type (数组,≤3): 基础内容生产/信息搜集整理/采访与信源关系/事实核查与证据/社会解释与议题发现/伦理责任与公共判断/平台运营与商业传播/其他
规则: 评论中提及或暗示的能力。如"写稿拍视频"→[基础内容生产];"找资料找数据"→[信息搜集整理];"做采访"→[采访与信源关系];"核实真伪"→[事实核查与证据];"分析社会问题"→[社会解释与议题发现];"伦理判断责任"→[伦理责任与公共判断];"做新媒体运营抖音"→[平台运营与商业传播]。无能力提及→空数组[]。

ability_property (数组,≤2): 新闻专业独特/大学教育通用/可跨职业迁移/可由AI替代/AI辅助强化/AI强化需求/未明确
规则: 如何定位上述能力。如"AI写稿比人快"→[可由AI替代];"任何专业都会写"→[大学教育通用];"只有新闻专业才教这个"→[新闻专业独特];"这能力转行也能用"→[可跨职业迁移];"AI是辅助工具"→[AI辅助强化];"AI越多越需要核实"→[AI强化需求]。无明确属性→空数组[]。

格式:[{"id":"cid","ability_type":[],"ability_property":[]}]。只输出JSON数组。"""


def encode_batch(batch):
    lines = [CODEBOOK]
    for _, r in batch.iterrows():
        lines.append(f"ID:{r['comment_id']} | [{r['note_title'][:60]}] | [{r['current_comment'][:150]}]")
    prompt = '\n'.join(lines)

    for attempt in range(RETRIES + 1):
        try:
            result = subprocess.run(
                ['codex', 'exec', '-m', 'gpt-5.5', '-c', 'model_reasoning_effort=low', prompt],
                capture_output=True, text=True, timeout=TIMEOUT, encoding='utf-8', errors='replace')
            m = re.search(r'\[\s*\{.*?\}\s*\]', result.stdout, re.DOTALL)
            if m:
                labels = json.loads(m.group(0))
                if len(labels) == len(batch):
                    return labels
            if attempt < RETRIES:
                time.sleep(8)
        except:
            pass
    return None


def main():
    begin = int(sys.argv[1])
    end = int(sys.argv[2])

    df = pd.read_csv(BASE / "analysis" / "04_embeddings" / "corpus_A_comments.csv")
    df['current_comment'] = df.get('text_v1_comment_only', '')
    for c in ['current_comment', 'note_title']:
        if c not in df.columns: df[c] = ''
        df[c] = df[c].fillna('')

    labels_dir = BASE / "analysis" / "07_labels"
    ab_dir = labels_dir / "ability_batches"
    ab_dir.mkdir(parents=True, exist_ok=True)

    total_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE
    done = 0; failed = 0

    for bi in range(begin, min(end, total_batches)):
        done_file = ab_dir / f"{bi:05d}.done"
        if done_file.exists():
            done += 1
            continue

        start = bi * BATCH_SIZE
        stop = min((bi + 1) * BATCH_SIZE, len(df))
        batch = df.iloc[start:stop]

        labels = encode_batch(batch)
        if labels:
            batch_file = labels_dir / f"ability_{bi:05d}.jsonl"
            with open(batch_file, 'w', encoding='utf-8') as f:
                for lb in labels:
                    f.write(json.dumps(lb, ensure_ascii=False) + '\n')
            done_file.touch()
            done += 1
            print(f"[{time.strftime('%H:%M:%S')}] batch {bi} OK ({done}/{end-begin})", flush=True)
        else:
            fail_file = ab_dir / f"{bi:05d}.fail"
            fail_file.touch()
            failed += 1
            print(f"[{time.strftime('%H:%M:%S')}] batch {bi} FAIL", flush=True)

    print(f"DONE range=[{begin},{end}) success={done} failed={failed}", flush=True)


if __name__ == "__main__":
    main()
