# -*- coding: utf-8 -*-
"""补全失败批次的ability编码 (按给定IDs重试)"""
import json, os, re, sys, time, subprocess
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
os.chdir(BASE)
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

BATCH_SIZE = 60
RETRIES = 3
TIMEOUT = 600
CODEBOOK = """你是内容编码员。对以下评论逐条编码ability_type和ability_property。仅输出JSON数组。

ability_type(数组≤3): 基础内容生产/信息搜集整理/采访与信源关系/事实核查与证据/社会解释与议题发现/伦理责任与公共判断/平台运营与商业传播/其他
规则: 评论中提及或暗示的能力。如"写稿拍视频"→[基础内容生产];"找资料"→[信息搜集整理];"采访"→[采访与信源关系];"核实"→[事实核查与证据];"分析社会问题"→[社会解释与议题发现];"伦理责任"→[伦理责任与公共判断];"做运营抖音"→[平台运营与商业传播]。无能力提及→空数组[]。

ability_property(数组≤2): 新闻专业独特/大学教育通用/可跨职业迁移/可由AI替代/AI辅助强化/AI强化需求/未明确
规则: "AI替"→[可由AI替代];"谁都会"→[大学教育通用];"只有新闻专业"→[新闻专业独特];"转行也能用"→[可跨职业迁移];"AI辅助"→[AI辅助强化];"AI越多越需核"→[AI强化需求]。无→空数组[]。

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
            if attempt < RETRIES: time.sleep(8)
        except: pass
    return None


def main():
    batch_ids = [int(x) for x in sys.argv[1].split(',')]

    df = pd.read_csv(BASE / "analysis" / "04_embeddings" / "corpus_A_comments.csv")
    df['current_comment'] = df.get('text_v1_comment_only', '')
    for c in ['current_comment', 'note_title']:
        if c not in df.columns: df[c] = ''
        df[c] = df[c].fillna('')

    labels_dir = BASE / "analysis" / "07_labels"
    ab_dir = labels_dir / "ability_batches"
    ok = 0; failed = 0

    for bi in batch_ids:
        fail_file = ab_dir / f"{bi:05d}.fail"
        start = bi * BATCH_SIZE
        stop = min((bi + 1) * BATCH_SIZE, len(df))
        batch = df.iloc[start:stop]

        labels = encode_batch(batch)
        if labels:
            batch_file = labels_dir / f"ability_{bi:05d}.jsonl"
            with open(batch_file, 'w', encoding='utf-8') as f:
                for lb in labels: f.write(json.dumps(lb, ensure_ascii=False) + '\n')
            # 移除fail标记，加done
            if fail_file.exists(): fail_file.unlink()
            (ab_dir / f"{bi:05d}.done").touch()
            ok += 1
            print(f"[{time.strftime('%H:%M:%S')}] batch {bi} OK ({ok}/{len(batch_ids)})", flush=True)
        else:
            failed += 1
            print(f"[{time.strftime('%H:%M:%S')}] batch {bi} FAIL", flush=True)

    print(f"DONE ok={ok} failed={failed}", flush=True)


if __name__ == "__main__":
    main()
