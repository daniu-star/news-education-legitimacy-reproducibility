# -*- coding: utf-8 -*-
"""Stage 4 Worker: 编码指定范围的批次 (batch_begin ~ batch_end)"""
import json, os, re, sys, time, subprocess
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
os.chdir(BASE)

BATCH_SIZE = 40
RETRIES = 3
TIMEOUT = 600
CODEBOOK = "你是内容编码员。对以下评论逐一编码，输出JSON数组。变量:evaluation_object(数组≤3):课程/知识/能力/专业/职业/行业/学科/未明确。evidence_basis(数组≤3):个人学习经历/实习就业经历/身边他人经历/招聘收入与市场结果/专业知识与理论/公共价值/AI与平台可替代性/身份情感/反讽或梗/无依据断言。stance:否定/条件性否定/中性/条件性认可/认可/无法判断。格式:[{\"id\":\"cid\",\"evaluation_object\":[],\"evidence_basis\":[],\"stance\":\"\"}]。只输出JSON数组。"

def encode_one(batch_id, batch_df):
    lines = [CODEBOOK]
    for _, r in batch_df.iterrows():
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
                if len(labels) == len(batch_df):
                    return labels
            if attempt < RETRIES:
                time.sleep(10)
        except subprocess.TimeoutExpired:
            pass
        except Exception as e:
            pass
    return None

def main():
    begin = int(sys.argv[1])
    end = int(sys.argv[2])

    df = pd.read_csv(BASE / "analysis" / "04_embeddings" / "corpus_A_comments.csv")
    df['current_comment'] = df.get('text_v1_comment_only', '')
    for c in ['current_comment', 'note_title', 'parent_comment']:
        if c not in df.columns: df[c] = ''
        df[c] = df[c].fillna('')

    labels_dir = BASE / "analysis" / "07_labels"
    ckpt_dir = labels_dir / "batches"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    done = 0; failed = 0
    for bi in range(begin, min(end, (len(df) + BATCH_SIZE - 1) // BATCH_SIZE)):
        done_file = ckpt_dir / f"{bi:05d}.done"
        if done_file.exists():
            done += 1
            continue

        start = bi * BATCH_SIZE
        stop = min((bi + 1) * BATCH_SIZE, len(df))
        batch = df.iloc[start:stop]

        labels = encode_one(bi, batch)
        if labels:
            batch_file = labels_dir / f"batch_{bi:05d}.jsonl"
            with open(batch_file, 'w', encoding='utf-8') as f:
                for lb in labels:
                    f.write(json.dumps(lb, ensure_ascii=False) + '\n')
            done_file.touch()
            done += 1
            print(f"[{time.strftime('%H:%M:%S')}] batch {bi} OK ({done}/{end-begin})", flush=True)
        else:
            fail_file = ckpt_dir / f"{bi:05d}.fail"
            fail_file.touch()
            # write empty labels
            batch_file = labels_dir / f"batch_{bi:05d}.jsonl"
            with open(batch_file, 'w', encoding='utf-8') as f:
                for _, r in batch.iterrows():
                    f.write(json.dumps({"id": r['comment_id'], "error": "failed"}, ensure_ascii=False) + '\n')
            failed += 1
            print(f"[{time.strftime('%H:%M:%S')}] batch {bi} FAIL", flush=True)

    print(f"DONE range=[{begin},{end}) success={done} failed={failed}", flush=True)

if __name__ == "__main__":
    main()
