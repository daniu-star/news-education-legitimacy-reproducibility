# -*- coding: utf-8 -*-
"""Stage 4: 全量编码 21,779条 — 分批调用codex gpt-5.5 low-effort"""
import json, os, re, signal, sys, time, subprocess
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)

df = pd.read_csv(os.path.join(BASE, "analysis", "04_embeddings", "corpus_A_comments.csv"))
for c in ['current_comment', 'note_title', 'parent_comment']:
    if c not in df.columns:
        df[c] = ''

df['current_comment'] = df.get('text_v1_comment_only', df['current_comment'])
for c in ['current_comment', 'note_title', 'parent_comment']:
    df[c] = df[c].fillna('')

OUT = os.path.join(BASE, "analysis", "07_labels", "all_encoded.jsonl")
CKPT = os.path.join(BASE, "analysis", "07_labels", "checkpoint.txt")
LOG = os.path.join(BASE, "analysis", "07_labels", "encode_log.txt")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

CODEBOOK = "你是内容编码员。对以下评论逐一编码，输出JSON数组。变量:evaluation_object(数组≤3):课程/知识/能力/专业/职业/行业/学科/未明确。evidence_basis(数组≤3):个人学习经历/实习就业经历/身边他人经历/招聘收入与市场结果/专业知识与理论/公共价值/AI与平台可替代性/身份情感/反讽或梗/无依据断言。stance:否定/条件性否定/中性/条件性认可/认可/无法判断。格式:[{\"id\":\"cid\",\"evaluation_object\":[],\"evidence_basis\":[],\"stance\":\"\"}]。只输出JSON数组。"

BATCH = 40

def log(msg):
    t = time.strftime("%H:%M:%S")
    line = f"[{t}] {msg}"
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')
    print(line, flush=True)

# 断点恢复
start = 0
if os.path.exists(CKPT):
    with open(CKPT) as f:
        start = int(f.read().strip())
    log(f"从批次{start}恢复")

total_batches = (len(df) + BATCH - 1) // BATCH
log(f"开始: {len(df)}条, {total_batches}批, batch={BATCH}")

total = start * BATCH
t0_all = time.time()

with open(OUT, 'a', encoding='utf-8') as fout:
    for bi in range(start, total_batches):
        batch = df.iloc[bi * BATCH: min((bi + 1) * BATCH, len(df))]
        lines = [CODEBOOK]
        for _, r in batch.iterrows():
            lines.append(f"ID:{r['comment_id']} | [{r['note_title'][:60]}] | [{r['current_comment'][:150]}]")
        prompt = '\n'.join(lines)

        ok = False
        for attempt in range(3):
            try:
                result = subprocess.run(
                    ['codex', 'exec', '-m', 'gpt-5.5', '-c', 'model_reasoning_effort=low', prompt],
                    capture_output=True, text=True, timeout=600,
                    encoding='utf-8', errors='replace'
                )
                m = re.search(r'\[\s*\{.*?\}\s*\]', result.stdout, re.DOTALL)
                if m:
                    labels = json.loads(m.group(0))
                    for lb in labels:
                        fout.write(json.dumps(lb, ensure_ascii=False) + '\n')
                    total += len(labels)
                    dt = time.time() - t0_all
                    eta = (dt / total * (len(df) - total) / 3600) if total else 0
                    log(f"批{bi+1}/{total_batches} ✓{len(labels)}条 {total}/{len(df)} ETA{eta:.1f}h")
                    ok = True
                    break
                else:
                    if attempt < 2:
                        log(f"  批{bi+1} 重试{attempt+2}/3...")
                        time.sleep(5)
            except subprocess.TimeoutExpired:
                log(f"  批{bi+1} 超时重试...")
            except Exception as e:
                log(f"  批{bi+1} 错误: {e}")
        if not ok:
            log(f"  批{bi+1} ❌ 全部失败")
            for _ in range(len(batch)):
                fout.write(json.dumps({"id": "error", "error": "failed"}, ensure_ascii=False) + '\n')
            total += len(batch)

        fout.flush()
        with open(CKPT, 'w') as fc:
            fc.write(str(bi + 1))

elapsed = (time.time() - t0_all) / 3600
log(f"DONE! {total}条 {elapsed:.1f}h")
