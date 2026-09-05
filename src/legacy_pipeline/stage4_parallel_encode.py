# -*- coding: utf-8 -*-
"""Stage 4: 并发全量编码 — 多进程并行调用Codex, I/O-bound任务"""
import json, os, re, sys, time, signal, subprocess
from pathlib import Path
from multiprocessing import Process, Lock, Value
from ctypes import c_int
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
os.chdir(BASE)

# 配置
BATCH_SIZE = 40
WORKERS = 6          # 并发worker数(I/O-bound, 6个足够)
RETRIES = 3
TIMEOUT = 600

LABELS_DIR = BASE / "analysis" / "07_labels"
CKPT_DIR = LABELS_DIR / "batches"
LABELS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(exist_ok=True)

CODEBOOK = "你是内容编码员。对以下评论逐一编码，输出JSON数组。变量:evaluation_object(数组≤3):课程/知识/能力/专业/职业/行业/学科/未明确。evidence_basis(数组≤3):个人学习经历/实习就业经历/身边他人经历/招聘收入与市场结果/专业知识与理论/公共价值/AI与平台可替代性/身份情感/反讽或梗/无依据断言。stance:否定/条件性否定/中性/条件性认可/认可/无法判断。格式:[{\"id\":\"cid\",\"evaluation_object\":[],\"evidence_basis\":[],\"stance\":\"\"}]。只输出JSON数组。"

# 全局锁
lock = Lock()


def encode_batch(batch_id, comments_batch):
    """编码一批评论, 返回label列表或None"""
    lines = [CODEBOOK]
    for _, r in comments_batch.iterrows():
        lines.append(f"ID:{r['comment_id']} | [{r['note_title'][:60]}] | [{r['current_comment'][:150]}]")
    prompt = '\n'.join(lines)

    for attempt in range(RETRIES + 1):
        try:
            result = subprocess.run(
                ['codex', 'exec', '-m', 'gpt-5.5', '-c', 'model_reasoning_effort=low', prompt],
                capture_output=True, text=True, timeout=TIMEOUT,
                encoding='utf-8', errors='replace'
            )
            m = re.search(r'\[\s*\{.*?\}\s*\]', result.stdout, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass
            if attempt < RETRIES:
                time.sleep(10)
        except subprocess.TimeoutExpired:
            pass
        except Exception:
            pass
    return None


def worker(worker_id, batch_ids, df, progress_counter, error_counter):
    """Worker进程: 从batch_ids中取任务并编码"""
    for bi in batch_ids:
        # 检查是否已完成
        done_file = CKPT_DIR / f"{bi:05d}.done"
        if done_file.exists():
            with lock:
                progress_counter.value += 1
            continue

        start = bi * BATCH_SIZE
        end = min((bi + 1) * BATCH_SIZE, len(df))
        batch = df.iloc[start:end]

        labels = encode_batch(bi, batch)
        done_file.parent.mkdir(parents=True, exist_ok=True)

        if labels and len(labels) == len(batch):
            # 写入单个批次文件
            batch_file = LABELS_DIR / f"batch_{bi:05d}.jsonl"
            with open(batch_file, 'w', encoding='utf-8') as f:
                for lb in labels:
                    f.write(json.dumps(lb, ensure_ascii=False) + '\n')
            done_file.touch()
            with lock:
                progress_counter.value += 1
        else:
            # 失败: 写错误标记
            fail_file = CKPT_DIR / f"{bi:05d}.fail"
            fail_file.touch()
            # 写入空标签以免丢失
            batch_file = LABELS_DIR / f"batch_{bi:05d}.jsonl"
            with open(batch_file, 'w', encoding='utf-8') as f:
                for _, r in batch.iterrows():
                    f.write(json.dumps({"id": r['comment_id'], "error": "failed"}, ensure_ascii=False) + '\n')
            with lock:
                progress_counter.value += 1
                error_counter.value += 1

        # 每10批更新汇总日志
        if progress_counter.value % 10 == 0:
            log_msg = f"[W{worker_id}] 进度 {progress_counter.value}/{len(batch_ids)} 批"
            with open(LABELS_DIR / "parallel_log.txt", 'a', encoding='utf-8') as lf:
                lf.write(f"{time.strftime('%H:%M:%S')} {log_msg}\n")


def main():
    # 加载数据
    df = pd.read_csv(BASE / "analysis" / "04_embeddings" / "corpus_A_comments.csv")
    df['current_comment'] = df.get('text_v1_comment_only', df.get('current_comment', ''))
    for c in ['current_comment', 'note_title', 'parent_comment']:
        if c not in df.columns:
            df[c] = ''
        df[c] = df[c].fillna('')

    total_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE
    all_batch_ids = list(range(total_batches))

    # 过滤已完成的
    remaining = [bi for bi in all_batch_ids if not (CKPT_DIR / f"{bi:05d}.done").exists()]
    print(f"总批: {total_batches}, 剩余: {len(remaining)}, Workers: {WORKERS}")
    print(f"并发编码启动: {time.strftime('%H:%M:%S')}")

    if not remaining:
        print("所有批次已完成!")
        return

    # 分配任务给workers (轮转分配以负载均衡)
    worker_batches = [[] for _ in range(WORKERS)]
    for i, bi in enumerate(remaining):
        worker_batches[i % WORKERS].append(bi)

    progress_counter = Value(c_int, total_batches - len(remaining))
    error_counter = Value(c_int, 0)

    processes = []
    for wid in range(WORKERS):
        p = Process(target=worker, args=(wid, worker_batches[wid], df, progress_counter, error_counter))
        p.start()
        processes.append(p)
        print(f"  Worker {wid}: {len(worker_batches[wid])} 批待处理")

    # 监控进度
    try:
        while any(p.is_alive() for p in processes):
            done = progress_counter.value
            eta_min = (total_batches - done) * 2.0 / WORKERS if done else 999
            print(f"\r[{time.strftime('%H:%M:%S')}] 进度: {done}/{total_batches} ({100*done//total_batches}%) "
                  f"错误:{error_counter.value} ETA:{eta_min:.0f}min  ", end='', flush=True)
            time.sleep(30)

        # 全部完成
        print(f"\n[{time.strftime('%H:%M:%S')}] 所有worker完成!")

        # 合并所有批次文件
        merged = LABELS_DIR / "all_encoded.jsonl"
        with open(merged, 'w', encoding='utf-8') as mf:
            for bi in sorted(all_batch_ids):
                bf = LABELS_DIR / f"batch_{bi:05d}.jsonl"
                if bf.exists():
                    mf.write(bf.read_text(encoding='utf-8'))

        total_lines = sum(1 for _ in open(merged, encoding='utf-8'))
        print(f"合并完成: {total_lines} 条 → {merged}")

    except KeyboardInterrupt:
        print("\n中断信号, 终止workers...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.join(timeout=5)
        print("已终止。重启时会从checkpoint恢复。")


if __name__ == "__main__":
    main()
