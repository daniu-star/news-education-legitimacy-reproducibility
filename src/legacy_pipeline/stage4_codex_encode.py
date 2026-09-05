# -*- coding: utf-8 -*-
"""
Stage 4: Codex LLM 编码管线
- 使用 gpt-5.5 + medium reasoning, 批量编码
- 先400专家样本 → 验证 → 全量21,779
"""
import os, sys, json, time, subprocess, re
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS_DIR = os.path.join(BASE, "analysis", "07_labels")
os.makedirs(LABELS_DIR, exist_ok=True)

CODEX = "codex"
MODEL = "gpt-5.5"
EFFORT = "medium"
BATCH_SIZE = 30

# ---- Codebook摘要 (精简版, 节省tokens) ----
CODEBOOK_PROMPT = """你是学术内容分析编码员。对以下评论逐一编码，仅输出JSON数组（无其他文字）。

编码变量：
1. evaluation_object (数组,≤3): 课程/知识/能力/专业/职业/行业/学科/未明确
   规则: 评论评价的具体对象。如"新传课水"→[课程];"找不到工作"→[职业];"这个专业没救了"→[专业]
2. evidence_basis (数组,≤3): 个人学习经历/实习就业经历/身边他人经历/招聘收入与市场结果/专业知识与理论/公共价值/AI与平台可替代性/身份情感/反讽或梗/无依据断言
   规则: 评论的论据来源。如"我985毕业现在..."→[实习就业经历];"就是垃圾"→[无依据断言];"张雪峰说得对"→[反讽或梗]
3. stance (单值): 否定/条件性否定/中性/条件性认可/认可/无法判断
   规则: 整体态度。如"但看个人，985还行双非算了"→[条件性否定];纯表情→[无法判断]
4. ability_type (数组,≤3,可选): 基础内容生产/信息搜集整理/采访与信源关系/事实核查与证据/社会解释与议题发现/伦理责任与公共判断/平台运营与商业传播/其他
5. ability_property (数组,≤2,可选): 新闻专业独特/大学教育通用/可跨职业迁移/可由AI替代/AI辅助强化/AI强化需求/未明确
6. reply_strategy (数组,≤2,仅回复层级): 同层直接回应/具体能力解释/个人经验回应/市场结果回应/公共价值回应/身份否定/反讽与情绪/话题转移/其他
7. translation_link (单值): 课程—能力连接/能力—职业连接/能力—公共价值连接/仅抽象价值表达/无转译

JSON格式: [{"id":"comment_id","evaluation_object":[],"evidence_basis":[],"stance":"","ability_type":[],"ability_property":[],"reply_strategy":[],"translation_link":"","ambiguity_flag":false}]

仅输出JSON数组，不输出任何其他内容。"""


def build_batch_prompt(comments_batch):
    """构建单批次的编码prompt"""
    lines = [CODEBOOK_PROMPT, "\n评论列表:"]
    for c in comments_batch:
        ctx = ""
        if c.get('note_title'):
            ctx += f"[笔记]{c['note_title'][:80]} "
        if c.get('parent_comment'):
            ctx += f"[父评论]{c['parent_comment'][:80]} "
        ctx += f"[评论]{c['current_comment'][:200]}"
        lines.append(f"ID:{c['comment_id']} Lv{c['comment_level']} AuthorReply:{c['is_note_author']} | {ctx}")
    return "\n".join(lines)


def call_codex(prompt, timeout=300):
    """调用 codex exec 并返回文本输出"""
    try:
        result = subprocess.run(
            [CODEX, "exec", "-m", MODEL, "-c", f"model_reasoning_effort={EFFORT}", prompt],
            capture_output=True, text=True, timeout=timeout,
            cwd=BASE, encoding='utf-8', errors='replace'
        )
        return result.stdout
    except subprocess.TimeoutExpired:
        return "[TIMEOUT]"
    except Exception as e:
        return f"[ERROR: {e}]"


def parse_codex_output(output):
    """从Codex输出中提取JSON数组"""
    # 找 [...]
    m = re.search(r'\[\s*\{.*\}\s*\]', output, re.DOTALL)
    if not m:
        # fallback: 找 {...} 行
        objs = re.findall(r'\{[^{}]*"id"[^{}]*\}', output)
        if objs:
            return json.loads("[" + ",".join(objs) + "]")
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def encode_batch(comments_batch, batch_id, retries=2):
    """编码一批评论, 返回标签列表"""
    prompt = build_batch_prompt(comments_batch)
    for attempt in range(retries + 1):
        print(f"  批次{batch_id} 尝试{attempt+1}/{retries+1}: {len(comments_batch)}条...")
        t0 = time.time()
        output = call_codex(prompt, timeout=300)
        dt = time.time() - t0
        labels = parse_codex_output(output)
        if labels:
            # 验证返回数量匹配
            if len(labels) == len(comments_batch):
                print(f"    ✅ {dt:.0f}s, {len(labels)}条")
                return labels
            else:
                print(f"    ⚠️ 数量不匹配: 期望{len(comments_batch)}, 得到{len(labels)}")
        else:
            print(f"    ❌ JSON解析失败 ({dt:.0f}s)")
            if attempt < retries:
                print(f"    重试中...")
    # 全部失败: 返回空标签
    print(f"    ❌ 全部失败, 返回空标签")
    return [{"id": c["comment_id"], "error": "encoding_failed"} for c in comments_batch]


def encode_all(comments_df, output_file, batch_size=BATCH_SIZE):
    """分批编码全部评论"""
    batches = []
    for i in range(0, len(comments_df), batch_size):
        batch = comments_df.iloc[i:i+batch_size].to_dict('records')
        batches.append(batch)

    print(f"开始编码: {len(comments_df)}条, {len(batches)}批, 每批{batch_size}条")

    all_labels = []
    with open(output_file, 'w', encoding='utf-8') as f:
        for bid, batch in enumerate(batches):
            labels = encode_batch(batch, bid)
            for label in labels:
                f.write(json.dumps(label, ensure_ascii=False) + '\n')
            all_labels.extend(labels)
            f.flush()
            # 短暂暂停避免rate limit
            if bid % 5 == 4:
                time.sleep(2)
    return all_labels


def main(mode="expert"):
    if mode == "expert":
        # 400专家样本
        input_file = os.path.join(BASE, "报告", "04_LLM编码", "data", "expert_sample_400.csv")
        output_file = os.path.join(LABELS_DIR, "expert_400_encoded.jsonl")
    else:
        # 全量
        input_file = os.path.join(BASE, "analysis", "04_embeddings", "corpus_A_comments.csv")
        output_file = os.path.join(LABELS_DIR, "all_encoded.jsonl")

    df = pd.read_csv(input_file)
    # 确保有需要的列
    if 'current_comment' not in df.columns and 'text_v1_comment_only' in df.columns:
        df['current_comment'] = df['text_v1_comment_only']
    if 'note_title' not in df.columns:
        df['note_title'] = ''
    if 'parent_comment' not in df.columns:
        df['parent_comment'] = ''
    if 'is_note_author' not in df.columns:
        df['is_note_author'] = 0
    if 'comment_level' not in df.columns:
        df['comment_level'] = 1

    # 填充空值
    for col in ['note_title', 'parent_comment', 'current_comment']:
        if col in df.columns:
            df[col] = df[col].fillna('')

    print(f"模式: {mode}, 评论数: {len(df)}")
    labels = encode_all(df, output_file)
    print(f"\n完成: {len(labels)}条标签, 保存至 {output_file}")
    return labels


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "expert"
    main(mode)
