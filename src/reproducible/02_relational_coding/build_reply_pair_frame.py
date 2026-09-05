#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rebuild direct reply relations from analysis_v2_frozen.db.

This script never modifies the input database. It reconstructs direct-parent
chains, separates thread roots from direct parents, and writes the Stage 2A
relation frame, exclusions, thread summaries, and blinded coding batches.
"""

from __future__ import annotations
import argparse
import json
import math
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260804)
    args = parser.parse_args()

    out = Path(args.out)
    batches = out / "batches"
    out.mkdir(parents=True, exist_ok=True)
    batches.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(args.db)
    comments = pd.read_sql_query("""
        SELECT comment_id,note_id,root_comment_id,parent_comment_id,comment_level,
               author_hash_anon,target_user_hash_anon,content_raw,content_clean,
               create_ts_ms,create_time_cn,like_count_num,sub_comment_count_num,
               semantic_eligible,nonverbal_only,short_text_flag,text_length,
               source_type,relation_eligible,is_note_author,author_identifiable,is_orphan
        FROM comments_anon
    """, con)
    notes = pd.read_sql_query("""
        SELECT note_id,title,description,source_type AS note_source_type,
               description_synthetic,has_original_desc,has_any_desc,
               author_hash_anon AS note_author_hash
        FROM notes_anon
    """, con)
    labels = pd.read_sql_query("SELECT * FROM comment_labels", con)
    by_id = comments.set_index("comment_id", drop=False).to_dict("index")

    def resolve(row: pd.Series, max_depth: int = 10) -> pd.Series:
        cid = row["comment_id"]
        parent = row["parent_comment_id"] if pd.notna(row["parent_comment_id"]) and row["parent_comment_id"] else row["root_comment_id"]
        seen = {cid}
        chain: list[str] = []
        direct = parent if parent else None
        top = None
        status = None
        cur = parent
        cycle = False
        while cur and len(chain) < max_depth:
            if cur in seen:
                cycle = True
                status = "cycle"
                break
            seen.add(cur)
            chain.append(cur)
            rec = by_id.get(cur)
            if rec is None:
                status = "missing_parent"
                break
            if rec.get("comment_level") == 1:
                top = cur
                status = "resolved"
                break
            cur = rec.get("parent_comment_id") or rec.get("root_comment_id")
        if status is None:
            status = "max_depth" if len(chain) >= max_depth else "missing_parent"
        direct_rec = by_id.get(direct) if direct else None
        top_rec = by_id.get(top) if top else None
        return pd.Series({
            "direct_parent_comment_id": direct,
            "direct_parent_resolved": int(direct_rec is not None),
            "direct_parent_level": direct_rec.get("comment_level") if direct_rec else None,
            "top_root_comment_id": top,
            "top_root_resolved": int(top_rec is not None),
            "relation_resolution_status": status,
            "relation_chain_depth": len(chain),
            "chain_ids_json": json.dumps(chain, ensure_ascii=False),
            "cycle_flag": int(cycle),
        })

    replies = comments[comments["comment_level"] >= 2].copy()
    rel = pd.concat([replies.reset_index(drop=True), replies.apply(resolve, axis=1)], axis=1)

    def field(cid: str | None, name: str):
        rec = by_id.get(cid)
        return rec.get(name) if rec else None

    for prefix, idcol in [("direct_source", "direct_parent_comment_id"), ("thread_root", "top_root_comment_id")]:
        rel[f"{prefix}_text"] = rel[idcol].map(lambda x: field(x, "content_clean"))
        rel[f"{prefix}_author_hash"] = rel[idcol].map(lambda x: field(x, "author_hash_anon"))
        rel[f"{prefix}_is_note_author"] = rel[idcol].map(lambda x: field(x, "is_note_author"))
        rel[f"{prefix}_semantic_eligible"] = rel[idcol].map(lambda x: field(x, "semantic_eligible"))
        rel[f"{prefix}_like_count"] = rel[idcol].map(lambda x: field(x, "like_count_num"))
        rel[f"{prefix}_create_ts_ms"] = rel[idcol].map(lambda x: field(x, "create_ts_ms"))

    rel = rel.rename(columns={
        "comment_id": "reply_comment_id",
        "content_clean": "reply_text",
        "content_raw": "reply_text_raw",
        "author_hash_anon": "reply_author_hash",
        "is_note_author": "reply_is_note_author",
        "semantic_eligible": "reply_semantic_eligible",
        "nonverbal_only": "reply_nonverbal_only",
        "short_text_flag": "reply_short_text",
        "text_length": "reply_text_length",
        "like_count_num": "reply_like_count",
        "create_ts_ms": "reply_create_ts_ms",
        "source_type": "reply_source_type",
        "is_orphan": "reply_is_orphan",
    })
    rel = rel.merge(notes, on="note_id", how="left")

    for prefix, key in [("reply_v1_", "reply_comment_id"), ("root_v1_", "top_root_comment_id"), ("direct_v1_", "direct_parent_comment_id")]:
        lab = labels.add_prefix(prefix).rename(columns={f"{prefix}comment_id": key})
        rel = rel.merge(lab, on=key, how="left")

    def relation_type(row: pd.Series) -> str:
        if row["reply_is_orphan"] == 1:
            return "原库标记孤立"
        if row["relation_resolution_status"] != "resolved":
            return "上文缺失不可恢复"
        if row["direct_parent_level"] == 1:
            return "一级评论—直接回复"
        if pd.notna(row["direct_parent_level"]) and row["direct_parent_level"] >= 2:
            return "嵌套回复—直接父级可恢复"
        return "已恢复但直接层级未知"

    rel["relation_type"] = rel.apply(relation_type, axis=1)
    rel["thread_context_available"] = ((rel["top_root_resolved"] == 1) & rel["thread_root_text"].notna()).astype(int)
    rel["direct_context_available"] = ((rel["direct_parent_resolved"] == 1) & rel["direct_source_text"].notna()).astype(int)
    rel["relation_model_eligible"] = (
        (rel["reply_is_orphan"] == 0)
        & (rel["relation_resolution_status"] == "resolved")
        & rel["reply_text"].notna()
        & rel["direct_source_text"].notna()
    ).astype(int)
    rel["core_semantic_relation_eligible"] = (
        (rel["relation_model_eligible"] == 1)
        & (rel["reply_semantic_eligible"] == 1)
        & (rel["direct_source_semantic_eligible"].fillna(0) == 1)
    ).astype(int)
    rel["reply_to_note_author"] = (rel["direct_source_is_note_author"].fillna(0).astype(int) == 1).astype(int)
    rel["same_author_as_source"] = (
        rel["reply_author_hash"].notna()
        & rel["direct_source_author_hash"].notna()
        & (rel["reply_author_hash"] == rel["direct_source_author_hash"])
    ).astype(int)
    rel["reply_delay_hours"] = np.where(
        rel["reply_create_ts_ms"].notna() & rel["direct_source_create_ts_ms"].notna(),
        (rel["reply_create_ts_ms"] - rel["direct_source_create_ts_ms"]) / 3_600_000,
        np.nan,
    )

    rel = rel.sort_values(["note_id", "top_root_comment_id", "reply_create_ts_ms", "reply_comment_id"], na_position="last")
    eligible = rel[rel["relation_model_eligible"] == 1].copy()
    eligible["thread_reply_index"] = eligible.groupby(["note_id", "top_root_comment_id"]).cumcount() + 1
    eligible["thread_reply_total"] = eligible.groupby(["note_id", "top_root_comment_id"])["reply_comment_id"].transform("count")

    rel.to_csv(out / "reply_pair_frame.csv.gz", index=False, compression="gzip", encoding="utf-8-sig")
    rel.to_json(out / "reply_pair_frame.jsonl", orient="records", lines=True, force_ascii=False)

    exclusions = rel[rel["relation_model_eligible"] == 0].copy()
    exclusions["exclusion_reason"] = np.select(
        [
            exclusions["reply_is_orphan"].eq(1),
            exclusions["relation_resolution_status"].eq("missing_parent"),
            exclusions["relation_resolution_status"].eq("cycle"),
            exclusions["direct_source_text"].isna(),
            exclusions["reply_text"].isna(),
        ],
        ["原库标记孤立", "上文评论缺失，无法恢复关系", "关系链循环", "直接上文文本缺失", "回复文本缺失"],
        default="其他",
    )
    exclusions.to_csv(out / "reply_pair_exclusions.csv", index=False, encoding="utf-8-sig")

    thread_summary = (
        eligible.groupby(["note_id", "top_root_comment_id"], dropna=False)
        .agg(
            note_title=("title", "first"),
            note_source_type=("note_source_type", "first"),
            thread_root_text=("thread_root_text", "first"),
            reply_count=("reply_comment_id", "count"),
            author_reply_count=("reply_is_note_author", "sum"),
            semantic_reply_count=("reply_semantic_eligible", "sum"),
            core_semantic_pair_count=("core_semantic_relation_eligible", "sum"),
            nested_reply_count=("relation_type", lambda s: (s == "嵌套回复—直接父级可恢复").sum()),
            total_reply_likes=("reply_like_count", "sum"),
            max_reply_likes=("reply_like_count", "max"),
        )
        .reset_index()
    )
    thread_summary.to_csv(out / "relation_thread_summary.csv", index=False, encoding="utf-8-sig")

    coding_cols = [
        "note_id", "title", "description", "note_source_type",
        "top_root_comment_id", "thread_root_text",
        "direct_parent_comment_id", "direct_source_text", "direct_parent_level", "relation_type",
        "reply_comment_id", "reply_text", "reply_is_note_author", "reply_to_note_author", "same_author_as_source",
        "reply_source_type", "reply_semantic_eligible", "reply_nonverbal_only",
        "reply_short_text", "reply_text_length", "thread_reply_index", "thread_reply_total",
    ]
    coding = eligible[coding_cols].copy()
    for name in [
        "relevance_status", "evaluations_json", "evidence_basis_json", "ability_mentions_json",
        "reply_strategy_json", "translation_link", "secondary_translation_link",
        "ambiguity_flag", "ambiguity_type_json", "confidence", "evidence_span",
        "review_priority", "coding_reason",
    ]:
        coding[name] = ""
    coding["label_version"] = "v2.1"
    coding["coder_type"] = "Agent"
    coding["prompt_version"] = "relational_v2.1"
    coding["review_status"] = "unreviewed"
    coding.to_csv(out / "reply_pair_coding_input.csv.gz", index=False, compression="gzip", encoding="utf-8-sig")

    for old in batches.glob("batch_*.csv"):
        old.unlink()
    for i in range(math.ceil(len(coding) / args.batch_size)):
        coding.iloc[i * args.batch_size:(i + 1) * args.batch_size].to_csv(
            batches / f"batch_{i + 1:03d}.csv", index=False, encoding="utf-8-sig"
        )

    print(json.dumps({
        "all_replies": int(len(replies)),
        "eligible_relations": int(len(eligible)),
        "core_semantic_relations": int(eligible["core_semantic_relation_eligible"].sum()),
        "batches": int(math.ceil(len(coding) / args.batch_size)),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
