#!/usr/bin/env python3
"""Recompute terminal-v2.2 controversy-network descriptive analyses.

The script deliberately separates two evidence layers:
1. Full-corpus object/evidence/stance associations from ``comment_labels``.
2. Reply-network structure from the frozen terminal relation frame.

These are descriptive associations, not causal estimates or population-level claims.
The full-corpus labels are the historical v1 labels retained for continuity; relation
strategy and translation labels are terminal v2.1 labels with explicit confidence tiers.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
from pathlib import Path
from typing import List

import networkx as nx
import pandas as pd

POSITIVE = {"认可", "条件性认可"}
NEGATIVE = {"否定", "条件性否定"}
NEUTRAL = {"中性"}
RELEVANT = {"核心相关", "语境相关"}


def parse_list(value: object) -> List[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    if not text:
        return []
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
    except Exception:
        pass
    return [text]


def polarity(stance: object) -> str:
    s = "" if stance is None else str(stance).strip()
    if s in POSITIVE:
        return "正向"
    if s in NEGATIVE:
        return "负向"
    if s in NEUTRAL:
        return "中性"
    return "未知"


def binary_entropy(pos: int, neg: int) -> float:
    total = pos + neg
    if total == 0:
        return float("nan")
    p = pos / total
    if p in (0.0, 1.0):
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def load_labels(db_path: Path) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    labels = pd.read_sql_query(
        "SELECT comment_id,evaluation_object,evidence_basis,stance FROM comment_labels", con
    )
    con.close()
    labels["objects"] = labels["evaluation_object"].map(parse_list)
    labels["evidences"] = labels["evidence_basis"].map(parse_list)
    labels["polarity"] = labels["stance"].map(polarity)
    return labels


def signed_bipartite(labels: pd.DataFrame):
    node_rows = []
    for typ, col in [("评价对象", "objects"), ("评价依据", "evidences")]:
        counts = {}
        for vals, pol in labels[[col, "polarity"]].itertuples(index=False, name=None):
            for val in dict.fromkeys(vals):
                bucket = counts.setdefault(val, {"正向": 0, "负向": 0, "中性": 0, "未知": 0})
                bucket[pol] += 1
        for node, c in counts.items():
            pos, neg = c["正向"], c["负向"]
            node_rows.append({
                "type": typ, "node": node, "n": sum(c.values()), "pos": pos, "neg": neg,
                "neutral": c["中性"], "unknown": c["未知"],
                "positive_share_among_directional": pos / (pos + neg) if pos + neg else float("nan"),
                "polarization_entropy": binary_entropy(pos, neg),
            })
    node_df = pd.DataFrame(node_rows).sort_values(["type", "n"], ascending=[True, False])

    edge_counts = {}
    for objects, evidences, pol in labels[["objects", "evidences", "polarity"]].itertuples(index=False, name=None):
        for obj in dict.fromkeys(objects):
            for ev in dict.fromkeys(evidences):
                bucket = edge_counts.setdefault((obj, ev), {"正向": 0, "负向": 0, "中性": 0, "未知": 0})
                bucket[pol] += 1
    edge_rows = []
    for (obj, ev), c in edge_counts.items():
        pos, neg = c["正向"], c["负向"]
        directional = pos + neg
        edge_rows.append({
            "object": obj, "evidence": ev, "n": sum(c.values()), "pos": pos, "neg": neg,
            "neutral": c["中性"], "unknown": c["未知"],
            "pos_share": pos / directional if directional else float("nan"),
            "signed_balance": (pos - neg) / directional if directional else float("nan"),
            "entropy": binary_entropy(pos, neg),
        })
    edge_df = pd.DataFrame(edge_rows).sort_values("n", ascending=False)
    return node_df, edge_df


def relation_stats(frame: pd.DataFrame):
    # Resolve terminal field names while retaining backwards compatibility.
    rel_col = "final_relevance_status_terminal" if "final_relevance_status_terminal" in frame else "final_relevance_status"
    strategy_col = "final_reply_strategy_terminal" if "final_reply_strategy_terminal" in frame else "final_reply_strategy"
    translation_col = "final_translation_link_terminal" if "final_translation_link_terminal" in frame else "final_translation_link"

    high_quality = frame[
        frame[rel_col].isin(RELEVANT)
        & ((frame["final_label_method"] == "agent_adjudicated")
           | (frame["final_review_status"] == "hybrid_calibrated_high_confidence"))
    ].copy()
    agent = high_quality[high_quality["final_label_method"] == "agent_adjudicated"].copy()

    for col in ["direct_v1_evaluation_object", "reply_v1_evaluation_object",
                "direct_v1_evidence_basis", "reply_v1_evidence_basis"]:
        high_quality[col + "_list"] = high_quality[col].map(parse_list)
    high_quality["source_pol"] = high_quality["direct_v1_stance"].map(polarity)
    high_quality["reply_pol"] = high_quality["reply_v1_stance"].map(polarity)

    complete = high_quality[
        high_quality["direct_v1_evaluation_object_list"].map(bool)
        & high_quality["reply_v1_evaluation_object_list"].map(bool)
        & high_quality["direct_v1_evidence_basis_list"].map(bool)
        & high_quality["reply_v1_evidence_basis_list"].map(bool)
    ].copy()
    complete["object_overlap"] = complete.apply(
        lambda r: bool(set(r.direct_v1_evaluation_object_list) & set(r.reply_v1_evaluation_object_list)), axis=1)
    complete["evidence_overlap"] = complete.apply(
        lambda r: bool(set(r.direct_v1_evidence_basis_list) & set(r.reply_v1_evidence_basis_list)), axis=1)
    complete["first_evidence_same"] = complete.apply(
        lambda r: r.direct_v1_evidence_basis_list[0] == r.reply_v1_evidence_basis_list[0], axis=1)

    directional = high_quality[
        high_quality["source_pol"].isin(["正向", "负向"])
        & high_quality["reply_pol"].isin(["正向", "负向"])
    ].copy()
    mixing = pd.crosstab(directional["source_pol"], directional["reply_pol"]).reindex(
        index=["正向", "负向"], columns=["正向", "负向"], fill_value=0)
    mixing.index.name = "parent_pol"
    total = mixing.to_numpy().sum()
    observed_same = (mixing.loc["正向", "正向"] + mixing.loc["负向", "负向"]) / total if total else float("nan")
    source_dist = directional["source_pol"].value_counts(normalize=True)
    reply_dist = directional["reply_pol"].value_counts(normalize=True)
    expected_same = sum(source_dist.get(k, 0) * reply_dist.get(k, 0) for k in ["正向", "负向"])
    assort = (observed_same - expected_same) / (1 - expected_same) if expected_same < 1 else float("nan")

    agent_translation_rate = (agent[translation_col].fillna("") != "无转译").mean()
    hq_translation_rate = (high_quality[translation_col].fillna("") != "无转译").mean()

    users = pd.unique(pd.concat([
        high_quality["direct_source_author_hash"], high_quality["reply_author_hash"]
    ], ignore_index=True).dropna().astype(str))
    edge_df = high_quality[["direct_source_author_hash", "reply_author_hash"]].dropna().astype(str).drop_duplicates()
    edge_df = edge_df[edge_df.iloc[:, 0] != edge_df.iloc[:, 1]]
    G = nx.DiGraph()
    G.add_edges_from(edge_df.itertuples(index=False, name=None))
    connected_nodes = G.number_of_nodes()
    unique_edges = G.number_of_edges()
    reciprocity = nx.reciprocity(G) if unique_edges else float("nan")
    weak_components = nx.number_weakly_connected_components(G) if connected_nodes else 0
    out_degrees = sorted((d for _, d in G.out_degree()), reverse=True)
    top_n = max(1, math.ceil(.05 * len(out_degrees))) if out_degrees else 0
    top_share = sum(out_degrees[:top_n]) / sum(out_degrees) if out_degrees and sum(out_degrees) else float("nan")

    summary = {
        "high_quality_relations_n": int(len(high_quality)),
        "agent_adjudicated_relevant_n": int(len(agent)),
        "label_complete_relation_n": int(len(complete)),
        "object_overlap_rate_in_label_complete": float(complete.object_overlap.mean()),
        "evidence_any_overlap_rate_in_label_complete": float(complete.evidence_overlap.mean()),
        "first_evidence_same_rate_in_label_complete": float(complete.first_evidence_same.mean()),
        "directional_relation_n": int(len(directional)),
        "observed_same_polarity_rate": float(observed_same),
        "expected_same_polarity_rate": float(expected_same),
        "polarity_assortativity": float(assort),
        "agent_translation_rate": float(agent_translation_rate),
        "high_quality_translation_rate": float(hq_translation_rate),
        "all_observed_users_in_hq_relations": int(len(users)),
        "connected_users_after_self_loop_removal": int(connected_nodes),
        "unique_nonself_directed_user_edges": int(unique_edges),
        "user_network_reciprocity": float(reciprocity) if reciprocity is not None else None,
        "user_network_weak_components": int(weak_components),
        "top_5pct_connected_nodes_outdegree_edge_share": float(top_share),
        "strategy_field_used": strategy_col,
        "translation_field_used": translation_col,
    }
    return mixing.reset_index(), summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, type=Path)
    ap.add_argument("--relation-frame", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    labels = load_labels(args.db)
    nodes, edges = signed_bipartite(labels)
    frame = pd.read_csv(args.relation_frame)
    mixing, summary = relation_stats(frame)

    nodes.to_csv(args.out / "node_polarity_summary_v2_2.csv", index=False, encoding="utf-8-sig")
    edges.to_csv(args.out / "object_evidence_signed_edges_v2_2.csv", index=False, encoding="utf-8-sig")
    mixing.to_csv(args.out / "reply_stance_mixing_hq_v2_2.csv", index=False, encoding="utf-8-sig")
    (args.out / "controversy_network_summary_v2_2.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
