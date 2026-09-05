"""Recompute manuscript-facing binary models with note-clustered GEE.

This runner intentionally does not overwrite historical aggregate files. It writes
formal GEE outputs to a separate directory so that a release audit can compare the
model family and provenance before replacing manuscript-facing result tables.

Required environment variable:
    NEWS_EDU_ROOT: private project root containing 02_views/analysis.db or
                   00_freeze/analysis_v2_frozen.db.

Optional:
    NEWS_EDU_GEE_OUT: output directory. Defaults to <root>/formal_gee_results.
"""
from __future__ import annotations

import ast
import json
import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.stats.multitest import multipletests


OBJECTS = ["课程", "知识", "能力", "专业", "职业", "行业", "学科"]
EVIDENCES = [
    "个人学习经历", "实习就业经历", "身边他人经历", "招聘收入与市场结果",
    "专业知识与理论", "公共价值", "AI与平台可替代性", "身份情感", "反讽或梗", "无依据断言",
]


def parse_list(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, dict)):
        return value
    text = str(value).strip()
    if not text:
        return []
    try:
        return json.loads(text)
    except Exception:
        try:
            return ast.literal_eval(text)
        except Exception:
            return []


def find_database(root: Path) -> Path:
    candidates = [
        root / "00_freeze" / "analysis_v2_frozen.db",
        root / "02_views" / "analysis.db",
        root / "analysis" / "02_views" / "analysis.db",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Could not find a private analysis database")


def load_comments(db_path: Path) -> pd.DataFrame:
    connection = sqlite3.connect(db_path)
    tables = {row[0] for row in connection.execute("select name from sqlite_master where type='table'")}
    if "comments_anon" not in tables or "comment_labels" not in tables:
        raise RuntimeError("Database must contain comments_anon and comment_labels")
    query = """
        select c.*, l.evaluation_object, l.evidence_basis, l.stance
        from comments_anon c
        left join comment_labels l using(comment_id)
        where c.semantic_eligible=1 and c.is_orphan=0
    """
    data = pd.read_sql_query(query, connection)
    connection.close()
    return data


def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["objects"] = data.evaluation_object.map(parse_list)
    data["evidences"] = data.evidence_basis.map(parse_list)
    data["stance"] = data.stance.fillna("无法判断")
    data["is_reply"] = (data.comment_level >= 2).astype(int)
    data["log_text_length"] = np.log1p(data.text_length.fillna(0))
    data["is_note_author"] = data.is_note_author.fillna(0).astype(int)
    data["judgment_clear"] = (data.stance != "无法判断").astype(int)
    data["negative"] = data.stance.isin(["否定", "条件性否定"]).astype(int)
    data["positive"] = data.stance.isin(["认可", "条件性认可"]).astype(int)
    for index, value in enumerate(OBJECTS):
        data[f"o{index}"] = data.objects.map(lambda items: int(value in items))
    for index, value in enumerate(EVIDENCES):
        data[f"e{index}"] = data.evidences.map(lambda items: int(value in items))
    return data


def fit_gee(data: pd.DataFrame, formula: str) -> pd.DataFrame:
    data = data.dropna(subset=["note_id"]).copy()
    model = smf.gee(
        formula,
        groups=data.note_id.astype(str),
        data=data,
        family=sm.families.Binomial(),
        cov_struct=Exchangeable(),
    ).fit(maxiter=100)
    table = pd.DataFrame({
        "term": model.params.index,
        "coef": model.params.values,
        "se": model.bse.values,
        "p_value": model.pvalues.values,
    })
    table["odds_ratio"] = np.exp(table.coef)
    table["ci_low"] = np.exp(table.coef - 1.96 * table.se)
    table["ci_high"] = np.exp(table.coef + 1.96 * table.se)
    table["n"] = len(data)
    valid = table.p_value.notna()
    if valid.any():
        table.loc[valid, "p_fdr_bh"] = multipletests(table.loc[valid, "p_value"], method="fdr_bh")[1]
    table["model_family"] = "binomial_gee_exchangeable"
    table["cluster"] = "note_id"
    return table


def fit_optional_relation_models(root: Path, output: Path, registry: list[dict]) -> None:
    relation_path = os.environ.get("NEWS_EDU_RELATION_FRAME")
    if not relation_path:
        return
    frame = pd.read_csv(relation_path)
    frame["strategies"] = frame.final_reply_strategy_terminal.map(parse_list)
    frame["translation_any"] = (frame.final_translation_link_terminal.fillna("无转译") != "无转译").astype(int)
    frame["is_author"] = frame.reply_is_note_author.fillna(0).astype(int)
    frame["source_negative"] = frame.direct_v1_stance.isin(["否定", "条件性否定"]).astype(int)
    frame["log_reply_length"] = np.log1p(frame.reply_text_length.fillna(0))
    for index, value in enumerate(EVIDENCES):
        frame[f"e{index}"] = frame.direct_v1_evidence_basis.map(parse_list).map(lambda items: int(value in items))
    strategies = ["同层直接回应", "具体能力解释", "个人经验回应", "市场结果回应", "公共价值回应", "身份否定", "反讽与情绪", "话题转移", "其他"]
    for index, value in enumerate(strategies):
        frame[f"s{index}"] = frame.strategies.map(lambda items: int(value in items))
    quality = np.select([
        frame.final_review_status.isin(["pilot_final_agent", "agent_adjudicated_final_double_pass", "agent_adjudicated_final_single_pass"]),
        frame.final_review_status.eq("hybrid_model_high_confidence"),
    ], ["adjudicated", "hybrid_high"], default="hybrid_low")
    frame["quality"] = quality
    rhs = "is_author + source_negative + is_author:source_negative + C(reply_source_type) + log_reply_length + " + " + ".join(f"e{i}" for i in range(len(EVIDENCES)))
    for scope, subset in {
        "adjudicated": frame[frame.quality == "adjudicated"],
        "adjudicated_plus_high": frame[frame.quality.isin(["adjudicated", "hybrid_high"])],
        "full_hybrid": frame,
    }.items():
        data = subset[subset.final_relevance_status_terminal.isin(["核心相关", "语境相关"])].copy()
        for index, strategy in enumerate(strategies):
            if data[f"s{index}"].sum() < 15 or data[f"s{index}"].nunique() < 2:
                continue
            formula = f"s{index} ~ {rhs}"
            table = fit_gee(data, formula)
            table["scope"] = scope
            table["strategy"] = strategy
            table.to_csv(output / "rq2" / f"{scope}_strategy_{index}_gee.csv", index=False)
            registry.append({"model_id": f"rq2_{scope}_strategy_{index}", "rq": "RQ2", "outcome": strategy,
                             "scope": scope, "model_family": "binomial_gee_exchangeable", "cluster": "note_id",
                             "n": len(data), "formula": formula})
        if data.translation_any.sum() >= 15:
            formula = f"translation_any ~ {rhs}"
            table = fit_gee(data, formula)
            table["scope"] = scope
            table.to_csv(output / "rq2" / f"{scope}_translation_any_gee.csv", index=False)
            registry.append({"model_id": f"rq2_{scope}_translation", "rq": "RQ2", "outcome": "translation_any",
                             "scope": scope, "model_family": "binomial_gee_exchangeable", "cluster": "note_id",
                             "n": len(data), "formula": formula})


def fit_optional_ability_models(output: Path, registry: list[dict]) -> None:
    ability_path = os.environ.get("NEWS_EDU_ABILITY_MENTIONS")
    if not ability_path:
        return
    ability = pd.read_csv(ability_path)
    ability["is_reply"] = (ability.comment_level >= 2).astype(int)
    for prop, count in ability.ability_property.value_counts().items():
        if prop == "未明确" or count < 15:
            continue
        ability["y"] = (ability.ability_property == prop).astype(int)
        formula = "y ~ C(ability_type) + ai_context + C(source_type) + is_reply"
        table = fit_gee(ability, formula)
        table["property"] = prop
        safe_name = str(prop).replace("/", "_")
        table.to_csv(output / "rq3" / f"property_{safe_name}_gee.csv", index=False)
        registry.append({"model_id": f"rq3_{prop}", "rq": "RQ3", "outcome": prop,
                         "model_family": "binomial_gee_exchangeable", "cluster": "note_id",
                         "n": len(ability), "formula": formula})


def main() -> None:
    root = Path(os.environ["NEWS_EDU_ROOT"])
    output = Path(os.environ.get("NEWS_EDU_GEE_OUT", root / "formal_gee_results"))
    (output / "rq1").mkdir(parents=True, exist_ok=True)
    (output / "rq4").mkdir(parents=True, exist_ok=True)

    data = add_indicators(load_comments(find_database(root)))
    rhs = " + ".join([f"e{i}" for i in range(len(EVIDENCES))] + [f"o{i}" for i in range(len(OBJECTS))] + [
        "C(source_type)", "is_reply", "is_note_author", "log_text_length"
    ])
    registry = []
    for outcome in ["judgment_clear", "negative", "positive"]:
        subset = data if outcome == "judgment_clear" else data[data.judgment_clear == 1]
        table = fit_gee(subset, f"{outcome} ~ {rhs}")
        path = output / "rq1" / f"{outcome}_gee.csv"
        table.to_csv(path, index=False)
        registry.append({"model_id": f"rq1_{outcome}", "rq": "RQ1", "outcome": outcome,
                         "model_family": "binomial_gee_exchangeable", "cluster": "note_id",
                         "n": len(subset), "formula": f"{outcome} ~ {rhs}"})

    rhs4 = " + ".join([f"e{i}" for i in range(len(EVIDENCES))] + [f"o{i}" for i in range(len(OBJECTS))] + [
        "C(source_type)", "is_reply", "is_note_author", "is_reply:is_note_author", "log_text_length"
    ])
    data["any_like"] = (data.like_count_num.fillna(0) > 0).astype(int)
    table = fit_gee(data, f"any_like ~ {rhs4}")
    table.to_csv(output / "rq4" / "any_like_gee.csv", index=False)
    registry.append({"model_id": "rq4_any_like", "rq": "RQ4", "outcome": "any_like",
                     "model_family": "binomial_gee_exchangeable", "cluster": "note_id",
                     "n": len(data), "formula": f"any_like ~ {rhs4}"})

    (output / "rq2").mkdir(parents=True, exist_ok=True)
    (output / "rq3").mkdir(parents=True, exist_ok=True)
    fit_optional_relation_models(root, output, registry)
    fit_optional_ability_models(output, registry)

    metadata = {
        "model_policy": "GEE for binary outcomes; negative-binomial models remain separate for count outcomes",
        "database": str(find_database(root)),
        "n_comments": int(len(data)),
        "cluster": "note_id",
        "working_correlation": "Exchangeable",
        "registry": registry,
    }
    (output / "model_registry_gee.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "n_comments": len(data), "models": len(registry)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
