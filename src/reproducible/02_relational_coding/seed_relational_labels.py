#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate high-precision, non-final seed labels for relation coding.

The output is a workload-routing artifact only. It must never be used as final
RQ2 labels unless every row has been semantically reviewed and review_status is
changed to final.
"""
from __future__ import annotations
import argparse
import json
import re
import unicodedata
from pathlib import Path
import pandas as pd

def norm(x: object) -> str:
    text = "" if pd.isna(x) else str(x)
    text = unicodedata.normalize("NFKC", text).strip()
    return re.sub(r"\s+", "", text)

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    df = pd.read_csv(args.input, compression="infer")

    pure = re.compile(r"^(谢谢|好的|好滴|收到|明白了|懂了|学到了|求|蹲|dd|码住|收藏了|哈哈+|呵呵+|笑死|(\[[^\]]+\]){1,4}|(@[^ ]+)+|[?？!！。,.，、~～]+)$", re.I)
    direct = re.compile(r"^(对|是的|确实|没错|同意|不同意|不是|不对|没有|有|可以|不可以|会|不会|能|不能|当然|未必|但|但是|不过|其实|所以|因为|哪有|为什么|看情况)")
    market = re.compile(r"(工资|薪资|收入|月薪|年薪|岗位|招聘|就业|失业|找工作|考公|编制|录取|分数线|调档线|门槛|名额|需求|供需|裁员|毕业去向|待遇)")
    public = re.compile(r"(真相|事实核查|公共利益|公共性|社会责任|新闻伦理|媒体伦理|舆论监督|新闻监督|知情权|弱者保护|记录社会|公信力|新闻理想)")
    identity = re.compile(r"(你(懂|学过|读过|做过|干过|从业过|什么学历|哪个学校|没学过|没做过|没资格)|外行人|不懂装懂|没资格发言|你懂还是我懂)")
    emotion = re.compile(r"(笑死|哈哈哈|呵呵|无语|离谱|逆天|绷不住|破防|急了|酸了|可笑|恶心|服了|救命|doge|\[(?:捂脸|扶额|哭惹|笑哭|失望|再见|微笑|完啦|偷笑|裂开|生气|流泪|大哭|可怜|尬笑)[^\]]*\])", re.I)
    exp = re.compile(r"((我|我们|本人|室友|同学|朋友|学长|学姐|同事|家人|女儿|儿子).{0,18}(毕业|在读|读了|学了|实习|工作|入职|离职|转行|考上|求职|面试|工资|薪资|收入|从业|做过|干过|待过))")
    ability_terms = ["写作","写稿","拍摄","摄影","剪辑","编辑","排版","采访","信源","核实","查证","调查","证据","搜索","整理","数据分析","判断","洞察","议题发现","策划","账号运营","投流","营销","增长","表达能力","沟通能力","伦理判断","媒介素养","建模","编程"]
    causal = re.compile(r"(因为|所以|通过|从而|导致|才能|可以|能够|帮助|有助于|用于|用来|需要|必须|学会|培养|形成|进入|对应|服务于|支撑)")
    course = re.compile(r"(课程|课堂|上课|老师|作业|实践课|实训|教学|培养|训练|教材|专业课)")
    career = re.compile(r"(岗位|职业|就业|工作|求职|薪资|工资|收入|考公|编制|转行|媒体|记者|运营|报社|电视台|行业)")

    rows = []
    for _, row in df.iterrows():
        text = norm(row.get("reply_text"))
        strategies = []
        confidence = 0.50
        if pure.fullmatch(text):
            strategies = ["话题转移"]
            confidence = 0.97
        else:
            if identity.search(text):
                strategies.append("身份否定")
            if exp.search(text):
                strategies.append("个人经验回应")
            if market.search(text):
                strategies.append("市场结果回应")
            if public.search(text):
                strategies.append("公共价值回应")
            count_ability = sum(term in text for term in ability_terms)
            if count_ability and (causal.search(text) or count_ability >= 2):
                strategies.append("具体能力解释")
            if emotion.search(text):
                strategies.append("反讽与情绪")
            if direct.search(text) or (text.endswith(("吗","呢","？","?")) and len(text) <= 80):
                strategies.insert(0, "同层直接回应")
            strategies = list(dict.fromkeys(strategies))
            if len(strategies) > 2:
                strategies = strategies[:2]
            if not strategies:
                strategies = ["其他"]
            confidence = 0.90 if len(strategies) == 1 and strategies != ["其他"] else (0.82 if strategies != ["其他"] else 0.50)

        has_ability = any(term in text for term in ability_terms)
        has_link = bool(causal.search(text))
        candidates = []
        if has_link and course.search(text) and has_ability:
            candidates.append("课程—能力连接")
        if has_link and has_ability and career.search(text):
            candidates.append("能力—职业连接")
        if has_link and has_ability and public.search(text):
            candidates.append("能力—公共价值连接")
        translation = candidates[0] if candidates else ("仅抽象价值表达" if public.search(text) else "无转译")
        secondary = candidates[1] if len(candidates) > 1 else ""
        priority = "低" if confidence >= 0.90 and translation in {"无转译","仅抽象价值表达"} else ("中" if confidence >= 0.80 else "高")
        rows.append({
            "reply_strategy_seed_json": json.dumps(strategies, ensure_ascii=False),
            "translation_seed": translation,
            "secondary_translation_seed": secondary,
            "seed_confidence": confidence,
            "seed_review_priority": priority,
            "seed_method": "high_precision_rules_v2",
            "review_status": "seed_only_not_final",
        })

    out = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, compression="infer", encoding="utf-8-sig")

if __name__ == "__main__":
    main()
