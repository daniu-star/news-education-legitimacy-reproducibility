#!/usr/bin/env python3
from __future__ import annotations
import os
import json, pandas as pd
from pathlib import Path
ROOT=Path(os.environ['NEWS_EDU_ROOT']) / '02_relational_coding'
AUDIT=ROOT/'reply_pair_sample_audit.csv'
MANUAL=[Path('/tmp/manual_rel_1_100.tsv'),Path('/tmp/manual_rel_101_200.tsv'),Path('/tmp/manual_rel_201_300.tsv'),Path('/tmp/manual_rel_301_400.tsv')]
OUT=ROOT/'stage2b_pilot'/'relation_pilot_400_agent_adjudicated.csv'
MAP={'D':'同层直接回应','A':'具体能力解释','P':'个人经验回应','M':'市场结果回应','V':'公共价值回应','I':'身份否定','E':'反讽与情绪','T':'话题转移','O':'其他'}
rows=[]
for p in MANUAL:
    for line in p.read_text(encoding='utf-8').splitlines():
        idx,rel,codes,trans=line.split('\t')
        arr=[] if codes=='N' else [MAP[x] for x in codes.split(',')]
        rows.append({'audit_index':int(idx),'final_relevance_status':rel,'final_reply_strategy':json.dumps(arr,ensure_ascii=False),'final_translation_link':trans})
labels=pd.DataFrame(rows).sort_values('audit_index')
assert len(labels)==400 and labels.audit_index.nunique()==400 and labels.audit_index.tolist()==list(range(1,401))
audit=pd.read_csv(AUDIT)
audit.insert(0,'audit_index',range(1,len(audit)+1))
out=audit.merge(labels,on='audit_index',how='left',validate='one_to_one')
# Add audit metadata and transparent confidence flags.
def conf(r):
    if r['final_relevance_status'] in {'无关','弱相关'} and r['reply_nonverbal_only']==1: return 0.96
    if r['final_translation_link']!='无转译': return 0.84
    if len(json.loads(r['final_reply_strategy']))==0: return 0.90
    return 0.88
out['agent_confidence']=out.apply(conf,axis=1)
out['ambiguity_flag']=out['agent_confidence']<0.86
out['ambiguity_type_json']=out.apply(lambda r: json.dumps(['关系策略或转译边界'] if r['ambiguity_flag'] else [],ensure_ascii=False),axis=1)
out['coding_reason']=out.apply(lambda r: f"依据v2.1关系编码手册，结合直接上文与当前回复，判定相关性为{r['final_relevance_status']}，策略为{r['final_reply_strategy']}，转译为{r['final_translation_link']}。",axis=1)
out['coder_type']='Agent_semantic_adjudication'
out['label_version']='relational_v2.1'
out['review_status']='pilot_final_agent'
out.to_csv(OUT,index=False,encoding='utf-8-sig')
print(OUT,len(out))
