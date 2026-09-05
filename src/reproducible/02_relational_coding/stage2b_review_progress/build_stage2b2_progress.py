import os
import pandas as pd, json, hashlib, os, datetime, zipfile
from pathlib import Path
BASE=Path(os.environ['NEWS_EDU_ROOT']) / '02_relational_coding'
FULL=BASE/'stage2b_pilot/relation_full_firstpass_v2.1.csv.gz'
QUEUE=BASE/'stage2b_pilot/relation_review_queue_v2.1.csv.gz'
URG=BASE/'stage2b_review_urgent/urgent_outstanding_493_agent_final.csv'
HIGH=BASE/'stage2b_review_high_tranche01/high_tranche01_500_agent_final.csv'
OUT=BASE/'stage2b_review_progress'

full=pd.read_csv(FULL,low_memory=False)
queue=pd.read_csv(QUEUE,low_memory=False)
urg=pd.read_csv(URG,low_memory=False)
high=pd.read_csv(HIGH,low_memory=False)
assert len(urg)==493 and len(high)==500
assert set(urg.reply_comment_id).isdisjoint(set(high.reply_comment_id))

# Harmonize final labels for writeback
new=pd.concat([urg,high],ignore_index=True)
map_df=new[['reply_comment_id','adjudicated_relevance_status','adjudicated_reply_strategy_json','adjudicated_translation_link','review_status','review_depth','agent_confidence','ambiguity_flag','ambiguity_type_json','coding_reason','final_change_reason','coder_type','label_version']].copy()
map_df=map_df.rename(columns={
 'adjudicated_relevance_status':'new_final_relevance_status',
 'adjudicated_reply_strategy_json':'new_final_reply_strategy',
 'adjudicated_translation_link':'new_final_translation_link',
 'review_status':'new_review_status',
 'label_version':'new_label_version'
})
full2=full.merge(map_df,on='reply_comment_id',how='left',validate='one_to_one')
m=full2.new_review_status.notna()
assert int(m.sum())==993
full2.loc[m,'final_relevance_status']=full2.loc[m,'new_final_relevance_status']
full2.loc[m,'final_reply_strategy']=full2.loc[m,'new_final_reply_strategy']
full2.loc[m,'final_translation_link']=full2.loc[m,'new_final_translation_link']
full2.loc[m,'review_status']=full2.loc[m,'new_review_status']
full2.loc[m,'label_version']=full2.loc[m,'new_label_version']
# Keep audit fields with explicit stage2b2 prefix
for c in ['review_depth','agent_confidence','ambiguity_flag','ambiguity_type_json','coding_reason','final_change_reason','coder_type']:
    full2['stage2b2_'+c]=full2[c]
# Drop transient merge columns and unprefixed copies imported from map
for c in list(full2.columns):
    if c.startswith('new_') or c in ['review_depth','agent_confidence','ambiguity_flag','ambiguity_type_json','coding_reason','final_change_reason','coder_type']:
        full2.drop(columns=c,inplace=True)

# Validation
allowed_rel={'核心相关','语境相关','弱相关','无关'}
allowed_trans={'课程—能力连接','能力—职业连接','能力—公共价值连接','仅抽象价值表达','无转译'}
allowed_str={'同层直接回应','具体能力解释','个人经验回应','市场结果回应','公共价值回应','身份否定','反讽与情绪','话题转移','其他'}
for _,r in full2[full2.review_status.str.contains('agent_adjudicated_final',na=False)].iterrows():
    assert r.final_relevance_status in allowed_rel
    assert r.final_translation_link in allowed_trans
    a=json.loads(r.final_reply_strategy); assert len(a)<=2 and set(a)<=allowed_str
assert full2.reply_comment_id.is_unique and len(full2)==10620

full2.to_csv(OUT/'relation_full_review_progress_stage2b2_v2.1.csv.gz',index=False,compression='gzip',encoding='utf-8')

# Updated queue excludes newly finalized and still excludes pilot 400 (as original review queue did)
finalized=set(new.reply_comment_id)
q2=queue[~queue.reply_comment_id.isin(finalized)].copy()
q2.to_csv(OUT/'relation_review_queue_after_stage2b2_v2.1.csv.gz',index=False,compression='gzip',encoding='utf-8')

# Progress ledger per status and priority
status=pd.crosstab(full2['review_priority_stage2b'],full2['review_status'],dropna=False)
status.to_csv(OUT/'stage2b2_status_by_priority.csv',encoding='utf-8-sig')
flow=[]
for pri in ['紧急复核','高','中','低']:
    g=full2[full2.review_priority_stage2b.eq(pri)]
    flow.append({'priority':pri,'total':len(g),'pilot_final_agent':int(g.review_status.eq('pilot_final_agent').sum()),'stage2b2_agent_final':int(g.review_status.str.contains('agent_adjudicated_final',na=False).sum()),'remaining_provisional':int(g.review_status.eq('provisional_model_not_for_inference').sum())})
pd.DataFrame(flow).to_csv(OUT/'stage2b2_review_flow.csv',index=False,encoding='utf-8-sig')

# Combined new final labels for this stage
new[['reply_comment_id','note_id','title','direct_source_text','reply_text','reply_is_note_author','relation_type','adjudicated_relevance_status','adjudicated_reply_strategy_json','adjudicated_translation_link','agent_confidence','ambiguity_flag','ambiguity_type_json','evidence_span','coding_reason','review_depth','review_status','final_change_reason','coder_type','label_version']].to_csv(OUT/'stage2b2_newly_adjudicated_993.csv.gz',index=False,compression='gzip',encoding='utf-8')

summary={
 'relation_frame_total':10620,
 'pilot_final_agent':int(full2.review_status.eq('pilot_final_agent').sum()),
 'newly_agent_adjudicated_stage2b2':int(full2.review_status.str.contains('agent_adjudicated_final',na=False).sum()),
 'total_agent_reviewed_including_pilot':int((full2.review_status.eq('pilot_final_agent')|full2.review_status.str.contains('agent_adjudicated_final',na=False)).sum()),
 'remaining_provisional':int(full2.review_status.eq('provisional_model_not_for_inference').sum()),
 'new_urgent_outstanding':493,
 'urgent_total_including_pilot':525,
 'new_high_tranche01':500,
 'high_total_reviewed_including_pilot':int(((full2.review_priority_stage2b=='高') & (full2.review_status.ne('provisional_model_not_for_inference'))).sum()),
 'high_remaining':int(((full2.review_priority_stage2b=='高') & (full2.review_status.eq('provisional_model_not_for_inference'))).sum()),
 'double_pass_new':int(new.review_depth.eq('double_pass_critical_or_sample').sum()),
 'single_pass_new':int(new.review_depth.eq('single_pass_noncritical').sum()),
 'remaining_queue_n':len(q2),
 'status_counts':full2.review_status.value_counts().to_dict(),
 'priority_flow':flow,
 'generated_at':'2026-08-04T08:00:00+08:00'
}
with open(OUT/'stage2b2_progress_summary.json','w',encoding='utf-8') as f:json.dump(summary,f,ensure_ascii=False,indent=2)
print(json.dumps(summary,ensure_ascii=False,indent=2))
