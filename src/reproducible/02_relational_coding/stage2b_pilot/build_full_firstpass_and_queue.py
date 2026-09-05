#!/usr/bin/env python3
from __future__ import annotations
import os
import json, sqlite3, math
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

ROOT=Path(os.environ['NEWS_EDU_ROOT'])
REL=ROOT/'02_relational_coding'
PILOT=REL/'stage2b_pilot/relation_pilot_400_agent_adjudicated.csv'
FRAME=REL/'reply_pair_frame.csv.gz'
SEED=REL/'reply_strategy_translation_seed_labels.csv.gz'
R2=ROOT/'01_reliability/round2/round2_adjudication_final.csv'
R3=ROOT/'01_reliability/round3/round3_final_adjudicated_labels.csv'
OUT=REL/'stage2b_pilot'

classes_all=['同层直接回应','具体能力解释','个人经验回应','市场结果回应','公共价值回应','身份否定','反讽与情绪','话题转移','其他']
model_classes=['同层直接回应','具体能力解释','个人经验回应','市场结果回应','公共价值回应','反讽与情绪','话题转移']

def safe(x):return '' if pd.isna(x) else str(x)
def parse(x):
 try:return json.loads(x) if isinstance(x,str) else []
 except:return []
def make_text(r):
 return ' [SEP] '.join([
  '标题 '+safe(r.get('title')),'上文 '+safe(r.get('source_text')),'回复 '+safe(r.get('reply_text')),
  '旧对象 '+safe(r.get('v1_object')),'旧依据 '+safe(r.get('v1_evidence')),'旧立场 '+safe(r.get('v1_stance')),
  '旧能力 '+safe(r.get('v1_ability')),'作者 '+safe(r.get('author_flag')),'语义 '+safe(r.get('semantic_flag'))])
# Full frame and seeds.
frame=pd.read_csv(FRAME,compression='infer')
full=frame[frame.relation_model_eligible==1].copy()
assert len(full)==10620
seed=pd.read_csv(SEED,compression='infer')
seed=seed[['reply_comment_id','reply_strategy_seed_json','translation_seed','secondary_translation_seed','seed_confidence','seed_review_priority','seed_reason']]
full=full.merge(seed,on='reply_comment_id',how='left',validate='one_to_one')
# Normalize full feature table.
F=pd.DataFrame({
 'note_id':full.note_id.astype(str),'comment_id':full.reply_comment_id.astype(str),'title':full.title,
 'source_text':full.direct_source_text,'reply_text':full.reply_text,'author_flag':full.reply_is_note_author,'semantic_flag':full.reply_semantic_eligible,
 'v1_object':full.reply_v1_evaluation_object,'v1_evidence':full.reply_v1_evidence_basis,'v1_stance':full.reply_v1_stance,'v1_ability':full.reply_v1_ability_type,
})
F['text']=F.apply(make_text,axis=1)
# Pilot labels.
p=pd.read_csv(PILOT)
P=pd.DataFrame({
 'note_id':p.note_id.astype(str),'comment_id':p.reply_comment_id.astype(str),'title':p.title,
 'source_text':p.direct_source_text,'reply_text':p.reply_text,'author_flag':p.reply_is_note_author,'semantic_flag':p.reply_semantic_eligible,
 'v1_object':'','v1_evidence':'','v1_stance':'','v1_ability':'',
 'relevance':p.final_relevance_status,'strategy':p.final_reply_strategy,'translation':p.final_translation_link,
})
# Merge V1 from full where available.
v1map=F.set_index('comment_id')[['v1_object','v1_evidence','v1_stance','v1_ability']].to_dict('index')
for col in ['v1_object','v1_evidence','v1_stance','v1_ability']:
 P[col]=P.comment_id.map(lambda x:v1map.get(x,{}).get(col,''))
P['text']=P.apply(make_text,axis=1)
# Stage1 auxiliary for relevance only.
aux=[]
for path in [R2,R3]:
 d=pd.read_csv(path)
 a=pd.DataFrame({'note_id':d.note_id.astype(str),'comment_id':d.comment_id.astype(str),'title':d.note_title,'source_text':d.context_comment,'reply_text':d.current_comment,'author_flag':d.is_note_author,'semantic_flag':1,'relevance':d.final_relevance_status})
 for col in ['v1_object','v1_evidence','v1_stance','v1_ability']:
  dbcol={'v1_object':'evaluation_object','v1_evidence':'evidence_basis','v1_stance':'stance','v1_ability':'ability_type'}[col]
  a[col]=a.comment_id.map(lambda x:v1map.get(x,{}).get(col,''))
 a['text']=a.apply(make_text,axis=1);aux.append(a)
A=pd.concat(aux,ignore_index=True).drop_duplicates('comment_id',keep='last')
A=A[~A.comment_id.isin(set(P.comment_id))]
# Shared vectorizer fit on all labeled text, then transform full.
train_text=P.text.tolist()+A.text.tolist()
vec=TfidfVectorizer(analyzer='char',ngram_range=(1,4),min_df=2,max_features=70000,sublinear_tf=True)
Xtrain_all=vec.fit_transform(train_text);Xfull=vec.transform(F.text.tolist())
# Relevance model uses pilot + aux.
yrel=np.concatenate([P.relevance.values,A.relevance.values])
rel_model=LinearSVC(class_weight='balanced',C=1.2).fit(Xtrain_all,yrel)
rel_scores=rel_model.decision_function(Xfull);rel_pred=rel_model.classes_[np.argmax(rel_scores,axis=1)]
rel_sorted=np.sort(rel_scores,axis=1);rel_margin=rel_sorted[:,-1]-rel_sorted[:,-2]
# Pilot-only matrix uses first len(P) rows of shared matrix.
Xp=Xtrain_all[:len(P)]
# Strategy model only classes with enough support.
mlb=MultiLabelBinarizer(classes=model_classes)
Ys=mlb.fit_transform(P.strategy.map(parse))
strat_model=OneVsRestClassifier(LinearSVC(class_weight='balanced',C=0.8)).fit(Xp,Ys)
strat_scores=strat_model.decision_function(Xfull)
strategy_pred=[];strategy_margin=[]
for scores in strat_scores:
 order=np.argsort(scores)[::-1]
 labels=[model_classes[i] for i in order if scores[i]>0][:2]
 if not labels and scores[order[0]]>-0.15:labels=[model_classes[order[0]]]
 strategy_pred.append(labels)
 # uncertainty around zero and gap between top labels
 strategy_margin.append(float(abs(scores[order[0]]) if len(order)==1 else min(abs(scores[order[0]]),abs(scores[order[1]]))))
# Translation pilot-only.
ytr=P.translation.values
trans_model=LinearSVC(class_weight='balanced',C=1.0).fit(Xp,ytr)
trans_scores=trans_model.decision_function(Xfull);trans_pred=trans_model.classes_[np.argmax(trans_scores,axis=1)]
trans_sorted=np.sort(trans_scores,axis=1);trans_margin=trans_sorted[:,-1]-trans_sorted[:,-2]
# Build candidate labels and priority.
manual_by_id=p.set_index('reply_comment_id')
rows=[]
rare={'身份否定','公共价值回应','具体能力解释'}
for i,r in full.reset_index(drop=True).iterrows():
 seed_s=parse(r.reply_strategy_seed_json)
 model_s=list(strategy_pred[i])
 # Retain explicit identity seed as a candidate because pilot lacked this rare class.
 if '身份否定' in seed_s and '身份否定' not in model_s:model_s=['身份否定']+model_s
 model_s=model_s[:2]
 seed_t=safe(r.translation_seed) or '无转译'
 mpred_t=str(trans_pred[i])
 # candidate is model, with seed positive preserved as secondary candidate when disagreement.
 candidate_t=mpred_t
 secondary_candidate=''
 if seed_t!='无转译' and seed_t!=mpred_t:secondary_candidate=seed_t
 seed_exact=set(seed_s)==set(model_s)
 trans_agree=seed_t==mpred_t
 score=0;reasons=[]
 if int(r.core_semantic_relation_eligible)==1:score+=3;reasons.append('双方语义有效')
 if int(r.reply_is_note_author)==1:score+=2;reasons.append('笔记作者回复')
 if r.relation_type=='嵌套回复—直接父级可恢复':score+=3;reasons.append('嵌套关系')
 if not seed_exact:score+=2;reasons.append('策略模型与规则不一致')
 if not trans_agree:score+=2;reasons.append('转译模型与规则不一致')
 if candidate_t!='无转译' or seed_t!='无转译':score+=3;reasons.append('价值转译候选')
 if any(x in rare for x in model_s+seed_s):score+=2;reasons.append('低频关键策略')
 if rel_margin[i]<0.35:score+=2;reasons.append('相关性模型低间隔')
 if strategy_margin[i]<0.20:score+=1;reasons.append('策略模型低间隔')
 if trans_margin[i]<0.40:score+=1;reasons.append('转译模型低间隔')
 if int(r.reply_short_text)==1:score+=1;reasons.append('短文本')
 if r.reply_source_type=='browser':score+=1;reasons.append('浏览器来源')
 if score>=10:priority='紧急复核'
 elif score>=7:priority='高'
 elif score>=4:priority='中'
 else:priority='低'
 status='provisional_model_not_for_inference'
 final_rel='';final_s='';final_t=''
 cid=str(r.reply_comment_id)
 if cid in manual_by_id.index:
  mr=manual_by_id.loc[cid]
  status='pilot_final_agent'
  final_rel=mr.final_relevance_status;final_s=mr.final_reply_strategy;final_t=mr.final_translation_link
 rows.append({
  'reply_comment_id':cid,'model_relevance':rel_pred[i],'model_relevance_margin':float(rel_margin[i]),
  'model_strategy_json':json.dumps(model_s,ensure_ascii=False),'model_strategy_margin':float(strategy_margin[i]),
  'seed_strategy_json':r.reply_strategy_seed_json,'strategy_seed_model_exact':int(seed_exact),
  'model_translation':mpred_t,'model_translation_margin':float(trans_margin[i]),'seed_translation':seed_t,'translation_seed_model_agree':int(trans_agree),
  'secondary_translation_candidate':secondary_candidate,'review_score':score,'review_priority_stage2b':priority,'review_reason':'；'.join(reasons) if reasons else '低风险常规复核',
  'final_relevance_status':final_rel,'final_reply_strategy':final_s,'final_translation_link':final_t,
  'label_version':'relational_v2.1','review_status':status,
 })
pred=pd.DataFrame(rows)
out=full.merge(pred,on='reply_comment_id',how='left',validate='one_to_one')
out.to_csv(OUT/'relation_full_firstpass_v2.1.csv.gz',index=False,compression='gzip',encoding='utf-8-sig')
# Review queue puts provisional first, then by priority and score.
rank={'紧急复核':0,'高':1,'中':2,'低':3}
q=out[out.review_status!='pilot_final_agent'].copy();q['_rank']=q.review_priority_stage2b.map(rank);q=q.sort_values(['_rank','review_score','reply_is_note_author','core_semantic_relation_eligible'],ascending=[True,False,False,False]).drop(columns='_rank')
q.to_csv(OUT/'relation_review_queue_v2.1.csv.gz',index=False,compression='gzip',encoding='utf-8-sig')
# Pilot final rows extracted from full first pass.
out[out.review_status=='pilot_final_agent'].to_csv(OUT/'relation_pilot_400_in_full_frame.csv',index=False,encoding='utf-8-sig')
summary={
 'total_relations':int(len(out)),'pilot_final_agent':int((out.review_status=='pilot_final_agent').sum()),'provisional_not_for_inference':int((out.review_status!='pilot_final_agent').sum()),
 'core_semantic_relations':int(out.core_semantic_relation_eligible.sum()),
 'review_priority':out.review_priority_stage2b.value_counts().to_dict(),
 'seed_model_strategy_exact_rate':float(out.strategy_seed_model_exact.mean()),'seed_model_translation_agreement_rate':float(out.translation_seed_model_agree.mean()),
 'model_relevance_distribution':out.model_relevance.value_counts().to_dict(),
 'model_translation_distribution':out.model_translation.value_counts().to_dict(),
 'status_warning':'除400条pilot_final_agent外，其余模型候选不得进入RQ2推断或论文比例。'
}
(OUT/'relation_full_firstpass_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
