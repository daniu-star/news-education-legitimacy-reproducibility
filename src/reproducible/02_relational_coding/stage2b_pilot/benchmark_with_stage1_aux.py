#!/usr/bin/env python3
from __future__ import annotations
import os
import json, sqlite3
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
ROOT=Path(os.environ['NEWS_EDU_ROOT'])
PILOT=ROOT/'02_relational_coding/stage2b_pilot/relation_pilot_400_agent_adjudicated.csv'
FRAME=ROOT/'02_relational_coding/reply_pair_frame.csv.gz'
DB=ROOT/'00_freeze/analysis_v2_frozen.db'
R2=ROOT/'01_reliability/round2/round2_adjudication_final.csv'
R3=ROOT/'01_reliability/round3/round3_final_adjudicated_labels.csv'
OUT=ROOT/'02_relational_coding/stage2b_pilot'
classes=['同层直接回应','具体能力解释','个人经验回应','市场结果回应','公共价值回应','身份否定','反讽与情绪','话题转移','其他']
mlb=MultiLabelBinarizer(classes=classes)

def safe(x):return '' if pd.isna(x) else str(x)
def parse(x):
 try:return json.loads(x) if isinstance(x,str) else []
 except:return []
def make_text(r):
 return ' [SEP] '.join([
  '标题 '+safe(r.get('title')),'上文 '+safe(r.get('source_text')),'回复 '+safe(r.get('reply_text')),
  '旧对象 '+safe(r.get('v1_object')),'旧依据 '+safe(r.get('v1_evidence')),'旧立场 '+safe(r.get('v1_stance')),
  '旧能力 '+safe(r.get('v1_ability')),'作者 '+safe(r.get('author_flag')),'语义 '+safe(r.get('semantic_flag'))])
# v1 labels
con=sqlite3.connect(DB); lab=pd.read_sql_query('select * from comment_labels',con).set_index('comment_id')
def v1(cid,col):
 try:return lab.at[str(cid),col]
 except:return ''
# pilot normalize
p=pd.read_csv(PILOT)
pn=pd.DataFrame({
 'note_id':p.note_id.astype(str),'comment_id':p.reply_comment_id.astype(str),'title':p.title,
 'source_text':p.direct_source_text,'reply_text':p.reply_text,'author_flag':p.reply_is_note_author,'semantic_flag':p.reply_semantic_eligible,
 'relevance':p.final_relevance_status,'strategy':p.final_reply_strategy,'translation':p.final_translation_link,
})
# aux normalize
aux=[]
for path in [R2,R3]:
 d=pd.read_csv(path)
 aux.append(pd.DataFrame({
  'note_id':d.note_id.astype(str),'comment_id':d.comment_id.astype(str),'title':d.note_title,
  'source_text':d.context_comment,'reply_text':d.current_comment,'author_flag':d.is_note_author,'semantic_flag':1,
  'relevance':d.final_relevance_status,'strategy':d.final_reply_strategy,'translation':d.final_translation_link,
 }))
a=pd.concat(aux,ignore_index=True).drop_duplicates('comment_id',keep='last')
# remove pilot duplicate ids
A=a[~a.comment_id.isin(set(pn.comment_id))].copy()
for d in [pn,A]:
 d['v1_object']=d.comment_id.map(lambda x:v1(x,'evaluation_object'))
 d['v1_evidence']=d.comment_id.map(lambda x:v1(x,'evidence_basis'))
 d['v1_stance']=d.comment_id.map(lambda x:v1(x,'stance'))
 d['v1_ability']=d.comment_id.map(lambda x:v1(x,'ability_type'))
pn['text']=pn.apply(make_text,axis=1);A['text']=A.apply(make_text,axis=1)
Yp=mlb.fit_transform(pn.strategy.map(parse)); Ya=mlb.transform(A.strategy.map(parse))
gkf=GroupKFold(5)
res=[]; rel_t=[];rel_p=[];tr_t=[];tr_p=[];s_t=[];s_p=[]
for fold,(tr,te) in enumerate(gkf.split(pn,pn.relevance,pn.note_id),1):
 train_text=A.text.tolist()+pn.iloc[tr].text.tolist();test_text=pn.iloc[te].text.tolist()
 vec=TfidfVectorizer(analyzer='char',ngram_range=(1,4),min_df=2,max_features=65000,sublinear_tf=True)
 Xtr=vec.fit_transform(train_text);Xte=vec.transform(test_text)
 rel_y=np.concatenate([A.relevance.values,pn.iloc[tr].relevance.values]); tran_y=np.concatenate([A.translation.values,pn.iloc[tr].translation.values]); strat_y=np.vstack([Ya,Yp[tr]])
 r=LinearSVC(class_weight='balanced',C=1.2).fit(Xtr,rel_y); rp=r.predict(Xte)
 t=LinearSVC(class_weight='balanced',C=1.0).fit(Xtr,tran_y); tp=t.predict(Xte)
 s=OneVsRestClassifier(LinearSVC(class_weight='balanced',C=0.8)).fit(Xtr,strat_y); sp=s.predict(Xte)
 rel_t.extend(pn.iloc[te].relevance);rel_p.extend(rp);tr_t.extend(pn.iloc[te].translation);tr_p.extend(tp);s_t.append(Yp[te]);s_p.append(sp)
 res.append({'fold':fold,'n':len(te),'rel_acc':accuracy_score(pn.iloc[te].relevance,rp),'rel_macro_f1':f1_score(pn.iloc[te].relevance,rp,average='macro',zero_division=0),'strat_micro_f1':f1_score(Yp[te],sp,average='micro',zero_division=0),'strat_macro_f1':f1_score(Yp[te],sp,average='macro',zero_division=0),'trans_acc':accuracy_score(pn.iloc[te].translation,tp),'trans_macro_f1':f1_score(pn.iloc[te].translation,tp,average='macro',zero_division=0)})
s_t=np.vstack(s_t);s_p=np.vstack(s_p)
summary={'pilot_n':len(pn),'aux_n':len(A),'split':'pilot GroupKFold by note_id; stage1 adjudicated samples fixed in training','relevance':{'accuracy':accuracy_score(rel_t,rel_p),'macro_f1':f1_score(rel_t,rel_p,average='macro',zero_division=0)},'strategy':{'micro_f1':f1_score(s_t,s_p,average='micro',zero_division=0),'macro_f1':f1_score(s_t,s_p,average='macro',zero_division=0),'exact_match':float(np.mean(np.all(s_t==s_p,axis=1)))},'translation':{'accuracy':accuracy_score(tr_t,tr_p),'macro_f1':f1_score(tr_t,tr_p,average='macro',zero_division=0)}}
summary['quality_gate']={'relevance_pass':summary['relevance']['macro_f1']>=.8,'strategy_pass':summary['strategy']['micro_f1']>=.8,'translation_pass':summary['translation']['macro_f1']>=.8}
(OUT/'pilot_model_benchmark_with_stage1_aux.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
pd.DataFrame(res).to_csv(OUT/'pilot_model_benchmark_with_stage1_aux_by_fold.csv',index=False,encoding='utf-8-sig')
p,r,f,s=precision_recall_fscore_support(s_t,s_p,average=None,zero_division=0)
pd.DataFrame({'class':classes,'precision':p,'recall':r,'f1':f,'support':s}).to_csv(OUT/'pilot_strategy_per_class_metrics_with_stage1_aux.csv',index=False,encoding='utf-8-sig')
print(json.dumps(summary,ensure_ascii=False,indent=2))
