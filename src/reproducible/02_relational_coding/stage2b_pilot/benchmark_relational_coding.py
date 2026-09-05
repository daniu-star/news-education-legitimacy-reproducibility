#!/usr/bin/env python3
from __future__ import annotations
import os
import json, re
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, classification_report

ROOT=Path(os.environ['NEWS_EDU_ROOT']) / '02_relational_coding'
PILOT=ROOT/'stage2b_pilot'/'relation_pilot_400_agent_adjudicated.csv'
FRAME=ROOT/'reply_pair_frame.csv.gz'
SEED=ROOT/'reply_strategy_translation_seed_labels.csv.gz'
OUT=ROOT/'stage2b_pilot'

def safe(x):
    return '' if pd.isna(x) else str(x)

def parse_list(x):
    try: return json.loads(x) if isinstance(x,str) else []
    except Exception: return []

df=pd.read_csv(PILOT)
frame=pd.read_csv(FRAME,compression='infer')
extra_cols=['reply_comment_id','reply_v1_evaluation_object','reply_v1_evidence_basis','reply_v1_stance','reply_v1_ability_type','reply_v1_ability_property']
df=df.merge(frame[extra_cols],on='reply_comment_id',how='left',validate='one_to_one')
seed=pd.read_csv(SEED,compression='infer')[['reply_comment_id','reply_strategy_seed_json','translation_seed','seed_review_priority','seed_confidence']]
df=df.merge(seed,on='reply_comment_id',how='left',validate='one_to_one')

def make_text(r):
    parts=[
        '标题 '+safe(r.get('title')),
        '直接上文 '+safe(r.get('direct_source_text')),
        '当前回复 '+safe(r.get('reply_text')),
        '旧对象 '+safe(r.get('reply_v1_evaluation_object')),
        '旧依据 '+safe(r.get('reply_v1_evidence_basis')),
        '旧立场 '+safe(r.get('reply_v1_stance')),
        '旧能力 '+safe(r.get('reply_v1_ability_type')),
        '作者回复 '+str(int(r.get('reply_is_note_author',0))),
        '语义资格 '+str(int(r.get('reply_semantic_eligible',0))),
    ]
    return ' [SEP] '.join(parts)
texts=df.apply(make_text,axis=1).tolist()
groups=df['note_id'].astype(str).values
classes=['同层直接回应','具体能力解释','个人经验回应','市场结果回应','公共价值回应','身份否定','反讽与情绪','话题转移','其他']
mlb=MultiLabelBinarizer(classes=classes)
Y_strategy=mlb.fit_transform(df['final_reply_strategy'].map(parse_list))
y_rel=df['final_relevance_status'].astype(str).values
y_trans=df['final_translation_link'].astype(str).values

# Seed benchmark against adjudicated pilot.
seed_Y=mlb.transform(df['reply_strategy_seed_json'].map(parse_list))
seed_strategy_metrics={
    'micro_f1':float(f1_score(Y_strategy,seed_Y,average='micro',zero_division=0)),
    'macro_f1':float(f1_score(Y_strategy,seed_Y,average='macro',zero_division=0)),
    'exact_match':float(np.mean(np.all(Y_strategy==seed_Y,axis=1))),
}
seed_translation_metrics={
    'accuracy':float(accuracy_score(y_trans,df['translation_seed'].astype(str))),
    'macro_f1':float(f1_score(y_trans,df['translation_seed'].astype(str),average='macro',zero_division=0)),
}

# Grouped CV; vectorizer fitted within each fold.
gkf=GroupKFold(n_splits=5)
rel_true=[]; rel_pred=[]; trans_true=[]; trans_pred=[]; strat_true=[]; strat_pred=[]; fold_rows=[]
for fold,(tr,te) in enumerate(gkf.split(texts,y_rel,groups),1):
    vec=TfidfVectorizer(analyzer='char',ngram_range=(1,4),min_df=2,max_features=50000,sublinear_tf=True)
    Xtr=vec.fit_transform([texts[i] for i in tr]); Xte=vec.transform([texts[i] for i in te])
    rel=LinearSVC(class_weight='balanced',C=1.2)
    rel.fit(Xtr,y_rel[tr]); rp=rel.predict(Xte)
    trans=LinearSVC(class_weight='balanced',C=1.0)
    trans.fit(Xtr,y_trans[tr]); tp=trans.predict(Xte)
    strat=OneVsRestClassifier(LinearSVC(class_weight='balanced',C=0.8))
    strat.fit(Xtr,Y_strategy[tr]); sp=strat.predict(Xte)
    rel_true.extend(y_rel[te]); rel_pred.extend(rp)
    trans_true.extend(y_trans[te]); trans_pred.extend(tp)
    strat_true.append(Y_strategy[te]); strat_pred.append(sp)
    fold_rows.append({
        'fold':fold,'test_n':len(te),
        'relevance_accuracy':accuracy_score(y_rel[te],rp),
        'relevance_macro_f1':f1_score(y_rel[te],rp,average='macro',zero_division=0),
        'strategy_micro_f1':f1_score(Y_strategy[te],sp,average='micro',zero_division=0),
        'strategy_macro_f1':f1_score(Y_strategy[te],sp,average='macro',zero_division=0),
        'translation_accuracy':accuracy_score(y_trans[te],tp),
        'translation_macro_f1':f1_score(y_trans[te],tp,average='macro',zero_division=0),
    })
strat_true=np.vstack(strat_true); strat_pred=np.vstack(strat_pred)
summary={
    'sample_n':len(df),
    'split':'5-fold GroupKFold by note_id',
    'seed_strategy':seed_strategy_metrics,
    'seed_translation':seed_translation_metrics,
    'model_relevance':{
        'accuracy':float(accuracy_score(rel_true,rel_pred)),
        'macro_f1':float(f1_score(rel_true,rel_pred,average='macro',zero_division=0)),
    },
    'model_strategy':{
        'micro_f1':float(f1_score(strat_true,strat_pred,average='micro',zero_division=0)),
        'macro_f1':float(f1_score(strat_true,strat_pred,average='macro',zero_division=0)),
        'exact_match':float(np.mean(np.all(strat_true==strat_pred,axis=1))),
    },
    'model_translation':{
        'accuracy':float(accuracy_score(trans_true,trans_pred)),
        'macro_f1':float(f1_score(trans_true,trans_pred,average='macro',zero_division=0)),
    },
    'quality_gate':{
        'relevance_pass':bool(f1_score(rel_true,rel_pred,average='macro',zero_division=0)>=0.80),
        'strategy_pass':bool(f1_score(strat_true,strat_pred,average='micro',zero_division=0)>=0.80),
        'translation_pass':bool(f1_score(trans_true,trans_pred,average='macro',zero_division=0)>=0.80),
    }
}
(OUT/'pilot_model_benchmark.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
pd.DataFrame(fold_rows).to_csv(OUT/'pilot_model_benchmark_by_fold.csv',index=False,encoding='utf-8-sig')
# Per-class strategy metrics.
p,r,f,s=precision_recall_fscore_support(strat_true,strat_pred,average=None,zero_division=0)
pd.DataFrame({'class':classes,'precision':p,'recall':r,'f1':f,'support':s}).to_csv(OUT/'pilot_strategy_per_class_metrics.csv',index=False,encoding='utf-8-sig')
# Confusion detail.
pd.DataFrame({'true':rel_true,'pred':rel_pred}).to_csv(OUT/'pilot_relevance_predictions.csv',index=False,encoding='utf-8-sig')
pd.DataFrame({'true':trans_true,'pred':trans_pred}).to_csv(OUT/'pilot_translation_predictions.csv',index=False,encoding='utf-8-sig')
print(json.dumps(summary,ensure_ascii=False,indent=2))
