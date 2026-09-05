"""Historical binary GLM pipeline.

This module is retained for audit history only. Its cluster-robust GLM outputs
are not the manuscript-facing GEE estimates. Use run_complete_pipeline.py or
run_formal_gee_models.py for the formal binary models.
"""
import os
from pathlib import Path
import pandas as pd, numpy as np, sqlite3, json, ast, warnings
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore')
ROOT=Path(os.environ['NEWS_EDU_ROOT']); OUT=ROOT/'final_analysis'; DB=ROOT/'00_freeze/analysis_v2_frozen.db'
for d in ['05_models/rq1','05_models/rq4','07_robustness']: (OUT/d).mkdir(parents=True,exist_ok=True)
def jl(x):
 if x is None or (isinstance(x,float) and np.isnan(x)): return []
 try:return json.loads(str(x))
 except:
  try:return ast.literal_eval(str(x))
  except:return []
def fit_bin(d,formula):
 try:
  m=smf.glm(formula,data=d,family=sm.families.Binomial()).fit(cov_type='cluster',cov_kwds={'groups':d.note_id.astype(str)},maxiter=100)
  t=pd.DataFrame({'term':m.params.index,'coef':m.params.values,'se':m.bse.values,'p_value':m.pvalues.values}); t['odds_ratio']=np.exp(t.coef);t['ci_low']=np.exp(t.coef-1.96*t.se);t['ci_high']=np.exp(t.coef+1.96*t.se);t['n']=len(d);return t,{'ok':1,'n':len(d),'formula':formula}
 except Exception as e:return pd.DataFrame([{'term':'model_error','error':str(e),'n':len(d)}]),{'ok':0,'n':len(d),'formula':formula,'error':str(e)}
def fit_nb(d,formula,offset=None):
 try:
  m=smf.glm(formula,data=d,family=sm.families.NegativeBinomial(alpha=1),offset=d[offset] if offset else None).fit(cov_type='cluster',cov_kwds={'groups':d.note_id.astype(str)},maxiter=100)
  t=pd.DataFrame({'term':m.params.index,'coef':m.params.values,'se':m.bse.values,'p_value':m.pvalues.values}); t['irr']=np.exp(t.coef);t['ci_low']=np.exp(t.coef-1.96*t.se);t['ci_high']=np.exp(t.coef+1.96*t.se);t['n']=len(d);return t,{'ok':1,'n':len(d),'formula':formula}
 except Exception as e:return pd.DataFrame([{'term':'model_error','error':str(e),'n':len(d)}]),{'ok':0,'n':len(d),'formula':formula,'error':str(e)}
def fdr(t):
 if 'p_value' in t:
  v=t.p_value.notna(); t.loc[v,'p_fdr_bh']=multipletests(t.loc[v,'p_value'],method='fdr_bh')[1] if v.sum() else np.nan
 return t
con=sqlite3.connect(DB)
d=pd.read_sql_query('''select c.*,l.evaluation_object,l.evidence_basis,l.stance from comments_anon c left join comment_labels l using(comment_id) where c.semantic_eligible=1 and c.is_orphan=0''',con);con.close()
d['objs']=d.evaluation_object.map(jl);d['evs']=d.evidence_basis.map(jl);d['stance']=d.stance.fillna('无法判断');d['is_reply']=(d.comment_level>=2).astype(int);d['log_text_length']=np.log1p(d.text_length.fillna(0));d['any_like']=(d.like_count_num.fillna(0)>0).astype(int);d['positive_likes']=d.like_count_num.fillna(0).clip(lower=0);d['judgment_clear']=(d.stance!='无法判断').astype(int);d['negative']=d.stance.isin(['否定','条件性否定']).astype(int);d['positive']=d.stance.isin(['认可','条件性认可']).astype(int)
objects=['课程','知识','能力','专业','职业','行业','学科']; evs=['个人学习经历','实习就业经历','身边他人经历','招聘收入与市场结果','专业知识与理论','公共价值','AI与平台可替代性','身份情感','反讽或梗','无依据断言']
for i,x in enumerate(objects):d[f'o{i}']=d.objs.map(lambda z:int(x in z))
for i,x in enumerate(evs):d[f'e{i}']=d.evs.map(lambda z:int(x in z))
ov=[f'o{i}' for i in range(len(objects))];ev=[f'e{i}' for i in range(len(evs))];rhs=' + '.join(ev+ov+['C(source_type)','is_reply','is_note_author','log_text_length'])
reg=[]
for y in ['judgment_clear','negative','positive']:
 dd=d if y=='judgment_clear' else d[d.judgment_clear==1]
 t,diag=fit_bin(dd,f'{y} ~ {rhs}');fdr(t).to_csv(OUT/f'05_models/rq1/{y}_cluster_glm.csv',index=False);reg.append({'model_id':'rq1_'+y,'rq':'RQ1','outcome':y,**diag})
for oi,obj in enumerate(['专业','职业','行业']):
 idx=objects.index(obj);dd=d[(d[f'o{idx}']==1)&(d.judgment_clear==1)]
 t,diag=fit_bin(dd,'negative ~ '+' + '.join(ev+['C(source_type)','is_reply','log_text_length']));fdr(t).to_csv(OUT/f'05_models/rq1/negative_within_{obj}.csv',index=False);reg.append({'model_id':'rq1_negative_'+obj,'rq':'RQ1','outcome':'negative','subset':obj,**diag})
# v2 adjudicated sensitivity rates
parts=[pd.read_csv(ROOT/'01_reliability/round2/round2_adjudication_final.csv'),pd.read_csv(ROOT/'01_reliability/round3/round3_final_adjudicated_labels.csv')];v2=pd.concat(parts,ignore_index=True);rows=[]
for e in evs:
 props=[]
 for r in v2.itertuples(index=False):
  if e in jl(r.final_evidence_basis):
   props.extend(jl(r.final_evaluations))
 rows.append({'evidence':e,'n_propositions':len(props),'negative_rate':np.mean([x.get('stance') in ['否定','条件性否定'] for x in props]) if props else np.nan,'positive_rate':np.mean([x.get('stance') in ['认可','条件性认可'] for x in props]) if props else np.nan})
pd.DataFrame(rows).to_csv(OUT/'05_models/rq1/v2_adjudicated_sample_sensitivity.csv',index=False)
# RQ4
freeze=pd.Timestamp('2026-08-03',tz='Asia/Shanghai').timestamp()*1000;d['exposure_days']=((freeze-d.create_ts_ms.fillna(freeze))/(86400*1000)).clip(lower=1);d['log_exposure']=np.log(d.exposure_days);d['author_reply']=((d.is_reply==1)&(d.is_note_author==1)).astype(int)
rhs4=' + '.join(ev+ov+['C(source_type)','is_reply','is_note_author','is_reply:is_note_author','log_text_length'])
t,diag=fit_bin(d,f'any_like ~ {rhs4}');fdr(t).to_csv(OUT/'05_models/rq4/any_like_cluster_glm.csv',index=False);reg.append({'model_id':'rq4_any_like','rq':'RQ4','outcome':'any_like',**diag})
pos=d[d.positive_likes>0];t,diag=fit_nb(pos,f'positive_likes ~ {rhs4}','log_exposure');fdr(t).to_csv(OUT/'05_models/rq4/positive_likes_nb.csv',index=False);reg.append({'model_id':'rq4_positive_likes','rq':'RQ4','outcome':'positive_likes',**diag})
# reply visibility root comments
con=sqlite3.connect(DB);children=pd.read_sql_query('select root_comment_id,count(*) n from comments_anon where comment_level>=2 group by root_comment_id',con);con.close();cm=dict(zip(children.root_comment_id,children.n));root=d[d.comment_level==1].copy();root['reply_count']=root.comment_id.map(cm).fillna(0);root['has_reply']=(root.reply_count>0).astype(int);rhsr=' + '.join(ev+ov+['C(source_type)','is_note_author','log_text_length'])
t,diag=fit_bin(root,f'has_reply ~ {rhsr}');fdr(t).to_csv(OUT/'05_models/rq4/root_has_reply_cluster_glm.csv',index=False);reg.append({'model_id':'rq4_root_has_reply','rq':'RQ4','outcome':'has_reply',**diag})
rp=root[root.reply_count>0];t,diag=fit_nb(rp,f'reply_count ~ {rhsr}');fdr(t).to_csv(OUT/'05_models/rq4/root_reply_count_nb.csv',index=False);reg.append({'model_id':'rq4_root_reply_count','rq':'RQ4','outcome':'reply_count',**diag})
# robustness key evidence by source/exclusion
cnt=d.groupby('note_id').size().sort_values(ascending=False);top1=set(cnt.head(max(1,int(np.ceil(len(cnt)*.01)))).index);top5=set(cnt.head(max(1,int(np.ceil(len(cnt)*.05)))).index);scopes={'all':d,'api':d[d.source_type=='api'],'browser':d[d.source_type=='browser'],'exclude_top1':d[~d.note_id.isin(top1)],'exclude_top5':d[~d.note_id.isin(top5)]};rob=[]
for sc,dd in scopes.items():
 dd=dd[dd.judgment_clear==1]
 for name in ['公共价值','招聘收入与市场结果','身份情感','反讽或梗']:
  x=f'e{evs.index(name)}'
  if dd[x].nunique()<2:continue
  t,di=fit_bin(dd,f'negative ~ {x} + C(source_type) + is_reply + log_text_length');rr=t[t.term==x]
  if len(rr):rob.append({'analysis':'RQ1_negative','scope':sc,'feature':name,'or':rr.odds_ratio.iloc[0],'ci_low':rr.ci_low.iloc[0],'ci_high':rr.ci_high.iloc[0],'p':rr.p_value.iloc[0],'n':len(dd)})
pd.DataFrame(rob).to_csv(OUT/'07_robustness/rq1_source_outlier_robustness.csv',index=False)
pd.DataFrame(reg).to_csv(OUT/'05_models/model_registry_rq1_rq4.csv',index=False)
print(json.dumps({'models':len(reg),'n':len(d),'root_n':len(root),'positive_like_n':len(pos)},ensure_ascii=False))
