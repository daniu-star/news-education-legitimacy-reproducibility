"""Historical cluster-robust GLM pipeline for RQ2/RQ3.

The output is retained for audit history only. It must not be described as
GEE. The manuscript-facing model policy is documented in
docs/methods/final_model_policy.md.
"""
import os
from pathlib import Path
import pandas as pd,numpy as np,json,ast,warnings
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore')
ROOT=Path(os.environ['NEWS_EDU_ROOT']);OUT=ROOT/'final_analysis'
for d in ['05_models/rq2','05_models/rq3','07_robustness']: (OUT/d).mkdir(parents=True,exist_ok=True)
def jl(x):
 if x is None or (isinstance(x,float) and np.isnan(x)):return []
 try:return json.loads(str(x))
 except:
  try:return ast.literal_eval(str(x))
  except:return []
def fit(d,form):
 try:
  m=smf.glm(form,data=d,family=sm.families.Binomial()).fit(cov_type='cluster',cov_kwds={'groups':d.note_id.astype(str)},maxiter=100)
  t=pd.DataFrame({'term':m.params.index,'coef':m.params.values,'se':m.bse.values,'p_value':m.pvalues.values});t['odds_ratio']=np.exp(t.coef);t['ci_low']=np.exp(t.coef-1.96*t.se);t['ci_high']=np.exp(t.coef+1.96*t.se);t['n']=len(d);v=t.p_value.notna();t.loc[v,'p_fdr_bh']=multipletests(t.loc[v,'p_value'],method='fdr_bh')[1] if v.sum() else np.nan;return t,{'ok':1,'n':len(d),'formula':form}
 except Exception as e:return pd.DataFrame([{'term':'model_error','error':str(e),'n':len(d)}]),{'ok':0,'n':len(d),'formula':form,'error':str(e)}
rf=pd.read_csv(OUT/'02_relational_final/relation_full_terminal_frame.csv.gz')
rf['strategies']=rf.final_reply_strategy_terminal.map(jl);rf['translation_any']=(rf.final_translation_link_terminal.fillna('无转译')!='无转译').astype(int);rf['is_author']=rf.reply_is_note_author.fillna(0).astype(int);rf['log_reply_length']=np.log1p(rf.reply_text_length.fillna(0));rf['source_negative']=rf.direct_v1_stance.isin(['否定','条件性否定']).astype(int)
evs=['个人学习经历','实习就业经历','身边他人经历','招聘收入与市场结果','专业知识与理论','公共价值','AI与平台可替代性','身份情感','反讽或梗','无依据断言']
for i,e in enumerate(evs):rf[f'e{i}']=rf.direct_v1_evidence_basis.map(jl).map(lambda z:int(e in z))
strategies=['同层直接回应','具体能力解释','个人经验回应','市场结果回应','公共价值回应','身份否定','反讽与情绪','话题转移','其他']
for i,s in enumerate(strategies):rf[f's{i}']=rf.strategies.map(lambda z:int(s in z))
rf['quality']=np.select([rf.final_review_status.isin(['pilot_final_agent','agent_adjudicated_final_double_pass','agent_adjudicated_final_single_pass']),rf.final_review_status.eq('hybrid_calibrated_high_confidence')],['adjudicated','hybrid_high'],'hybrid_low')
scopes={'adjudicated':rf[rf.quality=='adjudicated'],'adjudicated_plus_high':rf[rf.quality.isin(['adjudicated','hybrid_high'])],'full_hybrid':rf}
reg=[];desc=[];rob=[]
rhs='is_author + source_negative + is_author:source_negative + C(reply_source_type) + log_reply_length + '+' + '.join([f'e{i}' for i in range(len(evs))])
for sc,base in scopes.items():
 d=base[base.final_relevance_status_terminal.isin(['核心相关','语境相关'])].copy()
 for a,g in d.groupby('is_author'):
  row={'scope':sc,'is_author':int(a),'n':len(g),'translation_rate':g.translation_any.mean()}
  for i,s in enumerate(strategies):row[s]=g[f's{i}'].mean()
  desc.append(row)
 for i,s in enumerate(strategies):
  if d[f's{i}'].sum()<15 or d[f's{i}'].nunique()<2:continue
  t,di=fit(d,f's{i} ~ {rhs}');t['strategy']=s;t['scope']=sc;t.to_csv(OUT/f'05_models/rq2/{sc}_strategy_{i}.csv',index=False);reg.append({'model_id':f'rq2_{sc}_s{i}','rq':'RQ2','outcome':s,'scope':sc,**di});rr=t[t.term=='is_author']
  if len(rr):rob.append({'analysis':'RQ2_strategy','scope':sc,'feature':s,'or':rr.odds_ratio.iloc[0],'ci_low':rr.ci_low.iloc[0],'ci_high':rr.ci_high.iloc[0],'p':rr.p_value.iloc[0],'n':len(d)})
 if d.translation_any.sum()>=15:
  t,di=fit(d,f'translation_any ~ {rhs}');t['scope']=sc;t.to_csv(OUT/f'05_models/rq2/{sc}_translation_any.csv',index=False);reg.append({'model_id':f'rq2_{sc}_translation','rq':'RQ2','outcome':'translation_any','scope':sc,**di});rr=t[t.term=='is_author']
  if len(rr):rob.append({'analysis':'RQ2_translation','scope':sc,'feature':'作者回复','or':rr.odds_ratio.iloc[0],'ci_low':rr.ci_low.iloc[0],'ci_high':rr.ci_high.iloc[0],'p':rr.p_value.iloc[0],'n':len(d)})
pd.DataFrame(desc).to_csv(OUT/'05_models/rq2/rq2_strategy_translation_distribution.csv',index=False);pd.DataFrame(rob).to_csv(OUT/'07_robustness/rq2_scope_robustness.csv',index=False)
# translation type by author descriptive
pd.crosstab([rf.quality,rf.is_author],rf.final_translation_link_terminal).to_csv(OUT/'05_models/rq2/translation_type_by_quality_author.csv')
# RQ3
ab=pd.read_csv(OUT/'03_ability_pairing/ability_mentions_long_v2.1.csv.gz');pd.crosstab(ab.ability_type,ab.ability_property).to_csv(OUT/'05_models/rq3/ability_type_property_crosstab.csv');ab['is_reply']=(ab.comment_level>=2).astype(int)
for prop,n in ab.ability_property.value_counts().items():
 if prop=='未明确' or n<8:continue
 ab['y']=(ab.ability_property==prop).astype(int);t,di=fit(ab,'y ~ C(ability_type) + ai_context + C(source_type) + is_reply');t['property']=prop;t.to_csv(OUT/f'05_models/rq3/property_{prop}.csv',index=False);reg.append({'model_id':'rq3_'+prop,'rq':'RQ3','outcome':prop,**di})
# ability distributions and explicit AI contrast
rows=[]
for t,g in ab.groupby('ability_type'):
 for p,n in g.ability_property.value_counts().items():rows.append({'ability_type':t,'ability_property':p,'n':n,'share_within_type':n/len(g)})
pd.DataFrame(rows).to_csv(OUT/'05_models/rq3/ability_property_distribution.csv',index=False)
# chi-square overall
ct=pd.crosstab(ab.ability_type,ab.ability_property);from scipy.stats import chi2_contingency
try:
 chi,pv,dof,exp=chi2_contingency(ct);summary={'chi2':chi,'p':pv,'dof':dof,'n':len(ab),'min_expected':float(exp.min())}
except Exception as e:summary={'error':str(e)}
with open(OUT/'05_models/rq3/ability_chi_square.json','w',encoding='utf8') as f:json.dump(summary,f,ensure_ascii=False,indent=2)
pd.DataFrame(reg).to_csv(OUT/'05_models/model_registry_rq2_rq3.csv',index=False)
print(json.dumps({'models':len(reg),'scopes':{k:len(v) for k,v in scopes.items()},'ability_n':len(ab),'translation_positive':int(rf.translation_any.sum())},ensure_ascii=False))
