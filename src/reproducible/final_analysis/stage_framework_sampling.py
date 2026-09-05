import os
from pathlib import Path
import pandas as pd,numpy as np,sqlite3,json,re,warnings
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore')
ROOT=Path(os.environ['NEWS_EDU_ROOT']);OUT=ROOT/'final_analysis';DB=ROOT/'00_freeze/analysis_v2_frozen.db';(OUT/'06_framework_shift').mkdir(parents=True,exist_ok=True)
con=sqlite3.connect(DB);comments=pd.read_sql_query("select * from comments_anon where semantic_eligible=1 and is_orphan=0",con);notes=pd.read_sql_query('select * from v_note_base',con);con.close()
lin=pd.read_csv(OUT/'04_sampling_lineage/search_to_note_lineage.csv.gz')
lin['rank_log']=np.log1p(lin.min_global_rank.fillna(lin.min_global_rank.median()))
try:
 m=smf.glm('included ~ rank_log + C(query_category)',data=lin,family=sm.families.Binomial()).fit(cov_type='cluster',cov_kwds={'groups':lin.note_id.astype(str)})
 t=pd.DataFrame({'term':m.params.index,'coef':m.params.values,'se':m.bse.values,'p_value':m.pvalues.values});t['odds_ratio']=np.exp(t.coef);t['ci_low']=np.exp(t.coef-1.96*t.se);t['ci_high']=np.exp(t.coef+1.96*t.se);t.to_csv(OUT/'04_sampling_lineage/search_inclusion_model.csv',index=False)
except Exception as e:
 pd.DataFrame([{'error':str(e)}]).to_csv(OUT/'04_sampling_lineage/search_inclusion_model.csv',index=False)
frames={'就业职业':['就业','工作','岗位','工资','薪资','实习','考公','编制','大厂','运营','求职','毕业','offer','收入','转行'],
'课程知识':['课程','课堂','老师','教材','理论','新闻史','传播学','作业','学习','教学','知识','概论'],
'AI平台':['ai','人工智能','chatgpt','aigc','算法','平台','自媒体','短视频','豆包'],
'公共价值':['真相','公共利益','监督','伦理','责任','知情权','弱者','社会需要','新闻理想','记录时代','事实'],
'专业身份':['新传人','新闻人','热爱','后悔','骄傲','理想主义','情怀','热血','身份','归属'],
'升学报考':['考研','保研','跨考','上岸','报考','院校','硕士','研究生','分数线','择校'],
'能力训练':['写作','采访','核实','剪辑','拍摄','运营','选题','信源','调查','表达','能力']}
def score(x):
 s='' if pd.isna(x) else str(x).lower();d={k:sum(s.count(w) for w in ws) for k,ws in frames.items()};return d,sum(d.values())
nmap=notes.set_index('note_id');rows=[]
for nid,g in comments.groupby('note_id'):
 if len(g)<5 or nid not in nmap.index:continue
 nr=nmap.loc[nid];ns,nt=score(str(nr.get('title',''))+' '+str(nr.get('description','')));cs={k:0 for k in frames}
 for text in g.content_clean.fillna(''):
  z,_=score(text)
  for k,v in z.items():cs[k]+=v
 ct=sum(cs.values());npv={k:(v/nt if nt else 0) for k,v in ns.items()};cpv={k:(v/ct if ct else 0) for k,v in cs.items()};nf=max(npv,key=npv.get) if nt else '未识别';cf=max(cpv,key=cpv.get) if ct else '未识别';row={'note_id':nid,'title':nr.get('title',''),'n_comments':len(g),'note_frame':nf,'comment_frame':cf,'frame_shift_tv':.5*sum(abs(npv[k]-cpv[k]) for k in frames),'employment_attraction':cpv['就业职业']-npv['就业职业']}
 for k in frames:row['note_'+k]=npv[k];row['comment_'+k]=cpv[k]
 rows.append(row)
fs=pd.DataFrame(rows);fs.to_csv(OUT/'06_framework_shift/shared_frame_projection.csv.gz',index=False,compression='gzip');pd.crosstab(fs.note_frame,fs.comment_frame).to_csv(OUT/'06_framework_shift/transition_matrix.csv')
rng=np.random.default_rng(20260804);vals=fs.employment_attraction.values;boots=[rng.choice(vals,len(vals),replace=True).mean() for _ in range(2000)]
sumr={'n_notes':len(fs),'mean_frame_shift_tv':fs.frame_shift_tv.mean(),'median_frame_shift_tv':fs.frame_shift_tv.median(),'mean_employment_attraction':vals.mean(),'employment_attraction_ci95':[np.quantile(boots,.025),np.quantile(boots,.975)],'positive_employment_attraction_share':(vals>0).mean()}
with open(OUT/'06_framework_shift/framework_shift_summary.json','w',encoding='utf8') as f:json.dump(sumr,f,ensure_ascii=False,indent=2)
# dominant query category per note
qdom=lin.groupby('note_id').query_category.agg(lambda x:x.value_counts().index[0]).rename('dominant_query_category');fs2=fs.merge(qdom,left_on='note_id',right_index=True,how='left').merge(notes[['note_id','source_type']],on='note_id',how='left')
try:
 m=smf.ols('employment_attraction ~ C(note_frame) + C(dominant_query_category) + C(source_type) + np.log1p(n_comments)',data=fs2).fit(cov_type='HC3');t=pd.DataFrame({'term':m.params.index,'coef':m.params.values,'se':m.bse.values,'p_value':m.pvalues.values});v=t.p_value.notna();t.loc[v,'p_fdr_bh']=multipletests(t.loc[v,'p_value'],method='fdr_bh')[1];t.to_csv(OUT/'06_framework_shift/employment_attraction_model.csv',index=False)
except Exception as e:pd.DataFrame([{'error':str(e)}]).to_csv(OUT/'06_framework_shift/employment_attraction_model.csv',index=False)
print(json.dumps(sumr,ensure_ascii=False))
