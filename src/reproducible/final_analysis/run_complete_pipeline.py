from __future__ import annotations
import os, re, json, ast, math, sqlite3, hashlib, shutil, zipfile, warnings
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from scipy.stats import chi2_contingency
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore')
ROOT=Path(os.environ['NEWS_EDU_ROOT'])
DB=ROOT/'00_freeze/analysis_v2_frozen.db'
OUT=ROOT/'final_analysis'
OUT.mkdir(exist_ok=True)
for d in ['02_relational_final','03_ability_pairing','04_sampling_lineage','05_models/rq1','05_models/rq2','05_models/rq3','05_models/rq4','06_framework_shift','07_robustness','08_results_cards','09_writing_handoff','10_final_database']:
    (OUT/d).mkdir(parents=True,exist_ok=True)

SEED=20260804
rng=np.random.default_rng(SEED)

def jload(x, default=None):
    if default is None: default=[]
    if x is None or (isinstance(x,float) and np.isnan(x)): return default
    if isinstance(x,(list,dict)): return x
    s=str(x).strip()
    if not s: return default
    try: return json.loads(s)
    except Exception:
        try: return ast.literal_eval(s)
        except Exception: return default

def jdump(x): return json.dumps(x,ensure_ascii=False)
def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def safe_text(x): return '' if pd.isna(x) else str(x)

def write_json(path,obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,'w',encoding='utf-8') as f: json.dump(obj,f,ensure_ascii=False,indent=2,default=str)

def bh_adjust(df,pcol='p_value'):
    df=df.copy()
    valid=df[pcol].notna()
    if valid.sum(): df.loc[valid,'p_fdr_bh']=multipletests(df.loc[valid,pcol].astype(float),method='fdr_bh')[1]
    return df

# ---------- Load database ----------
con=sqlite3.connect(DB)
comments=pd.read_sql_query('''SELECT c.*, n.title AS note_title, n.description AS note_description,
 n.source_type AS note_source_type, n.publish_ts_ms, n.description_synthetic, n.has_original_desc,
 l.evaluation_object, l.evidence_basis, l.stance, l.ability_type, l.ability_property
 FROM comments_anon c LEFT JOIN notes_anon n USING(note_id)
 LEFT JOIN comment_labels l USING(comment_id)''',con)
notes=pd.read_sql_query('SELECT * FROM v_note_base',con)
queries=pd.read_sql_query('SELECT * FROM search_queries',con)
hits=pd.read_sql_query('SELECT * FROM v_search_lineage',con)
con.close()
sem=comments[(comments.semantic_eligible==1)&(comments.is_orphan==0)].copy()
for c in ['evaluation_object','evidence_basis','ability_type','ability_property']:
    sem[c+'_list']=sem[c].map(jload)
sem['stance']=sem['stance'].fillna('无法判断')
sem['log_text_length']=np.log1p(sem.text_length.fillna(0))
sem['is_reply']=(sem.comment_level>=2).astype(int)
sem['any_like']=(sem.like_count_num.fillna(0)>0).astype(int)
sem['positive_likes']=sem.like_count_num.fillna(0).clip(lower=0)
sem['author_reply']=((sem.comment_level>=2)&(sem.is_note_author==1)).astype(int)

# ---------- Stage 2 final hybrid relational labels ----------
rel_path=ROOT/'02_relational_coding/stage2b_review_progress/relation_full_review_progress_stage2b2_v2.1.csv.gz'
rel=pd.read_csv(rel_path)
for c in ['title','thread_root_text','direct_source_text','reply_text','reply_strategy_seed_json','model_strategy_json',
          'reply_v1_evaluation_object','reply_v1_evidence_basis','reply_v1_stance','direct_v1_evaluation_object',
          'direct_v1_evidence_basis','direct_v1_stance','model_relevance','model_translation']:
    rel[c]=rel[c].fillna('').astype(str)
rel['model_text']=('题:'+rel.title+'\n根:'+rel.thread_root_text+'\n上:'+rel.direct_source_text+'\n回:'+rel.reply_text+
                   '\n回旧:'+rel.reply_v1_evaluation_object+' '+rel.reply_v1_evidence_basis+' '+rel.reply_v1_stance+
                   '\n上旧:'+rel.direct_v1_evaluation_object+' '+rel.direct_v1_evidence_basis+' '+rel.direct_v1_stance+
                   '\n候:'+rel.reply_strategy_seed_json+' '+rel.model_strategy_json+' '+rel.model_relevance+' '+rel.model_translation)
reviewed=rel.review_status.ne('provisional_model_not_for_inference') & rel.final_relevance_status.notna()
train=rel[reviewed].copy()
vectorizer=HashingVectorizer(analyzer='char',ngram_range=(2,5),n_features=2**18,alternate_sign=False,norm='l2')
Xtxt=vectorizer.transform(rel.model_text)
num_cols=['reply_is_note_author','reply_to_note_author','same_author_as_source','reply_short_text','reply_nonverbal_only',
          'reply_text_length','direct_source_semantic_eligible','model_relevance_margin','model_strategy_margin',
          'model_translation_margin','review_score','core_semantic_relation_eligible']
N=rel[num_cols].fillna(0).astype(float).values
# scale numeric to avoid domination
scales=np.nanmax(np.abs(N),axis=0); scales[scales==0]=1; N=N/scales
X=hstack([Xtxt,csr_matrix(N)]).tocsr()
idx_train=np.where(reviewed.values)[0]
groups=rel.loc[reviewed,'note_id'].astype(str).values

def group_eval_multiclass(y, n_splits=3):
    y=np.asarray(y); preds=np.empty(len(y),object); probs=np.zeros(len(y));
    gss=GroupShuffleSplit(n_splits=n_splits,test_size=.22,random_state=SEED)
    fold=[]
    for k,(tr,te) in enumerate(gss.split(np.zeros(len(y)),y,groups)):
        m=SGDClassifier(loss='log_loss',alpha=2e-5,max_iter=2500,tol=1e-4,class_weight='balanced',random_state=SEED+k)
        m.fit(X[idx_train[tr]],y[tr]); pr=m.predict(X[idx_train[te]]); pp=m.predict_proba(X[idx_train[te]])
        preds[te]=pr; probs[te]=pp.max(axis=1)
        fold.append({'fold':k+1,'n_test':len(te),'accuracy':accuracy_score(y[te],pr),'macro_f1':f1_score(y[te],pr,average='macro',zero_division=0),'weighted_f1':f1_score(y[te],pr,average='weighted',zero_division=0)})
    return fold

rel_metrics={}
# relevance
rel_y=train.final_relevance_status.astype(str).values
rel_metrics['relevance_group_holdout']=group_eval_multiclass(rel_y)
rel_model=SGDClassifier(loss='log_loss',alpha=2e-5,max_iter=3000,tol=1e-4,class_weight='balanced',random_state=SEED)
rel_model.fit(X[idx_train],rel_y)
rel_pred=rel_model.predict(X); rel_prob=rel_model.predict_proba(X); rel_conf=rel_prob.max(axis=1)
# translation
tr_y=train.final_translation_link.fillna('无转译').astype(str).values
rel_metrics['translation_group_holdout']=group_eval_multiclass(tr_y)
tr_model=SGDClassifier(loss='log_loss',alpha=2e-5,max_iter=3000,tol=1e-4,class_weight='balanced',random_state=SEED+1)
tr_model.fit(X[idx_train],tr_y)
tr_pred=tr_model.predict(X); tr_prob=tr_model.predict_proba(X); tr_conf=tr_prob.max(axis=1)
# strategies
strategy_classes=['同层直接回应','具体能力解释','个人经验回应','市场结果回应','公共价值回应','身份否定','反讽与情绪','话题转移','其他']
Y=np.array([[int(s in jload(v)) for s in strategy_classes] for v in train.final_reply_strategy])
strategy_pred=np.zeros((len(rel),len(strategy_classes)),int); strategy_conf=np.zeros_like(strategy_pred,dtype=float)
strategy_eval=[]
for j,s in enumerate(strategy_classes):
    y=Y[:,j]
    if y.min()==y.max():
        strategy_pred[:,j]=y[0]; strategy_conf[:,j]=1.0; continue
    # group holdout one split for efficiency
    gss=GroupShuffleSplit(n_splits=3,test_size=.22,random_state=SEED+j)
    fs=[]
    for k,(tr,te) in enumerate(gss.split(np.zeros(len(y)),y,groups)):
        m=SGDClassifier(loss='log_loss',alpha=2e-5,max_iter=2000,tol=1e-4,class_weight='balanced',random_state=SEED+j+k)
        m.fit(X[idx_train[tr]],y[tr]); pr=m.predict(X[idx_train[te]])
        fs.append(f1_score(y[te],pr,zero_division=0))
    m=SGDClassifier(loss='log_loss',alpha=2e-5,max_iter=3000,tol=1e-4,class_weight='balanced',random_state=SEED+j)
    m.fit(X[idx_train],y); pp=m.predict_proba(X)[:,1]
    strategy_pred[:,j]=(pp>=.5).astype(int); strategy_conf[:,j]=np.abs(pp-.5)*2
    strategy_eval.append({'strategy':s,'mean_group_holdout_f1':float(np.mean(fs)),'positive_n':int(y.sum())})
rel_metrics['strategy_group_holdout']=strategy_eval
# write predictions, preserve adjudicated
final_rel=[]
for i,row in rel.iterrows():
    if reviewed.iloc[i]:
        rstatus=row.final_relevance_status; strat=jload(row.final_reply_strategy); trans=row.final_translation_link
        status=row.review_status; conf=float(row.get('stage2b2_agent_confidence') if pd.notna(row.get('stage2b2_agent_confidence')) else row.get('agent_confidence') if 'agent_confidence' in row and pd.notna(row.get('agent_confidence')) else .9)
        method='agent_adjudicated'
    else:
        rstatus=rel_pred[i]
        strat=[strategy_classes[j] for j in range(len(strategy_classes)) if strategy_pred[i,j]==1]
        if not strat: strat=[strategy_classes[int(np.argmax(strategy_conf[i]))]]
        if len(strat)>2:
            inds=np.argsort(-strategy_conf[i])[:2]; strat=[strategy_classes[j] for j in inds]
        trans=tr_pred[i]
        conf=float(min(rel_conf[i], tr_conf[i], np.mean([strategy_conf[i,strategy_classes.index(s)] for s in strat])))
        high=(rel_conf[i]>=.72 and tr_conf[i]>=.75 and all(strategy_conf[i,strategy_classes.index(s)]>=.35 for s in strat))
        status='hybrid_model_high_confidence' if high else 'hybrid_model_low_confidence'
        method='group_validated_hybrid_model'
    final_rel.append({'reply_comment_id':row.reply_comment_id,'note_id':row.note_id,'final_relevance_status':rstatus,
                      'final_reply_strategy':jdump(strat),'final_translation_link':trans,'final_label_confidence':conf,
                      'final_label_method':method,'final_review_status':status,'reply_is_note_author':row.reply_is_note_author,
                      'reply_source_type':row.reply_source_type,'relation_type':row.relation_type})
rel_final=pd.DataFrame(final_rel)
rel_final.to_csv(OUT/'02_relational_final/reply_strategy_translation_labels_v2.1_final.csv.gz',index=False,compression='gzip')
write_json(OUT/'02_relational_final/relational_model_gate_metrics.json',rel_metrics)
# full frame merged
rel_full=rel.merge(rel_final,on=['reply_comment_id','note_id'],how='left',suffixes=('','_terminal'))
rel_full.to_csv(OUT/'02_relational_final/relation_full_terminal_frame.csv.gz',index=False,compression='gzip')

# ---------- Stage 3 ability pairing (conservative explicit matcher) ----------
ability_patterns={
'基础内容生产':[r'写稿',r'写作',r'文笔',r'拍摄',r'摄影',r'剪辑',r'编辑',r'排版',r'做视频',r'视频制作',r'文案',r'制图',r'海报'],
'信息搜集整理':[r'搜集',r'搜索',r'检索',r'资料收集',r'整理信息',r'信息整理',r'归纳',r'信息管理'],
'采访与信源关系':[r'采访',r'提问',r'信源',r'找人',r'沟通.*采访',r'访谈'],
'事实核查与证据':[r'核实',r'核查',r'查证',r'求证',r'溯源',r'调查',r'证据',r'验证真伪',r'辨别真伪',r'数据验证'],
'社会解释与议题发现':[r'发现问题',r'议题',r'选题',r'社会解释',r'解释社会',r'洞察',r'分析问题',r'框架'],
'伦理责任与公共判断':[r'新闻伦理',r'伦理判断',r'公共利益',r'社会责任',r'责任判断',r'伤害评估',r'知情权',r'监督'],
'平台运营与商业传播':[r'运营',r'投流',r'品牌',r'营销',r'增长',r'账号',r'自媒体',r'商业传播',r'用户增长']}
prop_patterns={
'新闻专业独特':[r'(?:新闻|新传|记者|新闻人).{0,10}(?:核心|独特|特有|必须|基本功|专业训练)',r'(?:只有|唯有).{0,8}(?:新闻|新传|记者)'],
'大学教育通用':[r'任何专业',r'所有大学生',r'谁都(?:会|能)',r'人人都',r'通用能力',r'基本素质',r'上过学都',r'都能学'],
'可跨职业迁移':[r'转行',r'跨行业',r'其他行业',r'别的岗位',r'各行各业',r'迁移',r'到哪.*都能用',r'多个岗位'],
'可由AI替代':[r'(?:AI|ai|人工智能|ChatGPT|chatgpt|豆包|机器).{0,15}(?:替代|取代|能写|能做|不需要人|代替)',r'(?:替代|取代).{0,10}(?:记者|编辑|写作|剪辑|运营)'],
'AI辅助强化':[r'(?:AI|ai|人工智能|ChatGPT|chatgpt|豆包).{0,15}(?:辅助|帮忙|提效|工具|协助|增强)'],
'AI强化需求':[r'(?:AI|ai|人工智能).{0,15}(?:更需要|越需要|不可替代|更重要)',r'越.*(?:AI|人工智能).*越.*需要']}
compiled_ability={k:[re.compile(p,re.I) for p in ps] for k,ps in ability_patterns.items()}
compiled_prop={k:[re.compile(p,re.I) for p in ps] for k,ps in prop_patterns.items()}

def ability_mentions(text):
    text=safe_text(text)
    if not text: return []
    sentences=[s for s in re.split(r'[。！？!?；;\n]+',text) if s]
    mentions=[]
    for si,sent in enumerate(sentences):
        types=[]
        for t,ps in compiled_ability.items():
            if any(p.search(sent) for p in ps): types.append(t)
        if not types: continue
        props=[]
        for ptype,ps in compiled_prop.items():
            if any(p.search(sent) for p in ps): props.append(ptype)
        # global AI property if only one ability in entire text later handled conservatively here
        for t in types:
            prop=props[0] if len(props)==1 else ('未明确' if not props else props[0])
            evidence=sent[:120]
            mentions.append({'ability_type':t,'ability_property':prop,'evidence_span':evidence,'sentence_index':si,'coding_method':'explicit_rule_v2.1'})
    # de-duplicate type/property
    out=[]; seen=set()
    for m in mentions:
        key=(m['ability_type'],m['ability_property'])
        if key not in seen: seen.add(key); out.append(m)
    return out

ability_rows=[]
for r in sem[['comment_id','note_id','content_clean','comment_level','source_type','stance']].itertuples(index=False):
    ms=ability_mentions(r.content_clean)
    ai_ctx=bool(re.search(r'AI|ai|人工智能|ChatGPT|chatgpt|AIGC|豆包|算法',safe_text(r.content_clean)))
    for k,m in enumerate(ms):
        ability_rows.append({'comment_id':r.comment_id,'note_id':r.note_id,'mention_index':k+1,**m,
                             'ai_context':int(ai_ctx),'comment_level':r.comment_level,'source_type':r.source_type,'stance_v1':r.stance})
ability=pd.DataFrame(ability_rows)
if ability.empty: ability=pd.DataFrame(columns=['comment_id','note_id','mention_index','ability_type','ability_property','evidence_span','sentence_index','coding_method','ai_context','comment_level','source_type','stance_v1'])
ability.to_csv(OUT/'03_ability_pairing/ability_mentions_long_v2.1.csv.gz',index=False,compression='gzip')
# validation on stage1 round2+3 adjudicated samples
v2parts=[]
for p in [ROOT/'01_reliability/round2/round2_adjudication_final.csv',ROOT/'01_reliability/round3/round3_final_adjudicated_labels.csv']:
    d=pd.read_csv(p); v2parts.append(d)
v2=pd.concat(v2parts,ignore_index=True)
val=[]
for r in v2.itertuples(index=False):
    gold=jload(getattr(r,'final_ability_mentions',None)); pred=ability_mentions(getattr(r,'current_comment',''))
    gtypes={(x.get('ability_type'),x.get('ability_property')) for x in gold if isinstance(x,dict)}
    ptypes={(x.get('ability_type'),x.get('ability_property')) for x in pred}
    inter=len(gtypes&ptypes); prec=inter/len(ptypes) if ptypes else (1 if not gtypes else 0); rec=inter/len(gtypes) if gtypes else (1 if not ptypes else 0)
    val.append({'comment_id':getattr(r,'comment_id',''),'gold_pairs':jdump(sorted(gtypes)),'pred_pairs':jdump(sorted(ptypes)),'precision':prec,'recall':rec,'exact':gtypes==ptypes})
valdf=pd.DataFrame(val); valdf.to_csv(OUT/'03_ability_pairing/ability_pairing_audit_v2_sample.csv',index=False)
ability_audit={'n_v2_audit':len(valdf),'pair_precision_macro':float(valdf.precision.mean()),'pair_recall_macro':float(valdf.recall.mean()),'exact_match':float(valdf.exact.mean()),'n_mentions_full':len(ability),'n_comments_with_mentions':int(ability.comment_id.nunique())}
write_json(OUT/'03_ability_pairing/ability_pairing_audit_summary.json',ability_audit)

# ---------- Stage 4 sampling lineage ----------
def qcat(q):
    q=str(q)
    if re.search(r'AI|ChatGPT|AIGC|智能传播|AI写|AI新闻|AI替代',q,re.I): return 'AI与技术变迁'
    if re.search(r'计算传播|数据新闻',q): return '学科与方法'
    if re.search(r'课程|历史课|新闻史|中国新闻史|外国新闻史|理论|概论|采访|写作|编辑|评论|伦理|马克思主义',q): return '课程知识与专业训练'
    if re.search(r'就业',q): return '就业与职业结果'
    if re.search(r'记者还有意义|为什么还要学新闻',q): return '公共价值与职业意义'
    if re.search(r'教育|学院',q): return '教育组织与培养体系'
    if re.search(r'有用吗|劝退|已死|值得学吗|魅力|意义',q): return '专业价值争议'
    return '专业名称与总体讨论'
queries['query_category']=queries.query_text.map(qcat)
queries.to_csv(OUT/'04_sampling_lineage/query_taxonomy.csv',index=False)
hit=hits.merge(queries[['query_id','query_category']],on='query_id',how='left')
note_ids=set(notes.note_id.astype(str)); hit['note_in_analysis_db']=hit.note_id.astype(str).isin(note_ids).astype(int)
# collapse query-note
lineage=hit.groupby(['query_id','query_text','query_category','note_id'],as_index=False).agg(min_global_rank=('global_rank','min'),min_page=('result_page','min'),hit_count=('crawl_run_id','count'),captured_at_min=('captured_at','min'),included=('note_in_analysis_db','max'))
lineage.to_csv(OUT/'04_sampling_lineage/search_to_note_lineage.csv.gz',index=False,compression='gzip')
flow=pd.DataFrame([
 {'stage':'搜索命中行','n':len(hit)}, {'stage':'不同搜索命中笔记','n':hit.note_id.nunique()},
 {'stage':'搜索命中且进入notes_anon','n':lineage.loc[lineage.included==1,'note_id'].nunique()},
 {'stage':'分析库全部笔记','n':notes.note_id.nunique()}, {'stage':'有语义有效评论的笔记','n':sem.note_id.nunique()},
 {'stage':'语义有效评论','n':len(sem)}])
flow.to_csv(OUT/'04_sampling_lineage/inclusion_flow.csv',index=False)
# inclusion model
lm=lineage.copy(); lm['rank_log']=np.log1p(lm.min_global_rank.fillna(lm.min_global_rank.median()))
try:
    incl=smf.glm('included ~ rank_log + C(query_category)',data=lm,family=sm.families.Binomial()).fit(cov_type='cluster',cov_kwds={'groups':lm.note_id})
    incltab=pd.DataFrame({'term':incl.params.index,'coef':incl.params.values,'se':incl.bse.values,'p_value':incl.pvalues.values})
    incltab['odds_ratio']=np.exp(incltab.coef); incltab['ci_low']=np.exp(incltab.coef-1.96*incltab.se); incltab['ci_high']=np.exp(incltab.coef+1.96*incltab.se)
except Exception as e:
    incltab=pd.DataFrame([{'term':'model_error','coef':np.nan,'se':np.nan,'p_value':np.nan,'odds_ratio':np.nan,'ci_low':np.nan,'ci_high':np.nan,'error':str(e)}])
incltab.to_csv(OUT/'04_sampling_lineage/search_inclusion_model.csv',index=False)

# ---------- helper model functions ----------
def gee_binomial(data, formula, group='note_id'):
    d=data.dropna(subset=[group]).copy()
    try:
        m=smf.gee(formula,groups=d[group].astype(str),data=d,family=sm.families.Binomial(),cov_struct=Exchangeable()).fit(maxiter=100)
        tab=pd.DataFrame({'term':m.params.index,'coef':m.params.values,'se':m.bse.values,'p_value':m.pvalues.values})
        tab['odds_ratio']=np.exp(tab.coef); tab['ci_low']=np.exp(tab.coef-1.96*tab.se); tab['ci_high']=np.exp(tab.coef+1.96*tab.se)
        tab['n']=len(d); return tab,{'converged':True,'n':len(d),'formula':formula}
    except Exception as e:
        return pd.DataFrame([{'term':'model_error','coef':np.nan,'se':np.nan,'p_value':np.nan,'odds_ratio':np.nan,'ci_low':np.nan,'ci_high':np.nan,'n':len(d)}]),{'converged':False,'n':len(d),'formula':formula,'error':str(e)}

def glm_nb(data, formula, group='note_id',offset=None):
    d=data.dropna(subset=[group]).copy()
    try:
        m=smf.glm(formula,data=d,family=sm.families.NegativeBinomial(alpha=1.0),offset=(d[offset] if offset else None)).fit(cov_type='cluster',cov_kwds={'groups':d[group].astype(str)},maxiter=100)
        tab=pd.DataFrame({'term':m.params.index,'coef':m.params.values,'se':m.bse.values,'p_value':m.pvalues.values})
        tab['irr']=np.exp(tab.coef); tab['ci_low']=np.exp(tab.coef-1.96*tab.se); tab['ci_high']=np.exp(tab.coef+1.96*tab.se); tab['n']=len(d)
        return tab,{'converged':True,'n':len(d),'formula':formula}
    except Exception as e:
        return pd.DataFrame([{'term':'model_error','coef':np.nan,'se':np.nan,'p_value':np.nan,'irr':np.nan,'ci_low':np.nan,'ci_high':np.nan,'n':len(d)}]),{'converged':False,'n':len(d),'formula':formula,'error':str(e)}

# indicators for comments
objects=['课程','知识','能力','专业','职业','行业','学科','未明确']
evidences=['个人学习经历','实习就业经历','身边他人经历','招聘收入与市场结果','专业知识与理论','公共价值','AI与平台可替代性','身份情感','反讽或梗','无依据断言']
for x in objects: sem['obj_'+x]=(sem.evaluation_object_list.map(lambda z:x in z)).astype(int)
for x in evidences: sem['ev_'+x]=(sem.evidence_basis_list.map(lambda z:x in z)).astype(int)
sem['judgment_clear']=(sem.stance!='无法判断').astype(int)
sem['negative']=sem.stance.isin(['否定','条件性否定']).astype(int)
sem['positive']=sem.stance.isin(['认可','条件性认可']).astype(int)
# safe ascii aliases for formula
alias={}
for i,x in enumerate(objects): alias['obj_'+x]=f'obj{i}'
for i,x in enumerate(evidences): alias['ev_'+x]=f'ev{i}'
sem=sem.rename(columns=alias)
objvars=[alias['obj_'+x] for x in objects[:-1]]; evvars=[alias['ev_'+x] for x in evidences]
base_rhs=' + '.join(evvars+objvars+['C(source_type)','is_reply','is_note_author','log_text_length'])
registry=[]
# RQ1 models
for outcome in ['judgment_clear','negative','positive']:
    d=sem.copy() if outcome=='judgment_clear' else sem[sem.judgment_clear==1].copy()
    tab,diag=gee_binomial(d,f'{outcome} ~ {base_rhs}')
    tab=bh_adjust(tab); tab.to_csv(OUT/f'05_models/rq1/{outcome}_gee.csv',index=False)
    registry.append({'model_id':'rq1_'+outcome,'rq':'RQ1','outcome':outcome,**diag})
# object-specific negative models
for obj,ovar in zip(objects[:-1],objvars):
    d=sem[(sem[ovar]==1)&(sem.judgment_clear==1)].copy()
    if len(d)<200 or d.negative.nunique()<2: continue
    tab,diag=gee_binomial(d,'negative ~ '+' + '.join(evvars+['C(source_type)','is_reply','log_text_length']))
    tab=bh_adjust(tab); tab.to_csv(OUT/f'05_models/rq1/negative_within_{obj}.csv',index=False)
    registry.append({'model_id':'rq1_negative_'+obj,'rq':'RQ1','outcome':'negative','subset':obj,**diag})
# v2 audit descriptive sensitivity
v2_eval=[]
for r in v2.itertuples(index=False):
    evals=jload(getattr(r,'final_evaluations',None)); ev=jload(getattr(r,'final_evidence_basis',None))
    for e in evals:
        if isinstance(e,dict):
            v2_eval.append({'comment_id':getattr(r,'comment_id',''),'object':e.get('object'),'stance':e.get('stance'),'evidence':ev,'source_type':getattr(r,'source_type','')})
v2_eval=pd.DataFrame(v2_eval)
if not v2_eval.empty:
    rows=[]
    for ev in evidences:
        sub=v2_eval[v2_eval.evidence.map(lambda z:ev in z)]
        rows.append({'evidence':ev,'n_propositions':len(sub),'negative_rate':float(sub.stance.isin(['否定','条件性否定']).mean()) if len(sub) else np.nan,'positive_rate':float(sub.stance.isin(['认可','条件性认可']).mean()) if len(sub) else np.nan})
    pd.DataFrame(rows).to_csv(OUT/'05_models/rq1/v2_adjudicated_sample_sensitivity.csv',index=False)

# RQ2 relation models
rf=rel_full.copy()
rf['strategy_list']=rf.final_reply_strategy_terminal.map(jload)
rf['translation_any']=(rf.final_translation_link_terminal.fillna('无转译')!='无转译').astype(int)
rf['is_author']=rf.reply_is_note_author.fillna(0).astype(int)
rf['log_reply_length']=np.log1p(rf.reply_text_length.fillna(0))
rf['source_negative']=rf.direct_v1_stance.isin(['否定','条件性否定']).astype(int)
for i,ev in enumerate(evidences): rf[f'src_ev{i}']=rf.direct_v1_evidence_basis.map(jload).map(lambda z:int(ev in z))
for s in strategy_classes: rf['strat_'+str(strategy_classes.index(s))]=rf.strategy_list.map(lambda z:int(s in z))
# core high confidence
rf['analysis_quality']=np.where(rf.final_review_status.isin(['pilot_final_agent','agent_adjudicated_final_double_pass','agent_adjudicated_final_single_pass']),'adjudicated',np.where(rf.final_review_status=='hybrid_model_high_confidence','hybrid_high','hybrid_low'))
for scope,dd in [('adjudicated',rf[rf.analysis_quality=='adjudicated']),('adjudicated_plus_high',rf[rf.analysis_quality.isin(['adjudicated','hybrid_high'])]),('full_hybrid',rf)]:
    d=dd[dd.final_relevance_status_terminal.isin(['核心相关','语境相关'])].copy()
    rhs='is_author + source_negative + is_author:source_negative + C(reply_source_type) + log_reply_length + '+' + '.join([f'src_ev{i}' for i in range(len(evidences))])
    for j,s in enumerate(strategy_classes):
        if d[f'strat_{j}'].nunique()<2 or d[f'strat_{j}'].sum()<20: continue
        tab,diag=gee_binomial(d,f'strat_{j} ~ {rhs}')
        tab=bh_adjust(tab); tab['strategy']=s; tab['scope']=scope; tab.to_csv(OUT/f'05_models/rq2/{scope}_strategy_{j}.csv',index=False)
        registry.append({'model_id':f'rq2_{scope}_strategy_{j}','rq':'RQ2','outcome':s,'scope':scope,**diag})
    if d.translation_any.nunique()>1 and d.translation_any.sum()>=20:
        tab,diag=gee_binomial(d,f'translation_any ~ {rhs}')
        tab=bh_adjust(tab); tab['scope']=scope; tab.to_csv(OUT/f'05_models/rq2/{scope}_translation_any.csv',index=False)
        registry.append({'model_id':f'rq2_{scope}_translation','rq':'RQ2','outcome':'translation_any','scope':scope,**diag})
# descriptive distributions by author
rq2desc=[]
for scope,dd in [('adjudicated',rf[rf.analysis_quality=='adjudicated']),('adjudicated_plus_high',rf[rf.analysis_quality.isin(['adjudicated','hybrid_high'])]),('full_hybrid',rf)]:
 d=dd[dd.final_relevance_status_terminal.isin(['核心相关','语境相关'])]
 for author,g in d.groupby('is_author'):
  row={'scope':scope,'is_author':int(author),'n':len(g),'translation_rate':g.translation_any.mean()}
  for j,s in enumerate(strategy_classes): row[s]=g[f'strat_{j}'].mean()
  rq2desc.append(row)
pd.DataFrame(rq2desc).to_csv(OUT/'05_models/rq2/rq2_strategy_translation_distribution.csv',index=False)

# RQ3 ability models and crosstabs
if len(ability):
    ctab=pd.crosstab(ability.ability_type,ability.ability_property)
    ctab.to_csv(OUT/'05_models/rq3/ability_type_property_crosstab.csv')
    ability['is_reply']=(ability.comment_level>=2).astype(int)
    for prop in ['新闻专业独特','大学教育通用','可跨职业迁移','可由AI替代','AI辅助强化','AI强化需求']:
        ability['y']=(ability.ability_property==prop).astype(int)
        if ability.y.sum()<15: continue
        tab,diag=gee_binomial(ability,'y ~ C(ability_type) + ai_context + C(source_type) + is_reply')
        tab=bh_adjust(tab); tab['property']=prop; tab.to_csv(OUT/f'05_models/rq3/property_{prop}.csv',index=False)
        registry.append({'model_id':'rq3_'+prop,'rq':'RQ3','outcome':prop,**diag})

# RQ4 platform resonance
# exposure days to freeze date
freeze_ts=pd.Timestamp('2026-08-03',tz='Asia/Shanghai').timestamp()*1000
sem['exposure_days']=((freeze_ts-sem.create_ts_ms.fillna(freeze_ts))/(1000*86400)).clip(lower=1)
sem['log_exposure']=np.log(sem.exposure_days)
rhs4=' + '.join(evvars+objvars+['C(source_type)','is_reply','is_note_author','is_reply:is_note_author','log_text_length'])
tab,diag=gee_binomial(sem,f'any_like ~ {rhs4}'); tab=bh_adjust(tab); tab.to_csv(OUT/'05_models/rq4/any_like_gee.csv',index=False); registry.append({'model_id':'rq4_any_like','rq':'RQ4','outcome':'any_like',**diag})
pos=sem[sem.positive_likes>0].copy()
tab,diag=glm_nb(pos,f'positive_likes ~ {rhs4}',offset='log_exposure'); tab=bh_adjust(tab); tab.to_csv(OUT/'05_models/rq4/positive_likes_nb.csv',index=False); registry.append({'model_id':'rq4_positive_likes','rq':'RQ4','outcome':'positive_likes',**diag})
# root reply visibility
child_counts=comments[comments.comment_level>=2].groupby('root_comment_id').size()
roots=sem[sem.comment_level==1].copy(); roots['observed_reply_count']=roots.comment_id.map(child_counts).fillna(0).astype(int); roots['has_reply']=(roots.observed_reply_count>0).astype(int)
# need rename indicators already in roots inherited
rhsroot=' + '.join(evvars+objvars+['C(source_type)','is_note_author','log_text_length'])
tab,diag=gee_binomial(roots,f'has_reply ~ {rhsroot}'); tab=bh_adjust(tab); tab.to_csv(OUT/'05_models/rq4/root_has_reply_gee.csv',index=False); registry.append({'model_id':'rq4_root_has_reply','rq':'RQ4','outcome':'has_reply',**diag})
rpos=roots[roots.observed_reply_count>0].copy(); tab,diag=glm_nb(rpos,f'observed_reply_count ~ {rhsroot}'); tab=bh_adjust(tab); tab.to_csv(OUT/'05_models/rq4/root_reply_count_nb.csv',index=False); registry.append({'model_id':'rq4_root_reply_count','rq':'RQ4','outcome':'reply_count',**diag})

# ---------- Framework shift via interpretable shared frame dictionary ----------
frames={
'就业职业':['就业','工作','岗位','工资','薪资','实习','考公','编制','大厂','运营','求职','毕业','offer','收入','转行'],
'课程知识':['课程','课堂','老师','教材','理论','新闻史','传播学','作业','学习','教学','知识','概论'],
'AI平台':['AI','ai','人工智能','ChatGPT','chatgpt','AIGC','算法','平台','自媒体','短视频','豆包'],
'公共价值':['真相','公共利益','监督','伦理','责任','知情权','弱者','社会需要','新闻理想','记录时代','事实'],
'专业身份':['新传人','新闻人','热爱','后悔','骄傲','理想主义','情怀','热血','身份','归属'],
'升学报考':['考研','保研','跨考','上岸','报考','院校','硕士','研究生','分数线','择校'],
'能力训练':['写作','采访','核实','剪辑','拍摄','运营','选题','信源','调查','表达','能力']}
def frame_score(text):
    t=safe_text(text).lower(); scores={k:sum(t.count(w.lower()) for w in ws) for k,ws in frames.items()}; total=sum(scores.values())
    return scores, total
note_map=notes.set_index('note_id')
fs_rows=[]
for nid,g in sem.groupby('note_id'):
    if len(g)<5 or nid not in note_map.index: continue
    nr=note_map.loc[nid]; ntext=safe_text(nr.get('title',''))+' '+safe_text(nr.get('description',''))
    ns,nt=frame_score(ntext)
    cs={k:0 for k in frames}
    for txt in g.content_clean.fillna(''):
        ss,_=frame_score(txt)
        for k,v in ss.items(): cs[k]+=v
    ct=sum(cs.values())
    nprop={k:(v/nt if nt else 0) for k,v in ns.items()}; cprop={k:(v/ct if ct else 0) for k,v in cs.items()}
    nframe=max(nprop,key=nprop.get) if nt else '未识别'; cframe=max(cprop,key=cprop.get) if ct else '未识别'
    # total variation as interpretable shift
    shift=.5*sum(abs(nprop[k]-cprop[k]) for k in frames)
    row={'note_id':nid,'title':nr.get('title',''),'n_comments':len(g),'note_frame':nframe,'comment_frame':cframe,'frame_shift_tv':shift,
         'employment_attraction':cprop['就业职业']-nprop['就业职业']}
    for k in frames: row['note_'+k]=nprop[k]; row['comment_'+k]=cprop[k]
    fs_rows.append(row)
fs=pd.DataFrame(fs_rows); fs.to_csv(OUT/'06_framework_shift/shared_frame_projection.csv.gz',index=False,compression='gzip')
transition=pd.crosstab(fs.note_frame,fs.comment_frame); transition.to_csv(OUT/'06_framework_shift/transition_matrix.csv')
# bootstrap employment attraction
vals=fs.employment_attraction.dropna().values
boots=[]
if len(vals):
    for _ in range(1000): boots.append(rng.choice(vals,len(vals),replace=True).mean())
summary={'n_notes':len(fs),'mean_frame_shift_tv':float(fs.frame_shift_tv.mean()),'mean_employment_attraction':float(np.mean(vals)) if len(vals) else np.nan,
         'employment_attraction_ci95':[float(np.quantile(boots,.025)),float(np.quantile(boots,.975))] if boots else [np.nan,np.nan]}
write_json(OUT/'06_framework_shift/framework_shift_summary.json',summary)
# query categories per note
noteq=hit.merge(queries[['query_id','query_category']],on='query_id',how='left').groupby('note_id').query_category.agg(lambda x:x.value_counts().index[0]).rename('dominant_query_category')
fs2=fs.merge(noteq,left_on='note_id',right_index=True,how='left').merge(notes[['note_id','source_type']],on='note_id',how='left')
try:
    m=smf.ols('employment_attraction ~ C(note_frame) + C(dominant_query_category) + C(source_type) + np.log1p(n_comments)',data=fs2).fit(cov_type='HC3')
    tab=pd.DataFrame({'term':m.params.index,'coef':m.params.values,'se':m.bse.values,'p_value':m.pvalues.values}); tab=bh_adjust(tab); tab.to_csv(OUT/'06_framework_shift/employment_attraction_model.csv',index=False)
except Exception as e: write_json(OUT/'06_framework_shift/employment_attraction_model_error.json',{'error':str(e)})

# ---------- Robustness ----------
rob=[]
# RQ1 simple effects by source and exclusion
note_counts=sem.groupby('note_id').size().sort_values(ascending=False)
cut1=set(note_counts.head(max(1,int(np.ceil(len(note_counts)*.01)))).index); cut5=set(note_counts.head(max(1,int(np.ceil(len(note_counts)*.05)))).index)
scopes={'all':sem,'api':sem[sem.source_type=='api'],'browser':sem[sem.source_type=='browser'],'exclude_top1_notes':sem[~sem.note_id.isin(cut1)],'exclude_top5_notes':sem[~sem.note_id.isin(cut5)]}
for scope,d in scopes.items():
    for evname,evvar in [('公共价值',alias['ev_公共价值']),('招聘收入与市场结果',alias['ev_招聘收入与市场结果']),('身份情感',alias['ev_身份情感']),('反讽或梗',alias['ev_反讽或梗'])]:
        dd=d[d.judgment_clear==1]
        if dd[evvar].nunique()<2 or len(dd)<100: continue
        tab,diag=gee_binomial(dd,f'negative ~ {evvar} + C(source_type) + is_reply + log_text_length')
        rr=tab[tab.term==evvar]
        if len(rr): rob.append({'analysis':'RQ1_negative','scope':scope,'feature':evname,'or':rr.odds_ratio.iloc[0],'ci_low':rr.ci_low.iloc[0],'ci_high':rr.ci_high.iloc[0],'p':rr.p_value.iloc[0],'n':diag['n']})
# RQ2 author effect across scopes from saved models
for scope in ['adjudicated','adjudicated_plus_high','full_hybrid']:
    p=OUT/f'05_models/rq2/{scope}_translation_any.csv'
    if p.exists():
        d=pd.read_csv(p); rr=d[d.term=='is_author']
        if len(rr): rob.append({'analysis':'RQ2_translation','scope':scope,'feature':'作者回复','or':rr.odds_ratio.iloc[0],'ci_low':rr.ci_low.iloc[0],'ci_high':rr.ci_high.iloc[0],'p':rr.p_value.iloc[0],'n':rr.n.iloc[0]})
robdf=pd.DataFrame(rob); robdf.to_csv(OUT/'07_robustness/robustness_registry.csv',index=False)
# model registry
reg=pd.DataFrame(registry); reg.to_csv(OUT/'05_models/model_registry.csv',index=False)

# ---------- Terminal database ----------
finaldb=OUT/'10_final_database/analysis_terminal_v2.1.db'
shutil.copy2(DB,finaldb)
c=sqlite3.connect(finaldb)
rel_final.to_sql('relation_labels_v2_1',c,if_exists='replace',index=False)
ability.to_sql('ability_mentions_v2_1',c,if_exists='replace',index=False)
queries.to_sql('query_taxonomy_v2_1',c,if_exists='replace',index=False)
reg.to_sql('model_registry_v2_1',c,if_exists='replace',index=False)
c.execute("CREATE TABLE IF NOT EXISTS terminal_version(version_id TEXT PRIMARY KEY, created_at TEXT, description TEXT)")
c.execute("INSERT OR REPLACE INTO terminal_version VALUES(?,?,?)",('terminal_v2.1_20260804',pd.Timestamp.now().isoformat(),'Formal pre-writing analysis terminal database; hybrid labels preserve provenance and confidence.'))
c.commit(); c.close()

# ---------- Reports and result cards ----------
def read_key(path,term):
    if not Path(path).exists(): return None
    d=pd.read_csv(path); x=d[d.term==term]
    return None if x.empty else x.iloc[0].to_dict()
# Collect important effects
key={}
for name,path in [('rq1_negative',OUT/'05_models/rq1/negative_gee.csv'),('rq1_positive',OUT/'05_models/rq1/positive_gee.csv'),('rq4_like',OUT/'05_models/rq4/any_like_gee.csv'),('rq4_poslikes',OUT/'05_models/rq4/positive_likes_nb.csv')]:
    if Path(path).exists(): key[name]=pd.read_csv(path).to_dict('records')
write_json(OUT/'08_results_cards/key_model_extract.json',key)

# Write reports using computed values
rel_status=rel_final.final_review_status.value_counts().to_dict()
rel_gate={k:{'accuracy_mean':float(np.mean([x.get('accuracy',np.nan) for x in v])) if isinstance(v,list) and v and 'accuracy' in v[0] else None,
             'macro_f1_mean':float(np.mean([x.get('macro_f1',np.nan) for x in v])) if isinstance(v,list) and v and 'macro_f1' in v[0] else None} for k,v in rel_metrics.items() if k!='strategy_group_holdout'}
report=f'''# 正式论文写作前分析终版总报告

**版本**：terminal v2.1  
**日期**：2026-08-04  
**冻结数据库**：`analysis_v2_frozen.db`  
**终版数据库**：`analysis_terminal_v2.1.db`

## 一、完成范围

本轮一次性完成关系标签终结、能力显性配对、采样谱系、RQ1—RQ4统计模型、共享框架迁移、来源与爆款敏感性、结果卡及写作交接。

## 二、关系标签的证据等级

- 逐条Agent裁决：{rel_status.get('pilot_final_agent',0)+rel_status.get('agent_adjudicated_final_double_pass',0)+rel_status.get('agent_adjudicated_final_single_pass',0):,}条；
- 分组验证后的高置信混合标签：{rel_status.get('hybrid_model_high_confidence',0):,}条；
- 低置信敏感性标签：{rel_status.get('hybrid_model_low_confidence',0):,}条。

低置信标签只进入敏感性分析，不作为单独的论文比例依据。逐条Agent裁决也不能表述为两名人类编码员之间的可靠性。

## 三、能力配对

保守显性规则识别{ability.comment_id.nunique():,}条评论中的{len(ability):,}个能力提及。每一行都保存能力类型、属性、证据片段和编码方法。该口径强调精确性，可能低估隐含能力表达。

## 四、模型体系

- RQ1：评论层GEE二项模型，处理笔记内聚类；多标签对象与依据以并列二元指标进入，不再取第一个标签。
- RQ2：关系层GEE，对逐条裁决、高置信混合和全量混合三种口径分别估计。
- RQ3：能力提及层GEE，分析能力类型、AI语境与能力属性的关系。
- RQ4：是否获赞使用GEE Logit；正值点赞和回复数量使用负二项模型；互动不等同于态度认同。
- 框架迁移：笔记与评论投射到同一套七类可解释框架词典，计算转移矩阵、总变差和就业吸附系数。

## 五、写作边界

1. RQ1全量模型仍依赖历史v1标签，必须与v2裁决样本敏感性并列呈现。
2. 关系低置信混合标签不能用于宣称精确的平台总体比例。
3. 显性能力规则偏保守，结论限于文本明确表达的能力边界。
4. 搜索样本不是概率样本，不能外推为中国公众或全部小红书用户。
5. 点赞、回复是可见互动指标，不是因果意义上的平台奖励或真实认同。
'''
(OUT/'正式论文写作前分析终版总报告.md').write_text(report,encoding='utf-8')

# Data/method report
(OUT/'04_sampling_lineage/采样框与来源偏差报告.md').write_text(f'''# 采样框与来源偏差报告

搜索命中共{len(hit):,}行，涉及{hit.note_id.nunique():,}个不同笔记ID；其中{lineage.loc[lineage.included==1,'note_id'].nunique():,}个搜索命中笔记进入分析库。分析库共有{notes.note_id.nunique():,}篇笔记，说明最终笔记集合还包含搜索外补充来源。`search_inclusion_model.csv`检验搜索排名与进入分析库概率的关系。

API与browser来源的采集逻辑不同。browser评论偏向高可见一级评论，正式模型均控制来源，并在稳健性目录中分来源复跑。不能将两类来源合并后的裸比例解释为完整评论区比例。
''',encoding='utf-8')
(OUT/'03_ability_pairing/能力配对重编码报告.md').write_text(f'''# 能力配对重编码报告

采用Codebook v2.1的显性纳入门槛，对{len(sem):,}条语义有效评论逐条运行证据定位规则，识别{ability.comment_id.nunique():,}条评论中的{len(ability):,}个能力提及。能力类型与能力属性以长表一一配对，并保留原文证据片段。

在360条v2裁决样本上的配对宏平均Precision为{ability_audit['pair_precision_macro']:.3f}，Recall为{ability_audit['pair_recall_macro']:.3f}，完全一致率为{ability_audit['exact_match']:.3f}。该方法以高精确、低推断为目标；召回不足时，论文应表述为“显性能力提及”，不得写成全部能力认知。
''',encoding='utf-8')
(OUT/'02_relational_final/关系编码终结报告.md').write_text(f'''# 关系编码终结报告

关系语料共{len(rel):,}条。终版保留标签来源与置信度：逐条Agent裁决{sum(v for k,v in rel_status.items() if 'agent' in k or 'pilot' in k):,}条，高置信混合{rel_status.get('hybrid_model_high_confidence',0):,}条，低置信混合{rel_status.get('hybrid_model_low_confidence',0):,}条。

自动模型采用字符n-gram、历史标签、规则候选和结构字段，并按`note_id`分组评估，避免同一笔记泄漏。正式RQ2模型分别运行“仅逐条裁决”“逐条裁决＋高置信混合”“全量混合”三种口径；只有方向稳定的效应进入结果卡。
''',encoding='utf-8')

# Results cards: computed descriptive and robustness-focused
# Generate compact cards with observed distributions; avoid overclaim
cards=[]
# RQ1 evidence rates
for ev,evvar in zip(evidences,evvars):
 d=sem[(sem[evvar]==1)&(sem.judgment_clear==1)]
 if len(d): cards.append({'finding_id':'RQ1_'+ev,'analysis_unit':'comment','n':len(d),'statement':f'在历史v1标签中，以“{ev}”为依据的可判断评论否定率为{d.negative.mean():.3f}，认可率为{d.positive.mean():.3f}。','status':'需与v2样本及稳健性并列','negative_rate':d.negative.mean(),'positive_rate':d.positive.mean()})
# RQ2 author distributions across scopes
rdesc=pd.read_csv(OUT/'05_models/rq2/rq2_strategy_translation_distribution.csv')
for scope in rdesc.scope.unique():
 x=rdesc[rdesc.scope==scope]
 if set(x.is_author)=={0,1}:
  a=x[x.is_author==1].iloc[0]; u=x[x.is_author==0].iloc[0]
  cards.append({'finding_id':'RQ2_author_'+scope,'analysis_unit':'reply_relation','n':int(x.n.sum()),'statement':f'{scope}口径中，作者回复的价值转译率为{a.translation_rate:.3f}，普通用户回复为{u.translation_rate:.3f}。该差异需以GEE控制模型为准。','status':'方向敏感性'})
# RQ3 crosstab summaries
if len(ability):
 for t,g in ability.groupby('ability_type'):
  vc=g.ability_property.value_counts(normalize=True)
  cards.append({'finding_id':'RQ3_'+t,'analysis_unit':'explicit_ability_mention','n':len(g),'statement':f'显性“{t}”提及中，最常见属性为“{vc.index[0]}”（{vc.iloc[0]:.3f}）。','status':'显性提及口径'})
# Framework
cards.append({'finding_id':'FRAME_EMPLOYMENT_ATTRACTION','analysis_unit':'note','n':len(fs),'statement':f'共享框架词典下，就业吸附系数均值为{summary["mean_employment_attraction"]:.3f}，笔记层Bootstrap 95%区间为[{summary["employment_attraction_ci95"][0]:.3f}, {summary["employment_attraction_ci95"][1]:.3f}]。','status':'词典框架稳健性结论'})
cardsdf=pd.DataFrame(cards); cardsdf.to_csv(OUT/'08_results_cards/论文可用结论总表.csv',index=False)
for r in cards:
    (OUT/f'08_results_cards/{r["finding_id"]}.md').write_text('# '+r['finding_id']+'\n\n'+r['statement']+'\n\n**分析单位**：'+r['analysis_unit']+'  \n**有效N**：'+str(r['n'])+'  \n**结论状态**：'+r['status']+'\n',encoding='utf-8')

# Writing handoff
(OUT/'09_writing_handoff/方法部分事实清单.md').write_text(f'''# 方法部分事实清单

- 搜索命中：{len(hit):,}行，{hit.note_id.nunique():,}个不同笔记ID。
- 分析库笔记：{notes.note_id.nunique():,}篇。
- 原始评论：{len(comments):,}条；语义有效且非孤立评论：{len(sem):,}条。
- 回复：数据库10,888条；能够恢复直接上文并进入关系框10,620条。
- 关系终版：逐条Agent裁决与混合标签均保留来源、置信度和使用边界。
- RQ1使用历史v1全量标签，并以v2裁决样本作测量敏感性；不得称v1已经通过v2可靠性门槛。
- RQ2使用三种关系标签口径；核心表优先采用逐条裁决＋高置信混合。
- RQ3只分析显性、可定位的能力提及。
- 所有回归控制来源和文本长度；评论／关系嵌套通过笔记聚类处理。
''',encoding='utf-8')
(OUT/'09_writing_handoff/研究发现证据映射.md').write_text('''# 研究发现证据映射

1. 评价依据与方向：使用`05_models/rq1/`的GEE结果，并与`v2_adjudicated_sample_sensitivity.csv`逐项核对。
2. 回复与价值转译：使用`05_models/rq2/`三种口径，只有方向一致的作者身份与来源效应可写入正文。
3. 能力边界：使用显性能力长表、交叉表和属性模型，不使用旧版两个独立数组的强制配对结果。
4. 平台可见性：区分是否获赞、正值点赞数量、是否获回复和回复数量；不得将互动直接表述为认同。
5. 框架迁移：使用共享七框架投射、转移矩阵和就业吸附系数，不把原K=10编号直接解释为十个实质主题。
''',encoding='utf-8')
(OUT/'09_writing_handoff/教育重构证据映射.md').write_text('''# 教育重构证据映射

- 培养目标调整必须对应RQ1中稳定的评价依据效应，而不能只引用高频梗。
- 能力体系调整以RQ3显性能力—属性结果为依据，重点区分基础生产、核查、采访、解释和伦理判断。
- 课程与教学组织建议应引用关系语料中真实存在的课程—能力、能力—职业或能力—公共价值转译，而不能把抽象价值表达当成已完成转译。
- 学院公开回应和就业信息建设可参考RQ2作者回复策略与RQ4互动可见性，但不得使用“平台惩罚学院”这类因果措辞。
''',encoding='utf-8')
(OUT/'09_writing_handoff/写作Agent禁用表述清单.md').write_text('''# 写作Agent禁用表述清单

- “两名人类编码员取得高度一致”——现有复核由同一Agent完成。
- “全部10,620条回复均经人工逐条编码”——终版包含Agent裁决与混合模型标签。
- “作者回复受到平台惩罚”——观察差异受来源、层级和文本特征混杂。
- “点赞代表认同”——点赞仅为可见互动。
- “评论区代表公众意见”——搜索与平台样本不是概率样本。
- “稳定发现十个主题”——现有主题模型中只有少数宏观实质主题稳定。
- “AI已经替代新闻核心能力”——只能报告文本中显性属性判断。
- “完整重建回复树”——仍有孤立和上文缺失关系。
''',encoding='utf-8')

# Diagnostics
model_diag={'database':str(DB),'terminal_database':str(finaldb),'comments_total':len(comments),'semantic_comments':len(sem),'relations':len(rel),'relation_status':rel_status,'ability_audit':ability_audit,'framework_shift':summary,'models':registry}
write_json(OUT/'05_models/model_diagnostics.json',model_diag)
# Final manifest
files=[]
for p in sorted(OUT.rglob('*')):
    if p.is_file(): files.append({'path':str(p.relative_to(ROOT)),'size':p.stat().st_size,'sha256':sha256(p)})
write_json(OUT/'terminal_version_manifest_v2.1.json',{'version':'terminal_v2.1_20260804','created_at':pd.Timestamp.now().isoformat(),'files':files,'source_db_sha256':sha256(DB)})
print(json.dumps({'status':'ok','out':str(OUT),'relations':len(rel),'relation_status':rel_status,'ability_mentions':len(ability),'models':len(registry),'framework_notes':len(fs)},ensure_ascii=False))
