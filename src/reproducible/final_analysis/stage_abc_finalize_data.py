import os
from pathlib import Path
import pandas as pd, numpy as np, sqlite3, json, ast, re, shutil, hashlib
ROOT=Path(os.environ['NEWS_EDU_ROOT']); OUT=ROOT/'final_analysis'; DB=ROOT/'00_freeze/analysis_v2_frozen.db'
for d in ['02_relational_final','03_ability_pairing','04_sampling_lineage','10_final_database']: (OUT/d).mkdir(parents=True,exist_ok=True)
def jload(x):
    if x is None or (isinstance(x,float) and np.isnan(x)): return []
    if isinstance(x,list): return x
    try: return json.loads(str(x))
    except:
        try: return ast.literal_eval(str(x))
        except: return []
def jd(x): return json.dumps(x,ensure_ascii=False)
def wr(path,obj):
    with open(path,'w',encoding='utf8') as f: json.dump(obj,f,ensure_ascii=False,indent=2,default=str)
# DB
con=sqlite3.connect(DB)
comments=pd.read_sql_query('''select c.*,n.title note_title,n.description note_description,n.source_type note_source_type,n.publish_ts_ms,l.evaluation_object,l.evidence_basis,l.stance,l.ability_type,l.ability_property from comments_anon c left join notes_anon n using(note_id) left join comment_labels l using(comment_id)''',con)
notes=pd.read_sql_query('select * from v_note_base',con); queries=pd.read_sql_query('select * from search_queries',con); hits=pd.read_sql_query('select * from v_search_lineage',con); con.close()
sem=comments[(comments.semantic_eligible==1)&(comments.is_orphan==0)].copy()
# RELATION FINAL: calibrated fusion from existing candidates + reviewed labels
p=ROOT/'02_relational_coding/stage2b_review_progress/relation_full_review_progress_stage2b2_v2.1.csv.gz'; rel=pd.read_csv(p)
strategies=['同层直接回应','具体能力解释','个人经验回应','市场结果回应','公共价值回应','身份否定','反讽与情绪','话题转移','其他']
rows=[]
for r in rel.itertuples(index=False):
    reviewed=getattr(r,'review_status')!='provisional_model_not_for_inference' and pd.notna(getattr(r,'final_relevance_status'))
    if reviewed:
        rev=getattr(r,'final_relevance_status'); st=jload(getattr(r,'final_reply_strategy')); tr=getattr(r,'final_translation_link')
        status=getattr(r,'review_status'); method='agent_adjudicated'; conf=getattr(r,'stage2b2_agent_confidence',np.nan)
        if pd.isna(conf): conf=.90
    else:
        model_rel=str(getattr(r,'model_relevance')) if pd.notna(getattr(r,'model_relevance')) else '弱相关'
        rel_margin=float(getattr(r,'model_relevance_margin')) if pd.notna(getattr(r,'model_relevance_margin')) else 0
        model_st=jload(getattr(r,'model_strategy_json')); seed_st=jload(getattr(r,'reply_strategy_seed_json'))
        st_margin=float(getattr(r,'model_strategy_margin')) if pd.notna(getattr(r,'model_strategy_margin')) else 0
        model_tr=str(getattr(r,'model_translation')) if pd.notna(getattr(r,'model_translation')) else '无转译'
        seed_tr=str(getattr(r,'translation_seed')) if pd.notna(getattr(r,'translation_seed')) else '无转译'
        tr_margin=float(getattr(r,'model_translation_margin')) if pd.notna(getattr(r,'model_translation_margin')) else 0
        # relevance: model, downgraded when very uncertain short/noisy
        rev=model_rel
        if rel_margin<.10 and (getattr(r,'reply_short_text',0)==1 or getattr(r,'reply_nonverbal_only',0)==1): rev='弱相关'
        # strategy: exact agreement preferred; otherwise explicit seed + model intersection/strong candidate
        inter=[x for x in model_st if x in seed_st]
        if set(model_st)==set(seed_st) and model_st: st=model_st
        elif inter: st=inter[:2]
        elif st_margin>=.35 and model_st: st=model_st[:2]
        elif seed_st: st=seed_st[:2]
        else: st=['其他']
        # translation is conservative; positive only with seed/model agreement or strong margin
        if model_tr==seed_tr: tr=model_tr
        elif model_tr!='无转译' and tr_margin>=.80: tr=model_tr
        else: tr='无转译'
        agree=(set(model_st)==set(seed_st)) and model_tr==seed_tr
        high=(rel_margin>=.25 and (agree or st_margin>=.35) and (tr=='无转译' or tr_margin>=.80))
        status='hybrid_calibrated_high_confidence' if high else 'hybrid_calibrated_low_confidence'
        method='calibrated_existing_model_rule_fusion'
        conf=float(np.clip(.45+.12*min(rel_margin,1)+.12*min(st_margin,1)+.10*min(tr_margin,1)+(.10 if agree else 0),0,0.95))
    rows.append({'reply_comment_id':r.reply_comment_id,'note_id':r.note_id,'final_relevance_status':rev,'final_reply_strategy':jd(st),'final_translation_link':tr,'final_label_confidence':conf,'final_label_method':method,'final_review_status':status,'reply_is_note_author':r.reply_is_note_author,'reply_source_type':r.reply_source_type,'relation_type':r.relation_type})
relfin=pd.DataFrame(rows); relfin.to_csv(OUT/'02_relational_final/reply_strategy_translation_labels_v2.1_terminal.csv.gz',index=False,compression='gzip')
relfull=rel.merge(relfin,on=['reply_comment_id','note_id'],how='left',suffixes=('','_terminal')); relfull.to_csv(OUT/'02_relational_final/relation_full_terminal_frame.csv.gz',index=False,compression='gzip')
wr(OUT/'02_relational_final/relation_terminal_summary.json',{'n':len(relfin),'status':relfin.final_review_status.value_counts().to_dict(),'relevance':relfin.final_relevance_status.value_counts().to_dict(),'translation':relfin.final_translation_link.value_counts().to_dict()})
# Ability conservative explicit pairing
ability_p={
'基础内容生产':[r'写稿',r'写作',r'文笔',r'拍摄',r'摄影',r'剪辑',r'编辑',r'排版',r'做视频',r'视频制作',r'文案',r'制图',r'海报'],
'信息搜集整理':[r'搜集',r'搜索',r'检索',r'资料收集',r'整理信息',r'信息整理',r'归纳',r'信息管理'],
'采访与信源关系':[r'采访',r'提问',r'信源',r'访谈'],
'事实核查与证据':[r'核实',r'核查',r'查证',r'求证',r'溯源',r'调查',r'证据',r'验证真伪',r'辨别真伪'],
'社会解释与议题发现':[r'发现问题',r'议题',r'选题',r'社会解释',r'解释社会',r'洞察',r'分析问题',r'框架'],
'伦理责任与公共判断':[r'新闻伦理',r'伦理判断',r'公共利益',r'社会责任',r'责任判断',r'伤害评估',r'知情权',r'监督'],
'平台运营与商业传播':[r'运营',r'投流',r'品牌',r'营销',r'增长',r'账号',r'自媒体',r'商业传播']}
prop_p={
'新闻专业独特':[r'(?:新闻|新传|记者|新闻人).{0,10}(?:核心|独特|特有|必须|基本功|专业训练)',r'(?:只有|唯有).{0,8}(?:新闻|新传|记者)'],
'大学教育通用':[r'任何专业',r'所有大学生',r'谁都(?:会|能)',r'人人都',r'通用能力',r'基本素质',r'上过学都',r'都能学'],
'可跨职业迁移':[r'转行',r'跨行业',r'其他行业',r'别的岗位',r'各行各业',r'迁移',r'多个岗位'],
'可由AI替代':[r'(?:AI|ai|人工智能|ChatGPT|chatgpt|豆包|机器).{0,15}(?:替代|取代|能写|能做|代替)',r'(?:替代|取代).{0,10}(?:记者|编辑|写作|剪辑|运营)'],
'AI辅助强化':[r'(?:AI|ai|人工智能|ChatGPT|chatgpt|豆包).{0,15}(?:辅助|帮忙|提效|工具|协助|增强)'],
'AI强化需求':[r'(?:AI|ai|人工智能).{0,15}(?:更需要|越需要|不可替代|更重要)',r'越.*(?:AI|人工智能).*越.*需要']}
AP={k:[re.compile(x,re.I) for x in v] for k,v in ability_p.items()}; PP={k:[re.compile(x,re.I) for x in v] for k,v in prop_p.items()}
def mentions(text):
    text='' if pd.isna(text) else str(text); out=[]; seen=set()
    for si,s in enumerate(x for x in re.split(r'[。！？!?；;\n]+',text) if x):
        ts=[k for k,ps in AP.items() if any(p.search(s) for p in ps)]
        ps=[k for k,rr in PP.items() if any(p.search(s) for p in rr)]
        for t in ts:
            prop=ps[0] if ps else '未明确'; key=(t,prop)
            if key not in seen: seen.add(key); out.append({'ability_type':t,'ability_property':prop,'evidence_span':s[:120],'sentence_index':si,'coding_method':'explicit_rule_v2.1'})
    return out
ar=[]
for r in sem.itertuples(index=False):
    ms=mentions(r.content_clean); ai=int(bool(re.search(r'AI|ai|人工智能|ChatGPT|chatgpt|AIGC|豆包|算法','' if pd.isna(r.content_clean) else str(r.content_clean))))
    for i,m in enumerate(ms): ar.append({'comment_id':r.comment_id,'note_id':r.note_id,'mention_index':i+1,**m,'ai_context':ai,'comment_level':r.comment_level,'source_type':r.source_type,'stance_v1':r.stance})
ability=pd.DataFrame(ar); ability.to_csv(OUT/'03_ability_pairing/ability_mentions_long_v2.1.csv.gz',index=False,compression='gzip')
# validation 360
parts=[]
for q in [ROOT/'01_reliability/round2/round2_adjudication_final.csv',ROOT/'01_reliability/round3/round3_final_adjudicated_labels.csv']: parts.append(pd.read_csv(q))
v2=pd.concat(parts,ignore_index=True); va=[]
for r in v2.itertuples(index=False):
    gold={(x.get('ability_type'),x.get('ability_property')) for x in jload(getattr(r,'final_ability_mentions')) if isinstance(x,dict)}; pred={(x['ability_type'],x['ability_property']) for x in mentions(r.current_comment)}; inter=len(gold&pred)
    va.append({'comment_id':r.comment_id,'gold_pairs':jd(sorted(gold)),'pred_pairs':jd(sorted(pred)),'precision':inter/len(pred) if pred else int(not gold),'recall':inter/len(gold) if gold else int(not pred),'exact':gold==pred})
vadf=pd.DataFrame(va); vadf.to_csv(OUT/'03_ability_pairing/ability_pairing_audit_v2_sample.csv',index=False)
wr(OUT/'03_ability_pairing/ability_pairing_audit_summary.json',{'n_v2_audit':len(vadf),'pair_precision_macro':vadf.precision.mean(),'pair_recall_macro':vadf.recall.mean(),'exact_match':vadf.exact.mean(),'n_mentions_full':len(ability),'n_comments_with_mentions':ability.comment_id.nunique() if len(ability) else 0})
# Query taxonomy and lineage
def cat(q):
 q=str(q)
 if re.search(r'AI|ChatGPT|AIGC|智能传播|AI写|AI新闻|AI替代',q,re.I): return 'AI与技术变迁'
 if re.search(r'计算传播|数据新闻',q): return '学科与方法'
 if re.search(r'课程|历史课|新闻史|中国新闻史|外国新闻史|理论|概论|采访|写作|编辑|评论|伦理|马克思主义',q): return '课程知识与专业训练'
 if '就业' in q: return '就业与职业结果'
 if re.search(r'记者还有意义|为什么还要学新闻',q): return '公共价值与职业意义'
 if re.search(r'教育|学院',q): return '教育组织与培养体系'
 if re.search(r'有用吗|劝退|已死|值得学吗|魅力|意义',q): return '专业价值争议'
 return '专业名称与总体讨论'
queries['query_category']=queries.query_text.map(cat); queries.to_csv(OUT/'04_sampling_lineage/query_taxonomy.csv',index=False)
h=hits.merge(queries[['query_id','query_category']],on='query_id',how='left'); nids=set(notes.note_id.astype(str)); h['included']=h.note_id.astype(str).isin(nids).astype(int)
lin=h.groupby(['query_id','query_text','query_category','note_id'],as_index=False).agg(min_global_rank=('global_rank','min'),min_page=('result_page','min'),hit_count=('crawl_run_id','count'),included=('included','max'))
lin.to_csv(OUT/'04_sampling_lineage/search_to_note_lineage.csv.gz',index=False,compression='gzip')
flow=pd.DataFrame([['搜索命中行',len(h)],['不同搜索命中笔记',h.note_id.nunique()],['搜索命中且进入分析库',lin.loc[lin.included==1,'note_id'].nunique()],['分析库全部笔记',notes.note_id.nunique()],['有语义有效评论的笔记',sem.note_id.nunique()],['语义有效评论',len(sem)]],columns=['stage','n']); flow.to_csv(OUT/'04_sampling_lineage/inclusion_flow.csv',index=False)
# Terminal DB initial
finaldb=OUT/'10_final_database/analysis_terminal_v2.1.db'; shutil.copy2(DB,finaldb); cc=sqlite3.connect(finaldb); relfin.to_sql('relation_labels_v2_1',cc,if_exists='replace',index=False); ability.to_sql('ability_mentions_v2_1',cc,if_exists='replace',index=False); queries.to_sql('query_taxonomy_v2_1',cc,if_exists='replace',index=False); cc.commit(); cc.close()
print(json.dumps({'relations':len(relfin),'rel_status':relfin.final_review_status.value_counts().to_dict(),'ability_mentions':len(ability),'ability_comments':ability.comment_id.nunique() if len(ability) else 0,'lineage':len(lin)},ensure_ascii=False))
