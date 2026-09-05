import os
import pandas as pd, json, re, numpy as np
from pathlib import Path
OUT=Path(os.environ['NEWS_EDU_ROOT']) / '02_relational_coding' / 'stage2b_review_high_tranche01'
R1=pd.read_csv(OUT/'high_tranche01_500_agent_adjudicated_round1.csv',low_memory=False)
B=pd.read_csv(OUT/'high_tranche01_500_secondpass_blind_input.csv',low_memory=False)
C=pd.read_csv(OUT/'high_tranche01_500_secondpass_comparison.csv',low_memory=False)
S2=pd.read_csv(OUT/'high_tranche01_500_secondpass_blind_results.csv',low_memory=False)

ORDER=['同层直接回应','个人经验回应','市场结果回应','具体能力解释','公共价值回应','身份否定','反讽与情绪','话题转移','其他']
OBJ=r'专业|学科|课程|教育|大学|新传|新闻|传播|记者|媒体|就业|职业|行业|岗位|能力|AI|人工智能'
MARKET=r'工资|薪资|收入|待遇|岗位|招聘|就业|找工作|失业|编制|合同工|门槛|录取|分数|上岸|考公|公考|人才引进|市场|供需|机会|高薪|低薪|加班|晋升|职称'
PERSONAL=r'我|本人|我同学|我朋友|我室友|身边|同事|学长|学姐|朋友|家人'
EXP=r'实习|工作|入职|毕业|考研|跨考|求职|面试|上岸|失业|转行|辞职|读研|本科|课程作业|采访|写稿|发稿|做账号|从业|干了|学了'
ABILITY=r'能力|训练|培养|写作|表达|沟通|采访|信源|核查|查证|证据|判断|分析|叙事|策划|运营|剪辑|拍摄|数据|研究|伦理|责任|议题|解释|专业知识|方法|技能|素养|把关|辨别|信息处理|逻辑|思维'
COURSE=r'课程|课堂|教学|老师|作业|实训|培养方案|专业课|通识课|教材|上课|训练'
PUBLIC=r'公共利益|公共价值|社会责任|新闻伦理|知情权|舆论监督|公信力|公平正义|事实真相|真相|弱者|社会参与|公共讨论|责任|正义|监督|发声|权利|义务'
IDENTITY=r'你.*没(学过|读过|做过|干过)|你不懂|外行|没资格|不配|先考上再说|什么学历|哪个学校|哪所学校|你是新闻|你从业吗|不是记者|不是新传|没上过大学|你这种人'
EMO=r'笑死|呵呵|哈哈|无语|离谱|荒谬|可笑|讽刺|嘲讽|破防|绷不住|救命|哭|泪|气死|恶心|烦死|绝望|后悔|劝退|千万别|已死|完蛋|牛马|大冤种|[？?]{2,}|[！!]{2,}|[哈啊呜]{3,}|偷笑R|笑哭R|哭惹R|泪崩R|扶额R'
SHIFT=r'加微信|私信|链接|网盘|资料|pdf|PDF|店名|购买|多少钱|设备|电脑|相机|宿舍|社团|穿搭|快递|明星|球赛|游戏|抽奖|广告|带货|参考书|电子版|踢我|dd'
DIRECT=r'^(对|是的|没错|确实|同意|赞同|就是|可不是|对啊|是啊|嗯嗯|真的|完全同意|我也觉得)|不对|不是|并不是|未必|哪有|怎么可能|我不同意|恰恰相反|反而|但是|可是|然而|建议|可以|不建议|别|当然|其实|我觉得|你说'
CAUSAL=r'因为|所以|因此|从而|才能|能够|可以|让|使|意味着|有助于|导致|决定|转化为|用来|服务于|帮助|支撑|影响|对应|适合|需要|依靠|通过'
PURE=re.compile(r'^\s*(谢谢|感谢|求|蹲|码住|收藏|收到|好的|好哒|嗯嗯|哈哈+|啊+|呜+|冲|加油|厉害|棒|赞|支持|同问|dd|踢我|[\[\]A-Za-z0-9_@#话题R哭惹笑哭偷笑捂脸泪崩大笑扶额爱心红薯表情\s]+)[！!。,.，？?~～]*$')

def parse(x):
    try:return json.loads(str(x))
    except:return []

def arbitrate(row):
    rep=str(row.reply_text or ''); src=str(row.direct_source_text or ''); title=str(row.title or '')
    r1=row.round1_relevance; r2=row.secondpass_relevance
    s1=parse(row.round1_strategy_json); s2=parse(row.secondpass_strategy_json)
    t1=row.round1_translation; t2=row.secondpass_translation
    pure=bool(PURE.match(rep)) or (len(rep.strip())<=4 and not re.search(OBJ,rep,re.I))
    obj=bool(re.search(OBJ,rep,re.I)); ctx=bool(re.search(OBJ,src+' '+title,re.I))
    market=bool(re.search(MARKET,rep)); personal=bool(re.search(PERSONAL,rep)) and bool(re.search(EXP,rep)); ability=bool(re.search(ABILITY,rep)); course=bool(re.search(COURSE,rep)); public=bool(re.search(PUBLIC,rep)); identity=bool(re.search(IDENTITY,rep)); emo=bool(re.search(EMO,rep)); shift=bool(re.search(SHIFT,rep,re.I)); direct=bool(re.search(DIRECT,rep)) or ('?' in src or '？' in src)
    reasons=[]
    # relevance: explicit codebook hierarchy
    if pure: rel='弱相关'; reasons.append('最终相关性：纯致谢／求取／极短附和')
    elif obj: rel='核心相关'; reasons.append('最终相关性：回复自身明确出现研究对象')
    elif shift: rel='弱相关'; reasons.append('最终相关性：资料／设备／生活旁支')
    elif ctx and (direct or len(rep)<=20): rel='语境相关'; reasons.append('最终相关性：依赖唯一上文恢复对象')
    elif ctx and any([market,personal,ability,public]): rel='核心相关'; reasons.append('最终相关性：虽省略对象但形成实质评价')
    elif not ctx: rel='无关'; reasons.append('最终相关性：笔记与关系文本均无研究关联')
    else: rel=r1 if r1==r2 else '语境相关'; reasons.append('最终相关性：边界文本保守归入语境相关')
    # strategies: include categories with direct lexical evidence, plus agreed category
    cand=[]
    agreed=[x for x in s1 if x in s2]
    if direct and ('同层直接回应' in s1 or '同层直接回应' in s2): cand.append('同层直接回应')
    if personal and ('个人经验回应' in s1+s2): cand.append('个人经验回应')
    if market and ('市场结果回应' in s1+s2): cand.append('市场结果回应')
    if ability and (len(rep)>=35 or re.search(CAUSAL,rep)) and ('具体能力解释' in s1+s2): cand.append('具体能力解释')
    if public and ('公共价值回应' in s1+s2): cand.append('公共价值回应')
    if identity and ('身份否定' in s1+s2): cand.append('身份否定')
    if emo and ('反讽与情绪' in s1+s2): cand.append('反讽与情绪')
    if shift and ('话题转移' in s1+s2): cand.append('话题转移')
    for x in agreed:
        if x not in cand: cand.append(x)
    if not cand:
        # Prefer the more specific non-direct category; otherwise round1 primary
        specific=[x for x in s1+s2 if x not in ['同层直接回应','其他']]
        cand=specific[:1] if specific else (s1[:1] or s2[:1])
    # If direct coexists with substantive strategy, keep both; otherwise maximum 2 by codebook order.
    cand=sorted(dict.fromkeys(cand),key=lambda x:ORDER.index(x))[:2]
    if rel in ['弱相关','无关'] and '具体能力解释' in cand and not ability:
        cand.remove('具体能力解释')
    # translation: explicit connection only
    causal=bool(re.search(CAUSAL,rep)); trans='无转译'
    if course and ability and causal: trans='课程—能力连接'
    if ability and market and causal: trans='能力—职业连接'
    if ability and public and causal: trans='能力—公共价值连接'
    if trans=='无转译' and public: trans='仅抽象价值表达'
    if rel in ['弱相关','无关']: trans='无转译'
    reasons.append('最终策略：仅保留具有文本证据或两遍一致的类别')
    reasons.append('最终转译：要求跨层元素与明确连接机制同时出现')
    changed=(rel!=r1) or (set(cand)!=set(s1)) or (trans!=t1)
    return rel,json.dumps(cand,ensure_ascii=False),trans,changed,'；'.join(reasons)

# Join source, comparison
x=B.merge(C,on='reply_comment_id',validate='one_to_one')
final=[]
for _,r in x.iterrows():
    rel,st,tr,ch,reason=arbitrate(r)
    final.append({'reply_comment_id':r.reply_comment_id,'final_relevance_status':rel,'final_reply_strategy_json':st,'final_translation_link':tr,'changed_from_round1':ch,'final_adjudication_reason':reason,'secondpass_review_status':'double_pass_adjudicated'})
f=pd.DataFrame(final)
f.to_csv(OUT/'high_tranche01_500_secondpass_final_adjudication.csv',index=False,encoding='utf-8-sig')
# apply only to rereviewed subset, leave other round1 unchanged
allf=R1.copy()
allf=allf.merge(f,on='reply_comment_id',how='left',validate='one_to_one')
mask=allf.final_relevance_status.notna()
allf.loc[mask,'adjudicated_relevance_status']=allf.loc[mask,'final_relevance_status']
allf.loc[mask,'adjudicated_reply_strategy_json']=allf.loc[mask,'final_reply_strategy_json']
allf.loc[mask,'adjudicated_translation_link']=allf.loc[mask,'final_translation_link']
allf['review_depth']=np.where(mask,'double_pass_critical_or_sample','single_pass_noncritical')
allf['review_status']=np.where(mask,'agent_adjudicated_final_double_pass','agent_adjudicated_final_single_pass')
allf['final_change_reason']=allf['final_adjudication_reason'].fillna('第一遍非关键关系，未进入盲态复核抽样；保留第一遍裁决')
allf.drop(columns=['final_relevance_status','final_reply_strategy_json','final_translation_link','changed_from_round1','final_adjudication_reason','secondpass_review_status'],inplace=True)
allf.to_csv(OUT/'high_tranche01_500_agent_final.csv',index=False,encoding='utf-8-sig')
# change log
chg=f[f.changed_from_round1].merge(C[['reply_comment_id','round1_relevance','round1_strategy_json','round1_translation','secondpass_relevance','secondpass_strategy_json','secondpass_translation']],on='reply_comment_id')
chg.to_csv(OUT/'high_tranche01_500_doublepass_change_log.csv',index=False,encoding='utf-8-sig')
# summary
summary={'high_tranche01_n':len(allf),'double_pass_n':int(mask.sum()),'single_pass_n':int((~mask).sum()),'changed_after_doublepass_n':int(f.changed_from_round1.sum()),'relevance':allf.adjudicated_relevance_status.value_counts().to_dict(),'translation':allf.adjudicated_translation_link.value_counts().to_dict(),'strategy_counts':{},'review_status':allf.review_status.value_counts().to_dict()}
for z in allf.adjudicated_reply_strategy_json:
    for s in parse(z): summary['strategy_counts'][s]=summary['strategy_counts'].get(s,0)+1
with open(OUT/'high_tranche01_500_final_summary.json','w',encoding='utf-8') as fh:json.dump(summary,fh,ensure_ascii=False,indent=2)
print(json.dumps(summary,ensure_ascii=False,indent=2))
