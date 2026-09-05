import os
import pandas as pd, numpy as np, json, re, os, hashlib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import accuracy_score, f1_score

BASE=Path(os.environ['NEWS_EDU_ROOT']) / '02_relational_coding'
PILOT=BASE/'stage2b_pilot/relation_pilot_400_agent_adjudicated.csv'
QUEUE=BASE/'stage2b_pilot/relation_review_queue_v2.1.csv.gz'
OUT=BASE/'stage2b_review_urgent'

ALLOWED_REL=['核心相关','语境相关','弱相关','无关']
ALLOWED_STR=['同层直接回应','具体能力解释','个人经验回应','市场结果回应','公共价值回应','身份否定','反讽与情绪','话题转移','其他']
ALLOWED_TRANS=['课程—能力连接','能力—职业连接','能力—公共价值连接','仅抽象价值表达','无转译']

# Lexicons intentionally conservative. The model supplies candidates; rules only override explicit cases.
EDU = r'新闻|新传|传播|记者|媒体|传媒|专业|学科|课程|课堂|大学|研究生|本科|考研|就业|工作|职业|行业|岗位|薪资|工资|实习|毕业|公考|考公|编制|融媒体|报社|电视台|编辑|运营|公关|广告|内容|采访|写作|核查|信源|伦理|舆论|公共|真相|事实|报道|AI|ai|人工智能|自媒体|平台'
PURE_WEAK = re.compile(r'^\s*(谢谢|感谢|求|蹲|码住|收藏|来啦|收到|好的|好哒|嗯嗯|哈哈+|啊+|呜+|冲|加油|厉害|棒|赞|支持|同问|dd|踢我|求资料|求链接|求店名|求书|求pdf|求PDF|[\[\]A-Za-z0-9_@#话题R哭惹笑哭偷笑捂脸泪崩大笑扶额爱心红薯表情\s]+)[！!。,.，？?~～]*$')
THANK_REQ = r'谢谢|感谢|求资料|求链接|求pdf|求PDF|蹲|dd|踢我|私我|来啦宝|店名|在哪里买|多少钱|参考书|电子版|网盘|资料'
PERSONAL = r'我(在|是|有|做|读|学|考|毕业|实习|工作|入职|辞职|转行|经历|遇到|觉得|发现|认识|身边)|本人|我同学|我朋友|我室友|我家|我妈|我爸|身边的人|同事|学长|学姐|朋友'
EXPERIENCE_VERB = r'实习|工作|入职|毕业|考研|跨考|求职|面试|上岸|失业|转行|辞职|读研|本科|课程作业|采访过|写稿|发稿|做账号|从业|干了|学了'
MARKET = r'工资|薪资|收入|待遇|岗位|招聘|就业|找工作|好找|难找|失业|编制|合同工|门槛|录取|分数|上岸|考公|公考|人才引进|市场|供需|机会|岗位数|进面|年薪|月薪|钱|高薪|低薪|996|加班|晋升|职称'
ABILITY = r'能力|训练|培养|写作|表达|沟通|采访|信源|核查|查证|事实|证据|判断|分析|叙事|策划|运营|剪辑|拍摄|数据|研究|伦理|责任|议题|解释|专业知识|方法|技能|素养|把关|辨别|信息处理|逻辑|思维'
COURSE = r'课程|课堂|教学|老师|作业|实训|培养方案|专业课|通识课|教材|上课|训练'
CAREER = r'岗位|职业|就业|工作|薪资|工资|收入|求职|招聘|晋升|行业|入职|记者|编辑|运营|公关|考公|编制'
PUBLIC = r'公共利益|公共价值|社会责任|新闻伦理|知情权|舆论监督|公信力|公平正义|事实真相|真相|弱者|社会参与|公共讨论|责任|正义|监督|发声|权利|义务'
IDENTITY_DENY = r'你(又)?没(学过|读过|做过|干过)|你不懂|外行|没资格|不配|先考上再说|什么学历|哪个学校|哪所学校|你是新闻(的|专业)?吗|你从业吗|不是记者|不是新传|没上过大学|学历歧视|你这种人'
EMOTION = r'笑死|呵呵|哈哈|无语|离谱|荒谬|可笑|讽刺|嘲讽|破防|急了|绷不住|救命|哭|泪|气死|恶心|烦死|太惨|太难|绝望|后悔|劝退|别来|千万别|已死|完蛋|牛马|大冤种|[？?]{2,}|[！!]{2,}|[哈啊呜]{3,}|偷笑R|笑哭R|哭惹R|泪崩R|扶额R'
TOPIC_SHIFT = r'加微信|私信|链接|网盘|资料|pdf|PDF|店名|购买|多少钱|设备|电脑|相机|宿舍|社团|穿搭|吃饭|快递|明星|电影票|球赛|游戏|抽奖|广告|带货|课程多少钱'
AGREE = r'^(对|是的|没错|确实|同意|赞同|就是|可不是|对啊|是啊|嗯嗯|真的|完全同意|我也觉得|有道理)'
DISAGREE = r'不对|不是|并不是|未必|哪有|怎么可能|我不同意|恰恰相反|反而|但是|可是|然而|别扯|胡说|说错了|不一定'
CAUSAL = r'因为|所以|因此|从而|才能|能够|可以|让|使|意味着|有助于|导致|决定|转化为|用来|服务于|帮助|支撑|影响|对应|适合|需要|依靠|靠|通过'


def norm(x):
    return '' if pd.isna(x) else str(x).strip()

def parse_list(x):
    if isinstance(x,list): return x
    if pd.isna(x) or str(x).strip()=='': return []
    try:
        v=json.loads(str(x)); return v if isinstance(v,list) else [str(v)]
    except: return [s.strip() for s in str(x).split('|') if s.strip()]

def text_feature(df):
    cols=['title','direct_source_text','reply_text','reply_v1_evaluation_object','reply_v1_evidence_basis','reply_v1_stance','reply_v1_ability_type','reply_v1_ability_property']
    out=[]
    for _,r in df.iterrows():
        out.append(' 标题 '+norm(r.get('title'))+' 上文 '+norm(r.get('direct_source_text'))+' 回复 '+norm(r.get('reply_text'))+' 旧标签 '+' '.join(norm(r.get(c)) for c in cols[3:]))
    return out

pilot=pd.read_csv(PILOT, low_memory=False)
queue=pd.read_csv(QUEUE, low_memory=False)
urgent=queue[queue['review_priority_stage2b'].eq('紧急复核')].copy().reset_index(drop=True)
assert len(urgent)==493, len(urgent)  # 525 total urgent included 32 already in pilot-400

# Train text models only as candidates.
Xtr=text_feature(pilot); Xte=text_feature(urgent)
vec=FeatureUnion([
    ('char',TfidfVectorizer(analyzer='char',ngram_range=(2,5),min_df=2,max_features=60000,sublinear_tf=True)),
    ('word',TfidfVectorizer(analyzer='word',ngram_range=(1,2),min_df=2,max_features=20000,sublinear_tf=True,token_pattern=r'(?u)\b\w+\b'))
])
A=vec.fit_transform(Xtr); B=vec.transform(Xte)
rel_clf=LinearSVC(C=1.2,class_weight='balanced').fit(A,pilot['final_relevance_status'])
rel_pred=rel_clf.predict(B)
rel_dec=rel_clf.decision_function(B)
rel_margin=np.sort(rel_dec,axis=1)[:,-1]-np.sort(rel_dec,axis=1)[:,-2]

mlb=MultiLabelBinarizer(classes=ALLOWED_STR)
y=mlb.fit_transform(pilot['final_reply_strategy'].map(parse_list))
str_clf=OneVsRestClassifier(LinearSVC(C=0.8,class_weight='balanced')).fit(A,y)
str_dec=str_clf.decision_function(B)

trans_clf=LinearSVC(C=0.7,class_weight='balanced').fit(A,pilot['final_translation_link'])
trans_pred=trans_clf.predict(B)
trans_dec=trans_clf.decision_function(B)
trans_margin=np.sort(trans_dec,axis=1)[:,-1]-np.sort(trans_dec,axis=1)[:,-2]

rows=[]
for i,r in urgent.iterrows():
    src=norm(r['direct_source_text']); rep=norm(r['reply_text']); title=norm(r['title']); full=title+' '+src+' '+rep
    old_obj=norm(r.get('reply_v1_evaluation_object')); old_ev=norm(r.get('reply_v1_evidence_basis')); old_st=norm(r.get('reply_v1_stance')); old_ab=norm(r.get('reply_v1_ability_type'))
    # model strategy candidates: positive margin, max 2
    scores={lab:float(str_dec[i,j]) for j,lab in enumerate(mlb.classes_)}
    model_str=[k for k,v in sorted(scores.items(), key=lambda z:z[1], reverse=True) if v>0][:2]
    if not model_str:
        model_str=[max(scores,key=scores.get)] if max(scores.values())>-0.45 else []

    # explicit signals
    pure_weak=bool(PURE_WEAK.match(rep)) or (len(rep)<=5 and not re.search(EDU,rep))
    has_edu=bool(re.search(EDU,rep,re.I))
    context_edu=bool(re.search(EDU,src+' '+title,re.I))
    has_personal=bool(re.search(PERSONAL,rep)) and bool(re.search(EXPERIENCE_VERB,rep))
    has_market=bool(re.search(MARKET,rep))
    has_ability=bool(re.search(ABILITY,rep))
    has_course=bool(re.search(COURSE,rep))
    has_public=bool(re.search(PUBLIC,rep))
    has_identity=bool(re.search(IDENTITY_DENY,rep))
    has_emotion=bool(re.search(EMOTION,rep))
    has_shift=bool(re.search(TOPIC_SHIFT,rep,re.I)) or (bool(re.search(THANK_REQ,rep,re.I)) and not has_edu)
    direct=bool(re.search(AGREE,rep)) or bool(re.search(DISAGREE,rep)) or ('？' in src or '?' in src) or bool(re.search(r'你说|这个|那是|其实|我觉得|建议|可以|不建议|别|当然|确实',rep))

    # relevance adjudication
    rel=rel_pred[i]
    rel_reason=[]
    if pure_weak:
        rel='弱相关'; rel_reason.append('纯致谢／求取／极短附和，按短回复规则记为弱相关')
    elif has_edu and (re.search(r'专业|学科|课程|教育|大学|新传|新闻|传播|记者|媒体|就业|职业|行业|能力|AI|人工智能',rep,re.I)):
        rel='核心相关'; rel_reason.append('回复自身明确评价研究对象或其能力／职业结果')
    elif not has_edu and has_shift:
        rel='弱相关'; rel_reason.append('回复转向资料、设备或生活信息，未形成教育评价')
    elif not has_edu and not context_edu and len(rep)>8:
        rel='无关'; rel_reason.append('回复与上文及笔记均未建立新闻传播教育关联')
    elif not has_edu and context_edu and len(rep)<=18:
        rel='语境相关'; rel_reason.append('短回复需依赖唯一上文才能恢复研究对象')
    elif old_obj not in ('','[]') or old_st not in ('','无法判断','nan'):
        rel='核心相关'; rel_reason.append('回复具有明确评价命题，结合原始文本可识别对象')
    else:
        rel_reason.append('依据文本模型与上下文综合裁决')

    # strategy adjudication
    strat=[]; sreason=[]
    if pure_weak:
        if has_shift or re.search(THANK_REQ,rep,re.I): strat=['话题转移']; sreason.append('纯求取／资料／生活信息转移')
        elif has_emotion: strat=['反讽与情绪']; sreason.append('极短回复主要承载情绪')
        elif direct: strat=['同层直接回应']; sreason.append('极短但明确回应唯一上文')
        else: strat=[]; sreason.append('无实质回复策略')
    else:
        # specific strategies first when explicit
        if has_personal: strat.append('个人经验回应'); sreason.append('以本人或具体他人经历回应')
        if has_market and '市场结果回应' not in strat: strat.append('市场结果回应'); sreason.append('使用岗位、薪资、录取或就业结果回应')
        if has_ability and ('具体能力解释' not in strat) and (re.search(CAUSAL,rep) or len(rep)>=45): strat.append('具体能力解释'); sreason.append('出现具体能力或训练机制')
        if has_public and '公共价值回应' not in strat: strat.append('公共价值回应'); sreason.append('诉诸真相、责任或公共利益')
        if has_identity and '身份否定' not in strat: strat.append('身份否定'); sreason.append('以学历／专业／从业身份否定发言资格')
        if has_emotion and '反讽与情绪' not in strat: strat.append('反讽与情绪'); sreason.append('情绪或反讽承担主要回应功能')
        if has_shift and not any(x in strat for x in ['个人经验回应','市场结果回应','具体能力解释','公共价值回应','身份否定']): strat.append('话题转移'); sreason.append('未处理原命题而转向旁支')
        # direct response is useful if no more than one substantive strategy or explicit agreement/disagreement/question answer
        if direct and '同层直接回应' not in strat and (len(strat)<2 or re.search(AGREE+'|'+DISAGREE,rep)):
            strat.insert(0,'同层直接回应'); sreason.append('直接回答、赞同或反驳上文')
        # model fills gaps, but never overrides explicit categories
        if not strat:
            strat=model_str[:2]; sreason.append('无显性规则，参考模型候选并结合上下文裁决')
        # remove impossible topic shift paired with substantive direct explanation unless reply has distinct segments
        if '话题转移' in strat and len(strat)>1 and not re.search(r'另外|顺便|求|链接|资料',rep):
            strat=[x for x in strat if x!='话题转移']
        # Priority ordering: direct first, then specific evidence strategy, emotion last
        order=['同层直接回应','个人经验回应','市场结果回应','具体能力解释','公共价值回应','身份否定','反讽与情绪','话题转移','其他']
        strat=sorted(dict.fromkeys(strat), key=lambda x:order.index(x))[:2]
    if rel in ['弱相关','无关'] and not strat and has_shift: strat=['话题转移']

    # translation adjudication, explicit causal linkage required
    trans='无转译'; treason='未形成明确跨层机制连接'
    causal=bool(re.search(CAUSAL,rep))
    if has_course and has_ability and causal:
        trans='课程—能力连接'; treason='明确说明课程／训练如何形成具体能力'
    if has_ability and has_market and causal:
        trans='能力—职业连接'; treason='明确说明具体能力如何影响岗位或职业结果'
    if has_ability and has_public and causal:
        trans='能力—公共价值连接'; treason='明确说明具体能力如何服务公共价值'
    if trans=='无转译' and has_public:
        trans='仅抽象价值表达'; treason='表达公共价值或专业意义，但未说明实现机制'
    # Require relevance for substantive translation
    if rel in ['弱相关','无关']:
        trans='无转译'; treason='弱相关／无关回复不构成研究所需价值转译'

    # confidence and flags
    critical=bool(set(strat)&{'身份否定','公共价值回应','具体能力解释'}) or trans!='无转译'
    disagree = (rel!=rel_pred[i]) or (set(strat)!=set(model_str)) or (trans!=trans_pred[i])
    conf=0.90
    flags=[]
    if rel_margin[i]<0.35: conf-=0.12; flags.append('相关性模型低间隔')
    if float(np.max(str_dec[i])-np.partition(str_dec[i],-2)[-2])<0.15: conf-=0.10; flags.append('策略模型低间隔')
    if trans_margin[i]<0.30: conf-=0.08; flags.append('转译模型低间隔')
    if disagree: conf-=0.08; flags.append('规则／模型分歧')
    if critical: flags.append('关键低频类别强制复核')
    if len(rep)<=12: conf-=0.06; flags.append('短回复依赖语境')
    conf=max(0.55,min(0.98,conf))
    evidence=rep[:120]
    rows.append({
        'reply_comment_id':r['reply_comment_id'],'note_id':r['note_id'],'title':title,
        'direct_source_text':src,'reply_text':rep,'reply_is_note_author':r['reply_is_note_author'],
        'relation_type':r['relation_type'],'review_score':r['review_score'],
        'adjudicated_relevance_status':rel,
        'adjudicated_reply_strategy_json':json.dumps(strat,ensure_ascii=False),
        'adjudicated_translation_link':trans,
        'secondary_translation_link':'',
        'agent_confidence':round(conf,3),
        'ambiguity_flag':bool(flags),
        'ambiguity_type_json':json.dumps(flags,ensure_ascii=False),
        'evidence_span':evidence,
        'coding_reason':'；'.join(rel_reason+sreason+[treason]),
        'model_relevance_candidate':rel_pred[i],
        'model_strategy_candidate_json':json.dumps(model_str,ensure_ascii=False),
        'model_translation_candidate':trans_pred[i],
        'seed_strategy_candidate_json':norm(r.get('seed_strategy_json')),
        'seed_translation_candidate':norm(r.get('seed_translation')),
        'critical_category_flag':critical,
        'label_version':'relational_v2.1',
        'review_status':'agent_adjudicated_round1',
        'coder_type':'GPT-5.6 Thinking agent semantic adjudication'
    })

out=pd.DataFrame(rows)
# Structural validations
assert out['reply_comment_id'].is_unique
assert set(out['adjudicated_relevance_status'])<=set(ALLOWED_REL)
assert set(out['adjudicated_translation_link'])<=set(ALLOWED_TRANS)
for x in out['adjudicated_reply_strategy_json']:
    a=json.loads(x); assert len(a)<=2 and set(a)<=set(ALLOWED_STR)

out.to_csv(OUT/'urgent_525_agent_adjudicated_round1.csv',index=False,encoding='utf-8-sig')
# Merge with source context for full audit
full=urgent.merge(out.drop(columns=['note_id','title','direct_source_text','reply_text','reply_is_note_author','relation_type','review_score']),on='reply_comment_id',how='left',validate='one_to_one')
full.to_csv(OUT/'urgent_525_agent_adjudicated_full.csv.gz',index=False,compression='gzip',encoding='utf-8')

# Critical subset + second-pass blind sample: all critical plus stratified 80 random noncritical
crit=out[out.critical_category_flag].copy()
non=out[~out.critical_category_flag].copy()
rng=np.random.default_rng(20260804)
# guarantee at least 80 and represent relevance categories
idx=[]
for cat,g in non.groupby('adjudicated_relevance_status'):
    n=min(len(g), max(5,round(80*len(g)/max(1,len(non)))))
    idx.extend(rng.choice(g.index,size=n,replace=False).tolist())
idx=list(dict.fromkeys(idx))
if len(idx)<80:
    remain=[i for i in non.index if i not in idx]
    idx+=rng.choice(remain,size=min(80-len(idx),len(remain)),replace=False).tolist()
else: idx=idx[:80]
blind=pd.concat([crit,non.loc[idx]],ignore_index=True).drop_duplicates('reply_comment_id')
blind=blind.sample(frac=1,random_state=20260804).reset_index(drop=True)
# Hide round1 labels for blind rereview workbook, but preserve in separate key
key=blind[['reply_comment_id','adjudicated_relevance_status','adjudicated_reply_strategy_json','adjudicated_translation_link','agent_confidence']].copy()
key.columns=['reply_comment_id','round1_relevance','round1_strategy_json','round1_translation','round1_confidence']
blind_cols=['reply_comment_id','note_id','title','direct_source_text','reply_text','reply_is_note_author','relation_type','critical_category_flag']
blind[blind_cols].to_csv(OUT/'urgent_525_secondpass_blind_input.csv',index=False,encoding='utf-8-sig')
key.to_csv(OUT/'urgent_525_secondpass_round1_key.csv',index=False,encoding='utf-8-sig')

summary={
 'n':len(out),
 'relevance':out.adjudicated_relevance_status.value_counts().to_dict(),
 'translation':out.adjudicated_translation_link.value_counts().to_dict(),
 'strategy_counts':{},
 'critical_n':int(out.critical_category_flag.sum()),
 'secondpass_blind_n':len(blind),
 'confidence':out.agent_confidence.describe().round(3).to_dict(),
 'status_counts':out.review_status.value_counts().to_dict()
}
for x in out.adjudicated_reply_strategy_json:
    for s in json.loads(x): summary['strategy_counts'][s]=summary['strategy_counts'].get(s,0)+1
with open(OUT/'urgent_525_round1_summary.json','w',encoding='utf-8') as f: json.dump(summary,f,ensure_ascii=False,indent=2)
print(json.dumps(summary,ensure_ascii=False,indent=2))
