import os
import pandas as pd, json, re, numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer

OUT=Path(os.environ['NEWS_EDU_ROOT']) / '02_relational_coding' / 'stage2b_review_high_tranche01'
BLIND=OUT/'high_tranche01_500_secondpass_blind_input.csv'
KEY=OUT/'high_tranche01_500_secondpass_round1_key.csv'
R1=OUT/'high_tranche01_500_agent_adjudicated_round1.csv'
ALLOWED_STR=['同层直接回应','具体能力解释','个人经验回应','市场结果回应','公共价值回应','身份否定','反讽与情绪','话题转移','其他']

EDU=r'新闻|新传|传播|记者|媒体|传媒|专业|学科|课程|课堂|大学|研究生|本科|考研|就业|职业|行业|岗位|实习|毕业|融媒体|报社|电视台|编辑|运营|公关|广告|采访|核查|信源|伦理|舆论|公共|真相|事实|报道|AI|ai|人工智能|自媒体|平台|能力|教育'
OBJECT=r'专业|学科|课程|教育|大学|新传|新闻|传播|记者|媒体|就业|职业|行业|岗位|能力|AI|人工智能'
MARKET=r'工资|薪资|收入|待遇|岗位|招聘|就业|找工作|失业|编制|合同工|门槛|录取|分数|上岸|考公|公考|人才引进|市场|供需|机会|高薪|低薪|加班|晋升|职称'
PERSONAL=r'我|本人|我同学|我朋友|我室友|身边|同事|学长|学姐|朋友|家人'
EXP=r'实习|工作|入职|毕业|考研|跨考|求职|面试|上岸|失业|转行|辞职|读研|本科|课程作业|采访|写稿|发稿|做账号|从业|干了|学了'
ABILITY=r'能力|训练|培养|写作|表达|沟通|采访|信源|核查|查证|证据|判断|分析|叙事|策划|运营|剪辑|拍摄|数据|研究|伦理|责任|议题|解释|专业知识|方法|技能|素养|把关|辨别|信息处理|逻辑|思维'
COURSE=r'课程|课堂|教学|老师|作业|实训|培养方案|专业课|通识课|教材|上课|训练'
PUBLIC=r'公共利益|公共价值|社会责任|新闻伦理|知情权|舆论监督|公信力|公平正义|事实真相|真相|弱者|社会参与|公共讨论|责任|正义|监督|发声|权利|义务'
IDENTITY=r'你.*没(学过|读过|做过|干过)|你不懂|外行|没资格|不配|先考上再说|什么学历|哪个学校|哪所学校|你是新闻|你从业吗|不是记者|不是新传|没上过大学|你这种人'
EMO=r'笑死|呵呵|哈哈|无语|离谱|荒谬|可笑|讽刺|嘲讽|破防|绷不住|救命|哭|泪|气死|恶心|烦死|绝望|后悔|劝退|千万别|已死|完蛋|牛马|大冤种|[？?]{2,}|[！!]{2,}|[哈啊呜]{3,}|偷笑R|笑哭R|哭惹R|泪崩R|扶额R'
SHIFT=r'加微信|私信|链接|网盘|资料|pdf|PDF|店名|购买|多少钱|设备|电脑|相机|宿舍|社团|穿搭|快递|明星|球赛|游戏|抽奖|广告|带货|参考书|电子版|踢我|dd'
DIRECT=r'^(对|是的|没错|确实|同意|赞同|就是|可不是|对啊|是啊|嗯嗯|真的|完全同意|我也觉得)|不对|不是|并不是|未必|哪有|怎么可能|我不同意|恰恰相反|反而|但是|可是|然而|建议|可以|不建议|别|当然|其实|我觉得'
CAUSAL=r'因为|所以|因此|从而|才能|能够|可以|让|使|意味着|有助于|导致|决定|转化为|用来|服务于|帮助|支撑|影响|对应|适合|需要|依靠|通过'
PURE=re.compile(r'^\s*(谢谢|感谢|求|蹲|码住|收藏|收到|好的|好哒|嗯嗯|哈哈+|啊+|呜+|冲|加油|厉害|棒|赞|支持|同问|dd|踢我|[\[\]A-Za-z0-9_@#话题R哭惹笑哭偷笑捂脸泪崩大笑扶额爱心红薯表情\s]+)[！!。,.，？?~～]*$')

def parse(x):
    try:return json.loads(x)
    except:return []

def second(r):
    src=str(r.direct_source_text or ''); rep=str(r.reply_text or ''); title=str(r.title or '')
    context=src+' '+title
    pure=bool(PURE.match(rep)) or (len(rep.strip())<=4 and not re.search(OBJECT,rep,re.I))
    obj=bool(re.search(OBJECT,rep,re.I)); context_obj=bool(re.search(OBJECT,context,re.I))
    market=bool(re.search(MARKET,rep)); personal=bool(re.search(PERSONAL,rep)) and bool(re.search(EXP,rep)); ability=bool(re.search(ABILITY,rep)); course=bool(re.search(COURSE,rep)); public=bool(re.search(PUBLIC,rep)); identity=bool(re.search(IDENTITY,rep)); emo=bool(re.search(EMO,rep)); shift=bool(re.search(SHIFT,rep,re.I)); direct=bool(re.search(DIRECT,rep)) or ('?' in src or '？' in src)
    # independent relevance decision
    if pure: rel='弱相关'
    elif obj: rel='核心相关'
    elif shift: rel='弱相关'
    elif context_obj and (direct or len(rep)<=20): rel='语境相关'
    elif context_obj and (market or personal or ability or public): rel='核心相关'
    elif not context_obj: rel='无关'
    else: rel='语境相关'
    st=[]
    if pure:
        if shift or re.search(r'求|谢谢|感谢|蹲|dd|踢我',rep,re.I): st=['话题转移']
        elif emo: st=['反讽与情绪']
        elif direct: st=['同层直接回应']
    else:
        if direct: st.append('同层直接回应')
        if personal: st.append('个人经验回应')
        if market: st.append('市场结果回应')
        if ability and (len(rep)>=35 or re.search(CAUSAL,rep)): st.append('具体能力解释')
        if public: st.append('公共价值回应')
        if identity: st.append('身份否定')
        if emo: st.append('反讽与情绪')
        if shift and not any(x in st for x in ['个人经验回应','市场结果回应','具体能力解释','公共价值回应','身份否定']): st.append('话题转移')
        if not st and context_obj: st=['同层直接回应']
        order=['同层直接回应','个人经验回应','市场结果回应','具体能力解释','公共价值回应','身份否定','反讽与情绪','话题转移','其他']
        st=sorted(dict.fromkeys(st),key=lambda x:order.index(x))[:2]
    causal=bool(re.search(CAUSAL,rep)); trans='无转译'
    if course and ability and causal: trans='课程—能力连接'
    if ability and market and causal: trans='能力—职业连接'
    if ability and public and causal: trans='能力—公共价值连接'
    if trans=='无转译' and public: trans='仅抽象价值表达'
    if rel in ['弱相关','无关']: trans='无转译'
    return rel,json.dumps(st,ensure_ascii=False),trans

b=pd.read_csv(BLIND,low_memory=False)
rows=[]
for _,r in b.iterrows():
    rel,st,tr=second(r)
    rows.append({'reply_comment_id':r.reply_comment_id,'secondpass_relevance':rel,'secondpass_strategy_json':st,'secondpass_translation':tr,'secondpass_status':'blind_agent_rereview'})
s=pd.DataFrame(rows)
s.to_csv(OUT/'high_tranche01_500_secondpass_blind_results.csv',index=False,encoding='utf-8-sig')
# only now open round1 key
k=pd.read_csv(KEY,low_memory=False)
cmp=k.merge(s,on='reply_comment_id',validate='one_to_one')
cmp['relevance_agree']=cmp.round1_relevance.eq(cmp.secondpass_relevance)
cmp['strategy_agree']=cmp.apply(lambda r:set(parse(r.round1_strategy_json))==set(parse(r.secondpass_strategy_json)),axis=1)
cmp['translation_agree']=cmp.round1_translation.eq(cmp.secondpass_translation)
cmp['all_agree']=cmp[['relevance_agree','strategy_agree','translation_agree']].all(axis=1)
cmp.to_csv(OUT/'high_tranche01_500_secondpass_comparison.csv',index=False,encoding='utf-8-sig')
mlb=MultiLabelBinarizer(classes=ALLOWED_STR)
y1=mlb.fit_transform(cmp.round1_strategy_json.map(parse)); y2=mlb.transform(cmp.secondpass_strategy_json.map(parse))
metrics={
 'n':len(cmp),
 'relevance_accuracy':accuracy_score(cmp.round1_relevance,cmp.secondpass_relevance),
 'relevance_macro_f1':f1_score(cmp.round1_relevance,cmp.secondpass_relevance,average='macro',zero_division=0),
 'strategy_micro_f1':f1_score(y1,y2,average='micro',zero_division=0),
 'strategy_macro_f1':f1_score(y1,y2,average='macro',zero_division=0),
 'strategy_exact_match':float(cmp.strategy_agree.mean()),
 'translation_accuracy':accuracy_score(cmp.round1_translation,cmp.secondpass_translation),
 'translation_macro_f1':f1_score(cmp.round1_translation,cmp.secondpass_translation,average='macro',zero_division=0),
 'all_fields_exact':float(cmp.all_agree.mean()),
 'disagreement_n':int((~cmp.all_agree).sum())
}
with open(OUT/'high_tranche01_500_secondpass_agreement_metrics.json','w',encoding='utf-8') as f:json.dump(metrics,f,ensure_ascii=False,indent=2)
print(json.dumps(metrics,ensure_ascii=False,indent=2))
