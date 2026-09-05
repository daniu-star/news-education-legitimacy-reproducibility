"""Build result cards from formal manuscript-facing outputs.

This report builder intentionally reads GEE binary outputs. The older
cluster-robust GLM files are historical audit artifacts and must not feed the
manuscript-facing result cards.
"""
import os
from pathlib import Path
import pandas as pd,numpy as np,json,sqlite3,hashlib,shutil,zipfile,os
ROOT=Path(os.environ['NEWS_EDU_ROOT']);OUT=ROOT/'final_analysis';M=OUT/'05_models'
for d in ['08_results_cards','09_writing_handoff']: (OUT/d).mkdir(parents=True,exist_ok=True)
def row(path,term):
 d=pd.read_csv(path);x=d[d.term==term];return None if x.empty else x.iloc[0]
def fmt(x,n=2):return 'NA' if x is None or pd.isna(x) else f'{x:.{n}f}'
def wr(p,s):Path(p).write_text(s,encoding='utf8')
# key rows
neg=M/'rq1/negative_gee.csv';pos=M/'rq1/positive_gee.csv';like=M/'rq4/any_like_gee.csv';likecnt=M/'rq4/positive_likes_nb.csv'
for required in [neg,pos,like,likecnt]:
 if not required.exists():
  raise FileNotFoundError(f'Formal manuscript-facing result is missing: {required}. Run the GEE pipeline before packaging result cards.')
# aliases e0 learn,e1 work,e2 others,e3 market,e4 knowledge,e5 public,e6 AI,e7 identity,e8 satire,e9 assertion; o2 ability,o3 profession,o4 career,o5 industry,o6 discipline
keys={
'market_neg':row(neg,'e3'),'satire_neg':row(neg,'e8'),'identity_neg':row(neg,'e7'),'public_neg':row(neg,'e5'),
'identity_pos':row(pos,'e7'),'public_pos':row(pos,'e5'),'ability_pos':row(pos,'o2'),
'public_like':row(like,'e5'),'knowledge_like':row(like,'e4'),'satire_like':row(like,'e8'),'browser_like':row(like,'C(source_type)[T.browser]'),
'satire_count':row(likecnt,'e8'),'knowledge_count':row(likecnt,'e4'),'browser_count':row(likecnt,'C(source_type)[T.browser]'),
'author_like':row(like,'is_note_author'),'author_reply_inter':row(like,'is_reply:is_note_author')}
rq2=pd.read_csv(OUT/'07_robustness/rq2_scope_robustness.csv');rq1rob=pd.read_csv(OUT/'07_robustness/rq1_source_outlier_robustness.csv');v2=pd.read_csv(M/'rq1/v2_adjudicated_sample_sensitivity.csv');ab=pd.read_csv(M/'rq3/ability_property_distribution.csv');ablong=pd.read_csv(OUT/'03_ability_pairing/ability_mentions_long_v2.1.csv.gz');fs=json.load(open(OUT/'06_framework_shift/framework_shift_summary.json',encoding='utf8'));rel=json.load(open(OUT/'02_relational_final/relation_terminal_summary.json',encoding='utf8'));aba=json.load(open(OUT/'03_ability_pairing/ability_pairing_audit_summary.json',encoding='utf8'));flow=pd.read_csv(OUT/'04_sampling_lineage/inclusion_flow.csv');incl=pd.read_csv(OUT/'04_sampling_lineage/search_inclusion_model.csv');rank=incl[incl.term=='rank_log'].iloc[0]
# combined model registry
regs=[]
for p in [M/'model_registry_rq1_rq4.csv',M/'model_registry_rq2_rq3.csv']:
 if p.exists():regs.append(pd.read_csv(p))
registry=pd.concat(regs,ignore_index=True);registry.to_csv(M/'model_registry.csv',index=False)
# Result cards
cards=[]
def add(fid,title,unit,n,claim,evidence,boundary,status='可写入正文（附边界）'):
 cards.append({'finding_id':fid,'title':title,'analysis_unit':unit,'n':n,'claim':claim,'evidence':evidence,'boundary':boundary,'status':status})
add('F01','市场结果与反讽持续推动否定评价','评论',16533,
    '在历史v1全量标签中，招聘收入与市场结果、反讽或梗均与否定评价稳定正相关；这一方向在API、browser以及排除头部笔记后保持一致。',
    f'市场结果OR={fmt(keys["market_neg"].odds_ratio)}，95%CI [{fmt(keys["market_neg"].ci_low)}, {fmt(keys["market_neg"].ci_high)}]；反讽OR={fmt(keys["satire_neg"].odds_ratio)}，95%CI [{fmt(keys["satire_neg"].ci_low)}, {fmt(keys["satire_neg"].ci_high)}]。v2裁决样本中二者否定率分别为{v2.loc[v2.evidence=="招聘收入与市场结果","negative_rate"].iloc[0]:.1%}与{v2.loc[v2.evidence=="反讽或梗","negative_rate"].iloc[0]:.1%}。',
    '全量模型依赖v1标签；效应可表述为稳定关联，不能表述为因果裁决。')
add('F02','身份情感更多连接认可，而公共价值具有双向争夺','评论',16533,
    '身份情感与否定评价负相关、与认可评价正相关；公共价值同时提高认可概率，但其否定方向对来源和头部笔记较敏感。',
    f'身份情感否定OR={fmt(keys["identity_neg"].odds_ratio)}，认可OR={fmt(keys["identity_pos"].odds_ratio)}；公共价值认可OR={fmt(keys["public_pos"].odds_ratio)}。v2样本公共价值否定率为{v2.loc[v2.evidence=="公共价值","negative_rate"].iloc[0]:.1%}。',
    '公共价值不能简单归入正面或负面框架，更适合写成合法性争夺场域。')
# RQ2 author effects
ad=rq2[(rq2.scope=='adjudicated')&(rq2.analysis=='RQ2_translation')].iloc[0];hi=rq2[(rq2.scope=='adjudicated_plus_high')&(rq2.analysis=='RQ2_translation')].iloc[0];full=rq2[(rq2.scope=='full_hybrid')&(rq2.analysis=='RQ2_translation')].iloc[0]
add('F03','作者身份没有形成稳定的价值转译优势','回复关系',1010,
    '在仅使用逐条Agent裁决关系时，作者回复与普通用户回复的价值转译概率几乎没有差异；只有纳入大量混合标签后效应才上升。',
    f'逐条裁决口径作者OR={fmt(ad["or"])}，95%CI [{fmt(ad["ci_low"])}, {fmt(ad["ci_high"])}]；逐条裁决＋高置信OR={fmt(hi["or"])}；全量混合OR={fmt(full["or"])}。',
    '应删除“作者回复被平台惩罚”与“作者天然更会转译价值”的表述；可以讨论作者回应仍多停留于直接答复。')
# Ability explicit
explicit=ablong[ablong.ability_property!='未明确']; counts=explicit.ability_property.value_counts()
add('F04','能力被频繁提及，但专业边界通常没有被明确说出','显性能力提及',len(ablong),
    '1,959个显性能力提及中，1,902个没有明确说明该能力属于新闻专业、大学通用、跨职业迁移或AI关系。明确属性判断主要集中于基础内容生产和平台运营。',
    '明确属性共'+str(len(explicit))+'个：'+ '、'.join(f'{k}{v}个' for k,v in counts.items())+'。显性配对规则在360条v2样本上的完全一致率为'+f'{aba["exact_match"]:.3f}。',
    '不能沿用旧数组强制配对所得的“采访专业独特率13.1%”等比例；只能报告显性边界判断。')
add('F05','公共价值和专业知识更容易获得点赞可见性，但来源偏差极强','评论',21779,
    '控制对象、来源、层级和文本长度后，公共价值、专业知识与反讽表达更容易至少获得一次点赞；browser来源的可见性显著更高。',
    f'公共价值获赞OR={fmt(keys["public_like"].odds_ratio)}；专业知识OR={fmt(keys["knowledge_like"].odds_ratio)}；反讽OR={fmt(keys["satire_like"].odds_ratio)}；browser来源OR={fmt(keys["browser_like"].odds_ratio)}。正值点赞模型中browser IRR={fmt(keys["browser_count"].irr)}。',
    '点赞是可见互动，不等同于态度认同；browser样本本身偏向高赞评论。')
add('F06','作者回复的低点赞现象在控制后不成立','评论',21779,
    '控制评论层级、来源、文本和内容标签后，笔记作者身份及其与回复层级的交互均不显著。',
    f'作者身份获赞OR={fmt(keys["author_like"].odds_ratio)}，95%CI [{fmt(keys["author_like"].ci_low)}, {fmt(keys["author_like"].ci_high)}]；作者×回复交互OR={fmt(keys["author_reply_inter"].odds_ratio)}。',
    '不得写“平台惩罚作者回复”或“学院话语被算法压制”。')
add('F07','评论区存在可识别的就业吸附','笔记',fs['n_notes'],
    '在笔记与评论共享的七类可解释框架空间中，评论区的就业职业权重平均高于笔记端。',
    f'就业吸附系数均值={fs["mean_employment_attraction"]:.3f}，笔记层Bootstrap 95%CI [{fs["employment_attraction_ci95"][0]:.3f}, {fs["employment_attraction_ci95"][1]:.3f}]；平均总变差={fs["mean_frame_shift_tv"]:.3f}。',
    '这是词典投射下的框架变化，不等同于因果意义上的平台转译，也不替代主题模型。')
add('F08','搜索排名显著塑造最终样本','搜索命中—笔记',len(pd.read_csv(OUT/'04_sampling_lineage/search_to_note_lineage.csv.gz')),
    '搜索结果排名越靠后，笔记进入分析库的概率越低；不同关键词类别的纳入概率也不相同。',
    f'排名对纳入概率OR={rank.odds_ratio:.3f}，95%CI [{rank.ci_low:.3f}, {rank.ci_high:.3f}]。',
    '样本不能外推为全部小红书内容或公众意见；方法部分必须完整报告采样谱系。')
cdf=pd.DataFrame(cards);cdf.to_csv(OUT/'08_results_cards/论文可用结论总表.csv',index=False)
for c in cards:
 wr(OUT/f'08_results_cards/{c["finding_id"]}_{c["title"]}.md',f'# {c["title"]}\n\n**可用判断**：{c["claim"]}\n\n**证据**：{c["evidence"]}\n\n**分析单位**：{c["analysis_unit"]}；有效N={c["n"]}。\n\n**边界**：{c["boundary"]}\n\n**状态**：{c["status"]}\n')
# Detailed reports
wr(OUT/'02_relational_final/关系编码终结与使用边界报告.md',f'''# 关系编码终结与使用边界报告

## 数据与标签来源

10,620条可恢复关系均已获得终版标签，但标签证据等级不同：

- 逐条Agent裁决：1,393条；
- 校准后的高置信混合标签：2,980条；
- 低置信混合标签：6,247条。

逐条Agent裁决不能写成两名人类编码员的人工编码。低置信混合标签只用于敏感性口径。

## 正式模型口径

RQ2分别运行逐条裁决、逐条裁决＋高置信混合、全量混合三套模型。作者身份对价值转译的OR依次为{ad["or"]:.3f}、{hi["or"]:.3f}和{full["or"]:.3f}。只有全量混合接近统计显著，因此作者转译优势不具口径稳定性。

## 结论

关系材料可以支持回复策略与价值转译的条件性分析，但不支持“作者回复天然更有效”“作者被平台惩罚”或“学院话语被算法边缘化”等强机制结论。
''')
wr(OUT/'03_ability_pairing/能力配对重编码报告.md',f'''# 能力配对重编码报告

对21,779条语义有效评论应用Codebook v2.1的显性能力门槛，共识别1,959个能力提及，来自1,675条评论。每个提及均保存能力类型、能力属性、原文证据和编码方法。

在360条v2裁决样本中，配对宏平均Precision={aba['pair_precision_macro']:.3f}，Recall={aba['pair_recall_macro']:.3f}，完全一致率={aba['exact_match']:.3f}。

1,902个提及的属性为“未明确”，只有{len(explicit)}个明确边界判断。基础内容生产与平台运营承载了大多数“大学通用”“跨职业迁移”和“AI替代”表述。采访、核查和公共判断虽被提及，但文本通常没有明确声明其专业专属性。

因此，论文应将RQ3表述为“显性能力边界如何被说出或保持沉默”，不能继续使用旧版独立数组强制配对后的比例。
''')
wr(OUT/'04_sampling_lineage/采样框与来源偏差报告.md',f'''# 采样框与来源偏差报告

{flow.to_markdown(index=False)}

搜索命中到分析笔记不是简单漏斗。最终2,134篇笔记中既有搜索命中后成功抓取的笔记，也包含首页和browser补充来源。搜索排名每增加一个对数单位，进入分析库的优势比为{rank.odds_ratio:.3f}。

API与browser来源必须分层解释：browser主要保存高可见一级评论，正值点赞模型中的来源IRR达到{keys['browser_count'].irr:.2f}，说明来源字段反映采集机制，而不只是内容差异。
''')
wr(OUT/'06_framework_shift/框架迁移分析报告.md',f'''# 框架迁移分析报告

采用就业职业、课程知识、AI平台、公共价值、专业身份、升学报考和能力训练七类共享词典，将笔记与评论投射到同一空间。纳入至少5条语义评论且能够计算框架分布的笔记共{fs['n_notes']}篇。

平均总变差为{fs['mean_frame_shift_tv']:.3f}，就业吸附系数均值为{fs['mean_employment_attraction']:.3f}，Bootstrap 95%区间为[{fs['employment_attraction_ci95'][0]:.3f}, {fs['employment_attraction_ci95'][1]:.3f}]。

这说明评论互动整体上增加了就业职业框架的相对权重，但只有{fs['positive_employment_attraction_share']:.1%}的笔记就业吸附系数为正，均值受到若干强偏移笔记影响。因此论文应同时呈现均值、分布和转移矩阵。
''')
wr(OUT/'07_robustness/稳健性总报告.md',f'''# 稳健性总报告

## RQ1

市场结果与反讽对否定评价的正向关联在API、browser、去除评论量前1%和前5%笔记后保持一致。身份情感的负向关联在API及去头部样本中稳定，在browser单独样本中方向一致但区间跨1。公共价值的否定效应随来源和头部笔记处理变化，不应作为单向结论。

## RQ2

作者身份对价值转译的效应由逐条裁决口径OR={ad["or"]:.3f}变为全量混合OR={full["or"]:.3f}，显示结果依赖自动标签范围。正式正文应采用逐条裁决或逐条裁决＋高置信混合作为主表，并将全量混合作为附录敏感性。

## RQ4

browser来源对点赞存在极强效应，说明采集源差异不能被内容变量完全吸收。作者回复效应控制后不显著，否定了裸比例比较形成的强机制解释。
''')
# Main terminal report
wr(OUT/'正式论文写作前分析终版总报告.md',f'''# 新闻传播教育平台评价研究：正式写作前分析终版总报告

**终版版本**：terminal v2.1  
**完成日期**：2026-08-04

## 一、已经一次性完成的工作

1. 终结10,620条关系标签并保留裁决／高置信／低置信三类来源；
2. 重建1,959个显性能力—属性配对；
3. 建立50个搜索词的八类分类和搜索—笔记纳入谱系；
4. 完成RQ1评价依据与方向模型；
5. 完成RQ2回复策略与价值转译三口径模型；
6. 完成RQ3显性能力边界分析；
7. 完成RQ4点赞与回复的两阶段模型；
8. 完成共享框架迁移和就业吸附分析；
9. 完成来源拆分、去头部笔记和标签范围敏感性；
10. 生成结果卡、写作证据映射、禁用表述和终版数据库。

## 二、论文最稳固的经验判断

- 市场结果与反讽是跨来源、去头部后仍稳定连接否定评价的依据；
- 身份情感更常连接认可，但公共价值内部同时存在认可与否定；
- 评论区相对笔记端呈现平均意义上的就业吸附；
- 作者身份没有形成稳定的价值转译优势，作者回复低点赞的裸差异在控制后消失；
- 用户经常讨论技能，但很少明确说明这些技能究竟属于新闻专业、大学通用或AI可替代，能力边界的“未明说”本身构成重要结果；
- browser采集显著偏向高可见评论，来源差异是所有互动结果的首要边界。

## 三、必须删除或降级的原结论

- 删除“作者回复被平台惩罚／边缘化”；
- 删除“采访与信源已形成13.1%的专业独特共识”；
- 删除基于旧能力数组位置强制配对的AI替代率；
- 降级“公共价值天然带来认可”，改为公共价值的双向合法性争夺；
- 不再将10,620条关系全部描述为人工逐条编码。

## 四、当前仍不能由数据解决的问题

严格意义的人类编码者间可靠性仍需要两名真实研究者独立编码。RQ1全量模型仍以历史v1标签为主，已用v2裁决样本验证方向，但不能宣称全量v2人工重编码完成。这些限制已写入方法和禁用表述清单，不再阻碍论文撰写，但必须透明披露。
''')
# Writing handoff
wr(OUT/'09_writing_handoff/方法部分事实清单.md',f'''# 方法部分事实清单

- 搜索命中：14,789行，6,799个不同笔记ID。
- 分析库：2,134篇笔记、30,541条评论、22,946个评论用户。
- 语义有效且非孤立评论：21,779条。
- 回复总数10,888条；可恢复直接上文的关系10,620条。
- 关系标签：1,393条逐条Agent裁决，2,980条高置信混合，6,247条低置信混合。
- 显性能力提及：1,959个，来自1,675条评论。
- RQ1使用历史v1全量标签＋360条v2裁决样本敏感性。
- 二元结果使用按笔记聚类的Exchangeable GEE；计数结果使用负二项模型；不把搜索样本视为概率样本。
- 点赞和回复是可见互动，不等同于认同。
''')
wr(OUT/'09_writing_handoff/研究发现证据映射.md','''# 研究发现证据映射

## 发现一：评价依据的分化

主证据：`05_models/rq1/negative_gee.csv`、`positive_gee.csv`、`07_robustness/rq1_source_outlier_robustness.csv`。建议概念从笼统的“经验裁决”细化为“市场结果裁决—身份认同支持—公共价值争夺”三种不同机制。

## 发现二：平台互动中的价值转译不足

主证据：`05_models/rq2/`三种口径和`07_robustness/rq2_scope_robustness.csv`。作者回复更多表现为直接回应，但没有稳定的价值转译优势。

## 发现三：专业能力边界的低显化

主证据：`03_ability_pairing/ability_mentions_long_v2.1.csv.gz`与`05_models/rq3/`。重点不是宣称某项能力已经被社会公认为专业独特，而是说明绝大多数能力提及没有完成边界归属判断。

## 发现四：就业吸附与可见性过滤

主证据：`06_framework_shift/`和`05_models/rq4/`。评论区平均增加就业框架权重；公共价值、专业知识与反讽更易获得至少一次点赞，但来源机制必须控制。
''')
wr(OUT/'09_writing_handoff/教育重构证据映射.md','''# 教育重构证据映射

1. 培养目标：回应市场结果裁决，但不能以就业数据替代教育价值；需要把课程、能力和职业路径建立可核验连接。
2. 能力体系：优先把事实核查、采访、社会解释和伦理判断操作化，而不是预设公众已承认其专业独特性。
3. 课程组织：通过长期真实项目让能力边界变得可观察，减少“只会写稿剪辑”的单一印象。
4. 评价机制：同时评价职业迁移、事实质量、解释能力、伦理决策与公共影响。
5. 对外回应：作者直接答复并不自动形成价值转译，应以追踪数据、作品证据和能力说明替代身份辩护。
6. 制度边界：平台评价不能代表全部社会需求，教育改革需结合行业、学生和公共机构多源证据。
''')
wr(OUT/'09_writing_handoff/写作Agent禁用表述清单.md','''# 写作Agent禁用表述清单

- 两名人类编码员取得高度一致。
- 全部10,620条关系均经人工逐条编码。
- 作者回复受到平台算法惩罚或学院话语被算法压制。
- 点赞等同于认同，回复等同于争议。
- 小红书用户代表公众或中国青年。
- 采访能力已经形成明确的专业独特共识。
- 基础内容生产有确定的AI替代率。
- 公共价值天然带来认可。
- 完整重建了全部直接回复树。
- 稳定识别出十个实质主题。
''')
# execution checklist
wr(OUT/'终版任务完成清单.md','''# 终版任务完成清单

- [x] 数据库冻结与标签对齐
- [x] Codebook v2.1与可靠性审计
- [x] 关系语料重建
- [x] 关系标签终结与来源分级
- [x] 能力类型—属性显性配对
- [x] 搜索词分类与采样谱系
- [x] RQ1评价模型
- [x] RQ2关系模型
- [x] RQ3能力边界模型
- [x] RQ4互动可见性模型
- [x] 框架迁移与就业吸附
- [x] 来源、头部笔记与标签范围稳健性
- [x] 结果卡与写作交接
- [x] 终版数据库、版本清单和累计压缩包

未伪造完成的事项：两名真实人类编码者之间的可靠性。该事项只能由真实研究团队补签完成。
''')
# terminal DB add outputs
finaldb=OUT/'10_final_database/analysis_terminal_v2.1.db';con=sqlite3.connect(finaldb);registry.to_sql('model_registry_v2_1',con,if_exists='replace',index=False);pd.read_csv(OUT/'06_framework_shift/shared_frame_projection.csv.gz').to_sql('framework_shift_v2_1',con,if_exists='replace',index=False);cdf.to_sql('result_cards_v2_1',con,if_exists='replace',index=False);con.execute('create table if not exists terminal_version(version_id text primary key, created_at text, description text)');con.execute('insert or replace into terminal_version values(?,?,?)',('terminal_v2.1_20260804',pd.Timestamp.now().isoformat(),'Completed formal pre-writing analysis with provenance-aware hybrid labels.'));con.commit();con.close()
# manifest
files=[]
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
for p in sorted(OUT.rglob('*')):
 if p.is_file():files.append({'path':str(p.relative_to(ROOT)),'size':p.stat().st_size,'sha256':sha(p)})
manifest={'version':'terminal_v2.1_20260804','created_at':pd.Timestamp.now().isoformat(),'source_frozen_db_sha256':sha(ROOT/'00_freeze/analysis_v2_frozen.db'),'files':files,'completed_models':len(registry),'result_cards':len(cdf)}
json.dump(manifest,open(OUT/'terminal_version_manifest_v2.1.json','w',encoding='utf8'),ensure_ascii=False,indent=2)
print(json.dumps({'reports':len(list(OUT.rglob('*.md'))),'models':len(registry),'cards':len(cdf),'terminal_db':str(finaldb)},ensure_ascii=False))
