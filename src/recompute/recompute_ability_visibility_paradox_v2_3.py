from pathlib import Path
import json
import pandas as pd
from scipy.stats import fisher_exact

BASE = Path(__file__).resolve().parents[2]
INPUT = BASE / 'terminal_v2.2_升级补充' / '04_能力配对' / 'ability_mentions_long_v2.1.csv.gz'
OUT = Path(__file__).resolve().parent

df = pd.read_csv(INPUT)
foreground = {'基础内容生产', '平台运营与商业传播'}
df['ability_layer'] = df['ability_type'].map(lambda x: '前台技能' if x in foreground else '高阶能力')
df['property_explicit'] = df['ability_property'].ne('未明确')

cross = pd.crosstab(df['ability_layer'], df['property_explicit']).rename(columns={False: '属性未明确', True: '属性明确'})
cross.to_csv(OUT / 'ability_visibility_crosstab_v2.3.csv', encoding='utf-8-sig')

fg = df[df['ability_layer'] == '前台技能']
high = df[df['ability_layer'] == '高阶能力']
res = fisher_exact([
    [int(fg['property_explicit'].sum()), int((~fg['property_explicit']).sum())],
    [int(high['property_explicit'].sum()), int((~high['property_explicit']).sum())]
])
summary = {
    'total_mentions': int(len(df)),
    'foreground_mentions': int(len(fg)),
    'high_order_mentions': int(len(high)),
    'foreground_explicit': int(fg['property_explicit'].sum()),
    'high_order_explicit': int(high['property_explicit'].sum()),
    'foreground_explicit_rate': float(fg['property_explicit'].mean()),
    'high_order_explicit_rate': float(high['property_explicit'].mean()),
    'fisher_odds_ratio': float(res.statistic),
    'fisher_p_value': float(res.pvalue),
    'ai_replace_foreground': int(((fg['ability_property'] == '可由AI替代')).sum()),
    'ai_replace_high_order': int(((high['ability_property'] == '可由AI替代')).sum())
}
(OUT / 'ability_visibility_summary_v2.3.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2))
