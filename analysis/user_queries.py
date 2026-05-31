import pandas as pd
import json
import ast

df_res = pd.read_csv('../output/results.csv')

print("--- 1. Workshop prevalence ---")
if 'venue' in df_res.columns:
    ws_papers = df_res[df_res['venue'].astype(str).str.startswith('W') | df_res['venue'].astype(str).str.contains('workshop', case=False, na=False)]
else:
    ws_papers = df_res[df_res['paper_id'].astype(str).str.startswith('W') | df_res['paper_id'].astype(str).str.contains('workshop', case=False, na=False)]
print(f"Workshop HE prevalence: {ws_papers['has_human_eval'].mean():.1%}")

print("\n--- 2. Dip in 2000 ---")
y2k = df_res[df_res['year'] == 2000]
print(f"2000 HE prevalence: {y2k['has_human_eval'].mean():.1%}")
print(f"Total papers in 2000: {len(y2k)}")

print("\n--- 3. Story Gen & LLM Judge ---")
# Count task prevalence for LLM judge
has_llm = df_res['has_llm_judge'] == True
all_tasks = df_res['inferred_tasks'].dropna().apply(lambda x: ast.literal_eval(x) if x.startswith('[') else x.split(','))
df_tasks = all_tasks.explode()

for task in df_tasks.unique():
    task_idx = all_tasks.apply(lambda x: task in x if isinstance(x, list) else False)
    total_t = task_idx.sum()
    if total_t > 0:
        llm_t = (task_idx & has_llm).sum()
        if llm_t > 0:
            print(f"Task {task}: {llm_t} LLM judge papers out of {total_t} ({llm_t/total_t:.1%})")

print("\n--- 4. OpenAI and Gemini prevalence ---")
total_llm_judges = df_res['has_llm_judge'].sum()
print(f"Total LLM judge papers: {total_llm_judges}")

from collections import Counter
models = []
for m in df_res[df_res['has_llm_judge']]['llm_judge_model'].dropna():
    try:
        m_list = ast.literal_eval(m) if m.startswith('[') else [x.strip() for x in m.split(',')]
        models.extend(m_list)
    except:
        pass
counts = Counter(models)
openai_count = counts.get('gpt-4', 0) + counts.get('gpt-3.5', 0) + counts.get('chatgpt', 0) + counts.get('openai', 0)
gemini_count = counts.get('gemini', 0)
print(f"OpenAI prevalence: {openai_count} ({openai_count/total_llm_judges:.1%})")
print(f"Gemini prevalence: {gemini_count} ({gemini_count/total_llm_judges:.1%})")

print("\n--- 5. RQ3 Qualities > 75% tasks with > 10% prev ---")
all_crit = df_res[df_res['has_human_eval']]['human_eval_criteria'].dropna().apply(lambda x: ast.literal_eval(x) if x.startswith('[') else x.split(','))
df_crit = all_crit.explode()
unique_crits = df_crit.unique()

he_papers_idx = df_res['has_human_eval'] == True
unique_tasks = df_tasks.unique()

for c in unique_crits:
    c_idx = df_res['human_eval_criteria'].dropna().apply(lambda x: c in x if isinstance(x, (list, str)) else False)
    
    tasks_above_10 = 0
    total_tasks_with_he = 0
    
    for t in unique_tasks:
        t_idx = all_tasks.apply(lambda x: t in x if isinstance(x, list) else False)
        t_he_idx = t_idx & he_papers_idx
        total_he_for_t = t_he_idx.sum()
        
        if total_he_for_t >= 10: # Only count tasks with meaningful N
            total_tasks_with_he += 1
            c_for_t = (t_he_idx & c_idx).sum()
            prev = c_for_t / total_he_for_t
            if prev >= 0.10:
                tasks_above_10 += 1
                
    if total_tasks_with_he > 0 and (tasks_above_10 / total_tasks_with_he) > 0.75:
        print(f"Quality '{c}': meets condition ({tasks_above_10}/{total_tasks_with_he} tasks >10%)")

