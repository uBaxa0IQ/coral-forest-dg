"""Registered contrasts from saved predictions, with artifact provenance."""
from hashlib import sha256
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest,t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .design import methods,BASELINES,REPLICAS,contexts
from .engine import ROOT,RUN,REPORT,write,now


def summarize(values,planned=10):
    v=np.asarray(values,dtype=float);v=v[np.isfinite(v)];n=len(v)
    mean=float(v.mean()) if n else None
    half=float(t.ppf(.975,n-1)*v.std(ddof=1)/np.sqrt(n)) if n>1 else None
    nz=v[v!=0];p=float(binomtest(int((nz>0).sum()),len(nz),.5).pvalue) if len(nz) else 1.
    return dict(replicas=n,mean=mean,ci_low=mean-half if half is not None else None,ci_high=mean+half if half is not None else None,p_value=p if n==planned else None,nonzero_replicas=len(nz),status='complete' if n==planned else 'incomplete')


def source_summary():
    REPORT.mkdir(exist_ok=True);rows=[];trials=[]
    for p in sorted((RUN/'selection').glob('*.json')):
        s=json.loads(p.read_text(encoding='utf-8'));c=s['context']
        rows.append(dict(context=c['id'],task=c['task'],kind=c['kind'],method=s['method'],selected=s['selected'],reference=s['reference'],parameters=json.dumps(s['parameters'],sort_keys=True),reference_parameters=json.dumps(s['reference_parameters'],sort_keys=True),source_primary=s['source_primary'],reference_source_primary=s['reference_source_primary'],screen_failed=s['screen_failed'],full_failed=s['full_failed'],artifact=p.relative_to(ROOT).as_posix(),sha256=sha256(p.read_bytes()).hexdigest()))
    for p in (RUN/'search').rglob('*.json'):
        r=json.loads(p.read_text(encoding='utf-8'));trials.append(dict(context=r['context'],method=r['method'],stage=r['stage'],candidate=r['candidate'],status=r['status'],primary=r.get('mean_primary'),seconds=r['seconds'],reused='screen_reused_from' in r,artifact=p.relative_to(ROOT).as_posix(),sha256=sha256(p.read_bytes()).hexdigest()))
    pd.DataFrame(rows).to_csv(REPORT/'source_selection.csv',index=False);pd.DataFrame(trials).to_csv(REPORT/'source_trials.csv',index=False)
    summary=dict(at=now(),completed_method_contexts=len(rows),trial_records=len(trials),failed_trials=sum(r['status']!='complete' for r in trials),target_evaluations=0,scope='source-only; no target quality conclusions')
    write(REPORT/'source_summary.json',summary);return summary


def analyze():
    source_summary();rows=[];inventory=[];proof={}
    for p in sorted((RUN/'jobs').glob('*/result.json')):
        r=json.loads(p.read_text(encoding='utf-8'));rows.extend(r['metrics']);s=r['spec'];inventory.append(dict(job_id=s['job_id'],context=s['id'],rows=r['target_rows'],models=len(set(v['method']+':'+v['variant'] for v in r['metrics']))))
        proof[p.relative_to(ROOT).as_posix()]=sha256(p.read_bytes()).hexdigest()
    df=pd.DataFrame(rows);df.to_csv(REPORT/'target_metrics.csv',index=False)
    pd.DataFrame(inventory).to_csv(REPORT/'job_inventory.csv',index=False)
    keys=['job_id','context','task','kind','scenario','data_replica','model_seed','method','variant']
    job=df.groupby(keys,dropna=False).agg(primary=('primary','mean'),source_variance=('source_variance','first')).reset_index()
    job.to_csv(REPORT/'macro_metrics.csv',index=False)
    syn=job[job.kind=='synthetic'];contrasts=[];hypotheses=[]
    for task in ['classification','regression']:
        expected_scenarios=7 if task=='classification' else 6
        for method in methods(task):
            pivot=syn[(syn.task==task)&(syn.method==method)].pivot(index=['context','data_replica','model_seed'],columns='variant',values='primary')
            variance=syn[(syn.task==task)&(syn.method==method)].groupby(['context','data_replica','model_seed']).source_variance.first()
            for comparison,control in [('HPO','reference')]+([] if method in BASELINES else [('mechanism','control')]):
                values=[]
                for replica in range(REPLICAS):
                    valid=pd.Series(dtype=float)
                    if 'tuned' in pivot and control in pivot:
                        valid=(pivot.tuned-pivot[control])/(variance if task=='regression' else 1.)
                        valid=valid[valid.index.get_level_values('data_replica')==replica].dropna()
                    complete=len(valid)==expected_scenarios*2
                    value=float(valid.mean()) if complete else None
                    values.append(np.nan if value is None else value)
                    contrasts.append(dict(hypothesis=comparison,task=task,method=method,replica=replica,effect=value,scenario_seed_pairs=len(valid),expected_pairs=expected_scenarios*2,complete=complete))
                hypotheses.append(dict(hypothesis=comparison,task=task,method=method,**summarize(values)))
    tests=pd.DataFrame(hypotheses);assert len(tests)==24
    p=tests.p_value.fillna(1.).to_numpy();order=np.argsort(p,kind='stable');adjust=np.empty(len(p));running=0.
    for i,index in enumerate(order):running=max(running,min(1.,p[index]*(len(p)-i)));adjust[index]=running
    tests['p_holm']=adjust;tests.loc[tests.status!='complete','p_holm']=np.nan
    tests.to_csv(REPORT/'hypothesis_tests.csv',index=False);pd.DataFrame(contrasts).to_csv(REPORT/'replica_contrasts.csv',index=False)
    real=job[job.kind=='real'].groupby(['context','method','variant']).primary.mean().reset_index();real.to_csv(REPORT/'real_summary.csv',index=False)
    per_scenario=job.groupby(['kind','context','task','method','variant']).primary.mean().reset_index();per_scenario.to_csv(REPORT/'scenario_summary.csv',index=False)
    # Source-selected strong baseline: no max operation over target scores.
    strong=[]
    for p in (RUN/'jobs').glob('*/source.json'):
        r=json.loads(p.read_text(encoding='utf-8'));s=r['spec'];sub=job[(job.job_id==s['job_id'])&(job.variant=='tuned')]
        base=sub[sub.method==r['strong_baseline']]
        if base.empty:continue
        for _,row in sub.iterrows():strong.append(dict(job_id=s['job_id'],context=s['id'],method=row.method,strong_baseline=r['strong_baseline'],gain=(row.primary-float(base.iloc[0].primary))/(r['source_variance'] if s['task']=='regression' else 1.)))
    pd.DataFrame(strong).to_csv(REPORT/'strong_baseline_gains.csv',index=False)
    significant=tests[(tests.status=='complete')&(tests.p_holm<.05)]
    audit=dict(at=now(),final_jobs=len(inventory),metric_rows=len(df),independent_primary_metric_checks=len(df),max_auc_check_difference=float(df.independent_auc_difference.max()) if 'independent_auc_difference' in df else 0.,max_mse_check_difference=float(df.independent_mse_difference.max()) if 'independent_mse_difference' in df else 0.,planned_primary_tests=24,complete_primary_tests=int((tests.status=='complete').sum()),significant_holm_tests=len(significant),source_selection_artifacts=127,target_based_selection=False,scope='known DGPs, independently tuned once per development context; ACS two tasks same survey family; real resamples descriptive',result_artifact_sha256=proof)
    write(REPORT/'final_integrity.json',audit)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9})
    fig,ax=plt.subplots(figsize=(12,9))
    for i,r in tests.iterrows():
        if pd.isna(r['mean']):continue
        ax.errorbar(r['mean'],i,xerr=[[r['mean']-r.ci_low],[r.ci_high-r['mean']]],fmt='o' if r.status=='complete' else 'x',color='#a52a2a' if r.p_holm<.05 else '#326589',capsize=2)
    ax.set_yticks(range(len(tests)),[f'{r.hypothesis} {r.task[:3]} {r.method}' for _,r in tests.iterrows()]);ax.axvline(0,color='grey',lw=.7);ax.invert_yaxis();ax.set_xlabel('Mean paired gain; marginal t95% CI; red = global Holm(24) < .05');fig.tight_layout()
    (REPORT/'figures').mkdir(exist_ok=True);fig.savefig(REPORT/'figures/primary.png',dpi=170);fig.savefig(REPORT/'figures/primary.svg');plt.close(fig)
    svg=REPORT/'figures/primary.svg';svg.write_text('\n'.join(x.rstrip() for x in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8',newline='\n')
    text=f'''# Углублённая настройка и свежий финальный тест v4

## Выполненный дизайн

Все методы и strong baselines получили предзаданные пространства гиперпараметров: screening и full-capacity проверка по source domains, без target выбора. Полный дизайн и численные бюджеты: [source protocol](../../runs/tuned_v4/source_protocol.json). Итоговые выбранные настройки, source files и final source models зафиксированы до открытия targets: [final protocol](../../runs/tuned_v4/final_protocol.json). [Source selection](source_selection.csv), [все source trials и длительности](source_trials.csv).

HPO выбиралась один раз на development source каждого контекста. Затем параметры переносились на новые, более крупные source draws. Это оценка заранее настроенных pipelines, а не повтор настройки в каждой MC replica. Предел качества каждого метода и равные CPU budgets не заявляются. Референс использует исправленные реализации и ограниченный исходный grid; это новый контроль, а не подмена исторических v1/v2.

## Итоговые артефакты

Завершено jobs: {len(inventory)}; target-domain metric rows: {len(df)}. Основных тестов: 24, завершены: {audit['complete_primary_tests']}; Holm-significant: {len(significant)}. Источник каждого числа: [final integrity](final_integrity.json), [таблица тестов](hypothesis_tests.csv), [replica contrasts](replica_contrasts.csv). Статусы пропусков нельзя заменять нулями или игнорировать при complete-case тестировании.

![Основные предзаданные контрасты](figures/primary.png)

## Как читать статистику

HPO = tuned − source-selected restricted reference; mechanism = tuned − own matched-capacity control. Для classification используется разность AUROC; для regression — разность MSE с обратным знаком, делённая на full-source variance. Сначала macro target domains и model seeds, затем равновесное усреднение зарегистрированных сценариев внутри независимой MC replica. Exact two-sided sign test проверяет знаки, не нулевое среднее; нулевые contrasts исключаются. Один Holm family охватывает все основные тесты. CI — маргинальный t интервал среднего, не simultaneous CI. [Статистика и filters](hypothesis_tests.csv), [исходные per-domain метрики](target_metrics.csv).

## Ограничения и интерпретация

Свежие synthetic draws независимы от прежних draw, но относятся к уже известным механизмам. Это не подтверждение на новой семье реальных задач. ACSIncome/PublicCoverage используют одни новые штаты и принадлежат одной survey family; source resamples перекрываются, поэтому real результаты описательные. Для Beijing/Appliances выполнена source настройка, нового независимого real target в прежних данных нет. Старые v1/v2 targets не использовались.

Положительный HPO contrast показывает пользу данного source-only поиска при заданном бюджете; отрицательный — возможную ошибку переноса source ranking или избыточную capacity. Отсутствие значимости не доказывает эквивалентности. Mechanism gains не следует путать с победой над strong baselines: [source-selected baseline comparison](strong_baseline_gains.csv). Meta остаётся интерпретацией, IRF/CRF/MaxRM — независимыми реализациями, не точными репликациями авторских экспериментов.

Алгоритмы и выбранные настройки после target доступа не менялись. Сохраняются исходные failures, source receipts, матричные hashes и predictions. Локальные timestamps и hashes не являются OS-level независимым доказательством отсутствия любого ручного доступа. Полные predictions/models локальны; в Git находятся протоколы, таблицы, реестр hashes и отчёт.
'''
    (REPORT/'research_report.md').write_text(text,encoding='utf-8',newline='\n')
    import html,re
    body=[]
    for line in text.splitlines():
        if not line.strip():continue
        if line.startswith('!['):body.append('<img style="max-width:100%" src="figures/primary.png" alt="Primary tests">');continue
        s=html.escape(line);s=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'<a href="\2">\1</a>',s)
        body.append('<h1>'+s[2:]+'</h1>' if s.startswith('# ') else '<h2>'+s[3:]+'</h2>' if s.startswith('## ') else '<p>'+s+'</p>')
    (REPORT/'index.html').write_text('<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>НИР v4</title><style>body{font:17px/1.65 system-ui;max-width:1200px;margin:30px auto;padding:20px;color:#203548}table{border-collapse:collapse;font-size:13px}td,th{padding:8px;border:1px solid #ccc}section{overflow:auto}a{color:#176397}</style>'+''.join(body)+'<section>'+tests.to_html(index=False,float_format=lambda v:f'{v:.6g}',na_rep='—')+'</section></html>',encoding='utf-8',newline='\n')
    return audit
