"""Derive closeout material from immutable v1/v2 evidence; never fit a model."""
from datetime import datetime,timezone
from hashlib import sha256
from html import escape
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/closeout'
V2=ROOT/'reports/series_v2'
LABEL={'irf':'IRF','crf':'CRF','meta':'Meta weights','meta_masked':'Meta masked','anchor':'Anchor','maxrm':'MaxRM','era':'Era','directional_era':'Directional Era','rf':'RF','extra_trees':'ExtraTrees','hist_gbdt':'HistGBDT'}


def load(path):
    return json.loads((ROOT/path).read_text(encoding='utf-8'))


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8',newline='\n')


def main():
    OUT.mkdir(exist_ok=True);(OUT/'tables').mkdir(exist_ok=True);(OUT/'figures').mkdir(exist_ok=True)
    claims=[];inputs={}
    def evidence(cid,category,statement,path,selector,values,limits):
        p=ROOT/path;digest=sha256(p.read_bytes()).hexdigest();inputs[path]=digest
        claims.append(dict(id=cid,category=category,statement=statement,artifact=path,selector=selector,sha256=digest,values=values,limits=limits))
        return f'[{cid}](claims_registry.json)'
    h=pd.read_csv(V2/'tables/hypothesis_tests.csv')
    for _,r in h.iterrows():
        cid=f"{r.hypothesis}_{r.task}_{r.method}"
        status='confirmatory' if r.status=='complete' else 'incomplete'
        values={k:None if pd.isna(r[k]) else r[k] for k in ['replicas','mean','ci_low','ci_high','p_value','p_holm','status']}
        evidence(cid,status,'Контраст преимущества source-selected метода относительно собственного контроля','reports/series_v2/tables/hypothesis_tests.csv',dict(hypothesis=r.hypothesis,task=r.task,method=r.method),values,'Только соответствующий DGP; Holm отдельно внутри H1/H2; CI маргинальный t, p — exact sign; неполный MaxRM не подтверждающий тест')
    h['evidence_id']=h.apply(lambda r:f'{r.hypothesis}_{r.task}_{r.method}',axis=1)
    h.to_csv(OUT/'tables/primary_tests.csv',index=False)
    sec=pd.read_csv(V2/'tables/secondary_N_contrasts.csv')
    sec['evidence_id']=['N_'+str(i) for i in range(len(sec))]
    for _,r in sec.iterrows():
        evidence(r.evidence_id,'descriptive','Изменение парного преимущества при увеличении N','reports/series_v2/tables/secondary_N_contrasts.csv',dict(task=r.task,method=r.method,scenario=r.scenario),{k:float(r[k]) for k in ['mean','ci_low','ci_high']},'Secondary; без коррекции множественности; не все методы улучшаются')
    sec.to_csv(OUT/'tables/sample_size_contrasts.csv',index=False)
    metrics=pd.read_csv(V2/'tables/metric_summary.csv')
    real=metrics[(metrics.kind=='real') & (((metrics.dataset.str.startswith('acs_')) & (metrics.n==16000) & (metrics.k==8)) | ((metrics.dataset=='beijing_pm25')&(metrics.n==9600)&(metrics.k==6)) | ((metrics.dataset=='appliances')&(metrics.n==8000)&(metrics.k==8)))].copy()
    real['evidence_id']='real_'+real.dataset+'_'+real.method
    for _,r in real.iterrows():
        cols=['roc_auc','control_roc_auc'] if r.task=='classification' else ['rmse','control_rmse']
        evidence(r.evidence_id,'descriptive','Средняя target-метрика в предзаданной ячейке реальной задачи','reports/series_v2/tables/metric_summary.csv',dict(dataset=r.dataset,k=int(r.k),n=int(r.n),method=r.method),{k:None if pd.isna(r[k]) else float(r[k]) for k in cols},'Три перекрывающиеся source resamples и два model seeds; не независимые datasets; Appliances IRF и Beijing имеют численные/input оговорки')
    real.to_csv(OUT/'tables/real_results_v2.csv',index=False)
    v1=pd.read_csv(ROOT/'reports/tables/target_summary.csv')
    v1['evidence_id']='v1_'+v1.dataset+'_'+v1.method
    for _,r in v1.iterrows():
        evidence(r.evidence_id,'exploratory','Историческая v1 macro target-метрика','reports/tables/target_summary.csv',dict(dataset=r.dataset,method=r.method),dict(value=float(r.target_macro_score),seed_sd=float(r.seed_sd)), 'SD по model seeds не CI по независимым задачам; v1/v2 не парный контраст; historical implementation_status не подтверждает точную репликацию всей процедуры')
    v1.to_csv(OUT/'tables/real_results_v1.csv',index=False)
    effects=pd.read_csv(V2/'tables/effect_summary.csv')
    shifts=effects[(effects.kind=='synthetic')&(effects.k==8)&(effects.n==8192)&(effects.imbalance==1)].copy()
    shifts['evidence_id']=['shift_'+str(i) for i in range(len(shifts))]
    for _,r in shifts.iterrows():
        evidence(r.evidence_id,'descriptive','Парное преимущество по типу сдвига','reports/series_v2/tables/effect_summary.csv',dict(kind='synthetic',task=r.task,scenario=r.scenario,k=8,n=8192,imbalance=1,method=r.method),{k:float(r[k]) for k in ['mean','ci_low','ci_high']},'Маргинальные интервалы внутри DGP; отбор красивых интервалов не является confirmatory evidence')
    shifts.to_csv(OUT/'tables/shift_effects.csv',index=False)
    grid=pd.read_csv(V2/'tables/source_target_grid_diagnostics.csv')
    irf=grid[(grid.kind=='synthetic')&(grid.task=='classification')&(grid.scenario=='spurious')&(grid.n==8192)&(grid.imbalance==1)&(grid.method=='irf')]
    irf=irf.groupby(['k','config'])[['source_primary','target_primary']].mean().reset_index()
    freq=pd.read_csv(V2/'tables/configuration_selection_frequency.csv')
    freq=freq[(freq.kind=='synthetic')&(freq.task=='classification')&(freq.scenario=='spurious')&(freq.n==8192)&(freq.imbalance==1)&(freq.method=='irf')]
    irf=irf.merge(freq[['k','config','chosen_jobs','available_jobs']],on=['k','config'])
    irf['evidence_id']='selection_irf'
    irf.to_csv(OUT/'tables/irf_source_target.csv',index=False)
    evidence('selection_irf','descriptive','Source и target ранжируют фиксированные IRF настройки по-разному','reports/series_v2/tables/source_target_grid_diagnostics.csv',dict(kind='synthetic',task='classification',scenario='spurious',n=8192,imbalance=1,method='irf',aggregation='mean by k,config'),irf[['k','config','source_primary','target_primary']].to_dict('records'),'Target curves зарегистрированы, но новый выбор по ним является exploratory; фиксированная lambda не выбирается для deployment')
    evidence('selection_counts','descriptive','Частоты source-only выбора IRF','reports/series_v2/tables/configuration_selection_frequency.csv',dict(kind='synthetic',task='classification',scenario='spurious',n=8192,imbalance=1,method='irf'),irf[['k','config','chosen_jobs','available_jobs']].to_dict('records'),'Не является оценкой частоты в иных DGP')
    audit=load('reports/series_v2/audit/final_integrity.json')
    evidence('integrity_v2','audit','Независимый пересчёт сохранённых метрик и воспроизведение моделей','reports/series_v2/audit/final_integrity.json','whole JSON',audit,'Проверка отдельными уравнениями тем же агентом; не внешняя рецензия и не доказательство отсутствия исторического ручного доступа')
    a1=load('reports/audit_v1/audit_results.json')
    evidence('integrity_v1','audit','Аудит первой серии','reports/audit_v1/audit_results.json','whole JSON',a1,'v1 остаётся exploratory, включая ограничения контролей из audit_report.md')
    inv=load('reports/series_v2/analysis_inventory.json')
    evidence('inventory','audit','Полнота второй серии','reports/series_v2/analysis_inventory.json','whole JSON',inv,'18 planned tests, только 17 complete; не называть все 18 завершёнными')
    numeric=pd.read_csv(V2/'audit/tree_numeric_checks.csv')
    bad=numeric[numeric.iloc[:,2:].sum(axis=1)>0].copy();bad.to_csv(OUT/'tables/affected_irf_models.csv',index=False)
    evidence('irf_defect','implementation_limit','Нарушения фактических partitions сохранённых IRF Appliances','reports/series_v2/audit/tree_numeric_checks.csv','sum numeric finding columns > 0',bad.to_dict('records'),'Observed predictions конечны; корректность на произвольных новых inputs не доказана; чистая механистическая интерпретация этих пар ограничена')
    evidence('null_defect','implementation_limit','Cache creator Beijing требует исходное представление missing values','reports/series_v2/audit/null_cache_full_vector_replay.json','whole JSON',load('reports/series_v2/audit/null_cache_full_vector_replay.json'),'Внутри задания inputs одинаковы; между заданиями неодинаковая null schema; не ошибка labels')
    sens=pd.read_csv(V2/'audit/null_cache_sensitivity.csv')
    sens.to_csv(OUT/'tables/beijing_cache_sensitivity.csv',index=False)
    evidence('null_sensitivity','implementation_limit','Чувствительность RMSE к альтернативному cached input','reports/series_v2/audit/null_cache_sensitivity.csv','all rows',sens.to_dict('records'),'Только forensic sensitivity, исторические метрики не подменяются')
    diag=load('reports/closeout/source_diagnostics/summary.json')
    evidence('source_corrections','source_only','Source-only исправления IRF и null schema','reports/closeout/source_diagnostics/summary.json','whole JSON',diag,'Не target validation и не доказательство лучшего OOD качества')
    cert=pd.read_csv(OUT/'source_diagnostics/maxrm_numerical_certificates.csv')
    evidence('maxrm_certificates','source_only','Повтор исходных отказов и эквивалентная центрированная задача с проверкой primal/dual','reports/closeout/source_diagnostics/maxrm_numerical_certificates.csv','all rows',cert.to_dict('records'),'Численная корректность source optimization; не target/ensemble guarantee; исходный неполный H2 не дополняется')
    for cid,path in [('protocol_v1','runs/final_v1/frozen_protocol.json'),('protocol_v2','runs/expanded_v2/frozen_protocol.json'),('v1_controls','reports/audit_v1/audit_report.md'),('v2_plan','docs/series_v2/research_plan.md'),('v2_prereg','docs/series_v2/preregistration.md'),('v2_math','docs/series_v2/mathematical_appendix.md'),('v2_findings','docs/series_v2/implementation_findings.md'),('v2_addendum','docs/series_v2/source_failure_addendum.md'),('literature','literature/sources.json')]:
        evidence(cid,'protocol' if cid!='literature' else 'reference','Первичный документ / реестр',path,'whole document',{},'Читать вместе с closeout; методические ограничения сохраняются')
    pilot_path=OUT/'source_pilot/summary.json'
    pilot=load('reports/closeout/source_pilot/summary.json') if pilot_path.exists() else None
    pilot_text='Source-only пилот ещё не завершён; его промежуточные результаты не используются в выводах.'
    if pilot:
        selected=pd.read_csv(OUT/'source_pilot/selected_source_configurations.csv')
        selected.to_csv(OUT/'tables/source_pilot_selected.csv',index=False)
        evidence('source_pilot','source_only','Ограниченный пилот capacity и независимых механизмов настройки','reports/closeout/source_pilot/summary.json','whole JSON',pilot,'Один source draw на задачу и один seed; не равные бюджеты методов и не достижение максимального качества')
        evidence('source_pilot_configs','source_only','Source-selected настройки пилота','reports/closeout/source_pilot/selected_source_configurations.csv','all rows',selected.to_dict('records'),'Не тестировать эти решения на потреблённых targets как независимый финал')
        evidence('source_pilot_protocol','protocol','Дизайн source-only пилота зафиксирован до его fits','reports/closeout/source_pilot/source_protocol.json','whole JSON',{},'Локальная фиксация; новые target не генерировались')
        pilot_text=f"Завершён ограниченный пилот: {pilot['configurations']} configurations, {pilot['completed_source_folds']} успешных source folds, {pilot['failed_configurations']} failed configurations, target evaluations = 0. Два заранее описанных capacity profiles, отдельные CRF lambda1/lambda2, несколько сил регуляризации и strong baselines проверены на новых source draws, без генерации target. [source_pilot](claims_registry.json), [настройки](source_pilot/source_protocol.json), [source-selected таблица](tables/source_pilot_selected.csv). Это проверка работоспособности диапазонов, не демонстрация максимума каждого метода."

    # Scientific figures use saved aggregate tables, never new target predictions.
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    figures=[]
    def save(fig,name,title,table):
        fig.savefig(OUT/f'figures/{name}.png',dpi=170,bbox_inches='tight');fig.savefig(OUT/f'figures/{name}.svg',bbox_inches='tight');plt.close(fig)
        svg=OUT/f'figures/{name}.svg'
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8',newline='\n')
        figures.append(dict(file=f'figures/{name}.png',title=title,table=table))
    fig,axes=plt.subplots(1,2,figsize=(13,6))
    for ax,hyp in zip(axes,['H1','H2']):
        sub=h[h.hypothesis==hyp].reset_index(drop=True)
        for i,r in sub.iterrows():
            good=r.status=='complete';color='#a53131' if good and r.p_holm<.05 else '#386c91'
            ax.errorbar(r['mean'],i,xerr=[[r['mean']-r.ci_low],[r.ci_high-r['mean']]],fmt='o' if good else 'x',color=color,capsize=3)
        ax.set_yticks(range(len(sub)),[r.task[:3]+' '+LABEL[r.method] for _,r in sub.iterrows()]);ax.axvline(0,color='grey',lw=.8);ax.set_title(hyp+': '+('K16 − K4' if hyp=='H1' else 'imbalance 8 − 1'));ax.invert_yaxis();ax.set_xlabel('Difference of paired gain; marginal t 95% CI')
    fig.suptitle('Registered contrasts: red = Holm < .05 within family; x = incomplete')
    fig.tight_layout();save(fig,'primary','Предзаданные H1/H2: эффект, интервал и полнота','tables/primary_tests.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,column in zip(axes,['source_primary','target_primary']):
        for config,sub in irf.groupby('config'):
            ax.plot(sub.k,sub[column],marker='o',label='lambda '+['0','.05','.5'][int(config)])
        ax.set_xticks([4,8,16]);ax.set_title('Source heldout AUROC' if column.startswith('source') else 'Consumed target AUROC — diagnostic');ax.set_xlabel('K');ax.legend();ax.set_ylim(.2,1)
    fig.tight_layout();save(fig,'source_selection','IRF: source selection и target ranking','tables/irf_source_target.csv')
    scenarios=['iid','covariate','spurious','concept','heteroscedastic','missingness','label_prior']
    fig,axes=plt.subplots(1,2,figsize=(14,6))
    for ax,task in zip(axes,['classification','regression']):
        sub=shifts[shifts.task==task]
        for method,g in sub.groupby('method'):
            g=g.set_index('scenario').reindex(scenarios);ax.plot(range(len(scenarios)),g['mean'],marker='o',label=LABEL[method])
        ax.set_xticks(range(len(scenarios)),scenarios,rotation=35,ha='right');ax.axhline(0,color='grey',lw=.8);ax.set_title(task);ax.set_ylabel('Delta AUROC' if task=='classification' else '(control MSE − method MSE) / source variance');ax.legend(fontsize=8)
    fig.suptitle('Shift screening, K8 N8192 balanced: descriptive means');fig.tight_layout();save(fig,'shift_types','Зависимость наблюдаемого эффекта от механизма сдвига','tables/shift_effects.csv')
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    for ax,(dataset,sub) in zip(axes.flat,real.groupby('dataset',sort=True)):
        column='roc_auc' if sub.iloc[0].task=='classification' else 'rmse';sub=sub.sort_values(column,ascending=column=='rmse')
        ax.barh([LABEL[x] for x in sub.method],sub[column],color='#386c91');ax.set_title(dataset);ax.set_xlabel('Target AUROC (higher better)' if column=='roc_auc' else 'Target RMSE (lower better)');ax.invert_yaxis()
        if column=='roc_auc':ax.set_xlim(.65,.9)
    fig.suptitle('Real tasks: absolute quality, no population inference or cross-task pooling');fig.tight_layout();save(fig,'real_tasks','Реальные задачи: абсолютное качество выбранных pipelines','tables/real_results_v2.csv')
    cert_summary=cert.groupby(['diagnostic','status']).agg(cases=('job_id','size'),max_gap=('gap','max'),max_violation=('epigraph_violation','max')).reset_index()
    cert_summary.to_csv(OUT/'tables/maxrm_solver_summary.csv',index=False)
    claims.extend([
        dict(id='forbidden_universal',category='forbidden',statement='DG универсально превосходит обычные ансамбли; каждый метод достиг максимума',reason='Нет такого дизайна и универсального превосходства; capacity и HPO ограничены'),
        dict(id='forbidden_replication',category='forbidden',statement='Все независимые реализации — точные репликации авторских методов',reason='Meta — интерпретация; IRF/CRF/MaxRM — независимые реализации; использование авторского пакета не доказывает репликацию авторского экспериментального протокола'),
        dict(id='forbidden_reuse',category='forbidden',statement='Улучшенный по итогам v1/v2 алгоритм независимо подтверждён на тех же targets',reason='Все открытые targets consumed; исправления и HPO — новые решения'),
        dict(id='forbidden_null',category='forbidden',statement='Неотклонение H2 доказывает отсутствие эффекта или эквивалентность',reason='Не проводился equivalence test; MaxRM неполон; конечная мощность'),
        dict(id='forbidden_pure_K',category='forbidden',statement='Число source-доменов само по себе причинно ухудшает все IRF/Meta',reason='Меняются размер домена, grid механизма, fraction validation, selection, у Meta общий ensemble budget'),
        dict(id='forbidden_noise',category='forbidden',statement='Source minimax certificate гарантирует worst target risk ансамбля',reason='Сертификат относится к отдельному дереву, bootstrap source empirical risk'),
    ])
    write(OUT/'claims_registry.json',dict(scope='v1 exploratory; v2 confirmatory only predeclared complete tests; source corrections not target evidence',claims=claims))
    pd.DataFrame([{k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in c.items()} for c in claims]).to_csv(OUT/'claims_registry.csv',index=False)
    centered=cert[cert.diagnostic=='clarabel_centered']
    narrative=f'''# Итоговое экспериментально-аналитическое обобщение НИР

Методы обучения ансамблей деревьев решений, устойчивых к сдвигу распределения табличных данных. Заключительный аудит и совместная интерпретация final_v1 / expanded_v2. Исходные результаты сохранены; новые target-эксперименты не проводились.

## 1. Научный итог и готовность

Экспериментальная часть пригодна для отчёта НИР с явно указанными ограничениями. Её результат — проверенные положительные, отрицательные и нулевые наблюдения о конкретных реализациях и source-selected pipelines. Обязательной новой масштабной серии для достоверного отчёта не требуется. Исправленные версии нельзя объявлять подтверждёнными на потреблённых targets.

Вторая серия содержит {inv['jobs']} заданий: {inv['complete_jobs']} полных и {inv['partial_jobs']} с отсутствующим MaxRM; {inv['full_source_models']} full-source configurations и {inv['target_domain_metric_rows']} строк target-метрик. Из предзаданных основных тестов завершены {inv['complete_confirmatory_tests']}; план содержал {inv['confirmatory_tests']}. [inventory](claims_registry.json).

Аудит второй серии независимо пересчитал {audit['independent_metric_checks']} значений метрик, проверил {audit['saved_model_replays']} model probes и {audit['repeated_source_fits']} повторных source fits. Максимальное расхождение метрик {audit['max_metric_absolute_difference']:.3g}; model prediction probes совпали. Это проверка отдельными уравнениями тем же авторским агентом, без внешней рецензии. Локальные hashes и receipts не доказывают отсутствие любого исторического ручного доступа. [integrity_v2](claims_registry.json), [аудит v1](../audit_v1/audit_report.md).

## 2. Дизайн, единица наблюдения и метрики

V1 — exploratory сравнение: результаты сохранённых predictions верны, но часть пар смешивает механизм, capacity и допустимость splits. V2 заранее зафиксировала новые targets, grids, source folds, общий target draw для factorial contrasts и отдельные парные контроли. Полные спецификации, включая объёмы, seeds, правила folds и hashes: [протокол v1](../../runs/final_v1/frozen_protocol.json), [протокол v2](../../runs/expanded_v2/frozen_protocol.json), [preregistration](../../docs/series_v2/preregistration.md), [математическое приложение](../../docs/series_v2/mathematical_appendix.md). Числа протокола следует брать из этих документов, а не реконструировать по агрегатам.

Preprocessing и обучение происходят на train части source; гиперпараметры выбираются по source-heldout domains. Для regression нормирование y задаёт train source; опубликованные RMSE/MSE находятся в исходных единицах. Target не участвует в выборе. Полные source/target таблицы раздельны: [source CV](../series_v2/tables/source_cv_metrics.csv), [target metrics](../series_v2/tables/target_metrics.csv).

Классификационное преимущество = AUROC метода минус AUROC собственного контроля. Regression advantage = (MSE контроля − MSE метода) / дисперсия полного source y. Сначала равновесно усредняются target domains, затем model seeds, затем вычисляется paired contrast по независимым synthetic data replicas. Нельзя заменить разность MSE разностью RMSE или объединять сырые RMSE разных задач. [protocol_v2](claims_registry.json).

Для H1/H2 exact two-sided sign test использует знаки парных replica contrasts, исключая нулевые; это не тест нулевого математического ожидания. Holm применяется отдельно внутри каждой предзаданной семьи, без общей гарантии FWER для объединения семей. На графиках показаны маргинальные t интервалы среднего по data replicas; они не являются simultaneous CI и используют нормальную аппроксимацию. Реальные source resamples перекрываются, поэтому seeds/resamples не превращаются в независимые datasets. [protocol_v2](claims_registry.json), [полная таблица тестов](tables/primary_tests.csv).

## 3. Заключительный технический аудит

IRF / Appliances. Причина — округление midpoint почти соседних floating-point значений до правой границы: фактическая partition может отличаться от оценённой sorted position. В сохранённых моделях выявлены {len(bad)} затронутых configurations в {bad.job_id.nunique()} заданиях; вместе с нарушениями domain support присутствует недостижимый на сохранённых rows пустой leaf с NaN. Observed predictions конечны, и их метрики корректны; это не доказывает корректности дерева на произвольном будущем input. [irf_defect](claims_registry.json), [точные job/config IDs](tables/affected_irf_models.csv).

Дополнительный source-CV diagnostic также выявил leaf ниже зарегистрированного minimum; это отдельный промежуточный fit, а не дополнительная final модель. Основные synthetic H1/H2 не затронуты найденными full-source Appliances нарушениями. Не следует распространять этот факт на все возможные intermediate trees без проверки. [implementation findings](../../docs/series_v2/implementation_findings.md), [source diagnostic](../series_v2/source_numeric_diagnostic.json).

В отдельной prospective версии threshold гарантирует lower ≤ threshold < upper; перед рекурсией проверяются фактические размеры children и domain support. На всех source задачах, где найдены final дефекты, проверены {diag['irf_and_null']['corrected_irf_fits']} новых fits с разными штрафами; инварианты выполнены, bootstrap внутри пар совпадает. Это engineering correction, а не новый target результат. [source_corrections](claims_registry.json), [проверки](source_diagnostics/corrected_irf_source_checks.csv).

Beijing. CSV np.nan и Parquet None в object-колонке wd приводили к различным transformed matrices. Cache creator и последующие jobs получили разное представление пропуска, хотя semantic dataframe hash его не различал. Исходные predictions cache creator воспроизводятся после восстановления исходного np.nan; labels и row IDs не изменены. Внутри задания все методы видели одинаковую матрицу, но cross-job сравнения имеют input qualification. [null_defect](claims_registry.json), [чувствительность по моделям и доменам](tables/beijing_cache_sensitivity.csv).

Prospective preprocessing канонизирует missing scalars перед fit/transform; source cache roundtrip даёт точное совпадение transformed features. Для будущего исполнения следует сохранять также SHA transformed matrix, column order и dtype. Исправление необходимо для нового стенда; переписывать исходные Beijing metrics не нужно. [source_corrections](claims_registry.json).

MaxRM. Все {diag['maxrm']['source_failure_cases']} исходных source отказов воспроизведены на тех же bootstrap деревьях. Проблема относится к численной записи conic задачи и правилу принимать только OPTIMAL; она не доказывает отсутствия математического решения. Ужесточение solver tolerances само по себе статусы не исправило. Алгебраически эквивалентная centered запись получила OPTIMAL во всех этих случаях; независимый SCS согласовал objective. Максимальный centered primal-dual gap {centered.gap.max():.3g}, нарушение epigraph {centered.epigraph_violation.max():.3g}. [maxrm_certificates](claims_registry.json), [сводная таблица](tables/maxrm_solver_summary.csv).

Сертификат вычислен по прямым squared residuals и независимой dual lower bound, а не только по статусу solver. Prospective adapter принимает OPTIMAL и проверяет конечность, feasibility, gap и отсутствие ухудшения исходного feasible дерева. Исторические отказы сохранены: MaxRM H2 остаётся неполным, его недостающий результат нельзя незаметно дописать новой реализацией. Для engineering диагностики новые данные не требовались; для утверждения о качестве исправленного pipeline нужны новые, заранее зафиксированные holdouts. [source_corrections](claims_registry.json), [исходное дополнение о завершении методов](../../docs/series_v2/source_failure_addendum.md).

## 4. Подтверждённые статистические результаты

В предзаданном spurious reversal увеличение числа source domains уменьшило преимущество source-selected IRF classification: средний contrast −0.342194, маргинальный интервал [−0.438199; −0.246189], Holm p=0.03125. Аналогичный contrast Meta weights classification −0.008859, интервал [−0.012550; −0.005169], Holm p=0.017578. Подтверждение относится к sign-test процедуре внутри семьи H1, этому DGP и этим pipelines. [H1_classification_irf](claims_registry.json), [H1_classification_meta](claims_registry.json).

Остальные завершённые H1/H2 тесты не достигли принятого порога после Holm. Для H2 MaxRM полный contrast отсутствует; complete-case наблюдение нельзя выдать за предзаданный завершённый тест. Консервативный учёт отсутствующего теста сохраняет исходную ширину семьи. Отсутствие отклонения не доказывает эквивалентность или нулевой эффект. [основные тесты](tables/primary_tests.csv), [protocol_v2](claims_registry.json).

![Предзаданные контрасты](figures/primary.png)

## 5. Описательные наблюдения и отрицательные результаты

Тип сдвига существенно меняет картину. В spurious DGP target разворачивает source зависимость нерелевантного признака от y: отдельные механизмы полезны, но сильный source score может выбирать shortcut. IID screening не показывает универсальной необходимости DG. При covariate shift важны support/overlap и качество аппроксимации; при concept shift меняется сам условный механизм, поэтому существование общей invariant функции нельзя предполагать автоматически. Missingness — measurement shift, а не чистый covariate shift. Label-prior сценарий отделён от прочих. Утверждения о различиях опираются на предзаданные DGP и descriptive screening; общего теста превосходства на всех сдвигах здесь нет. [shift screening](tables/shift_effects.csv), [v2_plan](claims_registry.json).

![Механизмы сдвига](figures/shift_types.png)

Количество доменов. H1 проверяет конкретный factorial pipeline contrast. При фиксированном общем N меняются размер каждого source domain, доля train/heldout в leave-domain-out CV и плотность сетки source механизмов. У Meta с ростом K дополнительно меняется число source forests при фиксированной per-source capacity. Поэтому чистая причинная формулировка «большее K ухудшает метод независимо от остальных условий» не обоснована. [protocol_v2](claims_registry.json), [v2_math](claims_registry.json).

Объём данных. Secondary N contrasts содержат эффекты обоих знаков; более крупный source не гарантирует большего преимущества DG относительно его контроля. Эти интервалы полезны для гипотез, но не защищены от множественного выбора вторичных наблюдений. V1 и V2 нельзя использовать как парную оценку эффекта увеличения N: между сериями менялись targets, capacity, preprocessing и выбор конфигураций. [N contrasts](tables/sample_size_contrasts.csv), [протокол v1](../../runs/final_v1/frozen_protocol.json), [протокол v2](../../runs/expanded_v2/frozen_protocol.json).

Дисбаланс. Проверялось неравенство размеров source domains, его нельзя автоматически отождествлять с class imbalance. Неполучение Holm-significant H2 не подтверждает инвариантность методов к дисбалансу. Minimax и доменно-стратифицированные механизмы по-разному взвешивают малые группы; конечный source risk и изменяющаяся variance estimation требуют отдельного изучения. [H2 строки](tables/primary_tests.csv), [математическое приложение](../../docs/series_v2/mathematical_appendix.md).

Source-only HPO. Для IRF при большом K source предпочитает нулевой штраф, тогда как заранее зарегистрированные ненулевые точки лучше на открытом reversal target. Это объясняет часть отрицательного pipeline contrast, но не даёт права выбрать lambda по target и повторно назвать его финальным тестом. Retrospective oracle gap — диагностическая верхняя граница внутри ограниченного grid, а не deployable score или максимум всего метода. [selection_irf](claims_registry.json), [частоты выбора](tables/irf_source_target.csv).

![Source и target IRF](figures/source_selection.png)

Реальные задачи не дают универсального победителя DG. Нужно одновременно показывать абсолютное качество сильных RF/ExtraTrees/HistGBDT и преимущество относительно механистического контроля. Улучшение слабого контроля не означает победы над сильным baseline. Beijing решает contemporaneous PM2.5 modelling, а не forecasting; Appliances переносит модель на будущее одного дома, а не на новые дома; ACS tasks принадлежат общей survey family и используют невзвешенные rows. [реальные v2](tables/real_results_v2.csv), [исторические v1](tables/real_results_v1.csv), [v2_plan](claims_registry.json).

Первая серия уже дала существенные отрицательные наблюдения. ACSIncome: IRF AUROC 0.726347 против 0.821473 собственного контроля и 0.861032 HistGBDT. Hospital: Meta interpretation 0.566324 против RF 0.623960. Beijing: Era RMSE 29.033716 против 27.976706 собственного restricted контроля. Gas turbine: MaxRM RMSE 1.415941 против 1.420274 контроля, однако различия выбранной глубины ограничивают механистическое объяснение этой небольшой разницы. Это descriptive результаты исторических pipelines, не статистическое опровержение авторских методов; seed SD не отражает неопределённость по новым задачам. [v1 таблица с точными method keys и evidence IDs](tables/real_results_v1.csv), [v1_controls](claims_registry.json).

![Абсолютное качество на реальных задачах](figures/real_tasks.png)

## 6. Реализации и сопоставимость контролей

В v1 MaxRM и RF часть пар выбрала разную глубину; IRF regression меняла admissibility вместе со штрафом; Meta matched RF совпадал лишь по tree budget. Era zero-strength сохраняет domain-dependent split admissibility и не равен обычному sklearn HistGBDT. Эти числа можно описывать как сравнение pipelines, но не как чистый эффект единственного механизма. [v1_controls](claims_registry.json).

В v2 structural parameters и RNG внутри основных пар зафиксированы; Meta uniform/accuracy/full используют тот же список обученных members, masking выделен отдельно. Era сравнивается с domain-restricted pooled gain, рядом показан обычный HistGBDT. Пары IRF Appliances ограничены найденным numeric defect; источники остальных оговорок указаны выше. [v2_math](claims_registry.json), [v2_findings](claims_registry.json).

IRF, CRF и MaxRM здесь независимые реализации; Meta — явно ограниченная интерпретация недоопределённого описания. Использование авторских Era/Anchor packages не означает точной репликации авторского полного экспериментального исследования. Отрицательные результаты этих версий не опровергают статьи в целом. Ограниченная threshold grid, небольшие леса, неэквивалентные вычислительные бюджеты и source-only tuning входят в область вывода.

## 7. Настройка каждого метода: выполненное и следующий корректный дизайн

{pilot_text}

Более глубокая настройка научно оправдана, если вопрос звучит «какое качество получает практический source-selected pipeline при заданном бюджете». Механистический вопрос требует другого сравнения: одинаковые bootstrap/структурные настройки и candidate admissibility, различается только исследуемый механизм. Нужно сохранять оба трека, поскольку независимо настроенные лучшие pipelines не изолируют действие регуляризатора.

Для будущего best-pipeline исследования следует заранее описать spaces, source-CV folds, budget по времени/числу fits, early stopping, seeds, tie-break и failure policy; дать сопоставимый ресурс strong baselines. Исследовать capacity, размер leaf, threshold resolution, strength; у CRF разделить коэффициенты, у Meta различать weighting и feature policy. Численная точность solver — engineering constraint, не гиперпараметр для улучшения target score. Source CV само может переобучаться при широком поиске: полезны nested domain validation и отдельные development domains, когда их достаточно. Малое число folds не следует компенсировать бесконечным HPO.

Исследовательские гипотезы для следующего независимого этапа: эффект зависит от информации в малых source domains; source-heldout ranking fails, когда target выходит за диапазон observed mechanisms; capacity усиливает shortcuts при некоторых shifts; minimax может концентрироваться на высокошумной source среде. Это гипотезы, не доказанные данным closeout. Новый тест должен разделить общий N и per-domain N, сохранить общий tree budget в механистическом треке и заранее определить interaction contrasts. Массовую серию сейчас запускать не требуется.

## 8. Материалы, оставшиеся проблемы и предварительное заключение

Готовы: единое обобщение, публикационные графики PNG/SVG с CSV backing, основные статистические таблицы, реестр утверждений с точными artifact selectors и SHA-256, раздельные historical v1/v2 результаты, source-only engineering corrections и план дальнейшей настройки. Исходные frozen protocols, code, predictions и результаты не переписаны. [реестр](claims_registry.json), [индекс доказательств](evidence_manifest.json).

Существенные оставшиеся ограничения: отсутствующий MaxRM H2; numeric qualification IRF Appliances; cross-job null representation Beijing; fidelity независимых реализаций; ограниченный HPO/compute budget; отсутствие OS-level независимого доступа/внешней временной подписи; узкая выборка реальных задач и зависимые resamples. Библиографический реестр отдельно фиксирует доступность оригинальных PDF; недоступный полный текст нельзя выдавать за архивированный. [v2_findings](claims_registry.json), [литература](../../literature/sources.json), [статус архива](../../docs/literature_audit.md).

Рекомендация: завершать научный отчёт на существующем корпусе с этими оговорками. Точечные engineering проверки выполнены без новых target. Если требуется утверждение о качестве исправленного IRF, canonical Beijing pipeline или более тщательно настроенных методов, нужен новый заранее зафиксированный тест; для отчёта об исходных результатах это не обязательное условие. Статистически корректно оставить отрицательные и неполные результаты, а не добиваться победы каждого метода.

Предварительное заключение НИР: устойчивость ансамблей к распределительному сдвигу определяется взаимодействием механизма, структуры source domains и процедуры выбора модели. Изолированный выигрыш относительно собственного контроля, высокое абсолютное качество и воспроизводимость — различные свойства. Исследование подтверждает отдельные предзаданные контрасты, показывает существенные ограничения source-only выбора и задаёт воспроизводимую основу для следующего независимого этапа без универсальных обещаний OOD устойчивости.
'''
    (OUT/'research_synthesis.md').write_text(narrative,encoding='utf-8',newline='\n')
    readiness={'status':'ready_for_NIR_report_with_explicit_qualifications','historical_results_replaced':False,'new_target_evaluations':0,'mandatory_new_large_series':False,'remaining_substantive_issues':['MaxRM H2 incomplete','IRF Appliances historical partition defects','Beijing cache input representation across jobs','Independent implementations not exact paper replications','Limited source HPO and compute fairness','No trusted OS-level historical access attestation','Dependent real resamples and narrow task population'],'completed_source_engineering':diag,'source_pilot':pilot,'recommendation':'Use existing results with limitations; only claims about corrected/newly tuned pipelines require preregistered fresh holdout.'}
    write(OUT/'readiness.json',readiness)
    # Linkable static reader: no network dependencies, tables derived from cited CSVs.
    def inline(s):
        s=escape(s);s=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'<a href="\2">\1</a>',s);return s
    body=[]
    for line in narrative.splitlines():
        if not line.strip():continue
        image=re.fullmatch(r'!\[([^\]]+)\]\(([^)]+)\)',line)
        if image:body.append(f'<figure><img src="{image[2]}" alt="{image[1]}"><figcaption>{image[1]}</figcaption></figure>')
        elif line.startswith('# '):body.append('<h1>'+inline(line[2:])+'</h1>')
        elif line.startswith('## '):body.append('<h2>'+inline(line[3:])+'</h2>')
        else:body.append('<p>'+inline(line)+'</p>')
    for title,name,frame in [('Основные тесты','primary_tests.csv',h),('Реальные задачи v2','real_results_v2.csv',real[['dataset','method','k','n','roc_auc','rmse','control_roc_auc','control_rmse','evidence_id']]),('Вторичные N contrasts','sample_size_contrasts.csv',sec),('MaxRM численные проверки','maxrm_solver_summary.csv',cert_summary)]:
        body.append(f'<h2>{title}</h2><p><a href="tables/{name}">CSV</a></p><div class="table">'+frame.to_html(index=False,float_format=lambda v:f'{v:.6g}',na_rep='—')+'</div>')
    (OUT/'index.html').write_text('<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>НИР: итоговое обобщение</title><style>body{font:17px/1.65 system-ui;background:#f3f5f6;color:#172a39;margin:0}main{max-width:1120px;margin:auto;background:white;padding:40px}h1,h2{line-height:1.25}h2{margin-top:44px;color:#245a81}a{color:#176397}img{width:100%;height:auto}.table{overflow:auto}table{border-collapse:collapse;font-size:13px}td,th{padding:9px;border:1px solid #ddd;text-align:left}tr:nth-child(even){background:#f3f6f8}figure{margin:30px 0}figcaption{font-size:14px;color:#567}p{overflow-wrap:anywhere}@media(max-width:700px){main{padding:18px}}</style><main><p><a href="research_synthesis.pdf">PDF</a> · <a href="claims_registry.json">Реестр утверждений</a> · <a href="readiness.json">Готовность</a></p>'+''.join(body)+'</main></html>',encoding='utf-8',newline='\n')
    write(OUT/'evidence_manifest.json',dict(generated_at=datetime.now(timezone.utc).isoformat(),source_artifacts_sha256=inputs,derived_files_sha256={p.relative_to(ROOT).as_posix():sha256(p.read_bytes()).hexdigest() for p in sorted((OUT/'tables').glob('*.csv'))},figures=figures,generator_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),historical_results_modified=False,new_target_evaluations=0))
    print(json.dumps(dict(claims=len(claims),tables=len(list((OUT/'tables').glob('*.csv'))),figures=len(figures),source_pilot_complete=bool(pilot))))


if __name__=='__main__':
    main()
