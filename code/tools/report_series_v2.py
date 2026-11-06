"""Figures and a self-contained offline research reader from computed tables."""
import html
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analyze_series_v2 import OUT,TABLES,DIMS,ROBUST,interval

NAMES={"irf":"IRF v2","crf":"CRF","meta":"Meta weights","meta_masked":"Meta + masking","maxrm":"MaxRM posthoc","era":"Era","directional_era":"Directional Era","anchor":"Anchor","rf":"RF 200","extra_trees":"ExtraTrees 200","hist_gbdt":"HistGBDT 200"}
SCENARIOS=["iid","covariate","spurious","concept","heteroscedastic","missingness","label_prior"]
DATASETS={"acs_income":"ACSIncome","acs_public_coverage":"ACSPublicCoverage","beijing_pm25":"Beijing PM2.5","appliances":"Appliances (один дом)"}
FIG=OUT/"figures"


def save(fig,name):
    fig.savefig(FIG/f"{name}.png",dpi=180,bbox_inches="tight")
    fig.savefig(FIG/f"{name}.svg",bbox_inches="tight")
    plt.close(fig)


def figures(a,effects,summary):
    FIG.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False,"figure.facecolor":"white"})
    screen=effects[(effects.kind=="synthetic")&(effects.k==8)&(effects.n==8192)&(effects.imbalance==1)]
    fig,axes=plt.subplots(1,2,figsize=(14,4.5),layout="constrained")
    for ax,task in zip(axes,["classification","regression"]):
        f=screen[screen.task==task];cols=[c for c in SCENARIOS if c in f.scenario.unique()]
        methods=[m for m in ROBUST if m in f.method.unique()]
        mat=f.pivot(index="method",columns="scenario",values="mean").reindex(index=methods,columns=cols)
        limit=max(float(np.max(np.abs(mat.to_numpy()))),1e-4)
        image=ax.imshow(mat,cmap="RdBu",vmin=-limit,vmax=limit,aspect="auto")
        ax.set_xticks(range(len(cols)),cols,rotation=35,ha="right");ax.set_yticks(range(len(methods)),[NAMES[m] for m in methods])
        for i in range(len(methods)):
            for j in range(len(cols)):
                value=mat.iloc[i,j];ax.text(j,i,f"{value:.3f}",ha="center",va="center",fontsize=10,color="white" if abs(value)>.65*limit else "black")
        ax.set_title(f"{task}: source-selected gain vs own control")
        ax.set_xlabel("Registered synthetic shift");ax.set_ylabel("Method")
        fig.colorbar(image,ax=ax,label="AUROC gain" if task=="classification" else "Source-normalized MSE reduction",shrink=.7)
    fig.suptitle("K=8, N=8192, balanced · 10 data replicas · 2 model seeds · 3 target laws")
    save(fig,"mechanism_screen")
    fig,axes=plt.subplots(2,2,figsize=(12,9),layout="constrained")
    for row,hypothesis in enumerate(["H1","H2"]):
        for col,task in enumerate(["classification","regression"]):
            ax=axes[row,col];f=a["hypotheses"][(a["hypotheses"].hypothesis==hypothesis)&(a["hypotheses"].task==task)].copy()
            positions=np.arange(len(f));mean=f["mean"].to_numpy()
            ax.errorbar(mean,positions,xerr=[mean-f.ci_low.to_numpy(),f.ci_high.to_numpy()-mean],fmt="o",capsize=3,color="#245a81")
            ax.axvline(0,color="black",lw=.8);ax.set_yticks(positions,[f"{NAMES[m]} ("+("incomplete; descriptive CI" if pd.isna(p) else f"pH={p:.3g}")+")" for m,p in zip(f.method,f.p_holm)])
            ax.set_title(f"{hypothesis}: {'K16 minus K4' if hypothesis=='H1' else '8:1 minus balanced'} · {task}")
            ax.set_xlabel("Difference in AUROC gain" if task=="classification" else "Difference in normalized MSE reduction")
            ax.set_ylabel("Method · Holm p across all 9 tests in family")
    fig.suptitle("Spurious shift · N=8192 · 10 planned replicas · incomplete test is descriptive only")
    save(fig,"confirmatory_hypotheses")
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout="constrained")
    for row,scenario in enumerate(["spurious","heteroscedastic"]):
        for col,task in enumerate(["classification","regression"]):
            ax=axes[row,col];f=effects[(effects.kind=="synthetic")&(effects.scenario==scenario)&(effects.task==task)&(effects.n==8192)&(effects.imbalance==1)]
            for method,g in f.groupby("method",sort=False):
                g=g.sort_values("k");ax.plot(g.k,g["mean"],marker="o",label=NAMES[method]+f" (n={int(g.replicas.min())}–{int(g.replicas.max())})");ax.fill_between(g.k.to_numpy(),g.ci_low.to_numpy(),g.ci_high.to_numpy(),alpha=.10)
            ax.axhline(0,color="black",lw=.8);ax.set_xticks([4,8,16]);ax.set_xlabel("Number of source domains K (fixed total N)")
            ax.set_ylabel("AUROC gain" if task=="classification" else "Normalized MSE reduction");ax.set_title(f"{scenario} · {task}")
            ax.legend(fontsize=10,ncol=2)
    fig.suptitle("K effect · complete seed pairs only · 10 planned replicas · marginal 95% CI")
    save(fig,"source_domain_count")
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout="constrained")
    for ax,(dataset,label) in zip(axes.ravel(),DATASETS.items()):
        f=summary[(summary.dataset==dataset)&(summary.k==(8 if dataset!="beijing_pm25" else 6))]
        metric="roc_auc" if dataset.startswith("acs_") else "rmse"
        for method,g in f.groupby("method",sort=False):
            g=g.sort_values("n");ax.plot(g.n,g[metric],marker="o",label=NAMES[method],linewidth=1)
        ax.set_title(label);ax.set_xlabel("Total source training observations N");ax.set_ylabel("Macro target AUROC (higher better)" if metric=="roc_auc" else ("Macro target RMSE (µg/m³)" if dataset=="beijing_pm25" else "Macro target RMSE (Wh / 10-min interval)"))
        ax.legend(fontsize=9,ncol=2)
    fig.suptitle("New real targets · 3 overlapping source resamples × 2 seeds · descriptive means")
    save(fig,"real_training_size")
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout="constrained")
    diagnostics=a["selection_diagnostics"]
    for ax,task in zip(axes,["classification","regression"]):
        f=diagnostics[(diagnostics.kind=="synthetic")&(diagnostics.task==task)&(diagnostics.k==8)&(diagnostics.n==8192)&(diagnostics.imbalance==1)&(diagnostics.method.isin(ROBUST))]
        g=f.groupby("method").retrospective_oracle_regret.mean().sort_values()
        ax.barh([NAMES[m] for m in g.index],g.to_numpy(),color="#245a81")
        ax.set_title(f"{task}: registered-grid oracle gap");ax.set_xlabel("Mean AUROC gap" if task=="classification" else "Mean normalized MSE gap");ax.set_ylabel("Source-selected method")
    fig.suptitle("Retrospective diagnostic only · best target grid point is not deployable selection")
    save(fig,"selection_oracle_gap")
    fig,axes=plt.subplots(1,2,figsize=(14,4),layout="constrained")
    c=a["curves"]
    for ax,family in zip(axes,["meta","meta_masked"]):
        f=c[(c.kind=="synthetic")&(c.method==family)&(c.k==8)&(c.n==8192)&(c.imbalance==1)]
        g=f.groupby(["scenario","config"]).roc_auc.mean().unstack().reindex(SCENARIOS)
        for ci,label in enumerate(["Uniform same members","Accuracy weights","Accuracy + MMD"]):
            ax.plot(range(len(SCENARIOS)),g[ci],marker="o",label=label)
        ax.set_xticks(range(len(SCENARIOS)),SCENARIOS,rotation=35,ha="right");ax.set_ylabel("Macro target AUROC");ax.set_xlabel("Registered shift");ax.set_title("Same members · "+("masking" if family=="meta_masked" else "no masking"));ax.legend(fontsize=10)
    fig.suptitle("Meta ablations · all fixed configurations · 10 replicas × 2 seeds")
    save(fig,"meta_same_members")
    f=a["macro"]
    f=f[(f.kind=="synthetic")&(f.task=="regression")&(f.scenario=="heteroscedastic")&(f.k==8)&(f.n==8192)&(f.imbalance==1)&(f.method.isin(ROBUST))]
    fig,ax=plt.subplots(figsize=(8,4),layout="constrained")
    g=f.groupby(["method","replica"]).worst_gain.mean().reset_index()
    for i,(method,sub) in enumerate(g.groupby("method")):
        ci=interval(sub.worst_gain);ax.errorbar(ci["mean"],i,xerr=[[ci["mean"]-ci["ci_low"]],[ci["ci_high"]-ci["mean"]]],fmt="o",color="#245a81",capsize=3)
    methods=sorted(g.method.unique());ax.set_yticks(range(len(methods)),[NAMES[m] for m in methods]);ax.axvline(0,color="black",lw=.8)
    ax.set_xlabel("Reduction of worst target MSE / source variance");ax.set_ylabel("Source-selected method");ax.set_title("Heteroscedastic regression · K8,N8192 · marginal 95% t CI")
    save(fig,"worst_target_noise")
    for factor,condition,values,label,filename in [("n","imbalance",[2048,8192],"Total source observations N","synthetic_training_size"),("imbalance","n",[1,8],"Largest/smallest source-domain size ratio","source_imbalance")]:
        fig,axes=plt.subplots(2,2,figsize=(12,8),layout="constrained")
        for row,scenario in enumerate(["spurious","heteroscedastic"]):
            for col,task in enumerate(["classification","regression"]):
                ax=axes[row,col];f=effects[(effects.kind=="synthetic")&(effects.scenario==scenario)&(effects.task==task)&(effects.k==8)]
                f=f[f[condition]==(1 if condition=="imbalance" else 8192)]
                for method,g in f.groupby("method",sort=False):
                    g=g.sort_values(factor);ax.plot(g[factor],g["mean"],marker="o",label=NAMES[method]);ax.fill_between(g[factor].to_numpy(),g.ci_low.to_numpy(),g.ci_high.to_numpy(),alpha=.10)
                ax.axhline(0,color="black",lw=.8);ax.set_xticks(values);ax.set_xlabel(label);ax.set_ylabel("AUROC gain" if task=="classification" else "Normalized MSE reduction");ax.set_title(f"{scenario} · {task}");ax.legend(fontsize=10,ncol=2)
        fig.suptitle("Fixed K=8 · complete seed pairs only · 10 planned replicas · marginal 95% CI")
        save(fig,filename)
    grid=a["configuration_diagnostics"]
    grid=grid[(grid.task=="classification")&(grid.method=="irf")&(grid.scenario=="spurious")&(grid.n==8192)&(grid.imbalance==1)]
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout="constrained")
    for ax,column,title in zip(axes,["source_primary","target_primary"],["Source validation: configuration selection","New target: predeclared diagnostic points"]):
        for k,g in grid.groupby("k"):
            means=g.groupby("config")[column].mean()
            ax.plot(means.index,means.to_numpy(),marker="o",label=f"K={k}")
        ax.set_xticks([0,1,2],["0","0.05","0.5"]);ax.set_xlabel("IRF penalty λ (fixed registered grid)");ax.set_ylabel("Macro AUROC");ax.set_title(title);ax.legend()
    fig.suptitle("IRF · spurious reversal · N8192 · source/target rankings disagree · no target selection")
    save(fig,"irf_selection_failure")


def write_markdown(a,effects,summary,counts,frozen,audit):
    p=OUT/"research_report.md"
    lines=["# Расширенное исследование Domain Generalization для ансамблей деревьев",f"\nСерия expanded_v2. Завершение анализа: {counts['analyzed_at']}. Протокол SHA-256 `{frozen['sha256']}`. Code commit `{frozen['protocol']['source_code_commit']}`.","\n## 1. Что проверено",f"\nФактически завершены {counts['jobs']} заданий, {counts['ordinary_source_fits']:,} обычных source fits и {counts['repeated_source_fits']} повторов воспроизводимости. Сохранены {counts['full_source_models']:,} full-source конфигурации, {counts['target_domain_metric_rows']:,} per-target-domain строк и {counts['source_validation_rows']:,} source-validation строк. Все решения source-only зафиксированы до target соответствующего задания. Внешняя пререгистрация опубликована в commit b0325a5 до первого нового target-доступа.","\nПервая серия проверена отдельными уравнениями и артефактами: все 1008 метрик, 126 моделей и 42 source fits подтвердились. Выявлены ограничения исторических пар MaxRM/Era/IRF/Meta. Полный разбор — [аудит v1](../audit_v1/audit_report.md). Это независимый способ проверки тем же авторским агентом, **не внешнее рецензирование другим исследователем**.",f"\nИтоговый аудит v2: статус **{audit['status']}**, независимо проверены {audit['independent_metric_checks']:,} значений метрик и predictions {audit['saved_model_replays']:,} сохранённых конфигураций. Максимальное абсолютное расхождение метрик {audit['max_metric_absolute_difference']:.3g}, predictions {audit['max_model_prediction_difference']:.3g}. Проверены выбор конфигураций, пары, hashes, row IDs и последовательность source→receipt→target. [Машинные свидетельства](audit/final_integrity.json).", "\n## 2. Дизайн и область применимости", "\nСинтетика: 33 условия ×10 независимых data replicas ×2 model seeds, семь типов сдвига (в regression нет pure label-prior). Основной screen K8,N8192; парные контрасты K4/K16 при одном N, N2048/N8192 при одном K, дисбаланс1/8 при тех же K/N. Target draws совпадают внутри предзаданных сравнений; независимая единица — data replica, не seed и не severity.","\nРеальные задачи: ACSIncome/PublicCoverage N4000/16000,K4/8, Beijing N2400/9600,K4/6, Appliances N2000/4000,K4/8 плюс N8000,K8. Три source resamples пересекаются и не являются независимыми real datasets. Новые targets: MN/OH/PA/OR/VA, четыре ранее не открытые Beijing stations и апрель/май одного дома. Старые targets v1 не использованы как новый независимый тест.","\nПары разделяют исследуемый механизм и обучающую capacity: у Meta одни и те же trained members; у MaxRM те же bootstrap/ветви, только иные листья; у IRF/CRF одинаковы admissibility/RNG; Era сравнивается с domain-restricted pooled gain. Обычные RF/ExtraTrees/HistGBDT показаны отдельно. IRF v2 — зарегистрированный вариант с более строгими краевыми правилами; Meta — интерпретация. Это не точная репликация всех авторских экспериментов.","\nПолные настройки и математические объяснения: [пререгистрация](../../docs/series_v2/preregistration.md), [математическое приложение](../../docs/series_v2/mathematical_appendix.md). Source-validation включает две заранее заданные крайние среды; для Appliances — недели W09/W12 и embargo24h. Полного LODO по всем средам нет.","\n## 3. Основные H1/H2", "\nH1: изменение преимущества DG относительно своего контроля при K16 против K4, N8192,balanced,spurious. H2: изменение этого преимущества при дисбалансе8:1 против равных сред, K8,N8192,spurious. Для каждой гипотезы — family из девяти method-task contrasts; двусторонний exact sign test по десяти data replicas, Holm отдельно в каждой family. Показанные t-CI — marginal 95%, без simultaneous correction."]
    lines[3]=f"\nЗарегистрированы и предприняты {counts['jobs']} заданий: {counts['complete_jobs']} полностью завершены; в {counts['partial_jobs']} сохранены остальные заранее зарегистрированные методы при недоступном MaxRM. Сохранены {counts['ordinary_source_fits']:,} успешных обычных source fits и {counts['repeated_source_fits']} повторов, {counts['full_source_models']:,} full-source конфигурации, {counts['target_domain_metric_rows']:,} target-domain строк и {counts['source_validation_rows']:,} source-validation строк. Это retained successful fits; пилоты, диагностика и незавершённые вычисления не прибавлены к этому числу. Протокол опубликован в commit b0325a5 до первого нового target-доступа."
    lines.insert(6,"\nПять MaxRM pipelines остановились на source с CLARABEL optimal_inaccurate. Точность solver не ослаблялась, запасная модель не подставлялась. Остальные исходно зарегистрированные методы завершены по открытому дополнению [source failures](../../docs/series_v2/source_failure_addendum.md). Из 18 основных тестов полностью исполнимы 17; H2 MaxRM regression содержит только 9 полных replica pairs и не объявляется подтверждающим тестом. Неуспешность относится к этой реализации/решателю/критерию сертификации, а не доказывает принципиальную неразрешимость MaxRM.")
    numeric=audit.get("numeric_tree_findings",{})
    lines.insert(7,"\nЧисленная проверка деревьев: "+", ".join(f"{key}={value}" for key,value in numeric.items())+". Проверены фактические bootstrap partitions, а не только параметры конструктора. Source-only replay дополнительно выявил редкое нарушение min leaf в промежуточном Appliances CV-дереве при почти совпадающих floating-point календарных значениях; observed validation predictions конечны. Frozen код после открытия target не исправлялся. Это оговорка точности реализации; детали в source_numeric_diagnostic.json и audit/tree_numeric_checks.csv.")
    lines.insert(8,"\nВсе найденные full-source tree anomalies относятся к IRF на Appliances: девять нарушений фактической domain-support admissibility, включая один пустой leaf с NaN value. На сохранённых source/target observations predictions конечны и пересчитаны; корректность функции на произвольных новых входах этим не доказывается. IRF/Appliances pair differences нельзя интерпретировать как полностью чистый эффект заявленного penalty. В основных synthetic H1/H2 таких full-source нарушений не найдено.")
    raw_labels=json.loads((OUT/"audit/raw_target_labels.json").read_text(encoding="utf-8"));assert raw_labels["status"]=="passed"
    null_check=json.loads((OUT/"audit/null_cache_full_vector_replay.json").read_text(encoding="utf-8"))
    lines.insert(8,f"\nОтдельная реконструкция исходных CSV/ZIP подтвердила eligible rows, y и row IDs всех четырёх real tasks. Выявлен input-representation defect первого Beijing job: {null_check['missing_wd_rows']} пропусков wd представлены np.nan до cache и None после Parquet reload. При восстановлении исходной CSV-семантики полные векторы всех 17 моделей совпали с сохранёнными predictions точно; максимальное изменение per-domain RMSE при альтернативном cache-input для этих моделей равно {null_check['max_abs_domain_rmse_change']:.6g} µg/m³. Официальные метрики не заменялись. Across-job Beijing comparisons имеют эту оговорку; synthetic H1/H2 этим случаем не затронуты. [Разбор ограничений реализации](../../docs/series_v2/implementation_findings.md), [raw-label audit](audit/raw_target_labels.json), [input provenance](audit/replay_input_provenance.json).")
    significant=a["hypotheses"][a["hypotheses"].p_holm<.05]
    if significant.empty:
        lines.append("\nПосле Holm ни один из полностью выполненных основных тестов не достиг порога .05. Это не доказательство одинаковости эффектов: при 10 независимых повторах мощность ограничена. Нельзя объявлять общую закономерность о пользе большего K или дисбаланса на основании отдельных положительных средних.")
    else:
        lines.append(f"\nПорог Holm .05 прошли {len(significant)} из 18 предзаданных тестов. Это свидетельство только для указанных generator/method/contrast, не универсальный закон DG.")
    for _,r in a["hypotheses"].iterrows():
        if r.status!="complete":
            lines.append(f"\n- {r.hypothesis}, {r.task}, {NAMES[r.method]}: **не выполнен полностью**; {int(r.replicas)} полных replicas вместо 10. Descriptive complete-case Δgain={r['mean']:.6g}, t-CI [{r.ci_low:.6g}; {r.ci_high:.6g}]. Confirmatory p не объявляется; для коррекции остальных сохранена family=9 с консервативным input p=1.")
            continue
        lines.append(f"\n- {r.hypothesis}, {r.task}, {NAMES[r.method]}: Δgain={r['mean']:.6f}, 95% t-CI [{r.ci_low:.6f}; {r.ci_high:.6f}], sign p={r.p_value:.5g}, Holm p={r.p_holm:.5g}; ненулевых replicas {int(r.nonzero_replicas)}.")
    lines.extend(["\n![Основные контрасты](figures/confirmatory_hypotheses.png)","\n## 4. Механизмы сдвига и фиксированные кривые", "\nПоложительный gain означает улучшение относительно собственного нулевого контроля: ΔAUROC для binary и уменьшение MSE/source variance для regression. Нулевой gain часто означает source-selected отказ от регуляризации; это допустимый результат, а не пропущенный запуск. Степени severity равно взвешены. Внешний сильный baseline может оказаться лучше обеих моделей пары.","\n![Screen](figures/mechanism_screen.png)"])
    screen=effects[(effects.kind=="synthetic")&(effects.k==8)&(effects.n==8192)&(effects.imbalance==1)]
    for (task,scenario),f in screen.groupby(["task","scenario"],sort=False):
        text="; ".join(f"{NAMES[r.method]} {r['mean']:+.4f} [{r.ci_low:+.4f};{r.ci_high:+.4f}]" for _,r in f.iterrows())
        lines.append(f"\n{task}, **{scenario}**: {text}. Интервалы descriptive marginal, без поправки на все secondary comparisons.")
    lines.extend(["\n## 5. Число сред, размер выборки и дисбаланс", "\nH1 изолирует число сред при одном общем N и неизменных границах source-параметра. Каждая среда при большем K становится меньше; именно такой tradeoff проверяет дизайн. H3 меняет только N при K8, H2 — только распределение размеров при одинаковом K/N. Большой K не автоматически означает больше информации о стабильном механизме.","\n![K](figures/source_domain_count.png)","\n![N](figures/synthetic_training_size.png)","\n![Дисбаланс](figures/source_imbalance.png)","\nДля H3 результаты являются вторичными; exact paired effects/CI находятся в tables/secondary_N_contrasts.csv. Нормировка риска основана на source variance каждого условия; рядом сохранены исходные MSE/RMSE, чтобы видеть изменение масштаба.","\n## 6. Новые реальные домены", "\nНиже — описательный срез максимальных N и K каждой задачи. Порядок по target — ретроспективное представление уже зафиксированных pipelines, не процедура выбора будущего метода. Малое число зависимых доменов и overlapping source resamples не поддерживает популяционные p-values."])
    for dataset,label in DATASETS.items():
        f=summary[(summary.dataset==dataset)&(summary.k==(6 if dataset=="beijing_pm25" else 8))]
        metric="roc_auc" if dataset.startswith("acs_") else "rmse"
        f=f[f.n==f.n.max()].sort_values(metric,ascending=metric=="rmse")
        ranking="; ".join(f"{NAMES[r.method]}: {r[metric]:.5f}" for _,r in f.iterrows())
        lines.append(f"\n**{label}**, N={int(f.n.iloc[0])},K={int(f.k.iloc[0])}; {'AUROC ↑' if metric=='roc_auc' else 'macro RMSE ↓'}: {ranking}.")
        g=effects[(effects.dataset==dataset)&(effects.k==f.k.iloc[0])&(effects.n==f.n.iloc[0])]
        lines.append("\nMatched gains: "+"; ".join(f"{NAMES[r.method]} {r['mean']:+.5f}" for _,r in g.iterrows())+". Это means по трём source resamples и двум seeds; CI независимых datasets не рассчитывается.")
    lines.extend(["\n![Реальные задачи](figures/real_training_size.png)","\nBeijing — contemporaneous PM2.5 modelling с PM10/синхронными загрязнителями, не forecast. Станции разделяют погоду и временную ось. Appliances — перенос на будущее одного дома, не на новые дома; локальные T6/RH6 расположены снаружи у здания. ACSIncome/PublicCoverage используют один survey family и невзвешенные rows; результаты не являются survey-weighted population estimates.","\n## 7. Source selection, Meta и неодинаковый шум", "\nДля каждого frozen grid сохранены все target points. Разница с best-on-target grid point — retrospective oracle diagnostic: она показывает потерю source selection, но oracle нельзя выдавать за deployable pipeline. При единственном grid point такой gap тождественно нулевой и не свидетельствует о лучшей selection.","\n![Oracle gap](figures/selection_oracle_gap.png)","\nMeta uniform/accuracy/full сравниваются на буквально одинаковых forest members. Full-minus-accuracy изолирует добавление MMD в этой интерпретации, а masked/unmasked — отдельную политику признаков. Маскирование и MMD не объявляются полезными заранее.","\n![Meta](figures/meta_same_members.png)","\nHeteroscedastic regression имеет точно известный неустранимый target noise MSE примерно .192,1,5.207. Worst-risk оптимизация может расходовать свободу на наиболее шумную среду, не получая устойчивую среднюю функцию. Это проверяется по mean и worst target risk; per-tree bootstrap risk certificate MaxRM не является target или ensemble guarantee.","\n![Worst target](figures/worst_target_noise.png)","\n## 8. Что можно и нельзя заключать", "\nЧисленная корректность и сопоставимость controls проверяются отдельно от научного выигрыша. Отрицательный результат на selected pipeline не доказывает бесполезность всей статьи, особенно при независимой реализации/ограниченной capacity. Положительный secondary effect не становится confirmatory из-за красивого графика. Сдвиг условного механизма может разрушать предпосылку общей invariant функции; никаких гарантий произвольного OOD здесь нет.","\n10 MC replicas описывают неопределённость внутри этих DGP, а не выборку всех табличных задач. Две крайние source folds экономят вычисления, но могут пропускать внутренние механизмы. Folds, grids, severity и новые targets нельзя менять по этим результатам и снова называть те же targets независимым финалом. Дальнейшие решения потребуют нового зарегистрированного holdout либо честного exploratory статуса.","\nЛогическая цепочка hashes/receipts усилена и проверена; OS-аудита любого возможного ручного чтения нет. Нельзя превращать локальные timestamps в абсолютное доказательство исторического отсутствия доступа. Полные версии, source fingerprints, frozen runtime и journal доступны в репозитории; raw data/models/predictions локальные и исключены из Git.","\n## 9. Где полный материал", "\n- [Интерактивный локальный читатель](index.html): фильтры по задаче, сценарию, K/N, все summary metrics.\n- [Исходные per-domain метрики](tables/target_metrics.csv) и [source-CV](tables/source_cv_metrics.csv).\n- [Основные тесты](tables/hypothesis_tests.csv), [10 paired contrasts каждого теста](tables/hypothesis_replica_contrasts.csv).\n- [Все frozen points и matched gains](tables/all_grid_paired_effects.csv), [выбор конфигураций](tables/configuration_selection_frequency.csv).\n- [Выбор и oracle diagnostic](tables/selection_diagnostics.csv), [инвентарь](analysis_inventory.json), [аудит](audit/final_integrity.json).\n- Локально runs/expanded_v2/models и predictions: полные модели/исходные predictions; в Git только reports, configurations, hashes и журналы.\n- Полные тексты литературы локально literature/pdfs; ссылки/версии/недоступные тексты — literature/sources.json, download_manifest.json, docs/literature_audit.md."])
    irf_grid=a["configuration_diagnostics"]
    irf_grid=irf_grid[(irf_grid.task=="classification")&(irf_grid.method=="irf")&(irf_grid.scenario=="spurious")&(irf_grid.n==8192)&(irf_grid.imbalance==1)]
    g=irf_grid.groupby(["k","config"])[["source_primary","target_primary"]].mean()
    chosen=a["selected"]
    chosen=chosen[(chosen.task=="classification")&(chosen.method=="irf")&(chosen.scenario=="spurious")&(chosen.n==8192)&(chosen.imbalance==1)].drop_duplicates("job_id")
    k4=chosen[chosen.k==4];k16=chosen[chosen.k==16]
    at=lines.index("\n![Oracle gap](figures/selection_oracle_gap.png)")+1
    lines[at:at]=["\n### H5: конкретный случай смены source-выбора IRF",f"\nПри K4 source-validation выбрала λ=.05 в {int((k4.config==1).sum())} из {len(k4)} запусков; при K16 выбрала λ=0 в {int((k16.config==0).sum())} из {len(k16)}. При K16 mean source AUROC для λ0={g.loc[(16,0),'source_primary']:.6f}, для λ.05={g.loc[(16,1),'source_primary']:.6f}; mean target AUROC соответственно {g.loc[(16,0),'target_primary']:.6f} и {g.loc[(16,1),'target_primary']:.6f}. Это наблюдаемый механизм потери source-selected pipeline: он отказался от ненулевого штрафа, хотя зарегистрированная фиксированная точка лучше на данном target reversal. Поэтому отрицательный H1 для pipeline не равен доказательству, что каждый фиксированный регуляризованный IRF хуже при большем K. Эта secondary диагностика объясняет полученные результаты; выбор новой λ по открытому target потребовал бы нового независимого теста.","\n![IRF: source и target](figures/irf_selection_failure.png)"]
    negative_h1=a["hypotheses"][(a["hypotheses"].hypothesis=="H1")&(a["hypotheses"].p_holm<.05)&(a["hypotheses"]["mean"]<0)]
    lead="В предзаданном spurious-сценарии увеличение K4→K16 при N8192 снизило преимущество относительно собственного контроля: "+"; ".join(f"{NAMES[r.method]} ({r.task}) Δgain={r['mean']:.6f}, Holm p={r.p_holm:.5g}" for _,r in negative_h1.iterrows())+". Это вывод о конкретных source-selected pipelines и этом DGP, а не универсальное утверждение о числе доменов."
    lead2="Для H2 подтверждающих результатов после Holm не получено; MaxRM regression H2 неполон и не считается выполненным тестом. Реальные задачи не дают одного универсального победителя DG. У IRF показано расхождение source/target ranking настроек; этот диагностический вывод не превращает target в данные настройки."
    lines[2:2]=["\n## Основной результат","\n"+lead,"\n"+lead2]
    p.write_text("\n".join(lines)+"\n",encoding="utf-8")
    (OUT/"executive_summary.md").write_text("# Краткие выводы второй серии\n\n"+lead+"\n\n"+lead2+f"\n\nИнвентарь: {counts['complete_jobs']} полных заданий, {counts['partial_jobs']} с недоступным MaxRM; {counts['full_source_models']:,} full-source конфигурации. [Полный отчёт](research_report.md), [читатель](index.html), [аудит](audit/final_integrity.json).\n",encoding="utf-8")
    return lines


def build_report(a,effects,summary,per_rep,counts,frozen):
    audit=json.loads((OUT/"audit/final_integrity.json").read_text(encoding="utf-8"))
    assert audit["status"].startswith("passed")
    figures(a,effects,summary)
    lines=write_markdown(a,effects,summary,counts,frozen,audit)
    effect_columns=["kind","dataset","task","scenario","k","n","imbalance","method","mean","ci_low","ci_high"]
    joined=summary.merge(effects[effect_columns],on=effect_columns[:8],how="left",validate="one_to_one")
    records=json.loads(joined.to_json(orient="records",double_precision=10))
    tests=a["hypotheses"].copy();tests["method"]=tests.method.map(NAMES)
    tests=tests[["hypothesis","task","method","status","replicas","mean","ci_low","ci_high","p_value","p_holm","nonzero_replicas"]]
    hypotheses_html=tests.to_html(index=False,float_format=lambda x:f"{x:.5g}",border=0,classes="data",na_rep="не выполнен")
    plots=[("confirmatory_hypotheses","Предзаданные H1/H2: разности выигрыша и скорректированные p"),("mechanism_screen","Механизмы сдвига: source-selected pipeline против своего контроля"),("source_domain_count","Число сред при постоянном N"),("synthetic_training_size","Объём обучения при постоянном K"),("source_imbalance","Дисбаланс размеров сред при постоянных K/N"),("real_training_size","Новые реальные домены: увеличение source N"),("selection_oracle_gap","Диагностика source selection — oracle не является deployable методом"),("meta_same_members","Meta: разные веса одних и тех же компонентов"),("worst_target_noise","Неодинаковый шум: худший target")]
    plot_html="".join(f'<figure><h3>{html.escape(title)}</h3><img src="figures/{name}.png" alt="{html.escape(title)}"><figcaption>Источник: frozen expanded_v2; данные tables/. <a href="figures/{name}.svg">SVG</a></figcaption></figure>' for name,title in plots)
    plot_html+='<figure><h3>IRF: source-only выбор и target diagnostic curve</h3><img src="figures/irf_selection_failure.png" alt="IRF source-target ranking"><figcaption>Все точки зарегистрированы до target. Ретроспективная диагностика H5, не процедура настройки.</figcaption></figure>'
    files="".join(f'<li><a href="tables/{p.name}">{html.escape(p.name)}</a></li>' for p in sorted(TABLES.glob("*.csv")))
    body='''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Coral Forest DG · Полные результаты v2</title>
<style>body{font:16px/1.55 system-ui,sans-serif;color:#20262d;background:#fff;margin:0}main{max-width:1220px;margin:auto;padding:38px 28px}h1{font-size:30px;line-height:1.2;max-width:950px}h2{font-size:24px;margin-top:44px;border-top:1px solid #d8dce1;padding-top:22px}h3{font-size:18px}p{max-width:980px}nav{display:flex;flex-wrap:wrap;gap:20px;margin:28px 0}a{color:#245a81}small,figcaption{color:#5d6672}figure{margin:32px 0}figure img{max-width:100%;height:auto}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}th,td{text-align:left;border-bottom:1px solid #d8dce1;padding:9px 10px;white-space:nowrap}th{background:#f1f3f5;position:sticky;top:0}tr:hover{background:#f5f7f9}.filters{display:flex;flex-wrap:wrap;gap:14px;margin:18px 0}label{font-size:13px}select{display:block;padding:8px;border:1px solid #b7c0ca;background:white;border-radius:3px;min-width:140px}.lead{font-size:19px}.facts{border-left:3px solid #245a81;padding:6px 18px}.note{background:#f1f3f5;padding:18px}code{font-size:12px;overflow-wrap:anywhere}.good{color:#205b3a}.bad{color:#a13f34}@media print{nav,.filters{display:none}h2{break-before:page}figure{break-inside:avoid}main{padding:0}}</style>
<main><small>НИР · воспроизводимые эксперименты · expanded_v2</small><h1>Ансамбли деревьев при сдвиге распределения: полный разбор второй серии</h1>
<p class="lead">Новые целевые домены, проверяемые гипотезы о числе и размерах source-сред, сопоставимые контроли и отдельный аудит численных результатов.</p>
<nav><a href="#audit">Проверка</a><a href="#hypotheses">H1/H2</a><a href="#explorer">Все условия</a><a href="#figures">Графики</a><a href="#files">Файлы</a><a href="research_report.md">Полный текст</a><a href="research_report.pdf">PDF отчёта</a></nav>
<div class="facts">__FACTS__</div>
<h2 id="audit">Что подтверждено</h2><p>__AUDIT__</p><p>В первой серии подтверждены 1008 метрик, 126 моделей и 42 source fits; выявлены ограничения пар MaxRM/Era/IRF/Meta. <a href="../audit_v1/audit_report.md">Аудит первой серии</a>. Вторая серия использует исправленные пары и новые targets. Старые targets не стали новым независимым финалом.</p>
<p class="note">Проверки выполнены отдельными уравнениями тем же авторским агентом; это не внешнее peer review. Логические запреты и хеши не являются OS-доказательством отсутствия любого ручного доступа. Meta — интерпретация, IRF v2/CRF/MaxRM — независимые реализации.</p>
<h2 id="hypotheses">Основные тесты</h2><p>H1: gain(K16)−gain(K4), N8192,balanced,spurious. H2: gain(8:1)−gain(1:1), K8,N8192,spurious. Две отдельные семьи по девять exact sign tests. Holm внутри family; t-интервалы marginal 95%. Единица анализа — 10 независимых data replicas после усреднения seeds и target laws.</p><div class="scroll">__HYPOTHESES__</div>
<h2 id="explorer">Все условия и методы</h2><p>Средние по target-доменам, model seeds и source resamples/data replicas с равными весами на каждом уровне. AUROC выше — лучше; RMSE ниже — лучше. Gain: ΔAUROC или уменьшение MSE/source variance относительно собственного config-0 контроля. Для внешних RF/ExtraTrees/HistGBDT matched gain не определяется.</p><div class="filters" id="filters"></div><p id="rowcount"></p><div class="scroll"><table><thead><tr><th>Задача</th><th>Тип</th><th>Сценарий</th><th>K</th><th>N</th><th>Дисбаланс</th><th>Метод</th><th>AUROC</th><th>RMSE</th><th>Log loss</th><th>Gain</th><th>95% t-CI</th></tr></thead><tbody id="rows"></tbody></table></div><p><small>CI только по independent synthetic replicas; реальные source resamples пересекаются. CI не скорректированы одновременно. Полные сырые значения доступны в CSV.</small></p>
<h2 id="figures">Графики и диагностические сравнения</h2>__PLOTS__
<h2 id="files">Полные файлы</h2><p><a href="research_report.md">Исследовательский отчёт</a> · <a href="research_report.pdf">PDF</a> · <a href="../../docs/series_v2/preregistration.md">Пререгистрация</a> · <a href="../../docs/series_v2/mathematical_appendix.md">Математическое приложение</a> · <a href="audit/final_integrity.json">Итоговая целостность</a> · <a href="../../runs/expanded_v2/frozen_protocol.json">Frozen manifest</a></p><ul>__FILES__</ul>
<p>PDF источников находятся локально в literature/pdfs. На GitHub сохранены библиография, версии, hashes и журнал загрузок. Исходные models/predictions доступны локально в runs/expanded_v2; это не подмена агрегированными метриками.</p><small>Protocol SHA-256: <code>__HASH__</code></small></main>
<script>const DATA=__DATA__;const NAMES=__NAMES__;const DATASETS=__DATASETS__;
const fields=[['dataset','Датасет'],['task','Тип задачи'],['scenario','Сценарий'],['k','K'],['n','N'],['imbalance','Дисбаланс'],['method','Метод']];let controls={};
const escape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const fmt=x=>x==null?'—':Number(x).toFixed(5);
for(const [key,label] of fields){let l=document.createElement('label');l.textContent=label;let sel=document.createElement('select');let all=document.createElement('option');all.value='';all.textContent='Все';sel.append(all);for(const value of [...new Set(DATA.map(r=>r[key]))].sort((a,b)=>typeof a==='number'?a-b:String(a).localeCompare(b))){let o=document.createElement('option');o.value=value;o.textContent=key==='method'?(NAMES[value]||value):(key==='dataset'?(DATASETS[value]||value):value);sel.append(o)}sel.onchange=render;l.append(sel);controls[key]=sel;document.getElementById('filters').append(l)}
function render(){const selected=DATA.filter(r=>fields.every(([key])=>!controls[key].value||String(r[key])===controls[key].value));document.getElementById('rowcount').textContent=`Показано ${selected.length} условий × методов`;document.getElementById('rows').innerHTML=selected.map(r=>`<tr><td>${escape(DATASETS[r.dataset]||r.dataset)}</td><td>${escape(r.task)}</td><td>${escape(r.scenario)}</td><td>${r.k}</td><td>${r.n}</td><td>${r.imbalance}:1</td><td>${escape(NAMES[r.method]||r.method)}</td><td>${fmt(r.roc_auc)}</td><td>${fmt(r.rmse)}</td><td>${fmt(r.log_loss)}</td><td>${fmt(r.mean)}</td><td>${r.ci_low==null?'—':`[${fmt(r.ci_low)}; ${fmt(r.ci_high)}]`}</td></tr>`).join('')}render();</script></html>'''
    facts=f"<strong>{counts['jobs']} заданий: {counts['complete_jobs']} полных + {counts['partial_jobs']} с отказом MaxRM · {counts['ordinary_source_fits']+counts['repeated_source_fits']:,} сохранённых source fits</strong><p>{counts['full_source_models']:,} full-source конфигурации, {counts['target_domain_metric_rows']:,} per-domain оценок. {counts['complete_confirmatory_tests']} из 18 основных тестов полны. Все отказы сохранены; отсутствующие результаты не подменены контрольной моделью.</p>"
    evidence=f"Аудит v2: <strong>{audit['status']}</strong>. Независимо пересчитаны {audit['independent_metric_checks']:,} метрик, воспроизведены predictions {audit['saved_model_replays']:,} конфигураций. Максимальная разность метрик {audit['max_metric_absolute_difference']:.3g}, predictions {audit['max_model_prediction_difference']:.3g}. Проверены hashes, source selection, row IDs, matched pairs и журнал доступа."
    replacements={"__FACTS__":facts,"__AUDIT__":evidence,"__HYPOTHESES__":hypotheses_html,"__PLOTS__":plot_html,"__FILES__":files,"__HASH__":frozen["sha256"],"__DATA__":json.dumps(records,ensure_ascii=False),"__NAMES__":json.dumps(NAMES,ensure_ascii=False),"__DATASETS__":json.dumps(DATASETS,ensure_ascii=False)}
    for token,value in replacements.items():
        body=body.replace(token,value)
    body=body.replace('<th>Gain</th>','<th>Контроль AUROC / RMSE</th><th>Gain</th>')
    body=body.replace('<td>${fmt(r.mean)}</td>',"<td>${r.task==='classification'?fmt(r.control_roc_auc):fmt(r.control_rmse)}</td><td>${fmt(r.mean)}</td>")
    body=body.replace('<h2 id="hypotheses">Основные тесты</h2>','<h2 id="hypotheses">Основные тесты</h2><p><strong>Значимые H1-эффекты отрицательны: для IRF и Meta в classification преимущество относительно контроля снизилось при K16 против K4. H2 MaxRM regression неполон; остальные H2 не дали Holm-значимого результата.</strong> Область вывода — зарегистрированный spurious DGP и конкретные pipelines.</p>')
    body=body.replace('<h2 id="files">Полные файлы</h2>','<h2 id="files">Полные файлы</h2><p><a href="data_dictionary.md">Словарь полей, единиц и правил усреднения CSV</a></p>')
    body=body.replace('<h2 id="audit">Что подтверждено</h2>','<h2 id="audit">Что подтверждено</h2><p class="note">Аудит имеет оговорки: 5 source-отказов MaxRM; редкие floating-point нарушения min leaf в промежуточном CV; np.nan/None cache-различие в 357 wd rows первого Beijing job. Исходные полные векторы этого задания воспроизведены после восстановления исходной null-семантики. <a href="../../docs/series_v2/implementation_findings.md">Подробный разбор</a>.</p>')
    (OUT/"index.html").write_text(body,encoding="utf-8")
    (OUT/"README.md").write_text("# Полные результаты второй серии\n\nОткройте [читатель с фильтрами](index.html), [полный научный отчёт](research_report.md) или [PDF](research_report.pdf).\n\nCSV в tables/ содержат все конфигурации и исходные per-domain метрики. Графики доступны в PNG/SVG в figures/. Итоговый audit/final_integrity.json подтверждает фактически выполненные проверки.\n",encoding="utf-8")
