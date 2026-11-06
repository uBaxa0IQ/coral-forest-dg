"""Descriptive scientific reporting. Does not tune models or access raw targets."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from nir_dg.datasets import ROOT, load_source
from nir_dg.models import METHODS


def build_reports():
    reports = ROOT / "reports"
    (reports / "tables").mkdir(exist_ok=True)
    (reports / "figures").mkdir(exist_ok=True)
    source_paths = sorted((ROOT / "runs").glob("study_cv_v1_*/source_cv_metrics.csv"))
    source = pd.concat([pd.read_csv(p) for p in source_paths], ignore_index=True)
    source.to_csv(reports / "source_cv_metrics.csv", index=False)
    source_summary = source.groupby(["dataset", "task", "method", "trial", "parameters", "implementation_status"], dropna=False).agg(source_macro_score=("primary", "mean"), source_score_sd=("primary", "std"), fit_seconds_total=("fit_seconds", "sum"), fits=("primary", "size"), failures=("outcome", lambda s: int((s != "ok").sum()))).reset_index()
    source_summary.to_csv(reports / "tables/source_cv_summary.csv", index=False)
    method_rows = []
    papers = {"irf": "irf", "crf": "crf", "meta_forests_interpretation": "meta_forests", "maxrm_posthoc": "maxrm_rf", "era": "era_splitting", "directional_era": "era_splitting", "anchor": "anchor_boosting"}
    for method, spec in METHODS.items():
        rows = source[source.method == method]
        method_rows.append(dict(method=method, family=spec.family, implementation_status=spec.fidelity, paper_id=papers.get(method, ""), paired_baseline=spec.paired_baseline, source_successful_fits=int(rows.outcome.eq("ok").sum()), source_failed_fits=int(rows.outcome.ne("ok").sum()), task_types=";".join(sorted(rows.task.unique())), published_numbers_replicated=False))
    method_rows.append(dict(method="meta_forests_exact_replication", family="RF", implementation_status="blocked_unverified", paper_id="meta_forests", paired_baseline="meta_matched_rf", source_successful_fits=0, source_failed_fits=0, task_types="classification", published_numbers_replicated=False, reason="Eq.3 weight update, initial indices, kernel/bandwidth and masking edge cases underspecified; complete labelled interpretation available"))
    pd.DataFrame(method_rows).to_csv(reports / "method_matrix.csv", index=False)
    dataset_rows = []
    for name in ["acs_income", "hospital_readmission", "beijing_pm25", "gas_turbine_co"]:
        d = load_source(name)
        dataset_rows.append(dict(dataset=name, task=d.task, source_rows=len(d.y), source_environments=len(np.unique(d.environments)), temporal=d.temporal, num_raw_predictors=len(d.X.columns), source_fingerprint=d.fingerprint(), status="real_source_prepared", source_cap_per_domain=400))
    dataset_rows += [dict(dataset="tabred_cooking_time_delivery_eta", status="blocked_Kaggle_auth_and_unverified_eras"), dict(dataset="shifts_weather", status="network_TLS_timeout_official_storage")]
    pd.DataFrame(dataset_rows).to_csv(reports / "dataset_matrix.csv", index=False)
    target_path = ROOT / "runs/final_v1/target_metrics.csv"
    target = pd.read_csv(target_path) if target_path.exists() else pd.DataFrame()
    if not target.empty:
        # Domains first, then seed means. No pooling of incompatible dataset metrics.
        target.to_csv(reports / "target_metrics.csv", index=False)
        per_seed = target.groupby(["dataset", "task", "method", "seed", "implementation_status"]).primary.mean().reset_index()
        summary = per_seed.groupby(["dataset", "task", "method", "implementation_status"]).agg(target_macro_score=("primary", "mean"), seed_sd=("primary", "std"), seeds=("seed", "nunique")).reset_index()
        summary.to_csv(reports / "tables/target_summary.csv", index=False)
        worst_rows = []
        for (dataset, task, method, seed), group in target.groupby(["dataset", "task", "method", "seed"]):
            worst_rows.append(dict(dataset=dataset, task=task, method=method, seed=seed, worst_target_score=group.primary.min() if task == "classification" else group.primary.max(), num_target_domains=group.target_environment.nunique()))
        pd.DataFrame(worst_rows).to_csv(reports / "tables/worst_target_by_seed.csv", index=False)
        target.drop_duplicates("experiment_id").groupby(["dataset", "method"]).agg(mean_fit_seconds=("fit_seconds", "mean"), mean_predict_seconds=("predict_seconds", "mean"), model_seed_runs=("seed", "size")).reset_index().to_csv(reports / "tables/final_compute.csv", index=False)
        baseline = target[["dataset", "seed", "target_environment", "method", "primary"]].rename(columns={"method": "paired_baseline", "primary": "baseline_primary"})
        paired = target[target.paired_baseline.notna()].merge(baseline, on=["dataset", "seed", "target_environment", "paired_baseline"], how="left", validate="many_to_one")
        paired["improvement"] = np.where(paired.task == "classification", paired.primary - paired.baseline_primary, paired.baseline_primary - paired.primary)
        paired.to_csv(reports / "tables/paired_differences.csv", index=False)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for name, rows in summary.groupby("dataset"):
            task = rows.task.iloc[0]
            rows = rows.sort_values("target_macro_score", ascending=task == "regression")
            fig, ax = plt.subplots(figsize=(9, 6))
            colors = ["#52616B" if METHODS[m].fidelity == "baseline_only" else "#26685D" for m in rows.method]
            bars = ax.barh(rows.method, rows.target_macro_score, color=colors)
            for bar, fidelity in zip(bars, rows.implementation_status):
                if fidelity == "independent_interpretation":
                    bar.set_hatch("///")
            ax.invert_yaxis()
            unit = "AUROC (higher is better)" if task == "classification" else ("RMSE (micrograms/m³; lower is better)" if name == "beijing_pm25" else "RMSE (mg/m³; lower is better)")
            ax.set_xlabel(unit)
            ax.set_ylabel("Method and paired controls")
            ax.set_title(f"{name}: macro target performance, mean of 3 seeds")
            ax.spines[["right", "top"]].set_visible(False)
            if task == "classification":
                ax.set_xlim(0, 1)
            fig.text(.02, .015, "Source: preregistered final_v1; capped source (<=400/domain). Hatched: Meta-forests interpretation. No significance claim.", fontsize=8)
            fig.tight_layout(rect=(0, .04, 1, 1))
            fig.savefig(reports / "figures" / f"{name}_target.png", dpi=170)
            plt.close(fig)
    completed = 0 if target.empty else target.experiment_id.nunique()
    own = len(source)
    state = f"Выполнено {own} расширенных source-CV fits; успешных {int(source.outcome.eq('ok').sum())}. Финальных зарегистрированных model/seed запусков с метриками: {completed}/126."
    report = "# Методика и результаты НИР\n\n" + state + "\n\n"
    report += "## Экспериментальная постановка\n\nЧетыре реальные задачи: ACSIncome, Hospital Readmission, Beijing PM2.5 и Gas Turbine CO. Для первых двух оценивается бинарная классификация, для остальных регрессия. На каждой задаче есть RF и GBDT. Source ограничен 400 объектами на домен (в Hospital часть групп меньше). Расширенный подбор использует все доступные source folds и три seed 11/23/47. ACS и Hospital: LOEO; Beijing: leave-one-station-out; Gas Turbine: train 2011–2012, validation 2013. Внешний target не участвует. Все преобразования обучаются в fold.\n\n"
    report += "Главные метрики: AUROC и RMSE. Каждый target-домен имеет равный вес в macro. Порог бинарного решения 0.5 фиксирован. Для Hospital исключены неизвестные группы, повторные source-пациенты; в финальном target удаляются пациенты, присутствующие в source training. Это собственный протокол по исходным данным UCI, не точная репликация TableShift. Beijing — contemporaneous PM2.5 prediction с другими загрязнителями, не прогноз будущих часов. Gas Turbine — contemporaneous CO по датчикам на будущих годах, не прогноз их значений.\n\n"
    report += "## Достоверность методов\n\nIRF и MSE-posthoc MaxRM независимо реализованы по формулам. CRF реализует paper objective с явными отличиями от R-процедуры роста. Era/Directional и Anchor используют неизменённые авторские пакеты с зафиксированными зависимостями. Meta-forests реализован полностью как явно подписанная независимая интерпретация недоопределённых правил. Точная Meta-forests репликация остаётся blocked. Числа из статей не воспроизводились и не используются как собственные. Подробности в docs/methods/.\n\n"
    report += "## Результаты\n\nИсходные результаты — `source_cv_metrics.csv`; сводки — `tables/source_cv_summary.csv`. Они показывают перенос между source-средами и не являются финальными target-показателями.\n\n"
    if not target.empty:
        for name, rows in summary.groupby("dataset"):
            task = rows.task.iloc[0]
            primary = "AUROC" if task == "classification" else "RMSE"
            report += f"Для {name} ({primary}) получены следующие средние macro-показатели по трём seed:\n\n"
            for row in rows.sort_values("method").itertuples():
                report += f"- {row.method}: {row.target_macro_score:.6f}; SD по seed {row.seed_sd:.6f}; {row.implementation_status}.\n"
            report += "\n"
        report += "Парные различия для каждого target-домена и seed доступны в `tables/paired_differences.csv`; положительное improvement означает улучшение. Все отрицательные значения сохранены. Графики используют среднее по доменам, затем по seed.\n\n"
    else:
        report += "Target ещё не оценён. Численные target-выводы отсутствуют.\n\n"
    report += "## Интерпретация и ограничения\n\nЭто первая ограниченная по объёму source-выборки сравнительная серия. Она не устанавливает универсальный рейтинг механизмов DG. Различия движков, сеток и ёмкости моделей требуют парных контролей. HPO базовых режимов иногда содержит меньше вариантов; реальные затраты сохранены, равный compute не заявляется. RF внешний контроль имеет 100 деревьев и подбираемую глубину, независимые DG-леса — 12 деревьев глубины 4; сопоставление механизмов выполняется прежде всего с их собственными контролями. GBDT используют 100 итераций.\n\n"
    report += "SD по seed описывает случайность обучения, не доверительный интервал переноса. Несколько seed одного target не являются независимыми распределениями; две газотурбинные эпохи последовательны и могут зависеть. У Hospital один target, отдельная worst-domain статистика неинформативна. Формальные p-values не рассчитываются.\n\n"
    report += "Положительные результаты интерпретации Meta-forests не подтверждают точную статью. Для точной репликации нужны уточнения Eq. 3, MMD и masking. У MaxRM не реализованы local/global/NRW/regret. У CRF нет R-equivalence proof. Shifts недоступен по текущему сетевому маршруту; TabReD требует Kaggle и проверки eras/preprocessing. Их отсутствие объясняется до открытия target, не его качеством.\n\n"
    report += "## Источники и воспроизводимость\n\nБиблиография `literature/references.bib`, полный реестр `literature/sources.json`, фактические PDF/checksum `literature/download_manifest.json`. Ключевые источники: IRF DOI 10.1609/aaai.v38i12.29283; CRF PMLR 337; Meta-forests PMLR 222; MaxRM arXiv:2512.10445v2; Era DOI 10.1016/j.eswa.2025.129687; Anchor DOI 10.1093/rssdat/udag001. Протокол и hash в runs/final_v1/frozen_protocol.json, журналы в runs/, версии среды зафиксированы uv.lock.\n"
    (reports / "results_and_discussion.md").write_text(report, encoding="utf-8")
    (reports / "limitations.md").write_text("# Ограничения\n\n" + report.split("## Интерпретация и ограничения\n\n")[1].split("## Источники")[0], encoding="utf-8")
    (reports / "executive_summary.md").write_text("# Сводка НИР\n\n" + state + "\n\nЧетыре направления представлены на четырёх реальных задачах. Точная Meta-forests репликация остаётся неподтверждённой, реализована подписанная интерпретация. Численные результаты и все ограничения приведены в results_and_discussion.md. Это собственные измерения стенда, не результаты авторских статей.\n", encoding="utf-8")
    return dict(source_fits=own, source_successes=int(source.outcome.eq("ok").sum()), target_experiments=int(completed), target_metric_rows=len(target))
