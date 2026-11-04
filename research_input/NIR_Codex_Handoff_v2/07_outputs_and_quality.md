# Контракт результатов Codex: структура, доказательства, завершённость

## Требуемый репозиторий

```text
nir-dg-experiments/
  README.md
  pyproject.toml                 # также lockfile / отдельные R-окружения, если нужны
  configs/                      # YAML датасетов, методов, опытов, source-only tuning
  data/                         # не коммитить сырые/частные данные
  docs/
    preregistration.md
    data_audit.md
    leakage_audit.md
    method_audit.md
    decisions.md
    compute_budget.md
    methods/meta_forests_math.md
    methods/meta_forests_replication.md
    protocol_changes.md
  src/nir_dg/
    data/                         # зарегистрированные source/target handles
    methods/                      # все 6 методов; отдельные адаптеры baseline
    splits/                       # LOEO и causal/temporal rolling-origin
    preprocessing/                # fold-local fit
    selection/                    # source-only HPO
    evaluation/                   # protected final evaluation
    metrics/
    analysis/                     # парные сравнения, графики и статистика
  tests/
    test_no_target_access.py
    test_source_only_preprocessing.py
    test_domain_split_disjoint.py
    test_temporal_order.py
    test_baseline_equivalence.py
    test_meta_forests.py
  runs/<run_id>/
    config_frozen.yaml
    provenance.json
    source_cv_metrics.csv
    hpo_trials.csv
    frozen_protocol.json
    target_metrics.csv           # только после freeze, если проход реально состоялся
    target_predictions.parquet
    system_info.json
  reports/
    method_matrix.csv
    dataset_matrix.csv
    experiment_manifest.csv
    tables/
    figures/
    methods_and_protocol.md
    results_and_discussion.md
    limitations.md
    executive_summary.md
```

Структура ориентировочная; Codex может её рационально менять, но имена ключевых артефактов/их смысл должны оставаться однозначными.

## Обязательные поля таблиц

**`method_matrix.csv`**: метод, статья/URL/DOI, модельный тип, task, source availability, source version/commit, license, objective verified, task-support verified, pairing baseline, implementation status, unit tests, smoke tests, real source pilot, target evaluated, reasons if blocked, interpretation notes.

**`dataset_matrix.csv`**: dataset, original release/URL, license/access, task, canonical splits, domain definition, source count/size, target role (но без чтения целевых величин до freeze), feature provenance, potential leakage, missingness/categories, compatibility, final inclusion and reason.

**`experiment_manifest.csv`**: preregistered dataset, target IDs, source IDs, method, paired baseline, metric, seeds, HPO budget, expected artifact paths, run outcome, deviations. Manifest **сохранить до target results**.

**`source_cv_metrics.csv`**: experiment_id, dataset, model, environment/fold, seed, hyperparameter config, primary/secondary score, fit time, version, instance count (только source).

**`target_metrics.csv`**: experiment_id, dataset, target environment, model, seed, primary/secondary metrics, num_examples, paired baseline id, implementation status, frozen protocol hash, timestamp of evaluation. Никаких правок model selection по этой таблице.

## Тесты научной добросовестности

1. Train/HPO interfaces не содержат загрузчика target features/labels и проваливаются при попытке доступа до freeze.
2. Preprocessing, feature selection и imputation учатся внутри source-training fold, а не на объединённых source+target.
3. Исходные и целевые ID не пересекаются; для entity/time данных проверены повторные сущности и порядок времени.
4. HPO использует только source-validation, а source-target selection зафиксирован до целевой оценки.
5. У Meta-forests отдельный тест, что pseudo-meta-test — **source**, а не внешний target; unit-тесты компонентов MMD/accuracy weighting/маскирования.
6. Парные baseline сопоставимы по движку/ресурсному бюджету, либо в отчёте прямо оговорена невозможность такого сравнения.
7. Формулы original methods не изменены молча; при интерпретации это отражено в имени метода и таблицах.
8. Нет утверждений о статистической значимости только из повторных seed на одной target-среде; указывать реальную единицу независимости.
9. Все запусковые команды работают на чистой воспроизводимой среде или явно фиксируются ограничения лицензии/доступа.
10. Нулевые/отрицательные результаты и проваленные попытки не исключаются задним числом.

## Методология оценки

Для классификации основная метрика определяется **до target** (AUROC/другая подходящая заранее фиксированная) и регистрируется для каждого dataset; для регрессии RMSE/MAE с макроагрегацией по target domains. Дополнительно worst-target, source worst-risk, latency/RAM, calibration для вероятностей, по возможности статистические интервалы с правильной иерархией. Сравнение разных датасетов нормировать разумно; не усреднять несоизмеримые сырые RMSE.

Поддержать одинаковые заранее зафиксированные seeds и source folds для paired comparisons. Основные результаты — **метод против собственного baseline в одинаковом движке** и методы друг против друга при учёте архитектурных различий. Сила DG-метода не измеряется разностью опубликованных таблиц из разных статей.

## Что считается завершением

**Исследовательски завершено:** (1) исходники методов сверены, (2) код работает, (3) source-only protocol заморожен, (4) в каждой из четырёх линий проведено максимально возможное валидное сравнение, (5) все шесть основных методов получили достоверный статус и была предпринята их полноценная репликация, (6) выполнена frozen target evaluation series без подглядывания, (7) получены результаты, таблицы, графики и текст обсуждения.

**Частично завершено:** если хотя бы один из шести не удалось реализовать или запустить, требуется отчёт: какие действия предприняты, что именно заблокировано, какая замена рассмотрена, какие результаты относятся к остальным методам. Не подменять статус `blocked` на `completed`.

**Технически завершено, но научно невалидно:** если target использовался при настройке, нарушена хронология или отсутствует подтверждение математического критерия. Такие результаты не включать в основные выводы DG.

## Финальный ответ Codex должен содержать

- список реально созданных файлов/команд и test outputs;
- таблицу результатов по четырём направлениям, если финальные тесты действительно выполнены;
- состояние каждого из шести методов **включая Meta-forests**;
- ссылки на таблицы, figures и исследовательский отчёт;
- решения, принятые автономно, и их проверяемые обоснования;
- точный список непроверенного/блокеров и способ воспроизвести или продолжить;
- честное разграничение **«проверено по источнику» / «код создан» / «выполнен эксперимент»**.
