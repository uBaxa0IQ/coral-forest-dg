# Как читать файлы второй серии

## Идентификаторы и объём данных

- `job_id` — зарегистрированные context × model seed; `context_id` не включает seed.
- `replica` — независимая генерация synthetic data (0–9) либо пересекающийся source resample real data (0–2). Эти два значения термина статистически различны.
- `n` — **общий размер source training sample**, например 8192 или 16000. `target_rows` в плоском target CSV — число тестовых наблюдений одного target domain. В исходном вложенном result.json поле metrics[].n означает target count; spec.n означает source N. Эти поля намеренно разведены при экспорте.
- `k` — число source domains полного обучения; source-CV обучается на доступных train domains своей fold.
- `imbalance` — заданное отношение геометрических весов размеров сред. Фактические целочисленные counts и ratio восстанавливаются из source_checks/domains frozen manifest. Округление сохраняет общий N точно.
- Target draws одной synthetic replica совпадают между K/N/imbalance/seed. Их нельзя суммировать как новые независимые датасеты.

## Конфигурации, метрики и единицы

`config` — индекс в frozen_protocol.protocol.grids[task][method], начиная с нуля. `selected=True` означает выбор по двум source folds. Порядок сетки разрешает ties; target не выбирает config. Строки target_metrics уникальны по job_id/method/config/target_domain.

Binary: `roc_auc` — AUROC, `average_precision` — ступенчатая AP с корректной обработкой одинаковых score; `log_loss` — natural-log loss с clipping к машинному epsilon, `balanced_accuracy` — средняя recall двух классов при фиксированном пороге .5. Ориентация вероятности — class1, определённая в source; наблюдаемый target AUROC может быть ниже .5. Vote fractions IRF/CRF не объявляются калиброванными вероятностями.

Regression: `mse`, `rmse`, `mae`, `r2` рассчитываются в исходных единицах внутри target domain. Beijing PM2.5 — µg/m³; Appliances — Wh для десятиминутного интервала. В synthetic единицы задаются генератором. Macro RMSE — среднее per-domain RMSE; это не корень macro MSE и не pooled-row RMSE.

`source_y_scale` — population STD (ddof=0) полного source sample. Она отдельно пересчитана в job_inventory.csv. Внутри каждого fit regression labels нормируются только по **train части этого fit** и затем predictions возвращаются в исходные единицы. Полный source scale используется только как одинаковый для кандидатов нормировщик сравнительных scores.

`primary` = AUROC для binary и −MSE/source_y_scale² для regression. Чем выше, тем лучше. `gain` = primary(method config) − primary(own config0). Положительный regression gain — уменьшение MSE/source variance, не процент RMSE. `worst_primary` — минимум primary по target domains; `worst_gain` сравнивает собственные worst risks двух моделей.

Каждый target domain имеет одинаковый вес, затем одинаково усредняются model seeds. Это estimand «средний домен», а не «средний человек/час». Размеры target_rows веса не меняют.

## Контроли

Config0 IRF/CRF/Meta/MaxRM/Era/Directional/Anchor — контроль **внутри своего семейства**. Это не одна общая RF/GBDT. У Meta совпадают обученные forest members; у MaxRM bootstrap и ветви; у Era сохраняется ограничение domain support даже при нулевой robust strength. У IRF/CRF имеются зарегистрированные пороги/ограничения/агрегация. Внешние RF200/ExtraTrees200/HistGBDT200 — отдельные сильные pipelines; их matched gain в summary пустой.

## Пропуски и неопределённость

Пять source-stage отказов MaxRM сохранены в failures/. В этих заданиях отсутствуют оба MaxRM grid points, а остальные исходно зарегистрированные методы завершены по source_failure_addendum. Нет подмены на control. Для replica-method cells требуется **оба seed**; один seed не усредняется как полная replica.

`replicas` — фактическое число полных data replicas для effect/contrast. Для synthetic обычно 10; у некоторых MaxRM secondary points меньше. H2 MaxRM regression имеет 9 полных пар и status=incomplete_source_solver_failure. Пустые p_value/p_holm означают невыполненный полностью тест, не p=0 и не успешное подтверждение нуля. Complete-case estimate этого теста описателен; missingness может зависеть от трудности задачи.

95% CI — marginal Student t interval по независимым synthetic data replicas. Для real overlapping resamples такие CI не строятся. Holm применяется отдельно к девяти H1 и девяти H2 tests. При неполном MaxRM H2 для консервативной коррекции остальных сохранён input p=1 и family size9; самому тесту confirmatory p не приписывается.

В configuration_selection_frequency.csv отсутствующая строка config означает нулевое число выборов этой конфигурации. `available_jobs`, `planned_jobs`, `missing_jobs` показывают полноту family/condition. Число выбранных configs не равно числу обученных: все frozen grid points обучались для заранее заданных diagnostic curves.

## Диагностика и аудит

source_target_grid_diagnostics.csv содержит средние source-CV и target scores **каждой** зарегистрированной точки. Retrospective oracle gap не является deployable model selection. Его значения не используются для нового обучения на уже открытых targets.

audit/independent_metric_checks.csv — отдельные rank/NumPy уравнения; model_prediction_replays.csv — воспроизведение сохранённых моделей на target probe; access_and_pair_ledger.csv — hashes/даты/пары; tree_numeric_checks.csv — фактические bootstrap partitions. source_numeric_diagnostic.json отдельно фиксирует source-only исследование floating-point edge case в промежуточных Appliances CV деревьях.

fit_timings.csv: `seconds` — wall time fit, а в CV также validation predict. `source_seconds` включает source preparation/preprocessing/replays данного задания, но записывается до model export. Сумма этих times при параллельной работе не равна продолжительности всей серии. Исходные timestamps хранятся в UTC, ISO8601.

Полные индивидуальные source/target predictions и модели находятся локально в runs/expanded_v2/predictions и models; в Git они исключены. CSV reports — агрегированные метрики и проверочные свидетельства, не замена исходным predictions.
