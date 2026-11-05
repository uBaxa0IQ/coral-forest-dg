# Вычислительный бюджет

Машина: 12 логических CPU, около 32 GiB RAM; точная конфигурация и версии записаны в system_info.json каждого source run. Каждый fit ограничен одним потоком BLAS/OpenMP, независимые задачи могут исполняться одновременно. Платные ресурсы не используются.

## Архивная первая серия v1

Первый пилот: seed 11, максимум два source folds, 30 boosting rounds, 12 обычных деревьев. Расширение **v1**: seeds 11/23/47, все допустимые source folds, external RF 100 деревьев с выбором depth=4/None, GBDT 100 rounds. Независимые IRF/CRF/MaxRM — 12 деревьев depth=4 (MaxRM также depth=3), max 16 candidate thresholds, min leaf=12. Meta: 3 meta-итерации по 4 дерева на лес; matched RF имеет то же суммарное число деревьев. Это ограниченная интерпретационная проверка; недостаточность старого matched контроля установлена аудитом v1.

HPO: до двух содержательных вариантов для каждого специализированного метода; paired control иногда имеет один вариант. Это не равный суммарный compute, поэтому отдельно выводятся реальные fit/predict times и HPO totals. Параметры полностью перечислены в configs/study_source_cv.yaml. Первое расширенное source-CV: 1050 fit/predict сравнений, все успешны. Стоимость измерена wall-clock; rss_mb — снимок RSS после fit, **не peak RSS**. Измерения проводились при конкурирующих независимых source-задачах, время не является строгим аппаратным benchmark.

Финальная серия: 126 dataset/method/seed запусков, targets заранее перечислены; выбор параметров закончен на source. Число повторов 3 выбрано до target, ниже исходного предложения 5. Для причинного вывода о числе source domains и больших HPO budgets нужны отдельные заранее зарегистрированные расширения с новым независимым holdout.

## Вторая серия v2, 4–5 ноября 2026

Предзадано 381 contexts ×2 seeds =762 jobs. В synthetic 33 conditions ×10 data replicas; в real 17 conditions ×3 overlapping source resamples. Две source folds на configuration. Source sample N до8192 synthetic,16000 ACS,9600 Beijing,8000 Appliances. Основные гипотезы изолируют общий N, K и source size imbalance в явно описанных factorial contrasts; статистическая единица — data replica, а не model seed.

Сильные внешние RF/ExtraTrees —200 trees, HistGBDT200 rounds. Независимые IRF/CRF/MaxRM —16 trees,depth4; Meta —3 iterations ×4 trees per source member, отдельные same-member weighting controls; Era80 rounds, Anchor100. Полные grids, interpretation и source-CV holdouts — series_v2/preregistration.md. Compute/capacity между внешними baselines и DG не выравнивается; внутри mechanistic pair фиксируются структура/admissibility/bootstrap по смыслу метода.

Исходный six-process запуск завершил757 jobs за15535.7 wall seconds (около4ч19мин) и остановил5 jobs на source MaxRM optimal_inaccurate. Пять отказов сохранены; остальные исходно зарегистрированные методы этих jobs завершены отдельным явным дополнением без solver fallback. Из запланированных40026 обычных fits и875 replay fits сохранены39996+875 успешных fits. Незавершённые вычисления отказавших jobs, source pilots и extra diagnostics не прибавляются к retained inventory. Это не оценка всех вызовов fit за время работы над проектом.

Время отдельных fits при конкурирующих процессах не является строгим hardware benchmark. Фактические timings, число доступных моделей и статусы — runs/expanded_v2/execution_summary.json, completion_inventory.json и reports/series_v2/tables/job_inventory.csv. Аудит и forensic replays выполнялись отдельно от обучения. Платные вычислительные ресурсы не использовались.
