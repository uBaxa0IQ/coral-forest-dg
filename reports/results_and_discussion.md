# Методика и результаты НИР

Выполнено 1050 расширенных source-CV fits; успешных 1050. Финальных зарегистрированных model/seed запусков с метриками: 126/126.

## Экспериментальная постановка

Четыре реальные задачи: ACSIncome, Hospital Readmission, Beijing PM2.5 и Gas Turbine CO. Для первых двух оценивается бинарная классификация, для остальных регрессия. На каждой задаче есть RF и GBDT. Source ограничен 400 объектами на домен (в Hospital часть групп меньше). Расширенный подбор использует все доступные source folds и три seed 11/23/47. ACS и Hospital: LOEO; Beijing: leave-one-station-out; Gas Turbine: train 2011–2012, validation 2013. Внешний target не участвует. Все преобразования обучаются в fold.

Главные метрики: AUROC и RMSE. Каждый target-домен имеет равный вес в macro. Порог бинарного решения 0.5 фиксирован. Для Hospital исключены неизвестные группы, повторные source-пациенты; в финальном target удаляются пациенты, присутствующие в source training. Это собственный протокол по исходным данным UCI, не точная репликация TableShift. Beijing — contemporaneous PM2.5 prediction с другими загрязнителями, не прогноз будущих часов. Gas Turbine — contemporaneous CO по датчикам на будущих годах, не прогноз их значений.

## Достоверность методов

IRF и MSE-posthoc MaxRM независимо реализованы по формулам. CRF реализует paper objective с явными отличиями от R-процедуры роста. Era/Directional и Anchor используют неизменённые авторские пакеты с зафиксированными зависимостями. Meta-forests реализован полностью как явно подписанная независимая интерпретация недоопределённых правил. Точная Meta-forests репликация остаётся blocked. Числа из статей не воспроизводились и не используются как собственные. Подробности в docs/methods/.

## Результаты

Исходные результаты — `source_cv_metrics.csv`; сводки — `tables/source_cv_summary.csv`. Они показывают перенос между source-средами и не являются финальными target-показателями.

Для acs_income (AUROC) получены следующие средние macro-показатели по трём seed:

- anchor: 0.860396; SD по seed 0.000000; exact_author_code.
- anchor_paired: 0.860396; SD по seed 0.000000; baseline_only.
- crf: 0.809434; SD по seed 0.007915; independent_reimplementation.
- crf_paired: 0.818371; SD по seed 0.002593; baseline_only.
- hist_gbdt: 0.861032; SD по seed 0.000000; baseline_only.
- irf: 0.726347; SD по seed 0.023684; independent_reimplementation.
- irf_paired: 0.821473; SD по seed 0.006486; baseline_only.
- meta_forests_interpretation: 0.838951; SD по seed 0.002484; independent_interpretation.
- meta_matched_rf: 0.848962; SD по seed 0.001027; baseline_only.
- rf: 0.856182; SD по seed 0.000305; baseline_only.

Для beijing_pm25 (RMSE) получены следующие средние macro-показатели по трём seed:

- anchor: 25.970756; SD по seed 0.000000; exact_author_code.
- anchor_paired: 25.847518; SD по seed 0.000000; baseline_only.
- directional_era: 32.251172; SD по seed 0.000000; exact_author_code.
- era: 29.033716; SD по seed 0.000000; exact_author_code.
- era_paired: 27.976706; SD по seed 0.000000; baseline_only.
- hist_gbdt: 26.192257; SD по seed 0.000000; baseline_only.
- irf: 28.262893; SD по seed 0.134926; independent_reimplementation.
- irf_paired: 28.269190; SD по seed 0.099811; baseline_only.
- maxrm_paired: 28.447364; SD по seed 0.121210; baseline_only.
- maxrm_posthoc: 29.010080; SD по seed 0.176507; independent_reimplementation.
- rf: 27.154206; SD по seed 0.017934; baseline_only.

Для gas_turbine_co (RMSE) получены следующие средние macro-показатели по трём seed:

- anchor: 1.469326; SD по seed 0.000000; exact_author_code.
- anchor_paired: 1.465159; SD по seed 0.000000; baseline_only.
- directional_era: 1.813442; SD по seed 0.000000; exact_author_code.
- era: 1.510046; SD по seed 0.000000; exact_author_code.
- era_paired: 1.495436; SD по seed 0.000000; baseline_only.
- hist_gbdt: 1.449322; SD по seed 0.000000; baseline_only.
- irf: 1.433676; SD по seed 0.016532; independent_reimplementation.
- irf_paired: 1.430438; SD по seed 0.010083; baseline_only.
- maxrm_paired: 1.420274; SD по seed 0.007693; baseline_only.
- maxrm_posthoc: 1.415941; SD по seed 0.005563; independent_reimplementation.
- rf: 1.437259; SD по seed 0.003470; baseline_only.

Для hospital_readmission (AUROC) получены следующие средние macro-показатели по трём seed:

- anchor: 0.605090; SD по seed 0.000000; exact_author_code.
- anchor_paired: 0.605090; SD по seed 0.000000; baseline_only.
- crf: 0.543064; SD по seed 0.008650; independent_reimplementation.
- crf_paired: 0.574254; SD по seed 0.006754; baseline_only.
- hist_gbdt: 0.616094; SD по seed 0.000000; baseline_only.
- irf: 0.564579; SD по seed 0.009704; independent_reimplementation.
- irf_paired: 0.574597; SD по seed 0.001283; baseline_only.
- meta_forests_interpretation: 0.566324; SD по seed 0.012737; independent_interpretation.
- meta_matched_rf: 0.623656; SD по seed 0.003770; baseline_only.
- rf: 0.623960; SD по seed 0.002564; baseline_only.

Парные различия для каждого target-домена и seed доступны в `tables/paired_differences.csv`; положительное improvement означает улучшение. Все отрицательные значения сохранены. Графики используют среднее по доменам, затем по seed.

## Интерпретация и ограничения

Это первая ограниченная по объёму source-выборки сравнительная серия. Она не устанавливает универсальный рейтинг механизмов DG. Различия движков, сеток и ёмкости моделей требуют парных контролей. HPO базовых режимов иногда содержит меньше вариантов; реальные затраты сохранены, равный compute не заявляется. RF внешний контроль имеет 100 деревьев и подбираемую глубину, независимые DG-леса — 12 деревьев глубины 4; сопоставление механизмов выполняется прежде всего с их собственными контролями. GBDT используют 100 итераций.

SD по seed описывает случайность обучения, не доверительный интервал переноса. Несколько seed одного target не являются независимыми распределениями; две газотурбинные эпохи последовательны и могут зависеть. У Hospital один target, отдельная worst-domain статистика неинформативна. Формальные p-values не рассчитываются.

Положительные результаты интерпретации Meta-forests не подтверждают точную статью. Для точной репликации нужны уточнения Eq. 3, MMD и masking. У MaxRM не реализованы local/global/NRW/regret. У CRF нет R-equivalence proof. Shifts недоступен по текущему сетевому маршруту; TabReD требует Kaggle и проверки eras/preprocessing. Их отсутствие объясняется до открытия target, не его качеством.

## Источники и воспроизводимость

Библиография `literature/references.bib`, полный реестр `literature/sources.json`, фактические PDF/checksum `literature/download_manifest.json`. Ключевые источники: IRF DOI 10.1609/aaai.v38i12.29283; CRF PMLR 337; Meta-forests PMLR 222; MaxRM arXiv:2512.10445v2; Era DOI 10.1016/j.eswa.2025.129687; Anchor DOI 10.1093/rssdat/udag001. Протокол и hash в runs/final_v1/frozen_protocol.json, журналы в runs/, версии среды зафиксированы uv.lock.
