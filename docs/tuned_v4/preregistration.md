# Углублённая source-only настройка и свежий финальный тест

Пользователь разрешил настройку всех методов с последующим новым зафиксированным тестом. Исторические v1/v2 и source-only closeout v3 сохраняются. Это новый этап, а не исправление опубликованных target-чисел.

## Исследовательские вопросы

1. Улучшает ли расширенный source-only поиск итоговое качество относительно ограниченного source-selected reference grid той же исправленной реализации?
2. Даёт ли выбранный механизм преимущество относительно собственного контроля при одинаковых capacity, preprocessing, RNG/bootstrap и допустимости splits?
3. Как зависят эти наблюдения от механизма сдвига и как они соотносятся с сильным source-selected RF/ExtraTrees/HistGBDT? Этот разбор вторичный.

Вопросы о числе доменов и дисбалансе уже исследованы v2. Новая серия сосредоточена на настройке; она не заявляет чистый эффект K или N.

## Пространства и бюджет

Синтетика: classification/regression для IID, covariate, spurious, concept, heteroscedastic, missingness; дополнительно classification label-prior. Development N8192,K8,replica4001. Реальный development: source pools ACSIncome/PublicCoverage N16000,K8; Beijing N9600,K6; Appliances N8000,K8. Всего17 контекстов и127 method/task/context families.

Каждая семья получает24 заранее перечисленных candidate configurations, включая restricted reference grid. Screening: два заранее выбранных source-heldout domains, seed11, ограниченный tree/round budget, все source rows. Затем все reference configurations и четыре лучших новых candidates оцениваются с полной capacity на четырёх folds и двух seeds11/47. Tie-break — меньший candidate ID. Ранжирование по macro AUROC или negative MSE/full-development-source variance. Любой failed fold исключает соответствующий candidate. Если нет валидного reference/метода, продолжение на target блокируется. Точные параметры находятся в source_protocol.json; это конечный random search, не доказательство максимума метода.

RF/ExtraTrees: число деревьев, глубина, leaf size, feature subsampling. HistGBDT: rounds, leaves, leaf size, learning rate, L2. IRF/CRF: trees, depth, leaf size, threshold resolution и penalty; у CRF коэффициенты раздельны. MaxRM: trees/depth/leaf size и robust flag, фиксированные независимые сертификаты оптимизации. Meta: iterations, trees/source, depth, sampling ratio, weighting coefficients, feature policy; независимая интерпретация, не exact replication. Era: rounds/leaves/bins/leaf size/learning rate/strength. Anchor: rounds/leaves/leaf size/learning rate/gamma.

Число candidates и promotion rule сопоставимы, CPU budgets не равны; длительности сохраняются. Повторяющиеся screening configurations используют один и тот же fit. Source CV также способна переобучаться: итоговая проверка использует новые source draws и targets. HPO выполняется один раз на development context, а не заново в каждой финальной replica. Сравнение оценивает такую фиксированную процедуру переноса настроек.

## Свежий тест и границы доступа

Синтетические final sources: N16384,K8,replicas5000–5009, два model seeds. Новые target draws: отдельные root seed2026110504 и namespace sealed_v4_target, три target domains по2000 rows; один draw общий для сравниваемых методов и model seeds внутри task/scenario/replica. Это новые MC draws внутри известных DGPs, не новая семья задач.

Новые реальные targets: ACS2018 MI,NJ,WI,MD,CT, совместно зарезервированные для Income/PublicCoverage. До freeze не загружаются и не читаются. Final source N32000,K8, три перекрывающиеся resamples, два seeds. Development и final source используют только исторически разрешённые восемь source states. Два ACS tasks относятся к общей survey family; статистически независимыми datasets они не объявляются.

У Beijing и Appliances нет свежего независимого target среди исходных уже изученных данных: здесь только углублённая source-only настройка. Старые target не загружаются для новой оценки. Нельзя заменить свежий реальный домен новой перестановкой строк, seed или resample.

До любого нового target чтения сохраняются все selections и все final source models; их hashes включаются в final_protocol.json. Протокол публикуется в Git до открытия. Новый target loader требует этот freeze; evaluator проверяет опубликованный Git blob. Canonical null schema действует до fit/transform и после cache reload; сохраняются hashes фактических transformed matrices. Labels/IDs новых ACS проверяются отдельными прямыми masks исходного CSV.

Воспроизводимость: выбранная configuration каждой method/context семьи заново обучается на первом source fold с seed11; bytes predictions должны совпасть. Сохранённые final models повторно проверяются на256 первых target rows после окончания оценки, без выбора параметров. AUROC для каждой target строки независимо пересчитывается через rank sums с ties, MSE — отдельной residual формулой. Paired IRF/CRF/MaxRM bootstrap проверяется фактическими indices; MaxRM также сохраняет структуру деревьев. Это проверки конкретного окружения, не обещание bitwise совпадения между любыми CPU/версиями.

## Контроли и отказы

Два трека: (а) расширенно настроенный pipeline против restricted source-selected reference; (б) тот же выбранный structural config с отключённым механизмом. IRF/CRF сохраняют admissibility; Meta uniform использует буквально те же обученные members и выбранную feature policy; MaxRM сохраняет bootstrap/structure; Era zero-strength остаётся domain-restricted pooled gain, обычный HistGBDT показывается отдельно; Anchor gamma1.

Если final source fit метода или его обязательного контроля отказывает, отсутствие фиксируется до target, метод не подменяется baseline; остальные методы разрешено завершить. Такая replica делает соответствующий основной contrast неполным. Нет автоматических target-stage retries после receipt: прерывание требует явного forensic разбора. Никакой solver relaxation или quality-driven fallback. Допуск MaxRM фиксирован:1e-6·max(1,primal), OPTIMAL и независимые primal/dual/feasibility checks.

## Статистика

Предзадан один Holm family из24 two-sided exact sign tests:15 method/task HPO-minus-reference contrasts и9 nonbaseline method/task mechanism-minus-control contrasts. Единица — независимая synthetic data replica; target domains и seeds усредняются, затем зарегистрированные scenarios получают равный вес. Classification gain — AUROC difference; regression gain — (control MSE−method MSE)/full-final-source variance. Не строится общий сырой RMSE между задачами.

Каждый контраст требует все предусмотренные scenario/seed pairs во всех10 replicas; неполный тест не выдаётся за confirmatory. Для коррекции остальных отсутствующий p технически учитывается как1 с сохранением ширины семьи; его собственные p/adjusted p остаются отсутствующими. Нулевые contrasts исключаются из sign test. Маргинальный Student t95% CI среднего показывается отдельно: sign test проверяет знаки, не mean-zero, CI не одновременный. Пересечения CI с нулём и p могут расходиться.

Отдельные scenarios, сильный baseline и реальные штаты — descriptive/exploratory, без отбора положительных эффектов в подтверждающий вывод. Неотклонение не доказывает эквивалентности. Все отрицательные результаты и computational failures сохраняются. Код и планы после source freeze нельзя менять без явного нового source протокола; после target доступа новые решения потребуют нового holdout. Локальные hashes/receipts не заменяют OS-level независимый access audit.
