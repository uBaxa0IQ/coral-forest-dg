# Изменения протокола

До первой фиксации P2 разрешённые source-only решения перечислены в decisions.md, data_audit.md и compute_budget.md. Target не использовался при их принятии.

После фиксации любые исправления кода, изменение состава моделей/метрик/данных или повтор target-доступа требуют отдельной записи здесь. Предыдущие протоколы, receipts и результаты сохраняются. Тот же target после просмотра результата не считается новым holdout.

## 2026-11-04 После полного завершения final_v1

После всех 126 запусков исправлены раскрытые имена первых двух авторов Anchor Boosting в библиографическом реестре и константе инструмента загрузки: Malte Londschien, Manuel Burger. Проверено по первой странице сохранённого PDF. Формулы, реализации моделей, source/target loaders, выбранные параметры и evaluation не менялись; predictions не пересчитывались, target повторно не открывался.

Поскольку общий code_hash включает также Python-инструмент библиотеки, эта библиографическая правка меняет hash текущего checkout. Точный исполнявшийся frozen-код сохранён в Git commit `27a8663`. Его hash проверяется отдельно по Git blobs в reports/final_integrity.json. Для точного восстановления final_v1 нужен этот commit; текущий TargetVault закономерно откажет в повторном оценивании старого протокола с изменённым общим hash. Это не новая серия и не повторный тест.

## 2026-11-04 Вторая серия до target

После независимого пересчёта v1 исправлены условия mechanistic controls в отдельном namespace series_v2. Новые targets, гипотезы, statistical families, grids, source fingerprints, uv.lock и versions заморожены в protocol4b57843c27bb871d6b4ee494715024b99e37b6730420029943836ab7288be7a8. Исходный code commit bd3dca2; manifest опубликован в b0325a5 до новых target calls. Исторические файлы v1 не перезаписывались.

## 2026-11-05 Обработка source solver failures

Пять MaxRM jobs остановились до своих target receipts с optimal_inaccurate. Код дополнения, список IDs и omission policy опубликованы в9bc26d7 до их продолжения. Остальные семь ранее зарегистрированных families завершены при прежних данных, seeds, code и grids; оба MaxRM points этих jobs отсутствуют. Исходные training source files не редактировались; дополнительный исполнитель явно фильтрует MaxRM для указанных пяти jobs. Это не новый независимый тест решений, принятых по открытым targets. Полнота H2 MaxRM и консервативный Holm описаны в series_v2/source_failure_addendum.md.

## 2026-11-05 Аудит implementation/input edge cases

Source-only diagnostics обнаружили floating-point min-leaf violation в промежуточном Appliances CV tree. Frozen algorithm не исправлялся по результатам target. Полные saved trees проверяются отдельно; ограничения фиксируются в implementation_findings.md.

При model replay первого Beijing job обнаружено изменение null scalar np.nan→None после Parquet cache reload. Восстановление исходной CSV-null семантики воспроизводит полные vectors всех17 моделей точно. Официальные модели, predictions, метрики и выбранные parameters не заменялись; alternate-cache sensitivity — только forensic diagnostic. Аудитор явно фиксирует reconstructed input provenance. Новая canonical-null preprocessing версия потребует отдельной prospective оценки, а эти targets остаются уже открытыми.

Библиографические загрузки, проверочный код и тексты отчётов вне экспериментального runtime scope могут обновляться. Их изменения не объявляются заранее замороженным training code. Реестр потреблённых targets сохранён отдельно в configs/target_exposure_ledger_v2.json; frozen registry не переписывается.

При публикации отключена Git newline normalization для runs/expanded_v2 и reports/series_v2: receipts содержат SHA raw source_selection bytes, поэтому CRLF→LF изменило бы проверяемые файлы. Содержимое JSON-протокола и его canonical semantic digest не меняются. Предыдущие commit snapshots сохранены; публикация текущих artifacts сохраняет именно исходные локальные bytes.
