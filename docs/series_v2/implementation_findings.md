# Ограничения реализации, установленные аудитом второй серии

Это описание исходного замороженного исполнения. Исторический runtime, выбранные configurations, predictions и метрики не исправляются задним числом.

## MaxRM: source solver status

Пять из 374 зарегистрированных regression jobs остановились на source с CLARABEL optimal_inaccurate. Правило принимать только OPTIMAL сохранено. Отказы и явное дополнение о завершении остальных методов находятся в source_failure_addendum.md. H2 MaxRM не имеет всех десяти полных replica pairs и не объявляется полностью выполненным подтверждающим тестом. Отказ численной сертификации относится к выбранной реализации/solver settings, а не доказывает несуществование математического optimum.

## Почти совпадающие floating-point значения в split thresholds

В source-only replay Appliances обнаружен промежуточный IRF CV leaf размером 10 при заданном minimum12. При вычислении midpoint между очень близкими значениями hour_sin округление может изменить фактическую partition относительно candidate position. Это дополнительное численное условие, а не ошибка независимого пересчёта метрик. Наблюдаемые validation predictions этого replay конечны. Фактические partitions всех сохранённых full-source IRF/CRF дополнительно проверены в audit/tree_numeric_checks.csv; итоговые числа необходимо читать из этого файла, не переносить результат одного replay на все деревья.

Для будущей версии нужен устойчивый threshold с гарантией left_value≤threshold<right_value, повторная проверка реальных размеров/поддержки после split и source-only regression tests. Изменённый алгоритм нельзя оценить на уже открытых targets как на новом независимом финале.

Итог полного аудита: девять нарушений фактической domain support в сохранённых IRF/Appliances моделях; среди них один empty leaf с NaN value и нарушением minimum leaf. Все observed predictions конечны; leaf не был достигнут сохранёнными source/target rows. Поэтому проверка метрик прошла, но полная корректность реализации не заявляется. Эти Appliances пары не дают чистого механистического доказательства. В full-source synthetic моделях H1/H2 таких нарушений не обнаружено. Точные IDs и configs сохранены в tree_numeric_checks.csv.

## Тип пропуска при cache roundtrip Beijing

Target loader записывает Parquet и возвращает uncached DataFrame создавшему cache заданию; последующие задания читают Parquet. В CSV wd missing представлен np.nan, после Parquet reload object-колонка содержит None. SimpleImputer(missing_values=np.nan) различает эти scalars: один импутируется, другой может стать categorical value. pandas hash_pandas_object трактует оба как пропуск и не обнаруживает эту разницу семантики model input.

У cache-creating job 5259a196ef641ff1 (Beijing,K4,N2400,replica0,seed11) различаются 357 null wd rows. Сохранённые модели воспроизводят исходные predictions при восстановлении CSV np.nan; обычное чтение cache меняет некоторые predictions. Это input-representation defect, не изменение y или row IDs. Полные векторы дополнительно проверяются в audit/null_cache_full_vector_replay.json, а чувствительность RMSE — в null_cache_sensitivity.csv. Официальные predictions/метрики не подменяются пересчитанными альтернативами.

Все модели внутри этого задания использовали одну transformed matrix, поэтому mechanistic pairs получают одинаковый input. Между Beijing jobs cache path создал дополнительное различие представления пропусков; сравнения K/seed на реальных данных имеют эту оговорку. Основные H1/H2 synthetic contrasts этим случаем не затронуты. Source pools уже читались из Parquet одинаково для всех source fits.

Для будущего стенда необходима canonical null schema перед fit/transform и cache roundtrip test, сравнивающий также transformed features. Желателен SHA фактической transformed target matrix/column order до predict. Одного семантического dataframe fingerprint недостаточно. Исправленное preprocessing является новой версией pipeline; новые научные решения потребуют нового holdout.

## Сила вывода

Независимый пересчёт подтверждает метрики **сохранённых predictions**. Saved-model replay подтверждается для восстановленного исходного input representation. Это разные утверждения; наличие первого не гарантирует второго при изменении представления данных. Доверенная внешняя timestamp/OS access isolation также не подменяются локальными hashes и receipts.
