# Авторские Era Splitting и Anchor Boosting

## Era Splitting

Открытый препринт `literature/pdfs/era_splitting.pdf` сопоставляется с журнальной работой DOI 10.1016/j.eswa.2025.129687. Установлен авторский `erasplit==1.0.7`; исходники и commit журнала загрузки — `literature/upstream_era.json`. Для совместимости зафиксированы NumPy 1.26.4 и sklearn 1.3.2. Изменения алгоритма пакета не вносились.

Режимы (gamma, blama, vanna): Era=(1,0,0), Directional=(0,1,0), paired baseline=(0,0,1). Вход fit получает целочисленные source eras. Все temporal folds соблюдают порядок времени; early stopping выключен. Проверен только регрессионный API. В численной проверке на одном простом наборе paired baseline совпадает со стандартным HistGBDT при одинаковых bins/leaves/rounds; это не доказательство эквивалентности для всех данных и случаев равного gain.

Статус `exact_author_code` означает неизменённый авторский пакет, не репликацию таблиц статьи. Известны DeprecationWarning в Cython/Python интерфейсе era_indices; подавление предупреждения в отчёте тестов не изменяет вычисления.

## Anchor Boosting

Авторский `anchorboosting==0.4.0`, LightGBM 4.6.0; commit источника записан в `literature/upstream_anchor.json`. Источник DOI 10.1093/rssdat/udag001, открытая авторская версия — `literature/pdfs/anchor_boosting.pdf`.

Source environment ID передаётся как one-hot Z только при fit. Для регрессии — regression objective, для бинарной классификации — авторский probit. Параметр alpha_cvar=1 исключает добавленный CVaR-вариант. Gamma=1 служит парным контролем в том же пакете; не подменяется LightGBM logistic для бинарного probit. Target refit/DA не используется.

Статус `exact_author_code` для установленного пакета. Выполняются fit/predict обоих типов задач, позже сравнение gamma=1 regression с обычным LightGBM. Числа ICU из статьи не выдаются за собственные, а доступ к медицинским ICU базам не заявляется.
