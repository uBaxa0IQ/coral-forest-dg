# Пререгистрация расширенной серии

Написана до первого вызова новых target loaders. Машиночитаемый источник истины — runs/expanded_v2/frozen_protocol.json; настоящий документ объясняет дизайн. Статья/реализация и собственная инженерная интерпретация различаются явно.

## Объём и единицы анализа

381 context: 330 synthetic (33 условия × 10 независимых data replicas) и 51 real-data source resample. Два seed модели, 11 и 47: 762 задания. Синтетическая классификация: 17 условий; регрессия: 16. Семь типов сдвига, кроме отсутствующего регрессионного pure label-prior. Основной screen K=8,N=8192; дополнительные K=4/16, N=2048, дисбаланс 8:1 для spurious/heteroscedastic; K-контроль на IID.

ACSIncome и ACSPublicCoverage: N=4000/16000,K=4/8, три source resamples. Beijing: N=2400/9600,K=4/6. Appliances: N=2000/4000,K=4/8 и дополнительная точка N=8000,K=8. Это 5-кратное увеличение максимального обучения ACS относительно v1, 4-кратное Beijing; полные исходные пулы значительно больше. Новые ACS задачи используют один survey family, не независимые исследования.

18 конфигураций в классификации, 17 в регрессии. Две source-validation fits и одно full-source обучение каждой конфигурации: ожидаются 40 026 обычных fits и 875 повторов для воспроизводимости, всего 40 901. Ожидаются 13 342 full-source конфигурации, 41 652 per-target-domain строки метрик. Все количества проверяются фактическим inventory; неисполненные расчёты не называются результатом.

## Механизмы, пары и внешние модели

- IRF: независимый вариант v2, 16 деревьев, depth=4,min leaf=12, до 16 порогов на признак/узел, λ∈{0,.05,.5}; domain-stratified bootstrap. Сравнение с λ=0 при одинаковом RNG. Для регрессии **оба потомка** содержат каждую активную в узле среду при любом λ. Для всех λ требуется улучшение penalized parent objective. Эти правила исправляют сопоставимость, но отличаются от исторического v1 и не являются утверждением о точной авторской репликации. В классификации агрегируются голоса деревьев, smoothing .5/1 в ratio penalty сохранён.
- CRF: та же depth/tree/threshold capacity, pooled bootstrap, entropy + λ1 I(Y;E) − λ2 H(E); (λ1,λ2)∈{(0,0),(.05,.05),(.5,.5)}. Парный контроль — (0,0). Только binary classification; R-equivalence не подтверждается.
- Meta: интерпретация, 3 meta-iterations, 4 дерева на source forest, depth=5,min leaf=5, sampling ratio=.7. В каждой masking family одинаковые обученные members при uniform/accuracy/full weights. Full добавляет MMD к accuracy; MMD bandwidth=1/p после fold preprocessing, до 256 точек для MMD. Feature masking — отдельный фактор, не скрытое отличие baseline. Главный H1/H2 Meta без masking; masking и ablations вторичны.
- MaxRM: 16 фиксированных sklearn trees, depth=4,min leaf=12, одинаковые bootstrap и branch thresholds. Меняются только листья. Точное bootstrap empirical worst-domain MSE свёрнуто в leaf-level sufficient statistics, решается CLARABEL; после решения проверяется неухудшение исходного feasible worst risk. При неуспехе solver нет молчаливого fallback. Контроль — те же деревья до posthoc.
- Era и Directional Era: авторская библиотека 1.0.7, 80 итераций, 8 leaf nodes, 32 bins, min leaf=12, learning rate=.1. Strength∈{0,.1,1}; weights соответственно (s,0,1−s)/(0,s,1−s). Нулевая точка — **domain-restricted pooled-gain**, не обычный HistGBDT. Между двумя Era families нулевая точка одна и та же по коду/RNG.
- Anchor: авторская anchorboosting 0.4.0/LightGBM 4.6.0, 100 rounds, 8 leaves,min leaf=12, γ∈{1,2,8}, one-hot source anchors. Парный контроль γ=1. Objective binary/regression по задаче, alpha_cvar=1.
- Отдельные сильные практические baselines: RF и ExtraTrees, 200 деревьев,min leaf=5, max features=sqrt для binary/1 для regression; обычный HistGBDT,200 rounds,15 leaves. Это внешний уровень сравнения, а не одинаковая capacity с каждым DG-методом.

Regression labels центрируются/масштабируются **только на train части fit**, predictions возвращаются в исходные единицы. Категориальные признаки one-hot max_categories=16; median/most frequent imputation и standardization обучаются заново на train каждой fold. ID среды, row ID, временной timestamp и y не входят в признаки. ST у PublicCoverage удалён явно.

## Source-only выбор и новые targets

Две фиксированные leave-one-source-domain folds. В ACS остаются CA/AZ при обоих K; Beijing крайние станции в выбранном порядке; synthetic — крайние параметры. Выбор первой/последней среды обеспечивает сопоставимость K, но не заменяет полный leave-every-domain-out: внутренние механизмы в CV могут быть недопредставлены. Appliances validation всегда 2016-W09 и 2016-W12, train только раньше и с embargo 24h. Пропуски/категории никогда не оцениваются по target.

Primary selection: максимум среднего heldout AUROC, минимум среднего heldout MSE; regression MSE делится на full-source variance, одинаковую для всех кандидатов контекста. Порядок сетки разрешает ties в пользу baseline. Seed не подбирается. Все зарегистрированные full-source grid points обучаются и фиксируются до открытия target этого задания. Source selection JSON, model SHA и receipt дают проверяемую цепочку; receipt записывается перед чтением значений target.

Новые реальные targets: MN/OH/PA/OR/VA; Nongzhanguan/Tiantan/Wanliu/Wanshouxigong; будущие апрель/май одного дома. Старые target не вызываются v2 loaders. ACS sample rows внутри новых states не отбираются по качеству/метке — используются все официально eligible rows. Beijing исключает только неизвестный y. Appliances даты маршрутизированы по первому полю CSV до анализа остальных target полей, gap 31 марта–1 апреля исключён; данные после 27 мая не используются. Targets для синтетики: 3 distribution draws ×2000, отдельный RNG namespace. Они совпадают между K/N/imbalance/seed внутри data replica, чтобы разности были парными.

## Проверяемые гипотезы и статистика

H1/H2 — две отдельные confirmatory families по 9 method-task tests: classification IRF/CRF/Meta/Anchor; regression IRF/MaxRM/Era/Directional/Anchor. H1 сравнивает gain(K16)−gain(K4), N8192,balanced,spurious. H2 сравнивает gain(8:1)−gain(balanced), K8,N8192,spurious. Gain binary = AUROC(method)−AUROC(own zero control); gain regression = [MSE(control)−MSE(method)]/source variance.

Сначала равно усредняются target-домены и два model seeds, затем строятся 10 независимых replica contrasts. Двусторонний exact sign test, нули исключаются; при отсутствии ненулевых p=1. Holm отдельно для девяти H1 и девяти H2. 95% t-интервалы mean effects по 10 replicas — нормальная аппроксимация, не распределительно свободная гарантия. Результат ниже .05 после Holm нужен для confirmatory claim. При ограниченной мощности отсутствие значимости не означает равенства.

H3 (N), H4 (IID), H5 (выбор/кривые), H6 (Meta), H7 (шум) и все реальные сравнения — secondary descriptive. Retrospective best-on-target point, если показывается, называется oracle diagnostic и не выдаётся за результат выбранного метода. Одни и те же MC replicas внутри factorial сравниваются парно; model seeds и фиксированные target severity не создают новые независимые units. Реальные resamples пересекаются, states/stations зависимы, в Appliances один дом. Популяционные p-values по ним не рассчитываются.

## Целостность и ограничения

Перед target замораживаются code SHA каждого src файла и runner, uv.lock, package versions, source pools и row/content hashes всех contexts, grids, targets, statistical families. Исходный code commit фиксируется отдельно. JSON пишутся атомарно, модели и predictions хранятся локально. Повтор запуска завершённых заданий не открывает target; прерывание после target receipt блокирует автоматический retry.

Повторяем все конфигурации первой source fold для replica=0,seed=11 (875 fits), сравниваем predictions на 256 source probe rows с допуском 1e−8. Итоговый аудит отдельно пересчитает метрики по сохранённым predictions, проверит source/target IDs и модели. Нет OS sandbox против произвольного ручного чтения данных и независимой доверенной timestamp-службы; запрет логический, а не криптографическая гарантия. Нельзя скрывать неуспешные cases или переписывать runtime после target.

Beijing оценивает перенос contemporaneous PM2.5 modelling между станциями, с PM10 и другими синхронными загрязнителями; это не forecast. Appliances оценивает будущие периоды contemporaneous modelling, не причинное воздействие датчиков и не перенос на новые дома. ACS оценки невзвешенные, не survey-weighted population estimates.
