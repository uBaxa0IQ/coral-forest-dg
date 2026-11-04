# Аудит данных до открытия target

Дата: 4 ноября 2026, Europe/Moscow. Фактические source shapes, схемы и SHA-256 приведены в `data/source/<dataset>/source_metadata.json`, копии метаданных — в `runs/study_cv_v1_*/`. Ниже описан собственный протокол, а не заявление о точной репликации всех исходных benchmark.

## ACSIncome

Источник: Folktables 0.0.12, Census ACS PUMS 2018 1-Year person. Задача и фильтрация — официальные ACSIncome. Source: CA, TX, NY, FL, IL, WA, GA, AZ. По 400 eligible наблюдений на штат, общий source 3200. Target заранее предложен в пакете: CO, NC, MA; до P2 эти файлы не загружались. State ID не является предиктором. Категориальные коды COW/MAR/OCCP/POBP/RELP/SEX/RAC1P обрабатываются как категории. Это наш state split, не официальный benchmark split.

## Hospital Readmission

Открытые UCI Diabetes 130-US Hospitals 1999–2008, тот же исходный набор, что использует TableShift. Официальный код TableShift подтверждает admission_source_id=7 как Emergency Room и y=(readmitted != NO), то есть повторная госпитализация вообще, а не только <30 дней.

Архив механически маршрутизируется по одному опубликованному ID домена в source и `data/target_locked/hospital.csv`. В target строки переносятся как непрозрачные байты; не создаются target X/y, class counts или статистики. Это операция раздельного хранения, не обучение/аналитическое чтение target. Source-обработчик затем читает только source CSV.

Source: удалить missing race, оставить первую запись каждого patient_nbr, исключить неизвестные/неотображённые источники 9/15/17/20/21, оставить группы с >=150 объектами и двумя классами. Итог: 6 сред 1–6, 2159 наблюдений после cap=400. Используются 18 явно заданных предикторов, доступных к моменту выписки; это собственная компактная схема, не точная feature engineering TableShift. ID пациента/encounter и admission_source исключены из X. Финальное target правило заранее: missing race удаляются, первый encounter пациента, пациенты из source training исключаются из target. Source после target не меняется.

## Beijing Multi-Site Air Quality

Официальный UCI 501, DOI 10.24432/C5RK5G, CC BY 4.0. Архив содержит отдельные station CSV; открыты только 6 source станций Aotizhongxin, Changping, Dingling, Dongsi, Guanyuan, Gucheng. По 400 записей с известным PM2.5, общий source 2400. Huairou/Shunyi выбраны заранее как target, их CSV до P2 не открывались. Другие станции в этой версии не используются. Station и No исключены из предикторов, 15 остальных признаков описаны в коде. Задача — contemporaneous regression PM2.5, включая другие загрязнители; не прогноз будущего загрязнения. Пространственный mixed shift, без утверждения чистого covariate shift.

## Gas Turbine CO

Официальный UCI 551, DOI 10.24432/C5WC95, CC BY 4.0. Набор содержит отдельные годовые CSV, согласно документации строки хронологически упорядочены, точных дат нет. Следуем официальному правилу: source 2011/2012/2013, target 2014/2015. Открыты только первые три файла. По 400 source записей на год, общий source 1200. Девять sensor predictors, CO — outcome; NOX и year исключены из X. Год хранится как time/environment ID. Source validation: train 2011–2012 → validation 2013. Доступен только один допустимый forward fold, что ограничивает надёжность выбора. Это перенос модели связи датчиков с CO на будущие годы, не прогноз датчиков вперёд.

## Изначальные кандидаты и замены

TabReD: текущая официальная инструкция `github.com/yandex-research/tabred` требует Kaggle authentication. Учётные данные не запрашивались, доступ не обходился; естественные eras и предобработка не подтверждены. Shifts: официальный storage.yandexcloud.net не прошёл TLS/read timeout в доступном локальном сетевом маршруте. Блокер зафиксирован до target, а не по итоговому качеству. Чтобы сохранить две реальные регрессионные задачи и разные механизмы shift, заранее включены открытые Beijing и Gas Turbine. Shifts/TabReD остаются направлениями расширения.

## Source sampling и предобработка

Cap=400 на домен, pandas sample с фиксированным random_state=20261104; target не сэмплируется. Этот source cap относится к первой ограниченной серии, а не к полным объёмам данных. Все imputers/scalers/OneHotEncoder обучаются внутри конкретного source fold, максимум 16 категорий на колонку, unknown=ignore. Source дубликаты/группы проверяются до обучения; target-признаки не используются для выбора схемы.
