# Проверка изоляции target

Train и source-CV API не имеют target аргумента или target loader. `datasets.py` регистрирует только source-наборы; защищённый модуль target_evaluation отделён. Механическая маршрутизация Hospital читает только domain ID и копирует непрозрачные строки; клинические значения target в аналитический код не поступают.

Перед вызовом final loader `TargetVault` проверяет наличие frozen_protocol.json, его SHA-256, fingerprint Python-кода и preregistered experiment ID. Source fingerprint проверяется ещё раз. Доступ записывается эксклюзивным receipt для model/seed запуска; повторный доступ без оформленного восстановления запрещён. Модель и preprocessing обучаются и сохраняются до чтения target. Raw predictions хранятся локально и исключены из Git.

Это логическая защита интерфейса и проверяемый журнал, не ОС-песочница: процесс с произвольным Python-доступом технически может открыть локальный raw архив. Такой обход нарушил бы протокол. Рабочие команды не используют его; тесты проверяют штатные границы и отсутствие вызова loader до freeze.

Проверены непересечение source domain IDs между train/validation, порядок времени в Gas Turbine, fold-local imputation/scaling и поведение unknown categories. Hospital сохраняет patient IDs для контроля; source пациенты уникальны, final target исключает их согласно заранее принятому правилу. ACS states и Beijing stations disjoint по конструкции.

Не используем target mean/std, class balance, missingness, список категорий, early stopping или score для выбора решений. Все synthetic/meta-test среды в тестах/пилотах исходные. Seed, domains, grids, selected source configurations и feature schemas фиксируются до первого target inference. После final допустим только анализ уже полученных чисел, без retuning.
