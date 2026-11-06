# Выводы свежего теста v4

Серия завершена 6 ноября 2026 года. Настройки и final source models опубликованы до открытия новых target. Исторические v1/v2 не подменены. Здесь обобщаются результаты именно исправленных и настроенных v4 pipelines, а не точных репликаций авторских исследований.

## Основной результат

Углублённая source-only настройка помогла многим моделям. Положительный вклад собственно DG-механизма оказался существенно уже. Большой выигрыш относительно ограниченного reference grid сам по себе не означает победы над сильным baseline или устойчивости ко всем видам сдвига.

Все 272 final jobs завершены; выполнены все 24 предзаданных теста. После единой Holm-коррекции прошли 15 контрастов: 11 положительных HPO-versus-reference, два положительных mechanism-versus-control и два отрицательных mechanism-versus-control. Источник: [hypothesis_tests.csv](hypothesis_tests.csv), filters `hypothesis`, `task`, `method`; полнота — [final_integrity.json](final_integrity.json).

## Что дала настройка

На зарегистрированной синтетической классификации средний HPO gain IRF составляет +0.035031 AUROC, CRF +0.035034, то есть примерно +3.50 процентного пункта AUROC относительно source-selected restricted reference. У RF +0.003766, ExtraTrees +0.003002, Meta +0.002850. Для этих контрастов Holm p=0.046875. Anchor classification также прошёл тест, но mean gain всего +0.0000277 AUROC: статистическая значимость не делает этот эффект практически большим. [Таблица](hypothesis_tests.csv), `hypothesis=HPO, task=classification`.

В regression HPO положительно прошла Holm для RF, HistGBDT, IRF, MaxRM и Era. Единица regression gain — снижение MSE, делённое на full-source variance; это не процент снижения исходной ошибки. У Anchor regression tuned и reference совпали по итоговому результату; у Directional Era подтверждающего HPO выигрыша не получено. [Таблица](hypothesis_tests.csv), `hypothesis=HPO, task=regression`.

## Что дали сами механизмы

Положительные механизм-контрасты после Holm: Meta classification +0.002372 AUROC и Era regression +0.014899 в единицах source-normalized MSE gain. Meta сравнивается с uniform weights тех же обученных members; Era — с domain-restricted pooled gain при тех же structural settings. Это не сравнение Era с обычным sklearn HistGBDT. [Таблица](hypothesis_tests.csv), `hypothesis=mechanism, method=meta/era`.

У IRF classification средний mechanism gain −0.000431 AUROC, у CRF −0.001086, оба Holm p=0.046875. Следовательно, их большие HPO улучшения нельзя приписать положительному действию DG-регуляризации. Для остальных механизмов подтверждающего выигрыша после Holm нет; это не доказательство эквивалентности или нулевого эффекта. [Таблица](hypothesis_tests.csv), `hypothesis=mechanism`.

## Свежие реальные домены и отрицательные результаты

На новых ACS штатах лучший средний tuned AUROC среди зарегистрированных методов имеет HistGBDT: Income 0.875572, PublicCoverage 0.821724. Для Income настроенные IRF/CRF получили 0.841363/0.840639; для PublicCoverage — 0.763127/0.763932. Улучшение относительно их reference существенно, но абсолютное качество остаётся ниже сильного baseline. Эти реальные means описательные: source resamples перекрываются, оба tasks относятся к одной survey family. [real_summary.csv](real_summary.csv), filters `context`, `method`, `variant=tuned`; [source parameters](source_selection.csv).

В синтетическом spurious reversal средние tuned AUROC всех классификационных методов остались ниже 0.4. Следовательно, положительные macro results по набору сценариев не означают решения проблемы shortcut reversal. [scenario_summary.csv](scenario_summary.csv), `context=classification_spurious, variant=tuned`. Сравнение с сильным baseline также опубликовано отдельно, baseline выбран по source: [strong_baseline_gains.csv](strong_baseline_gains.csv).

## Воспроизводимость и границы вывода

Сохранено 11952 успешных source-CV fits, отказавших source trials и final source methods нет. Проверены 127 выбранных source configurations повторным обучением и 5256 сохранённых model probes; максимальное расхождение replay predictions равно нулю. Независимые primary metric checks: 16200 строк; максимальное AUROC расхождение 3.33e−16, MSE 3.55e−15. Полные значения, статусы и result artifact hashes: [final_integrity.json](final_integrity.json), [evidence_manifest.json](evidence_manifest.json).

Основные тесты относятся к десяти независимым MC data replicas внутри известных DGPs, при настройке один раз на development source каждого контекста. Holm p у прошедших тестов равен 0.046875; ограниченное число replicas даёт дискретную шкалу sign-test p. Это не универсальная гарантия и не независимая выборка всех табличных задач. Sign test проверяет знаки, CI описывает среднее при отдельной t аппроксимации. Равные CPU budgets и глобальные максимумы качества не заявляются. Новые решения по этим уже открытым targets потребуют другого holdout. [Пререгистрация](../../docs/tuned_v4/preregistration.md), [final protocol](../../runs/tuned_v4/final_protocol.json).

Вывод для НИР: необходимо отдельно оценивать пользу настройки capacity, пользу конкретного DG-механизма и абсолютное качество сильных baseline. Текущие результаты подтверждают необходимость этого разделения и дают как положительные, так и существенные отрицательные наблюдения.
