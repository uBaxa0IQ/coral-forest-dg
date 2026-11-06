# Углублённая настройка v4

[Пререгистрация](../../docs/tuned_v4/preregistration.md) описывает гипотезы, source-only поиск, свежие данные, контроли, отказы и статистику. Runtime и grids заморожены в [source protocol](../../runs/tuned_v4/source_protocol.json). Исторические v1/v2/v3 не меняются.

Порядок исполнения:

```powershell
.venv/Scripts/python.exe tools/run_tuned_v4.py --stage prepare
# Опубликовать source protocol и код.
.venv/Scripts/python.exe tools/continue_tuned_v4.py
```

Coordinator последовательно выполняет source search и selected-source replays, публикует selections, обучает final source models, фиксирует и публикует final protocol, затем открывает новые targets, проверяет результаты и строит отчёт. Ошибка любого этапа останавливает продолжение; scientific runtime и параметры не редактируются во время выполнения. Target-stage interruption требует forensic проверки, автоматического retry нет.

Прогресс: [локальная страница состояния](../../reports/tuned_v4/status.html) обновляется автоматически без обращений к модели; `runs/tuned_v4/coordinator.json` — текущий этап; `runs/tuned_v4/progress.json` — законченные задания внутри этапа. Логи и готовые материалы: `reports/tuned_v4/`. Промежуточные source scores не являются target результатами. Код coordinator и журнал публикаций доступны отдельно; работа вычислительного процесса не требует постоянных обращений к языковой модели.

Итоговые файлы после завершения: `research_report.md/.pdf`, `index.html`, `method_cards.md`, `hypothesis_tests.csv`, `source_selection.csv`, `target_metrics.csv`, `final_integrity.json`, `evidence_manifest.json`. PDF проходит автоматический render/bounds check; contact sheet дополнительно требует визуальной проверки. Сырые данные, модели и predictions хранятся локально и исключены из Git.

Новые synthetic draws проверяют известные DGPs. Новые реальные targets — только зарезервированные ACS штаты, без старых target. Beijing и Appliances здесь получают source-only HPO. Нельзя утверждать, что конечный поиск нашёл глобальный максимум каждого метода, или что CPU бюджеты методов строго равны.
