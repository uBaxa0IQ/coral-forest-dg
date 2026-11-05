# Проспективная версия: source-only инженерная проверка

Этот каталог изолирован от `src/nir_dg`: изменение исторического runtime нарушило бы frozen hash expanded_v2. Модели здесь **не использованы для новых target predictions** и не заменяют результатов final_v1/expanded_v2.

Исправления:

- IRF/CRF: midpoint реализует именно оценённую sorted partition; фактические children повторно проверяются на minimum leaf и, где применимо, domain support.
- Preprocessing: canonical missing values до fit/transform; проверяется cache roundtrip transformed matrix.
- MaxRM: эквивалентная centered conic форма; только OPTIMAL плюс независимые primal/dual, feasibility и feasible-baseline certificates. Допуск сертификата зафиксирован до source-only пилота. Сертификат относится к bootstrap source risk каждого дерева, не к ансамблю или target risk.
- Tuning: два capacity profiles, раздельные коэффициенты CRF, несколько regularization strengths, strong baselines. Это ограниченный пилот работоспособности, не оптимальное или равнобюджетное сравнение.

Итоговый научный материал: [обобщение](../../reports/closeout/research_synthesis.md), [реестр](../../reports/closeout/claims_registry.json). Исходные дефекты и отказавшие jobs сохраняются в историческом аудите.

Воспроизведение из корня репозитория:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_prospective_corrections.py -q
.venv/Scripts/python.exe tools/run_closeout_source_pilot.py
.venv/Scripts/python.exe tools/verify_closeout.py
```

Пилот фиксирует source manifest до fits, запрещает обычные target entrypoints в workers и сохраняет per-configuration results. Повторный вызов использует существующие jobs; для проверки детерминизма отдельный verifier повторяет первый fold выбранных source configurations и сравнивает bytes predictions. Логический запрет не является OS sandbox. Для любой новой научной оценки исправленных или настроенных pipelines нужен новый preregistered holdout.
