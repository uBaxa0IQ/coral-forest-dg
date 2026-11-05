# Coral Forest DG

НИР «Методы обучения ансамблей деревьев решений, устойчивых к сдвигу распределения табличных данных».

Исследуем существующие RF и GBDT в классификации и регрессии при строгой multi-source, source-only Domain Generalization. Собственный новый алгоритм в текущую НИР не входит. Ни один метод заранее не считается лучше обычной модели.

## Где что хранится

- `research_input/NIR_Codex_Handoff_v2/` — неизменённый пакет передачи и исходные обзоры.
- `docs/` — наша рабочая документация: постановка, аудит, решения, спецификации методов и протокол.
- `literature/pdfs/` — локальная библиотека полных текстов; `literature/sources.json` — реестр источников и адресов; `literature/download_manifest.json` — фактические загрузки, SHA-256 и ошибки.
- `src/nir_dg/` — стенд, модели, разбиения, защита target, оценка.
- `configs/` — явные конфигурации экспериментов.
- `tests/` — проверки математики и научного протокола.
- `data/` — локальные данные, не попадают в Git.
- `runs/` — журналы и результаты отдельных запусков.
- `reports/` — сводные результаты и текст НИР.

PDF сохраняются локально без изменения. Публичный Git содержит ссылки, библиографию и контрольные суммы; массовое переопубликование чужих PDF не предполагается. Библиотека восстанавливается командой загрузки. Отсутствующий PDF не заменяется аннотацией или выдуманной копией.

## Начало работы

```powershell
uv sync --extra research --extra test
uv run python tools/fetch_literature.py
uv run pytest -q
```

Рабочий статус и границы фактически выполненного описываются в [docs/status.md](docs/status.md). Полный цикл: P0 аудит → P1 source-only пилоты → P2 фиксация протокола → P3 предзаданная серия target-оценок → P4 научный анализ. Наличие кода не означает выполненного эксперимента, а доступный репозиторий метода не означает проверенной репликации.

## Полученные результаты

Вторая серия завершена 5 ноября 2026 с явными оговорками аудита: **757 полных заданий и 5 с source-отказом MaxRM**, 40 871 сохранённый успешный source fit, 13 332 full-source конфигурации и 41 622 target-domain оценки. Независимо пересчитаны 273 144 метрики; saved-model predictions воспроизведены при исходном представлении входов. Основные результаты и ограничения:

- [Полный отчёт PDF](reports/series_v2/research_report.pdf), [текст](reports/series_v2/research_report.md), [читатель с фильтрами](reports/series_v2/index.html).
- [Краткие выводы](reports/series_v2/executive_summary.md), [CSV и словарь полей](reports/series_v2/data_dictionary.md), [итоговый аудит](reports/series_v2/audit/final_integrity.json).
- [Независимый пересчёт первой серии](reports/audit_v1/audit_report.md), [пререгистрация v2](docs/series_v2/preregistration.md), [ограничения реализации](docs/series_v2/implementation_findings.md).

В предзаданном spurious-сценарии H1 увеличение K4→K16 при одном N снизило преимущество IRF и Meta в classification после Holm. Это результат о конкретных source-selected pipelines. Source/target ranking настроек может расходиться; универсального победителя DG на реальных задачах нет. H2 MaxRM неполон, неуспешные fits не подменялись baseline. Обнаружены также IRF/Appliances numerical split defects и np.nan/None cache-различие первого Beijing job. Числа не исправлялись задним числом.

Библиотека теперь содержит 63 оригинальных PDF-файла, 52 уникальных hashes; две старые статьи остаются недоступными. Web-документация UCI хранится отдельно и не выдаётся за оригинальный PDF. [Аудит библиотеки](docs/literature_audit.md).

Уже использованные targets перечислены в [реестре экспозиции v2](configs/target_exposure_ledger_v2.json). Они не являются новыми независимыми финальными тестами для дальнейших решений.

### Архивная первая серия

Первая ограниченная серия завершена: 1050 расширенных source-CV fits, 60 Meta-forests ablations, 126 финальных model/seed запусков и 252 per-domain оценки. Проверены 15 тестов. См. [научный разбор](reports/research_findings.md), [полные результаты](reports/results_and_discussion.md), [статусы методов](reports/method_matrix.csv), [ограничения](reports/limitations.md) и [пререгистрацию](docs/preregistration.md).

Это capped-source исследование, не завершённая точная репликация всех авторских статей. Meta-forests остаётся независимой интерпретацией. Библиотека содержит также исторические кандидаты; отсутствующие PDF перечислены в [аудите библиотеки](docs/literature_audit.md).

## Воспроизведение источников и вычислений

```powershell
uv sync --extra research --extra test
uv run python tools/prepare_sources.py
uv run python -c "from nir_dg.datasets import prepare_gas; prepare_gas()"
foreach ($dataset in @('acs_income','hospital_readmission','beijing_pm25','gas_turbine_co')) {
  uv run nir-dg source-cv --config configs/study_source_cv.yaml --run-dir "runs/reproduction_$dataset" --datasets $dataset
}
```

Source-подготовка не требует target. Свежие source-CV результаты сохраняются отдельно в `runs/reproduction_*`, их можно сравнить с архивными `runs/study_cv_v1_*`. Уже сохранённые run directories не перезаписывают результаты несовместимым кодом. `tools/build_reports.py` собирает **архивную первую серию** из study_cv_v1 и final_v1, не из произвольных reproduction directories. Meta-ablations уже сохранены в runs/meta_ablations_v1; скрипт добавляет записи при повторном вызове, поэтому его не следует запускать поверх архива.

Для точной архивной среды вычислений final_v1 используйте commit `27a8663`; после него была только библиографическая правка Python-инструмента, описанная в docs/protocol_changes.md. Первоначальная серия была запущена командами `tools/final_study.py freeze` и `tools/final_study.py evaluate`; повторно запускать её на том же target для подбора моделей нельзя. Проверка готовых отчётов не открывает raw target.
