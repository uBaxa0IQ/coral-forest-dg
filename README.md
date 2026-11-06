# Coral Forest DG

НИР «Методы обучения ансамблей деревьев решений, устойчивых к сдвигу распределения табличных данных».

## Четыре раздела

- [code/](code/README.md) — весь программный проект, тесты, конфигурации, окружение, локальные данные и журналы экспериментов.
- [materials/](materials/README.md) — общие документы, литература/PDF, библиография, исходный пакет и происхождение шаблонов.
- [nir/](nir/README.md) — отчёт НИР и полный архив экспериментальных результатов.
- [vkr/](vkr/README.md) — отдельные материалы будущей ВКР.

## НИР: текст и шаблон

[Редактируемый LaTeX-проект](nir/report/README.md), [собранный PDF](nir/report/report.pdf), [главный файл](nir/report/report.tex), [поля титульного листа](nir/report/metadata.tex).

Оформление взято из курсовой `E:/CW/report/`. Из `E:/nir/` взят только порядок частей и основных глав; его класс/preamble не используется. В шаблоне есть начальное содержание и настоящая таблица результатов. Это каркас для дальнейшего написания, а не законченный научный отчёт. [Происхождение](materials/templates/provenance.json).

## Экспериментальные результаты

[Навигатор по всем материалам](nir/index.html). Серия v4 завершена: [научные выводы](nir/results/tuned_v4/results_brief.md), [PDF](nir/results/tuned_v4/research_report.pdf), [HTML/таблицы](nir/results/tuned_v4/index.html), [статистические тесты](nir/results/tuned_v4/hypothesis_tests.csv), [аудит](nir/results/tuned_v4/final_integrity.json).

Предыдущие серии сохранены отдельно: [обобщение v1/v2](nir/results/closeout/research_synthesis.md), [PDF](nir/results/closeout/research_synthesis.pdf), [аудит v1](nir/results/audit_v1/audit_report.md), [отчёт v2](nir/results/series_v2/research_report.md). Числа не подменены при реорганизации. Новые решения требуют свежего holdout; использованные target повторно независимыми не становятся.

## Запуск программного проекта

```powershell
python code/bootstrap_workspace.py setup
cd code
uv sync --frozen --extra research --extra test
uv run python -m pytest -q
uv run python bootstrap_workspace.py verify
```

Bootstrap создаёт только локальные совместимые представления для замороженных исторических путей. Физические документы и литература остаются в `materials/`, результаты — в `nir/results/`. [Карта переноса и проверки сохранности](materials/repository_layout/README.md).

Сборка документа из общего корня:

```powershell
./nir/report/build.ps1
```

Можно открыть [workspace](coral-forest-dg.code-workspace) в VS Code с четырьмя отдельными разделами.

## Данные и литература

Локальные raw data, модели и predictions исключены из Git. Оригинальные PDF находятся в `materials/literature/pdfs/`; в Git сохраняются [реестр](materials/literature/sources.json), [библиография](materials/literature/references.bib), адреса и hashes. Недоступные статьи не подменяются фиктивными копиями. Исходный handoff сохранён в `materials/research_input/`.
