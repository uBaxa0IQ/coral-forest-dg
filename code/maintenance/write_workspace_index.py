"""Canonical entry points for the relocated workspace."""
from pathlib import Path
import json
import re

ROOT=Path(__file__).resolve().parents[2]


def main():
    readme='''# Coral Forest DG

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
'''
    (ROOT/'README.md').write_text(readme,encoding='utf-8',newline='\n')
    links=[
        ('Отчёт НИР: редактируемый LaTeX','report/README.md'),('PDF шаблона НИР','report/report.pdf'),('Титульные реквизиты','report/metadata.tex'),
        ('V4: научные выводы','results/tuned_v4/results_brief.md'),('V4: PDF','results/tuned_v4/research_report.pdf'),('V4: HTML и таблицы','results/tuned_v4/index.html'),('V4: статистические тесты','results/tuned_v4/hypothesis_tests.csv'),('V4: проверка целостности','results/tuned_v4/final_integrity.json'),('V4: зафиксированный протокол','../code/runs/tuned_v4/final_protocol.json'),
        ('V1/V2: обобщение','results/closeout/research_synthesis.md'),('V1/V2: PDF','results/closeout/research_synthesis.pdf'),('V2: полный отчёт','results/series_v2/research_report.md'),('V1: аудит','results/audit_v1/audit_report.md'),('Рабочая документация','../materials/docs'),('Литература и PDF','../materials/literature'),('Программный проект','../code/README.md'),('Материалы ВКР','../vkr/README.md'),('Проверка переноса','../materials/repository_layout/README.md')]
    items=''.join(f'<li><a href="{path}">{label}</a></li>' for label,path in links)
    page='<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>НИР: материалы и результаты</title><style>body{font:18px/1.65 system-ui;color:#203548;max-width:1050px;margin:40px auto;padding:24px}a{color:#176397}li{margin:9px 0}h1{line-height:1.2}</style><h1>Ансамбли деревьев при сдвиге распределения</h1><p>Общий навигатор: новый отчёт НИР, завершённые серии и исходные доказательства.</p><p>LaTeX-проект содержит каркас и начальный текст. Полные экспериментальные результаты находятся отдельно; их исходные файлы сохранены.</p><ul>'+items+'</ul></html>'
    (ROOT/'nir/index.html').write_text(page,encoding='utf-8',newline='\n')
    missing=[]
    for label,path in links:
        if not (ROOT/'nir'/path).exists():missing.append(path)
    for label,path in re.findall(r'\[([^\]]+)\]\(([^)]+)\)',readme):
        if not (ROOT/path).exists():missing.append(path)
    validation=dict(nir_navigation_links=len(links),root_readme_links_checked=len(re.findall(r'\[([^\]]+)\]\(([^)]+)\)',readme)),missing=missing)
    (ROOT/'materials/repository_layout/navigation_validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8',newline='\n')
    print(json.dumps(validation));assert not missing


if __name__=='__main__':main()
