"""Local status-file renderer. No API/model calls and no experiment mutations."""
import html
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/tuned_v4';OUT=ROOT/'reports/tuned_v4'


def read(name):
    path=RUN/name
    try:return json.loads(path.read_text(encoding='utf-8'))
    except (FileNotFoundError,json.JSONDecodeError):return {}


def main():
    previous=None
    while True:
        state=read('coordinator.json');progress=read('progress.json');access=read('global_target_access.json')
        current=json.dumps([state,progress,bool(access)],sort_keys=True)
        if current!=previous:
            stage=state.get('stage','starting');status=state.get('status','starting')
            body=f'<h1>НИР v4: ход исследования</h1><p>Состояние: <b>{html.escape(status)}</b>. Этап: <b>{html.escape(stage)}</b>.</p>'
            if progress:
                body+=f"<p>Завершённые задания этапа {html.escape(progress['stage'])}: <b>{progress['completed']} / {progress['planned']}</b>. Ошибки: {progress['failed']}.</p><p>Последнее обновление UTC: {html.escape(progress['updated_at'])}.</p>"
            body+='<p>Новые target '+('открыты после фиксации моделей и публикации final protocol.' if access else 'пока закрыты. Текущие source scores не являются финальными результатами.')+'</p>'
            if state.get('error'):body+='<h2>Продолжение остановлено</h2><pre>'+html.escape(state['error'])+'</pre>'
            body+='<p><a href="../../docs/tuned_v4/preregistration.md">Протокол</a> · <a href="../../runs/tuned_v4/coordinator.json">Состояние процесса JSON</a> · <a href="../../runs/tuned_v4/progress.json">Прогресс JSON</a></p>'
            if (OUT/'index.html').exists():body+='<h2>Результаты</h2><p><a href="index.html">HTML отчёт и таблицы</a> · <a href="research_report.md">Научный текст</a> · <a href="hypothesis_tests.csv">Основные тесты CSV</a></p>'
            if (OUT/'research_report.pdf').exists():body+='<p><a href="research_report.pdf">Полный PDF</a></p>'
            body+='<p>Страница обновляется автоматически. Вычисления и этот локальный индикатор не обращаются к языковой модели.</p>'
            doc='<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="20"><title>НИР v4: прогресс</title><style>body{font:18px/1.7 system-ui;max-width:1000px;margin:50px auto;padding:24px;color:#203548}a{color:#176397}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}</style>'+body+'</html>'
            (OUT/'status.html').write_text(doc,encoding='utf-8',newline='\n');previous=current
        if status_finished(state):break
        time.sleep(20)


def status_finished(state):return state.get('status') in {'complete','stopped_on_error'}


if __name__=='__main__':main()
