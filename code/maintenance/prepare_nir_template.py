"""Use coursework typesetting and NIR outline, not the NIR class/preamble."""
from hashlib import sha256
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'nir/report'
TEMPLATES=ROOT/'materials/templates'
PIN='a2e54a1f3fe933c2c7ec20e902f8e0571b412c28'


def put(name,text):
    p=REPORT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text.strip()+'\n',encoding='utf-8',newline='\n')


def main():
    REPORT.mkdir(parents=True,exist_ok=True)
    cw=Path('E:/CW/report');nir=Path('E:/nir/main')
    sources={}
    for src,dest in [(cw/'preambula.tex',TEMPLATES/'coursework/preambula.original.tex'),(cw/'report.tex',TEMPLATES/'coursework/report.original.tex'),(nir/'report.tex',TEMPLATES/'nir_structure/report.original.tex')]:
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(src.read_bytes());sources[str(src)]=dict(sha256=sha256(src.read_bytes()).hexdigest(),archived_at=dest.relative_to(ROOT).as_posix())
    outline={}
    for name in ['02-essay.tex','10-introduction.tex','20-analysis.tex','30-conclusion.tex','40-appendix.tex']:
        src=nir/name;text=src.read_text(encoding='utf-8')
        sources[str(src)]=dict(sha256=sha256(src.read_bytes()).hexdigest(),use='outline only; no body/style copied')
        outline[name]=re.findall(r'\\(?:chapter|section)\*?\{([^}]*)\}',text)
    (TEMPLATES/'nir_structure/source_outline.json').write_text(json.dumps(outline,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
    (REPORT/'preambula.tex').write_bytes((cw/'preambula.tex').read_bytes())
    provenance=dict(style_source='E:/CW/report/preambula.tex',structure_source='E:/nir/main',structure=['title','essay','contents','introduction','subject analysis','methodology and evaluation','conclusion','references','appendices'],nir_bmstu_class_or_preamble_copied=False,coursework_preamble_byte_identical=True,source_files=sources,scientific_data_source='nir/results/tuned_v4',artifact_permalink_commit=PIN,status='editable template with initial research content, not a finished NIR manuscript')
    (TEMPLATES/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
    put('metadata.tex',r'''
% Fill these fields before submitting. No signatures/approvals are fabricated.
\newcommand{\ReportUniversity}{Наименование образовательной организации}
\newcommand{\ReportFaculty}{Факультет / институт}
\newcommand{\ReportDepartment}{Кафедра}
\newcommand{\ReportStudent}{Фамилия Имя Отчество}
\newcommand{\ReportGroup}{Номер группы}
\newcommand{\ReportSupervisor}{Фамилия Имя Отчество}
\newcommand{\ReportCity}{Москва}
\newcommand{\ReportYear}{2026}
\newcommand{\ReportTitle}{Методы обучения ансамблей деревьев решений, устойчивых к сдвигу распределения табличных данных}
''')
    put('report.tex',r'''
% Typesetting source: coursework report; chapter structure: NIR sample.
\input{preambula}
\usepackage{longtable,booktabs}
\input{metadata}
\input{generated/v4_data}
\hypersetup{hidelinks,pdftitle={Отчёт о НИР: ансамбли деревьев при сдвиге распределения},pdfauthor={}}
\addto\captionsrussian{\renewcommand{\bibname}{СПИСОК ИСПОЛЬЗОВАННЫХ ИСТОЧНИКОВ}}
\begin{document}
\input{title}
\setcounter{page}{2}
\include{02-essay}
\clearpage
\begin{center}\LARGE\bfseries\MakeUppercase{СОДЕРЖАНИЕ}\end{center}
\vspace*{-31mm}
\begingroup\let\contentsname\relax\tableofcontents\endgroup
\include{10-introduction}
\include{20-analysis}
\include{25-methodology}
\include{30-conclusion}
\include{35-references}
\include{40-appendix}
\end{document}
''')
    put('title.tex',r'''
\begin{titlepage}
\begin{center}
\ReportUniversity\\[3mm]
\ReportFaculty\\[3mm]
\ReportDepartment
\vfill
{\large\bfseries ОТЧЁТ О НАУЧНО-ИССЛЕДОВАТЕЛЬСКОЙ РАБОТЕ}\\[8mm]
по теме\\[5mm]
{\large\bfseries \ReportTitle}
\vfill
\end{center}
\noindent\begin{tabularx}{\textwidth}{@{}lX@{}}
Выполнил: & \ReportStudent \\[4mm]
Группа: & \ReportGroup \\[4mm]
Научный руководитель: & \ReportSupervisor
\end{tabularx}
\vfill
\begin{center}\ReportCity{} --- \ReportYear\end{center}
\end{titlepage}
''')
    put('02-essay.tex',r'''
\chapter*{РЕФЕРАТ}
\addcontentsline{toc}{chapter}{РЕФЕРАТ}
\noindent\textbf{Ключевые слова:} ансамбли деревьев решений, случайный лес, градиентный бустинг, сдвиг распределения, обобщение на новые домены, табличные данные.

Объект исследования --- ансамбли деревьев решений для классификации и регрессии табличных данных. Предмет исследования --- методы повышения устойчивости этих ансамблей к изменению распределения между исходными и целевыми доменами.

Цель работы --- экспериментально сопоставить существующие методы при выборе модели только по исходным доменам и установить границы обоснованных выводов об их качестве. Рассматриваются IRF, CRF, интерпретация Meta-forests, MaxRM, Era splitting и Anchor boosting, а также сильные базовые ансамбли.

В отчёте разграничиваются эффект настройки гиперпараметров и вклад исследуемого механизма. Приводятся результаты предварительных серий и нового теста с заранее зафиксированным протоколом. Положительные, отрицательные и неполученные статистические результаты сохраняются; универсальное превосходство методов не предполагается.
''')
    put('10-introduction.tex',r'''
\chapter*{ВВЕДЕНИЕ}
\addcontentsline{toc}{chapter}{ВВЕДЕНИЕ}

Модели табличных данных могут терять качество при переносе между организациями, регионами, временными интервалами и условиями измерения. Высокое качество на случайно выделенной части исходной выборки само по себе не характеризует качество на новом домене.

Цель работы --- исследовать методы обучения ансамблей деревьев решений, устойчивых к сдвигу распределения, и экспериментально оценить их применимость при source-only выборе модели. Для достижения цели требуется:
\begin{enumerate}
\item определить виды сдвига и условия переноса модели;
\item систематизировать рассматриваемые методы и их реализации;
\item задать воспроизводимый протокол source/target разделения;
\item разделить сравнение настроенных моделей и сравнение отдельных механизмов;
\item проанализировать экспериментальные результаты, отрицательные наблюдения и ограничения.
\end{enumerate}

Работа носит экспериментально-аналитический характер. Разработка нового алгоритма не заявляется. Исследовательские утверждения связываются с конкретными реализациями, протоколами и экспериментальными артефактами.
''')
    put('20-analysis.tex',r'''
\chapter{Анализ предметной области и существующих решений}
\section{Основные определения и постановка задачи}
Пусть доступны исходные домены $\mathcal{E}_s$ с распределениями $P_e(X,Y)$. Обучение и выбор гиперпараметров используют только эти домены. Для нового домена $t\notin\mathcal{E}_s$ исследуется риск
\begin{equation}
R_t(f)=\mathbb{E}_{(X,Y)\sim P_t}\ell(f(X),Y).
\end{equation}
Наблюдения целевого домена не участвуют в выборе модели. Качество source-валидации и качество переноса рассматриваются как различные характеристики процедуры обучения.

\section{Типы распределительного сдвига}
В эксперименте различаются изменения распределения признаков, условного механизма ответа, долей классов, шума, пропусков и ложных корреляций. Сценарий с пропусками характеризует изменение измерений; он не объявляется чистым covariate shift. При изменении условного механизма существование общей инвариантной функции нельзя предполагать без дополнительных условий.

\section{Рассматриваемые методы}
IRF изменяет критерий построения деревьев с учётом неоднородности доменов~\cite{irf}. CRF использует ограничения, связывающие качество классификации и доменную информацию~\cite{crf}. Meta-forests объединяет леса, обученные по исходным доменам~\cite{meta}.

MaxRM рассматривает минимизацию наибольшего эмпирического риска~\cite{maxrm}. Era splitting меняет критерий разбиения с использованием групп наблюдений~\cite{era}. Anchor boosting использует анкерную постановку для переноса модели~\cite{anchor}.

\section{Реализации и критерии сравнения}
IRF, CRF и MaxRM в стенде являются независимыми реализациями; Meta-forests --- явно обозначенной интерпретацией. Они не называются точными репликациями авторских исследований. Отдельно учитываются capacity, допустимые разбиения, bootstrap, политика признаков и source-only настройка. В качестве сильных baseline рассматриваются Random Forest, ExtraTrees и HistGBDT.

\section{Вывод по аналитической части}
Сравнение требует раздельно оценивать абсолютное качество модели, преимущество над собственным контролем и воспроизводимость. Совпадение только числа деревьев недостаточно для изоляции механизма.
''')
    put('25-methodology.tex',r'''
\chapter{Методология и результаты экспериментального исследования}
\section{Протокол и источники данных}
Серии final\_v1 и expanded\_v2 сохранены как исторические результаты. После их аудита выполнены source-only инженерные исправления, а затем проведена серия tuned\_v4 с новым протоколом. Настройки и модели v4 опубликованы до открытия целевых данных~\cite{v4protocol}.

Новые синтетические выборки относятся к известным механизмам генерации. Новые реальные домены --- заранее зарезервированные штаты ACS для задач Income и PublicCoverage. Эти задачи принадлежат общей survey family. Для Beijing и Appliances в v4 выполнена только source-настройка: прежние target уже открыты и не являются независимым тестом новых решений.

\section{Выбор гиперпараметров и парные контроли}
Гиперпараметры выбираются по выделенным исходным доменам, без целевых оценок. Настройка выполняется один раз на development source каждого контекста; затем параметры переносятся на новые обучающие выборки. Сравниваются расширенно настроенный pipeline и ограниченный source-selected reference, а также выбранная модель и её структурно сопоставимый контроль.

Meta uniform использует те же обученные members. MaxRM сохраняет bootstrap и структуру деревьев. У Era контроль с нулевой силой критерия сохраняет доменные ограничения допустимости разбиений и не равен обычному HistGBDT.

\section{Метрики и статистический анализ}
Для классификации используется разность AUROC; для регрессии --- снижение MSE, делённое на дисперсию полного source. Сначала усредняются целевые домены и model seeds, затем сценарии внутри независимой MC replica.

Основные контрасты проверяются двусторонним точным sign test с единой коррекцией Holm. Нулевые контрасты исключаются из sign test. Маргинальный t-интервал среднего не является одновременным интервалом и относится к иному параметру, чем проверка знаков. Реальные перекрывающиеся resamples анализируются описательно~\cite{v4protocol,v4tests}.

\section{Результаты и обсуждение}
Завершено \FinalJobs{} заданий и \PrimaryTests{} основных тестов; после Holm значимы \SignificantTests{} контрастов~\cite{v4audit}. Избранные контрасты представлены в таблице~\ref{tab:v4}. Полная таблица и фильтры строк сохранены в экспериментальных артефактах~\cite{v4tests}.

\input{generated/results_table}

Расширение настройки улучшило ряд моделей относительно ограниченного reference. При этом положительный вклад механизма подтверждён не у всех методов. Большой HPO gain не означает, что выигрыш обусловлен DG-регуляризацией, и не гарантирует победы над сильным baseline.

\section{Ограничения исследования}
Синтетические MC replicas не являются выборкой всех табличных задач. Ограниченный поиск не устанавливает глобальный максимум каждого метода; CPU бюджеты различаются. Положительные средние контрасты не означают устойчивости при каждом отдельном сдвиге. Локальные hashes и timestamps не подменяют независимый системный аудит доступа.

\section{Вывод по экспериментальной части}
Экспериментальные данные обосновывают раздельную оценку настройки, механизмов и абсолютного качества. Окончательная редакция главы должна включать полный разбор v1/v2/v4, таблицы реальных задач и анализ каждого вида сдвига; исходные материалы сохранены в репозитории.
''')
    put('30-conclusion.tex',r'''
\chapter*{ЗАКЛЮЧЕНИЕ}
\addcontentsline{toc}{chapter}{ЗАКЛЮЧЕНИЕ}

В работе сформулирована source-only постановка переноса ансамблей деревьев между доменами, выделены механизмы распределительного сдвига и заданы условия сопоставимого экспериментального сравнения.

Проведённые серии показывают пользу тщательной настройки отдельных моделей и более ограниченную полезность самих DG-механизмов. Универсальное превосходство над сильными базовыми ансамблями не установлено. Обоснованность вывода зависит от реализации, протокола выбора модели и типа сдвига.

Дальнейшие новые решения должны проверяться на новых заранее зафиксированных целевых данных. Уже открытые домены не могут повторно использоваться как независимое подтверждение.
''')
    put('40-appendix.tex',r'''
\chapter*{ПРИЛОЖЕНИЕ А}
\addcontentsline{toc}{chapter}{ПРИЛОЖЕНИЕ А. Реестр экспериментальных материалов}
\begin{center}\bfseries Реестр экспериментальных материалов\end{center}
\noindent Программный стенд и журналы находятся в \texttt{code/}; документация и литература --- в \texttt{materials/}; результаты серий и материалы отчёта --- в \texttt{nir/}; материалы будущей ВКР --- в \texttt{vkr/}.

\begin{itemize}
\item протоколы и source receipts: \texttt{code/runs/};
\item исходные таблицы, графики и отчёты: \texttt{nir/results/};
\item реестр источников и локальные полные тексты: \texttt{materials/literature/};
\item карта переноса и проверка сохранности: \texttt{materials/repository\_layout/}.
\end{itemize}

Точные значения, определения эффектов, строки таблиц и контрольные суммы следует проверять по исходным артефактам~\cite{v4tests,v4audit}. Начальное наполнение настоящего шаблона не заменяет полный итоговый текст отчёта НИР.
''')
    library=json.loads((ROOT/'materials/literature/sources.json').read_text(encoding='utf-8'))
    keys={'irf':'irf','crf':'crf','meta_forests':'meta','maxrm_rf':'maxrm','era_splitting':'era','anchor_boosting':'anchor'}
    bib=[r'\begin{thebibliography}{99}',r'\addcontentsline{toc}{chapter}{\bibname}']
    def tex(s):return s.replace('&',r'\&').replace('_',r'\_').replace('æ',r'{\ae}').replace('ü',r'{\"u}').replace('ä',r'{\"a}')
    for row in library:
        if row['id'] in keys:
            bib.append(r'\bibitem{'+keys[row['id']]+'} '+tex(row['authors'])+'. '+tex(row['title'])+'. '+str(row.get('year',''))+'. '+r'\url{'+row['url']+'}.')
    for key,label,path in [('v4protocol','Протокол свежего теста v4','runs/tuned_v4/final_protocol.json'),('v4tests','Предзаданные контрасты и результаты v4','reports/tuned_v4/hypothesis_tests.csv'),('v4audit','Проверка целостности и воспроизводимости v4','reports/tuned_v4/final_integrity.json')]:
        url=f'https://github.com/uBaxa0IQ/coral-forest-dg/blob/{PIN}/{path}'
        bib.append(r'\bibitem{'+key+'} '+label+r'. Версия репозитория \texttt{'+PIN[:7]+r'}. \url{'+url+'}.')
    bib.append(r'\end{thebibliography}');put('35-references.tex','\n\n'.join(bib))
    import csv
    rows=list(csv.DictReader((ROOT/'nir/results/tuned_v4/hypothesis_tests.csv').open(encoding='utf-8',newline='')))
    wanted=[('HPO','classification','irf'),('HPO','classification','crf'),('mechanism','classification','meta'),('mechanism','regression','era'),('mechanism','classification','irf'),('mechanism','classification','crf')]
    selected=[next(r for r in rows if (r['hypothesis'],r['task'],r['method'])==key) for key in wanted]
    table=[r'\begin{table}[htbp]',r'\caption{Избранные результаты v4; источник: \cite{v4tests}}',r'\label{tab:v4}',r'\centering',r'\small',r'\begin{tabularx}{\textwidth}{@{}lXrr@{}}',r'\toprule',r'Метод & Контраст и задача & Эффект & Holm $p$ \\',r'\midrule']
    for r in selected:
        label=('Настройка' if r['hypothesis']=='HPO' else 'Механизм')+', '+('классификация' if r['task']=='classification' else 'регрессия')
        table.append(r['method'].upper()+' & '+label+' & '+f"{float(r['mean']):+.6f}"+' & '+f"{float(r['p_holm']):.6f}"+r' \\')
    table += [r'\bottomrule',r'\end{tabularx}',r'\par\smallskip\footnotesize Классификация: разность AUROC; регрессия: снижение MSE / source variance.',r'\end{table}']
    put('generated/results_table.tex','\n'.join(table))
    audit=json.loads((ROOT/'nir/results/tuned_v4/final_integrity.json').read_text(encoding='utf-8'))
    put('generated/v4_data.tex',f"\\newcommand{{\\FinalJobs}}{{{audit['final_jobs']}}}\n\\newcommand{{\\PrimaryTests}}{{{audit['planned_primary_tests']}}}\n\\newcommand{{\\SignificantTests}}{{{audit['significant_holm_tests']}}}")
    (REPORT/'generated/data_provenance.json').write_text(json.dumps(dict(source_csv='nir/results/tuned_v4/hypothesis_tests.csv',source_csv_sha256=sha256((ROOT/'nir/results/tuned_v4/hypothesis_tests.csv').read_bytes()).hexdigest(),selected_row_keys=wanted,audit_json_sha256=sha256((ROOT/'nir/results/tuned_v4/final_integrity.json').read_bytes()).hexdigest()),indent=2),encoding='utf-8',newline='\n')
    print(json.dumps(dict(template=str(REPORT),coursework_preamble_unchanged=True,nir_style_not_copied=True,selected_results=len(selected))))


if __name__=='__main__':main()
