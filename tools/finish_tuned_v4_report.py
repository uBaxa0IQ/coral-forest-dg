"""Postprocessing only: evidence inventory, method cards and PDF rendering."""
from datetime import datetime,timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import csv

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/tuned_v4';RUN=ROOT/'runs/tuned_v4'
RUNTIME=Path('C:/Users/uBaxa/.cache/codex-runtimes/codex-primary-runtime/dependencies')
FLAGS=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0


def main():
    import pandas as pd
    tests=pd.read_csv(OUT/'hypothesis_tests.csv');selection=pd.read_csv(OUT/'source_selection.csv')
    integrity=json.loads((OUT/'final_integrity.json').read_text(encoding='utf-8'))
    selected_replays=[json.loads(p.read_text(encoding='utf-8')) for p in (RUN/'source_replays').glob('*.json')]
    model_replays=[json.loads(p.read_text(encoding='utf-8')) for p in (RUN/'model_replays').glob('*.json')]
    assert len(selected_replays)==127 and len(model_replays)==272
    assert all(r['source_predictions_bytes_identical'] for r in selected_replays)
    assert all(r['max_prediction_difference']<=1e-12 for r in model_replays)
    issues=[]
    for p in (RUN/'jobs').glob('*/source.json'):
        r=json.loads(p.read_text(encoding='utf-8'))
        for method,error in r.get('failed_methods',{}).items():issues.append(dict(job_id=r['spec']['job_id'],method=method,error=error))
    successful_fits=0;failed_trials=0
    for p in (RUN/'search').rglob('*.json'):
        r=json.loads(p.read_text(encoding='utf-8'))
        if 'screen_reused_from' not in r:successful_fits+=len(r['folds'])
        failed_trials+=int(r['status']!='complete')
    integrity.update(source_cv_successful_fits=successful_fits,source_trial_failures=failed_trials,final_source_failed_methods=issues,source_replays=len(selected_replays),saved_model_replays=sum(r['models'] for r in model_replays),max_model_replay_difference=max(r['max_prediction_difference'] for r in model_replays),status='passed_with_missing_method_qualifications' if issues else 'passed_with_stated_scope_qualifications')
    (OUT/'final_integrity.json').write_text(json.dumps(integrity,indent=2,ensure_ascii=False),encoding='utf-8',newline='\n')
    cards=['# Карточки методов: source настройка и свежий тест','', 'Все source choices сделаны до нового target доступа. Значения и exact selectors находятся в source_selection.csv и hypothesis_tests.csv; полный provenance — evidence_manifest.json.']
    for (task,method),sub in selection.groupby(['task','method']):
        cards.extend(['',f'## {method} / {task}','',f'Контексты source HPO: {len(sub)}. Все строки фильтруются task={task}, method={method} в [source selection](source_selection.csv). Параметры каждого контекста сохранены целиком; единственный универсальный лучший config не заявляется.'])
        for _,r in tests[(tests.task==task)&(tests.method==method)].iterrows():
            text=f"{r.hypothesis}: mean={r['mean']:.6g}; CI=[{r.ci_low:.6g}; {r.ci_high:.6g}]; Holm p={r.p_holm:.6g}; status={r.status}."
            cards.append(text+' [Точная строка: hypothesis='+r.hypothesis+', task='+task+', method='+method+'](hypothesis_tests.csv).')
    (OUT/'method_cards.md').write_text('\n'.join(cards)+'\n',encoding='utf-8',newline='\n')
    freeze=json.loads((RUN/'final_protocol.json').read_text(encoding='utf-8'))
    ledger=dict(recorded_at=datetime.now(timezone.utc).isoformat(),series='tuned_v4',final_protocol_sha256=freeze['sha256'],consumed_real_targets=dict(acs_2018=dict(states=freeze['protocol']['new_acs_states'],joint_tasks=['ACSIncome','ACSPublicCoverage'])),consumed_synthetic_namespace='sealed_v4_target',root_seed=2026110504,replicas=list(range(10)),policy='Consumed targets cannot independently validate new decisions; new MC draws are within known DGPs')
    (ROOT/'configs/target_exposure_ledger_v4.json').write_text(json.dumps(ledger,indent=2,ensure_ascii=False),encoding='utf-8',newline='\n')
    # Marker is emitted once, immediately before authoring this new PDF.
    marker=ROOT/'data/tuned_v4/pdf_marker_done'
    if not marker.exists():
        subprocess.run([str(RUNTIME/'node/bin/node.exe'),'C:/Users/uBaxa/.codex/plugins/cache/openai-primary-runtime/pdf/26.1026.10807/skills/pdf/container_tools/mark_artifact_operation_started.mjs','--operation-kind','create','--expected-output-count','1','--output-format','pdf'],check=True,creationflags=FLAGS)
        marker.write_text('done',encoding='utf-8')
    subprocess.run([str(RUNTIME/'python/python.exe'),str(Path(__file__)), '--pdf'],check=True,cwd=ROOT,creationflags=FLAGS)
    verify_pdf()
    records=[]
    for p in [*sorted(OUT.glob('*.csv')),*sorted(RUN.glob('*.json'))]:
        records.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=sha256(p.read_bytes()).hexdigest()))
    (OUT/'evidence_manifest.json').write_text(json.dumps(dict(artifacts=records,source_trial_selectors='context/method/stage/candidate; per-trial artifacts and SHA in source_trials.csv',target_selectors='job_id/method/variant/target_domain in target_metrics.csv; result artifacts and SHA in final_integrity.json',generator_sha256=sha256(Path(__file__).read_bytes()).hexdigest()),indent=2),encoding='utf-8',newline='\n')
    print(json.dumps(dict(status='complete',source_cv_fits=successful_fits,source_replays=127,model_replays=sum(r['models'] for r in model_replays),final_source_failed_methods=len(issues))))


def pdf():
    from html import escape
    import posixpath,re
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.pagesizes import A4,landscape
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Image,PageBreak,LongTable,TableStyle
    for name,file in [('Arial','arial.ttf'),('Arial-Bold','arialbd.ttf')]:pdfmetrics.registerFont(TTFont(name,str(Path('C:/Windows/Fonts')/file)))
    body=ParagraphStyle('Body',fontName='Arial',fontSize=10,leading=15,spaceAfter=9)
    heading=ParagraphStyle('Heading',parent=body,fontName='Arial-Bold',fontSize=16,leading=22,spaceBefore=14,keepWithNext=True)
    cell=ParagraphStyle('Cell',parent=body,fontSize=7,leading=10,spaceAfter=0)
    def mark(s):
        def link(m):
            label,path=m.groups();url='https://github.com/uBaxa0IQ/coral-forest-dg/blob/main/'+posixpath.normpath('reports/tuned_v4/'+path)
            return f'<link href="{url}" color="#245a81">{label}</link>'
        return re.sub(r'\[([^\]]+)\]\(([^)]+)\)',link,escape(s))
    story=[]
    for line in (OUT/'research_report.md').read_text(encoding='utf-8').splitlines():
        if not line.strip():continue
        if line.startswith('!['):continue
        story.append(Paragraph(mark(line.lstrip('# ')),heading if line.startswith('#') else body))
    story.extend([PageBreak(),Paragraph('Предзаданные основные тесты',heading)])
    image=Image(str(OUT/'figures/primary.png'));factor=min(170*mm/image.imageWidth,210*mm/image.imageHeight);image.drawWidth=image.imageWidth*factor;image.drawHeight=image.imageHeight*factor;story.extend([image,PageBreak()])
    columns=['hypothesis','task','method','replicas','mean','ci_low','ci_high','p_holm','status']
    rows=list(csv.DictReader((OUT/'hypothesis_tests.csv').open(encoding='utf-8',newline='')))
    table=[[Paragraph(x,cell) for x in ['test','task','method','n','mean','CI low','CI high','Holm p','status']]]
    for r in rows:
        values=[]
        for key in columns:
            val=r[key]
            try:val=f'{float(val):.5g}'
            except ValueError:pass
            values.append(Paragraph(escape(val or '—'),cell))
        table.append(values)
    t=LongTable(table,colWidths=[19*mm,22*mm,26*mm,8*mm,20*mm,20*mm,20*mm,18*mm,21*mm],repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('GRID',(0,0),(-1,-1),.3,colors.lightgrey),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e5edf3')),('VALIGN',(0,0),(-1,-1),'TOP')]))
    story.append(t)
    for line in (OUT/'method_cards.md').read_text(encoding='utf-8').splitlines():
        if line.strip():story.append(Paragraph(mark(line.lstrip('# ')),heading if line.startswith('#') else body))
    def footer(canvas,doc):
        canvas.setFont('Arial',8);canvas.drawRightString(A4[0]-18*mm,11*mm,str(doc.page))
    SimpleDocTemplate(str(OUT/'research_report.pdf'),pagesize=A4,leftMargin=17*mm,rightMargin=17*mm,topMargin=17*mm,bottomMargin=17*mm,title='НИР: углублённая настройка и свежий тест v4').build(story,onFirstPage=footer,onLaterPages=footer)


def verify_pdf():
    import fitz
    from PIL import Image,ImageDraw
    review=OUT/'pdf_review';review.mkdir(exist_ok=True);pdf=OUT/'research_report.pdf';doc=fitz.open(pdf)
    subprocess.run([str(RUNTIME/'native/poppler/Library/bin/pdftoppm.exe'),'-png','-r','90',str(pdf),str(review/'page')],check=True,creationflags=FLAGS)
    checks=[];thumbs=[]
    for i,page in enumerate(doc):
        assert page.get_text().strip()
        for b in page.get_text('dict')['blocks']:
            if b['type']==0:
                x0,y0,x1,y1=b['bbox'];assert x0>=0 and y0>=0 and x1<=page.rect.width+.5 and y1<=page.rect.height+.5
        checks.append(dict(page=i+1,text_characters=len(page.get_text())))
        image=Image.open(review/f"page-{i+1:0{len(str(len(doc)))}}.png").convert('RGB');image.thumbnail((250,350));card=Image.new('RGB',(270,380),'#e6e9ed');card.paste(image,((270-image.width)//2,12));ImageDraw.Draw(card).text((12,362),str(i+1),fill='black');thumbs.append(card)
    sheet=Image.new('RGB',(1080,380*((len(thumbs)+3)//4)),'white')
    for i,card in enumerate(thumbs):sheet.paste(card,((i%4)*270,(i//4)*380))
    sheet.save(review/'contact_sheet.png')
    (OUT/'pdf_validation.json').write_text(json.dumps(dict(pages=len(doc),all_pages_rendered=True,text_bounds_verified=True,visual_review='pending human/agent contact-sheet inspection',checks=checks,pdf_sha256=sha256(pdf.read_bytes()).hexdigest()),indent=2),encoding='utf-8',newline='\n')


if __name__=='__main__':
    if '--pdf' in sys.argv:pdf()
    else:main()
