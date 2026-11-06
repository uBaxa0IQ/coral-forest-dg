"""Publication PDF of the closeout, including artifact-linked tables and figures."""
from html import escape
from pathlib import Path
import posixpath
import re
import csv
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4,landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import BaseDocTemplate,PageTemplate,Frame,NextPageTemplate,Paragraph,Spacer,Image,PageBreak,LongTable,TableStyle

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/closeout'


def markup(s):
    s=escape(s)
    def link(m):
        label,path=m.groups()
        url=path if path.startswith('http') else 'https://github.com/uBaxa0IQ/coral-forest-dg/blob/main/'+posixpath.normpath('reports/closeout/'+path)
        return f'<link href="{url}" color="#245a81">{label}</link>'
    return re.sub(r'\[([^\]]+)\]\(([^)]+)\)',link,s)


def header(canvas,doc):
    canvas.saveState();w,h=canvas._pagesize;canvas.setFont('Arial',8);canvas.setFillColor(colors.HexColor('#64707b'))
    canvas.drawString(18*mm,h-12*mm,'Coral Forest DG · заключительный аудит и обобщение НИР');canvas.drawRightString(w-18*mm,11*mm,str(doc.page));canvas.restoreState()


def main():
    for name,file in [('Arial','arial.ttf'),('Arial-Bold','arialbd.ttf')]:
        pdfmetrics.registerFont(TTFont(name,str(Path('C:/Windows/Fonts')/file)))
    pdfmetrics.registerFontFamily('Arial',normal='Arial',bold='Arial-Bold',italic='Arial',boldItalic='Arial-Bold')
    body=ParagraphStyle('Body',fontName='Arial',fontSize=10,leading=15,spaceAfter=9,allowWidows=0,allowOrphans=0)
    heading=ParagraphStyle('Heading',parent=body,fontName='Arial-Bold',fontSize=15,leading=21,spaceBefore=14,spaceAfter=10,keepWithNext=True)
    title=ParagraphStyle('Title',parent=heading,fontSize=23,leading=31)
    small=ParagraphStyle('Small',parent=body,fontSize=8,leading=11)
    cell=ParagraphStyle('Cell',parent=body,fontSize=8,leading=11,spaceAfter=0)
    story=[];figures=[]
    for line in (OUT/'research_synthesis.md').read_text(encoding='utf-8').splitlines():
        if not line.strip():continue
        image=re.fullmatch(r'!\[([^\]]+)\]\(([^)]+)\)',line)
        if image:
            figures.append((image[1],image[2]));story.append(Paragraph('Рисунок '+str(len(figures))+': '+escape(image[1])+'. См. графическое приложение.',small))
        elif line.startswith('# '):story.extend([Paragraph(markup(line[2:]),title),Spacer(1,8*mm)])
        elif line.startswith('## '):story.append(Paragraph(markup(line[3:]),heading))
        else:story.append(Paragraph(markup(line),body))
    wide=landscape(A4);story.extend([NextPageTemplate('wide'),PageBreak()])
    for i,(caption,path) in enumerate(figures):
        story.append(Paragraph(f'Рисунок {i+1}. '+escape(caption),heading))
        img=Image(str(OUT/path));factor=min((wide[0]-36*mm)/img.imageWidth,(wide[1]-60*mm)/img.imageHeight);img.drawWidth=img.imageWidth*factor;img.drawHeight=img.imageHeight*factor
        story.extend([img,Paragraph('Исходные значения и artifact selectors: tables/ и claims_registry.json. Исторические target scores; новых target evaluations нет.',small),PageBreak()])
    specs=[('Основные предзаданные тесты','primary_tests.csv',['hypothesis','task','method','replicas','mean','ci_low','ci_high','p_holm','status'],[16,26,33,15,25,25,25,25,45]),('Реальные задачи v2: абсолютные target-метрики','real_results_v2.csv',['dataset','method','k','n','roc_auc','rmse','control_roc_auc','control_rmse'],[43,37,10,20,28,28,32,32]),('Исторические v1: exploratory target-метрики','real_results_v1.csv',['dataset','task','method','target_macro_score','seed_sd'],[43,28,65,45,30]),('Source-only pilot: выбранные настройки','source_pilot_selected.csv',['task','method','profile','parameters','mean_source_primary'],[28,38,17,105,42])]
    for title_,file,columns,widths in specs:
        p=OUT/'tables'/file
        if not p.exists():continue
        story.append(Paragraph(title_,heading));story.append(Paragraph(markup(f'[CSV: {file}](tables/{file}); provenance: [реестр](claims_registry.json).'),small))
        rows=list(csv.DictReader(p.open(encoding='utf-8',newline='')))
        def text(v):
            if v=='':return '—'
            if v=='incomplete_source_solver_failure':return 'incomplete: solver'
            try:return f'{float(v):.6g}'
            except ValueError:return v
        labels={'hypothesis':'H','replicas':'n','ci_low':'CI lower','ci_high':'CI upper','p_holm':'Holm p','target_macro_score':'target score','seed_sd':'seed SD'}
        data=[[Paragraph(escape(labels.get(c,c.replace('_',' '))),cell) for c in columns]]+[[Paragraph(escape(text(r[c])),cell) for c in columns] for r in rows]
        table=LongTable(data,colWidths=[v*mm for v in widths],repeatRows=1,hAlign='LEFT')
        table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dfeaf1')),('GRID',(0,0),(-1,-1),.3,colors.HexColor('#c4ced5')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f4f7f9')])]))
        story.extend([table,PageBreak()])
    if isinstance(story[-1],PageBreak):story.pop()
    doc=BaseDocTemplate(str(OUT/'research_synthesis.pdf'),pagesize=A4,title='НИР — итоговый аудит и научное обобщение',author='Coral Forest DG',pageCompression=1)
    doc.addPageTemplates([PageTemplate(id='portrait',pagesize=A4,frames=[Frame(18*mm,18*mm,A4[0]-36*mm,A4[1]-39*mm,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)],onPage=header),PageTemplate(id='wide',pagesize=wide,frames=[Frame(15*mm,17*mm,wide[0]-30*mm,wide[1]-38*mm,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)],onPage=header)])
    doc.build(story);print(OUT/'research_synthesis.pdf')


if __name__=='__main__':main()
