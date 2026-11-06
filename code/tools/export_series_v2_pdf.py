"""Export the already computed research report using the bundled PDF runtime."""
from html import escape
import json
from pathlib import Path
import posixpath
import re
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4,landscape
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import BaseDocTemplate,PageTemplate,Frame,NextPageTemplate,Paragraph,Spacer,Image,PageBreak

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"reports/series_v2"


def markup(text):
    text=escape(text)
    def link(match):
        label,path=match.groups()
        if path.startswith("http"):
            url=path
        else:
            relative=posixpath.normpath("reports/series_v2/"+path)
            url="https://github.com/uBaxa0IQ/coral-forest-dg/blob/main/"+relative
        return f'<link href="{url}" color="#245a81">{label}</link>'
    text=re.sub(r"\[([^\]]+)\]\(([^)]+)\)",link,text)
    text=re.sub(r"\*\*([^*]+)\*\*",r"<b>\1</b>",text)
    text=re.sub(r"`([^`]+)`",r'<font name="Arial">\1</font>',text)
    return text


def header(canvas,doc):
    canvas.saveState();w,h=canvas._pagesize
    canvas.setFont("Arial",8);canvas.setFillColor(colors.HexColor("#64707b"))
    canvas.drawString(19*mm,h-12*mm,"Coral Forest DG · НИР · расширенная серия v2")
    canvas.drawRightString(w-19*mm,12*mm,str(doc.page))
    canvas.restoreState()


def main():
    for font,file in [("Arial","arial.ttf"),("Arial-Bold","arialbd.ttf"),("Arial-Italic","ariali.ttf")]:
        pdfmetrics.registerFont(TTFont(font,str(Path("C:/Windows/Fonts")/file)))
    pdfmetrics.registerFontFamily("Arial",normal="Arial",bold="Arial-Bold",italic="Arial-Italic",boldItalic="Arial-Bold")
    styles=getSampleStyleSheet()
    body=ParagraphStyle("ResearchBody",fontName="Arial",fontSize=9.5,leading=14,spaceAfter=8,alignment=TA_LEFT,allowWidows=0,allowOrphans=0)
    title=ParagraphStyle("ResearchTitle",parent=body,fontName="Arial-Bold",fontSize=22,leading=29,spaceAfter=18)
    heading=ParagraphStyle("ResearchHeading",parent=body,fontName="Arial-Bold",fontSize=15,leading=21,spaceBefore=15,spaceAfter=10,keepWithNext=True)
    caption=ParagraphStyle("ResearchCaption",parent=body,fontSize=8,leading=11,textColor=colors.HexColor("#64707b"))
    bullet=ParagraphStyle("ResearchBullet",parent=body,leftIndent=12,firstLineIndent=-8)
    source=(OUT/"research_report.md").read_text(encoding="utf-8")
    inventory=json.loads((OUT/"analysis_inventory.json").read_text(encoding="utf-8"))
    story=[Spacer(1,20*mm),Paragraph("Ансамбли деревьев при сдвиге распределения",title),Paragraph("Расширенное исследование · аудит и воспроизводимые эксперименты",heading),Paragraph("Полный научный отчёт. Новые целевые домены, синтетические механизмы с известными законами, парные контроли и заранее сформулированные гипотезы.",body),Spacer(1,10*mm),Paragraph(f"Обработано {inventory['jobs']} заданий: {inventory['complete_jobs']} полных и {inventory['partial_jobs']} с отказом MaxRM. Сохранены {inventory['ordinary_source_fits']+inventory['repeated_source_fits']:,} успешных source fits, {inventory['full_source_models']:,} full-source конфигурации, {inventory['target_domain_metric_rows']:,} per-target-domain строк.",body),Paragraph(f"Протокол SHA-256: {inventory['protocol_sha256']}",caption),Paragraph("Этот отчёт проверяет конкретные реализации. Он не является внешним рецензированием или утверждением об универсальном превосходстве DG. Полные данные анализа и исходные predictions доступны в структуре проекта.",body),PageBreak()]
    illustrations=[]
    for raw in source.splitlines():
        line=raw.strip()
        if not line:
            continue
        if line.startswith("# "):
            continue
        if line.startswith("## "):
            story.append(Paragraph(markup(line[3:]),heading));continue
        image=re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)",line)
        if image:
            label,path=image.groups();illustrations.append((label,path))
            story.append(Paragraph("Рисунок «"+escape(label)+"» — см. графическое приложение, лист "+str(len(illustrations))+".",caption));continue
        story.append(Paragraph(("• "+markup(line[2:])) if line.startswith("- ") else markup(line),bullet if line.startswith("- ") else body))
    wide=landscape(A4)
    story.extend([NextPageTemplate("landscape"),PageBreak()])
    for i,(label,path) in enumerate(illustrations):
        story.append(Paragraph(f"Графическое приложение {i+1}. "+escape(label),heading))
        img=Image(str(OUT/path));factor=min((wide[0]-34*mm)/img.imageWidth,(wide[1]-59*mm)/img.imageHeight)
        img.drawWidth=img.imageWidth*factor;img.drawHeight=img.imageHeight*factor
        story.extend([img,Paragraph("Источник: frozen expanded_v2. Полные подписи/оси на рисунке. Все points/CI в tables/, оригиналы PNG/SVG в figures/.",caption)])
        if i<len(illustrations)-1:
            story.append(PageBreak())
    document=BaseDocTemplate(str(OUT/"research_report.pdf"),pagesize=A4,title="Coral Forest DG — расширенное исследование v2",author="Coral Forest DG research project",pageCompression=1)
    portrait_frame=Frame(19*mm,19*mm,A4[0]-38*mm,A4[1]-41*mm,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
    landscape_frame=Frame(15*mm,16*mm,wide[0]-30*mm,wide[1]-34*mm,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
    document.addPageTemplates([PageTemplate(id="portrait",frames=[portrait_frame],pagesize=A4,onPage=header),PageTemplate(id="landscape",frames=[landscape_frame],pagesize=wide,onPage=header)])
    document.build(story)
    print(OUT/"research_report.pdf")


if __name__=="__main__":
    main()
