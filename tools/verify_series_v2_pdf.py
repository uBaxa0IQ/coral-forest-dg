"""Render every output page and verify Cyrillic text/page bounds."""
import json
import subprocess
from pathlib import Path
import fitz
from PIL import Image,ImageOps,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"reports/series_v2"
PAGES=OUT/"pdf_review"


def main():
    PAGES.mkdir(exist_ok=True)
    doc=fitz.open(OUT/"research_report.pdf")
    poppler=Path("C:/Users/uBaxa/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin/pdftoppm.exe")
    subprocess.run([str(poppler),"-png","-r","110",str(OUT/"research_report.pdf"),str(PAGES/"page")],check=True)
    checks=[];thumbs=[]
    for i,page in enumerate(doc):
        rendered=PAGES/f"page-{i+1:0{len(str(len(doc)))}}.png"
        rendered.replace(PAGES/f"page_{i+1:02d}.png")
        text=page.get_text();assert text.strip()
        for block in page.get_text("dict")["blocks"]:
            if block["type"]!=0:
                continue
            x0,y0,x1,y1=block["bbox"]
            assert x0>=0 and y0>=0 and x1<=page.rect.width+.5 and y1<=page.rect.height+.5
        checks.append(dict(page=i+1,width=page.rect.width,height=page.rect.height,text_characters=len(text),images=len(page.get_images())))
        image=Image.open(PAGES/f"page_{i+1:02d}.png").convert("RGB");image.thumbnail((250,350))
        card=Image.new("RGB",(270,380),"#e6e9ed");card.paste(image,((270-image.width)//2,12));ImageDraw.Draw(card).text((12,362),f"Page {i+1}",fill="black");thumbs.append(card)
    alltext="\n".join(p.get_text() for p in doc)
    assert "Расширенное" in alltext or "расширенное" in alltext
    assert "Графическое приложение 10" in alltext
    columns=4;rows=(len(thumbs)+columns-1)//columns
    sheet=Image.new("RGB",(270*columns,380*rows),"white")
    for i,thumb in enumerate(thumbs):
        sheet.paste(thumb,((i%columns)*270,(i//columns)*380))
    sheet.save(PAGES/"contact_sheet.png")
    (OUT/"pdf_validation.json").write_text(json.dumps(dict(pages=len(doc),checks=checks,all_pages_rendered=True,text_bounds_verified=True,cyrillic_text_verified=True,visual_review="contact_sheet_and_selected_full_pages_require_inspection"),indent=2),encoding="utf-8")
    print("Rendered",len(doc),"pages",PAGES/"contact_sheet.png")


if __name__=="__main__":
    main()
