"""Render all closeout PDF pages and check text bounds and linked appendices."""
import json
from pathlib import Path
import subprocess
import fitz
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/closeout';PAGES=OUT/'pdf_review'


def main():
    PAGES.mkdir(exist_ok=True)
    pdf=OUT/'research_synthesis.pdf';doc=fitz.open(pdf)
    poppler='C:/Users/uBaxa/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin/pdftoppm.exe'
    subprocess.run([poppler,'-png','-r','100',str(pdf),str(PAGES/'page')],check=True)
    checks=[];thumbs=[]
    for i,page in enumerate(doc):
        text=page.get_text();assert text.strip()
        for b in page.get_text('dict')['blocks']:
            if b['type']==0:
                x0,y0,x1,y1=b['bbox'];assert min(x0,y0)>=0 and x1<=page.rect.width+.5 and y1<=page.rect.height+.5,(i,b['bbox'])
        checks.append(dict(page=i+1,text_characters=len(text),images=len(page.get_images()),links=len(page.get_links())))
        p=PAGES/f"page-{i+1:0{len(str(len(doc)))}}.png";dest=PAGES/f'page_{i+1:02d}.png'
        p.replace(dest);image=Image.open(dest).convert('RGB');image.thumbnail((250,350))
        card=Image.new('RGB',(270,380),'#e6e9ed');card.paste(image,((270-image.width)//2,12));ImageDraw.Draw(card).text((12,362),f'Page {i+1}',fill='black');thumbs.append(card)
    alltext='\n'.join(p.get_text() for p in doc)
    for phrase in ['Итоговое','Основные предзаданные тесты','Source-only pilot','0.03125','0.017578']:
        assert phrase in alltext,phrase
    assert sum(c['images'] for c in checks)>=4
    sheet=Image.new('RGB',(1080,380*((len(thumbs)+3)//4)),'white')
    for i,card in enumerate(thumbs):sheet.paste(card,((i%4)*270,(i//4)*380))
    sheet.save(PAGES/'contact_sheet.png')
    (OUT/'pdf_validation.json').write_text(json.dumps(dict(pages=len(doc),all_pages_rendered=True,text_bounds_verified=True,cyrillic_and_key_tables_verified=True,checks=checks,visual_review='pending contact sheet inspection'),indent=2),encoding='utf-8',newline='\n')
    print(json.dumps(dict(pages=len(doc),status='passed',contact_sheet=str(PAGES/'contact_sheet.png'))))


if __name__=='__main__':main()
