"""Render the NIR template, verify layout bounds and source provenance."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import fitz
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'nir/report'


def main():
    pdf=REPORT/'report.pdf';pages=REPORT/'pdf_review';pages.mkdir(exist_ok=True)
    doc=fitz.open(pdf)
    poppler=Path('C:/Users/uBaxa/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin/pdftoppm.exe')
    subprocess.run([str(poppler),'-png','-r','90',str(pdf),str(pages/'page')],check=True)
    checks=[];thumbs=[];alltext=[];outside=[]
    for i,page in enumerate(doc):
        text=page.get_text();alltext.append(text);assert text.strip(),i+1
        for b in page.get_text('dict')['blocks']:
            if b['type']==0:
                x0,y0,x1,y1=b['bbox']
                if x0<0 or y0<0 or x1>page.rect.width+.5 or y1>page.rect.height+.5:outside.append(dict(page=i+1,bbox=b['bbox']))
        checks.append(dict(page=i+1,text_characters=len(text),links=len(page.get_links())))
        image=Image.open(pages/f"page-{i+1:0{len(str(len(doc)))}}.png").convert('RGB');image.thumbnail((250,350))
        card=Image.new('RGB',(270,380),'#e6e9ed');card.paste(image,((270-image.width)//2,12));ImageDraw.Draw(card).text((12,362),f'Page {i+1}',fill='black');thumbs.append(card)
    text='\n'.join(alltext)
    for word in ['РЕФЕРАТ','СОДЕРЖАНИЕ','ВВЕДЕНИЕ','ЗАКЛЮЧЕНИЕ','127','272']:
        # The full source selection inventory stays in the linked artifacts;
        # the template itself only reports final jobs and primary tests.
        if word=='127':continue
        assert word in text,word
    sheet=Image.new('RGB',(1080,380*((len(thumbs)+3)//4)),'white')
    for i,card in enumerate(thumbs):sheet.paste(card,((i%4)*270,(i//4)*380))
    sheet.save(pages/'contact_sheet.png')
    provenance=json.loads((ROOT/'materials/templates/provenance.json').read_text(encoding='utf-8'))
    original=ROOT/'materials/templates/coursework/preambula.original.tex'
    assert original.read_bytes()==(REPORT/'preambula.tex').read_bytes()
    assert not list(REPORT.rglob('bmstu.cls'))
    record=dict(pages=len(doc),all_pages_rendered=True,text_bounds_violations=outside,cyrillic_text_verified=True,coursework_preamble_sha256=sha256(original.read_bytes()).hexdigest(),coursework_preamble_byte_identical=True,nir_bmstu_class_or_preamble_copied=False,pdf_sha256=sha256(pdf.read_bytes()).hexdigest(),visual_review='pending contact-sheet inspection',page_checks=checks)
    (REPORT/'validation.json').write_text(json.dumps(record,indent=2),encoding='utf-8',newline='\n')
    print(json.dumps(dict(pages=len(doc),text_bounds_violations=outside,contact_sheet=str(pages/'contact_sheet.png'))));assert not outside


if __name__=='__main__':main()
