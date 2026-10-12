from collect_official import ROOT,fetch,Links
from pathlib import Path
from urllib.parse import urljoin,quote,urlsplit,unquote
from html.parser import HTMLParser
from html import unescape
from concurrent.futures import ThreadPoolExecutor,as_completed
from io import BytesIO
from pypdf import PdfReader
import json,re,csv,hashlib,time,zipfile

class Text(HTMLParser):
    def __init__(self,s):
        super().__init__(); self.parts=[]; self.feed(s)
    def handle_data(self,s): self.parts.append(s)
    def text(self): return re.sub(r'\s+',' ',' '.join(self.parts)).strip()

def get(url):
    for n in range(3):
        try: return fetch(url.replace(' ','%20'))
        except Exception:
            if n==2: raise
            time.sleep(1+n)

rows=[]; catalog=[]; jobs=[]; topics=[]
matrix=json.loads((ROOT/'matrix_links.json').read_text(encoding='utf-8'))
public=json.loads((ROOT/'public_manifest_links.json').read_text(encoding='utf-8'))

def source_year(y):
    base=f'https://www.contest.comap.com/undergraduate/contests/mcm/contests/{y}/'
    d=ROOT/str(y); (d/'官方网页').mkdir(parents=True,exist_ok=True)
    b,final=get(base+'results/'); (d/'官方网页/获奖名单.html').write_bytes(b)
    result_links=[urljoin(final,l) for l in Links(b.decode('utf-8',errors='replace')).links]
    t=Text(b.decode('utf-8',errors='replace')).text().replace('\ufffd','-')
    headers=list(re.finditer(r'Outstanding Winners of (?:THE |the )?(MCM|ICM)\s*\(([A-FYZ])\)\s*Problem',t,re.I))
    for i,h in enumerate(headers):
        p=h.group(2).upper(); segment=t[h.end():headers[i+1].start() if i+1<len(headers) else t.find('Now Available',h.end())]
        if p in 'YZ': continue
        teams=list(re.finditer(r'\b(\d{7})\b',segment))
        entries=[]
        for k,m in enumerate(teams):
            info=segment[m.end():teams[k+1].start() if k+1<len(teams) else len(segment)].strip()
            info=re.split(rf'{y}\s+(?:MCM|ICM)|Problem [A-F] Results|Download the complete',info)[0].strip()
            if '(Finalist)' in info: continue
            entry={'year':y,'problem':p,'team':m.group(1),'institution_awards':info,'award':'Outstanding Winner','official_result_url':final}
            urls=[l for l in public if f'/{y}-{p}/outstanding/{m.group(1)}.pdf' in l]
            entry['download_url']=urls[0] if urls else ''
            entries.append(entry)
        entries.sort(key=lambda e: (-(4*('COMAP Scholarship' in e['institution_awards'])+2*('SIAM Award' in e['institution_awards'])+('Award' in e['institution_awards'])),e['team']))
        selected=[e['team'] for e in entries if e['download_url']][:2 if p in 'ABC' else 1]
        for e in entries:
            e['selected']=e['team'] in selected; catalog.append(e)
            if e['selected']:
                path=d/'获奖论文_公开资料库'/f'{p}题'/f"{y}_{p}_O奖_{e['team']}.pdf"
                jobs.append(dict(e,kind='paper',path=str(path)))
    b,final=get(base+'problems/'); (d/'官方网页/赛题入口.html').write_bytes(b)
    links=[urljoin(final,l) for l in Links(b.decode('utf-8',errors='replace')).links]
    members=[l for l in links if '/membership/member-resources/item/' in l]
    # Primary six contest questions only; 2023 Y/Z are a separate later contest.
    if members:
        for p,url in zip('ABCDEF',members[:6]): topics.append((y,p,url))
    else:
        for p in 'ABCDEF':
            label='MCM' if p in 'ABC' else 'ICM'
            wanted=f'{y}_{label}_Problem_{p}.pdf'
            urls=[l for l in links if l.endswith(wanted)]
            if urls: jobs.append({'year':y,'problem':p,'kind':'problem','download_url':urls[0],'path':str(d/'官方赛题'/wanted)})
        for l in links:
            if l.lower().endswith(('.zip','.csv','.xlsx')):
                jobs.append({'year':y,'problem':'','kind':'data','download_url':l,'path':str(d/'官方数据'/unquote(urlsplit(l).path.split('/')[-1]))})
    for p in 'ABCDEF':
        matches=[l for l in result_links if f'/contests/{y}/results/' in l and re.search(rf'Problem[_ ]{p}_',unquote(l),re.I)]
        if not matches: matches=[l for l in matrix if f'/contests/{y}/results/' in l and re.search(rf'Problem[_ ]{p}_',unquote(l),re.I)]
        if matches: jobs.append({'year':y,'problem':p,'kind':'result','download_url':matches[0],'path':str(d/'官方评奖结果'/f'{y}_{p}_Results.pdf')})
    print('SOURCE',y,'parsed teams',sum(e['year']==y for e in catalog),flush=True)

def topic(item):
    y,p,url=item; b,final=get(url)
    d=ROOT/str(y); (d/f'官方网页/{p}题官方资源.html').write_bytes(b)
    links=[urljoin(final,l) for l in Links(b.decode('utf-8',errors='replace')).links]
    out=[]
    for l in links:
        if 'method=download' in l:
            if '71992483-3d3b-47dc-a146-3c901bdea04f' in l:
                kind='problem'; suffix='.pdf'
            elif '979148b8-68eb-4e88-83df-a28439d5b0e5' in l:
                kind='data'; suffix='.zip'
            else: continue
            label='MCM' if p in 'ABC' else 'ICM'
            path=d/('官方赛题' if kind=='problem' else '官方数据')/f'{y}_{label}_Problem_{p}{suffix}'
            out.append({'year':y,'problem':p,'kind':kind,'download_url':l,'official_resource_url':url,'path':str(path)})
    return out

def download(j):
    r=dict(j); path=Path(j['path']); path.parent.mkdir(parents=True,exist_ok=True)
    try:
        if path.exists(): b=path.read_bytes(); final=j['download_url']
        else: b,final=get(j['download_url'])
        if path.suffix=='.pdf':
            if not b.startswith(b'%PDF'): raise ValueError('Not a PDF')
            pdf=PdfReader(BytesIO(b)); r['pages']=len(pdf.pages)
            if not r['pages']: raise ValueError('Empty PDF')
            if j['kind']=='paper':
                text=' '.join(page.extract_text() or '' for page in pdf.pages[:2])
                if j['team'] not in re.sub(r'\s+','',text):
                    import pymupdf
                    with pymupdf.open(stream=b,filetype='pdf') as alt:
                        text=' '.join(page.get_text() for page in list(alt)[:2])
                    if j['team'] not in re.sub(r'\s+','',text): raise ValueError('Team number not verified in first two pages')
                    r['verification_note']='PyMuPDF verified; pypdf font encoding extraction differs'
                r['team_verified']=True
        elif path.suffix=='.zip':
            with zipfile.ZipFile(BytesIO(b)) as z:
                if z.testzip(): raise ValueError('ZIP CRC failed')
                r['data_files']=len(z.namelist())
        if not path.exists(): path.write_bytes(b)
        r['bytes']=len(b); r['sha256']=hashlib.sha256(b).hexdigest(); r['status']='ok'
        # Avoid retaining transient signed CDN links; stable public URL is recorded.
    except Exception as e: r['status']='failed'; r['error']=str(e)
    print(r['status'],j['year'],j['problem'],j['kind'],path.name,flush=True)
    return r

if __name__=='__main__':
    for y in range(2021,2026): source_year(y)
    with ThreadPoolExecutor(max_workers=5) as ex:
        for f in as_completed([ex.submit(topic,t) for t in topics]): jobs.extend(f.result())
    (ROOT/'全部O奖论文来源索引.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2),encoding='utf-8')
    (ROOT/'下载任务.json').write_text(json.dumps(jobs,ensure_ascii=False,indent=2),encoding='utf-8')
    print('DOWNLOAD JOBS',len(jobs),flush=True)
    with ThreadPoolExecutor(max_workers=5) as ex:
        for f in as_completed([ex.submit(download,j) for j in jobs]):
            rows.append(f.result())
            (ROOT/'下载验证记录.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    print('DONE',len(rows),'OK',sum(r['status']=='ok' for r in rows),flush=True)
