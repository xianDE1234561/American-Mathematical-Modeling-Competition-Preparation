from pathlib import Path
import json,re,csv,hashlib
import pymupdf

OUT=Path(__file__).resolve().parent
ARCHIVE=OUT.parent/'历年官方资料_2021-2025'
TEXT=OUT/'证据/逐页文本'; TEXT.mkdir(parents=True,exist_ok=True)
ROWS=json.loads((ARCHIVE/'下载验证记录.json').read_text(encoding='utf-8'))
PATTERNS={
 'assumptions':r'\bassumptions?\b',
 'sensitivity':r'\bsensitiv(?:ity|ities)\b',
 'validation':r'\b(?:validation|validate[ds]?|cross[- ]validation|back[- ]test(?:ing)?|out[- ]of[- ]sample)\b',
 'uncertainty':r'\b(?:uncertainty|confidence interval|prediction interval|credible interval)\b',
 'limitations':r'\b(?:weakness(?:es)?|limitations?|shortcomings?)\b',
 'references':r'\b(?:references|bibliography)\b',
 'baseline':r'\b(?:baseline|benchmark|comparison|compare[ds]?)\b',
 'recommendations':r'\b(?:recommendations?|memorandum|memo|letter)\b',
}
DEEP=['2100454','2102199','2107870','2212245','2213970','2229059','2303950','2300136','2314151','2424371','2419984','2401445','2511565','2501687','2513314','2124668','2410889','2507789']

def squash(t):return re.sub(r'\s+',' ',t).strip()

if __name__=='__main__':
 papers=[]; scan=[]; focused=[]
 for r in ROWS:
  if r['kind']!='paper':continue
  path=Path(r['path']); team=r['team']
  with pymupdf.open(path) as doc:
   pages=[p.get_text(sort=True) for p in doc]
   blocks=[]
   for p in doc:
    blocks.append({'page':p.number+1,'width':p.rect.width,'height':p.rect.height,'images':len(p.get_images()),'text_chars':len(p.get_text())})
  txt=TEXT/f'{r["year"]}_{r["problem"]}_{team}.txt'
  txt.write_text('\n\n'.join(f'=== PDF PAGE {i+1} ===\n{t}' for i,t in enumerate(pages)),encoding='utf-8')
  entry={**r,'text_path':str(txt),'deep_read':team in DEEP,'page_stats':blocks,'features':{}}
  for name,pat in PATTERNS.items():
   matches=[]
   for i,t in enumerate(pages):
    s=squash(t)
    ms=list(re.finditer(pat,s,re.I))
    if ms:
     m=ms[-1] if name in ['assumptions','limitations'] else ms[0]
     matches.append({'page':i+1,'excerpt':s[max(0,m.start()-120):m.end()+650]})
   entry['features'][name]=matches
  papers.append(entry)
  line={'year':r['year'],'problem':r['problem'],'team':team,'pages':r['pages'],'deep_read':entry['deep_read'],'source_path':str(path)}
  for name in PATTERNS:line[name+'_pages']=';'.join(str(m['page']) for m in entry['features'][name])
  scan.append(line)
  if team in DEEP:
   focused += [f'## {r["year"]} {r["problem"]} {team}',f'FILE: {path}',f'ABSTRACT p1: {squash(pages[0])[:4800]}']
   for feat in ['assumptions','validation','sensitivity','uncertainty','limitations','baseline']:
    matches=[m for m in entry['features'][feat] if m['page']>2]
    if not matches:matches=entry['features'][feat]
    focused.append(feat+': '+'\n'.join(f'p{m["page"]}: {m["excerpt"]}' for m in matches[-2:]))
  print(r['year'],r['problem'],team,r['pages'],' '.join(f'{k}:'+','.join(str(x['page']) for x in v if x['page']>2)[-35:] for k,v in entry['features'].items()),flush=True)
 (OUT/'证据/论文特征检索.json').write_text(json.dumps(papers,ensure_ascii=False,indent=2),encoding='utf-8')
 with (OUT/'45篇论文结构检索.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,list(scan[0]));w.writeheader();w.writerows(scan)
 (OUT/'证据/18篇重点阅读摘录.txt').write_text('\n\n'.join(focused),encoding='utf-8')
 print('DONE',len(papers),'deep',len(DEEP))
