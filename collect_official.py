from pathlib import Path
from urllib.request import urlopen, Request
from urllib.parse import urljoin
import re, json
from html.parser import HTMLParser

ROOT = Path(__file__).parent / '历年官方资料_2021-2025'
ROOT.mkdir(exist_ok=True)

def fetch(url):
    with urlopen(Request(url, headers={'User-Agent':'Mozilla/5.0'}), timeout=60) as r:
        return r.read(), r.url

class Links(HTMLParser):
    def __init__(self, text):
        super().__init__(); self.links=[]; self.feed(text)
    def handle_starttag(self, tag, attrs):
        if tag=='a':
            d=dict(attrs)
            if d.get('href'): self.links.append(d['href'])

if __name__=='__main__':
    for name,url in [('mathmodels','https://www.comap.org/?id=67&view=article'),('matrix','https://www.contest.comap.com/undergraduate/contests/matrix/index.html'),('public_manifest','https://wxj630.github.io/Math-Modeling-BAO/reference/report-pdf-library.html')]:
        b,final=fetch(url); (ROOT/(name+'.html')).write_bytes(b)
        links=Links(b.decode('utf-8',errors='replace')).links
        (ROOT/(name+'_links.json')).write_text(json.dumps([urljoin(final,l) for l in links],ensure_ascii=False,indent=2),encoding='utf-8')
        print(name, len(links), flush=True)
        print('\n'.join(l for l in links if any(w in l for w in ['2513314','2401298','2025_MCM','2021_MCM'])),flush=True)
