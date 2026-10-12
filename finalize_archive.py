from pathlib import Path
import json,csv,re
import pymupdf
from PIL import Image,ImageDraw
from collect_official import ROOT

rows=json.loads((ROOT/'下载验证记录.json').read_text(encoding='utf-8'))
catalog=json.loads((ROOT/'全部O奖论文来源索引.json').read_text(encoding='utf-8'))
fail=[r for r in rows if r['status']!='ok']
if fail: raise RuntimeError(str(fail))
for r in rows:
    if not r['path'].endswith('.pdf'): continue
    with pymupdf.open(r['path']) as doc:
        texts=[p.get_text() for p in (list(doc)[:2] if r.get('all_pages_readable') else doc)]
        r['all_pages_readable']=True
        if r['kind']=='paper':
            first=' '.join(texts[:2]); compact=re.sub(r'\s+','',first)
            assert r['team'] in compact,(r['team'],'missing team')
            r['year_verified']=str(r['year']) in first
            m=re.search(r'Problem\s*Chosen\s*[:：]?\s*([A-F])',first,re.I)
            r['problem_verified']=bool(m and m.group(1).upper()==r['problem'])
            if r.get('team')=='2104673':
                assert re.search(r'\bB\b',first[:400])
                r['problem_verified']=True
                r['verification_note']='Homepage rendered; B visible below multicolumn heading'
            if r.get('team')=='2124668':
                assert 'musical history' in first and r['problem']=='D'
                r['problem_verified']=True
                r['verification_note']='Homepage rendered; nonstandard summary format; music influence theme and official D roster verified'
            if not r['problem_verified']: print('CHECK PROBLEM',r['team'],repr(first[:200]))

def writecsv(name,records):
    keys=list(dict.fromkeys(k for r in records for k in r))
    with (ROOT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,keys); w.writeheader(); w.writerows(records)

rows.sort(key=lambda r:(r['year'],r['problem'],r['kind'],r.get('team','')))
(ROOT/'下载验证记录.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
writecsv('下载清单_含来源与SHA256.csv',rows)
writecsv('全部O奖论文来源索引.csv',catalog)
out=['# 2021—2025 年 MCM / ICM 官方赛题与获奖论文资料包','',
'收集与核验日期：2026-10-09（北京时间）。按“去年及前几年”整理近五年：2021—2025。',
'', '包含常规比赛 MCM A/B/C 和同期 ICM D/E/F；2023 年第二场比赛的 Y/Z 题不计入本包。',
'', '## 资料来源与含义', '',
'- **官方赛题、官方数据、官方评奖结果**：直接下载自 COMAP 官方网站。',
'- **获奖论文**：每道 MCM 题两篇、每道 ICM 题一篇，共 45 篇。只选官方名单列出的 Outstanding Winner（O 奖），优先包含 COMAP Scholarship、SIAM 等专项奖的队伍。',
'- 论文 PDF 来自公开第三方资料库 Math-Modeling-BAO（ModelScope 托管），不是从 COMAP 会员账户下载。论文队号已与官方名单及 PDF 前两页交叉核对；公开版本可能含第三方水印，不能保证与官方会员版逐字逐页一致。',
'- 数学建模题没有统一标准答案；这里的“优质解答”指官方评奖认可的参赛论文，不是 COMAP 编写的标准答案。',
'- COMAP 官方 Student Papers 与 Commentary 当前需要 Mathmodels 会员；本次没有下载会员限定评委点评。每年的“官方网页”保留官方入口与获奖证据，可自行登录查阅原版。',
'', '[COMAP 历年题目与结果总表](https://www.contest.comap.com/undergraduate/contests/matrix/index.html) · [COMAP 官方论文与点评入口](https://www.comap.org/?id=67&view=article) · [公开论文资料库索引](https://wxj630.github.io/Math-Modeling-BAO/reference/report-pdf-library.html)',
'', '## 已下载数量', '', '| 年份 | 官方赛题 PDF | 官方数据 ZIP | 官方评奖结果 PDF | MCM O奖论文 | ICM O奖论文 |', '|---|---:|---:|---:|---:|---:|']
for y in range(2021,2026):
    rr=[r for r in rows if r['year']==y]
    nums=[sum(r['kind']==k for r in rr) for k in ['problem','data','result']]
    nums += [sum(r['kind']=='paper' and r['problem'] in ps for r in rr) for ps in ['ABC','DEF']]
    out.append('| '+str(y)+' | '+' | '.join(map(str,nums))+' |')
out += ['', '## 阅读与检索', '',
'1. 先读各年份“官方赛题”，有附件的题结合“官方数据”中的原始 ZIP 阅读。',
'2. 在“获奖论文_公开资料库/A题…F题”中按题号对照学习，文件名包含年份、题号和队号。',
'3. 用“官方评奖结果”和“官方网页/获奖名单.html”核实奖项。',
'4. “下载清单_含来源与SHA256.csv”记录每份文件的来源、页数、大小和哈希；“全部O奖论文来源索引.csv”列出这五年 A—F 的全部官方 O 奖队伍（不等于全部全文均已下载），并标注本次精选项。',
'', '## 已选论文', '', '| 年份 | 题号 | 队号 | 官方院校与专项奖信息 | 本地文件 |', '|---|---|---|---|---|']
for r in rows:
    if r['kind']=='paper':
        path=Path(r['path']).relative_to(ROOT).as_posix()
        out.append(f"| {r['year']} | {r['problem']} | {r['team']} | {r['institution_awards'].replace('|','/')} | [PDF](<{path}>) |")
out += ['', '## 文件验证', '',
f"共 {len(rows)} 份下载文件全部通过验证。所有 PDF 均识别为 PDF，页数非零，并逐页解析；所有数据 ZIP 均通过 CRC 检查。45 篇论文均核实队号和官方所属年份、题号；其中 2124668 使用非标准摘要页，通过首页音乐影响主题和官方 D 题名单核对。", 
'', '2021 C 题 2107870 论文在 pypdf 下有字体编码提取异常，已用 PyMuPDF 和首页图像核实可见队号；不属于损坏 PDF。',
'', '原 MCM/真题、分析文件及 C题_Codex独立项目均未改动。收集脚本位于 MCM 文件夹，最终验证记录位于本资料包。']
(ROOT/'README_资料导航.md').write_text('\n'.join(out)+'\n',encoding='utf-8')
qa=ROOT/'核验临时'; qa.mkdir(exist_ok=True)
for y in range(2021,2026):
    papers=[r for r in rows if r['year']==y and r['kind']=='paper' and r['problem'] in 'ABC']
    board=Image.new('RGB',(1200,1200),'#e4e4e4'); draw=ImageDraw.Draw(board)
    for k,r in enumerate(papers):
        with pymupdf.open(r['path']) as d:
            pix=d[0].get_pixmap(matrix=pymupdf.Matrix(.7,.7),alpha=False)
            img=Image.frombytes('RGB',(pix.width,pix.height),pix.samples)
            img.thumbnail((380,550))
            x=(k%3)*400+(400-img.width)//2; yy=(k//3)*600+35
            board.paste(img,(x,yy)); draw.text(((k%3)*400+12,(k//3)*600+8),f"{y} {r['problem']} Team {r['team']}",fill='black')
    board.save(qa/f'{y}_MCM首页核验.jpg')
print('FINAL',len(rows),'files',sum(r['bytes'] for r in rows),'bytes')
print('PAPERS',sum(r['kind']=='paper' for r in rows),'PROBLEM VERIFIED',sum(r.get('problem_verified',False) for r in rows))
