"""Academic layout implementing project-local latex-paper-en format/table/caption rules.
The native compiler is used separately. The ReportLab export is an independent rendering.
"""
from pathlib import Path
import html,re,json
import numpy as np
import pandas as pd
from reportlab.platypus import (SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,
    PageBreak,Image,KeepTogether,CondPageBreak)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'report';F=ROOT/'outputs/figures';V=ROOT/'outputs/validation'
TITLE='A Budget-Constrained Forecast of Olympic Medal Tables'
CAPTIONS={
 '01_model_comparison':'Rolling validation compares per-delegation errors across the 2008–2020 forecasts; lower MAE is better.',
 '03_2028_total_intervals':'Expected total medals and 95% prediction intervals under the retained-2024 programme and roster scenario.',
 '06_first_medal_count':'The first-medal count distribution includes shared bootstrap parameters and a fixed joint medal budget.',
 '07_sport_heatmap':'Recent sport proxy shares describe programme exposure; they are not exact official sport-level medal allocations.',
 '09_coaching_scenarios':'Conditional gains follow an assumed 20% sport-strength uplift; these are not estimated coach causal effects.',
 '08_specialization_volatility':'Sport concentration and official medal volatility are positively associated in the selected 65-delegation sample.',
 '11_sensitivity':'Budget-normalized point forecasts vary with training windows and the USA host indicator.'}
TABLE_CAPTIONS={
 '1.':'Supplied inputs and their modelling roles.',
 '3.':'Rolling validation errors and first-medal Brier scores; lower values are better.',
 '4.':'2028 expected gold and total medals with 95% prediction intervals.',
 '5.':'Leading first-medal probabilities in the retained participant roster.',
 '7.':'Coaching investment scenarios assuming a 20% sport-strength uplift.',
 '9.':'Ablation errors averaged over the four validation editions.',
 '11.':'Joint-budget retrospective forecasts retaining the preceding edition budget.'}
EQUATIONS=[
 (r'W_{i,t}=0.6T_{i,t-4}+0.3T_{i,t-8}+0.1T_{i,t-12}',
  'Weighted official totals define the historical reference W. Edition labels are four years apart; 2020 denotes the postponed Games held in 2021.'),
 (r'\mu_{i,t}=(W_{i,t}+0.05)\exp(\beta_0+\beta_H H_{i,t}+\beta_P H_{i,t-4})',
  'For previously medalled delegations, H is the supplied current or preceding host indicator. This mean feeds the negative-binomial outcome draw.'),
 (r'q_{i,t}=\mathrm{logistic}(\boldsymbol{\gamma}^{\mathsf{T}}\mathbf{x}_{i,t})',
  'For historically unmedalled delegations, x contains an intercept, log participant scale, clipped participant trend and log observed Games. q is the pre-allocation first-medal probability.'),
 (r'Y_{i,t}\sim\mathrm{NB2}(\mu_{i,t},\alpha),\quad\mathrm{Var}(Y)=\mu+\alpha\mu^2',
  'NB2 describes overdispersed counts. For rare delegations, the general historical-feature NB mean and dispersion feed a zero-truncated draw instead of the anchored established-delegation mean.'),
 (r'\mathrm{E}[Y_{i,t}\mid\mathrm{rare}]=q_{i,t}\frac{\mu_{i,t}}{1-(1+\alpha\mu_{i,t})^{-1/\alpha}}',
  'The hurdle sets Y to zero when no breakthrough is drawn, and otherwise draws a positive NB count. The expectation is before global allocation.'),
 (r'X_{i,t}\mid Y_{i,t}\sim\mathrm{Binomial}(Y_{i,t},s_{i,t})',
  'The recent weighted gold share s, clipped to [0.001,0.999] with pooled fallback 1/3, produces raw gold X. Gold and non-gold weights then enter separate integer allocations.'),
 (r'\sum_i G_{i,t}=328,\quad\sum_i T_{i,t}=1039,\quad0\leq G_{i,t}\leq T_{i,t}',
  'Final G and T are allocated gold and total medals. These budgets are the supplied 2024 reported-delegation aggregates, excluding the unreported AIN aggregate.')]

def enhance_sections(sections,data):
    sections[:]=[s for s in sections if not s['title'].startswith('12. Claim')]
    table_no=0;fig_no=0
    for sec in sections:
        sec['equations']=[];sec['table_caption']=None;sec['figure_caption']=None
        if sec['title']=='2. Forecasting model':
            sec['code']=None;sec['equations']=EQUATIONS
            sec['paragraphs'].insert(1,'The method proceeds from verified national targets to lagged features, established/first-medal count draws, conditional gold draws, and joint budget allocation. Rolling comparisons evaluate the predictive modules; budget tests evaluate outcome coherence. The equations below follow this same input-to-output order.')
        if sec['title']=='4. The 2028 medal table':sec['figure']='03_2028_total_intervals'
        if sec['title']=='11. Reproducibility and references':
            sec['references']=[x for x in sec['paragraphs'] if re.match(r'^\[\d\]',x)]
            sec['paragraphs']=[x for x in sec['paragraphs'] if not re.match(r'^\[\d\]',x)]
            joint=data['joint']
            sec['table']=[['Edition','Total MAE','Gold MAE','Total 95% coverage']]+[[str(int(x.year)),f'{x.total_mae:.3f}',f'{x.gold_mae:.3f}',f'{x.total_coverage95:.1%}'] for x in joint.itertuples()]
        if sec['title'].startswith('Appendix:'):
            sec['title']='Appendix: '+sec['title'].split('Appendix: ',1)[1]
        if sec['title']=='Contents':
            sec['table']=None;sec['paragraphs']=[];continue
        if sec['table']:
            table_no+=1;sec['table_no']=table_no
            cap=next((cap for prefix,cap in TABLE_CAPTIONS.items() if sec['title'].startswith(prefix)),None)
            sec['table_caption']=cap or 'Complete delegation forecasts: '+sec['title'].split('(',1)[-1].rstrip(')')+'.'
            sec['table_note']='Note. Point estimates are simulation means; intervals are conditional predictive quantiles.' if sec['title'].startswith(('4.','Appendix:')) else ('Note. Gains are assumed scenarios, not identified coaching effects.' if sec['title'].startswith('7.') else None)
        if sec['figure']:
            fig_no+=1;sec['figure_no']=fig_no;sec['figure_caption']=CAPTIONS[sec['figure']]
            sec['paragraphs'].append(f'Figure {fig_no} provides the corresponding visual comparison.')
        if sec['table']:sec['paragraphs'].append(f'Table {table_no} reports the numerical results.' if not sec['title'].startswith('1.') else f'Table {table_no} records the input contract.')
    # Project-level method rewrite blueprint and evidence anchors.
    (V/'skill_rewrite_review.md').write_text('''# latex-paper-en 重构记录

技能范围：format / tables / caption；补充 Method section-writing 以闭合已有公式的符号与上下游关系。按顺序在单个代理中执行。

方法部分目标：读者能够从历史标签追踪到未来联合奖牌抽样。

段落角色与模块连接：

| 模块 | 输入 → 输出 | 后续用途 | 证据 |
|---|---|---|---|
| 数据审计 | 原始文件 → 官方国家届次标签 / 项目代理 | lag 特征 / 项目情景 | data_audit.json |
| lag 特征 | 过去届次 → 历史参考和参赛特征 | 已获牌 / 首牌子模型 | 未来标签扰动检查 |
| 计数模型 | 特征 → 原始总牌与金牌抽样 | 联合预算分配 | backtest_metrics.csv |
| 联合分配 | 金牌 / 非金牌权重 → 整数结果 | 均值、区间、首牌数量 | forecast_draws.npz |
| 情景分析 | 项目代理份额 → 假设机会和实力扰动 | 投资与暴露比较 | sport_scenarios.csv |

Claim: 加权基线验证 MAE 最低 | Evidence: model_selection.csv | Status: supported
Claim: 每次联合抽样满足预算、金牌不超过总牌 | Evidence: forecast_draws.npz / tests | Status: supported
Claim: 教练因果贡献 | Evidence: 缺少任职记录 | Status: unsupported；保留为条件情景
Claim: 2028 条件预测 | Evidence: predictions_2028.csv | Status: supported within stated assumptions

自检：结构与计算顺序一致；国家目标 / 代理 / 情景术语保持区分；没有修改统计指标、模型或输入数据；保留基线胜出、2024 回溯和教练无法因果识别的限制。

% FORMAT [Minor] [P2]: Replace manual page-number contents with generated contents and real section hierarchy.
% TABLES [Minor] [P2]: Add captions above tables, labels, top/mid/bottom rules, and right-aligned numeric columns.
% CAPTION [Minor] [P2]: Add concise sentence-case captions with metric and scenario boundaries.
% METHOD [Minor] [P2]: Replace verbatim formula block with numbered equations, symbol glosses and downstream uses.
% Changed: hierarchy, equations, captions, table layout, references and page flow.
% Protected: all saved numerical results, raw data and model implementations.
% Meaning-Check: PRESERVED
% Risk-Flags: none
''',encoding='utf-8')

def _fractions(n):
    return {6:[.075,.305,.10,.15,.11,.16],5:[.12,.18,.21,.22,.27],4:[.34,.22,.22,.22],3:[.27,.4,.33],2:[.5,.5]}[n]

def _math_images():
    dest=R/'equation_art';dest.mkdir(exist_ok=True)
    for i,(formula,_) in enumerate(EQUATIONS,1):
        fig=plt.figure(figsize=(8,.55));fig.text(.5,.5,'$'+formula+'$',ha='center',va='center',fontsize=13)
        fig.savefig(dest/f'eq_{i}.png',dpi=240,bbox_inches='tight',pad_inches=.12);plt.close(fig)
    return dest

class AcademicDoc(SimpleDocTemplate):
    def afterFlowable(self,flowable):
        if isinstance(flowable,Paragraph) and getattr(flowable,'toc_level',None) is not None:
            name=flowable.getPlainText();key=flowable.anchor_key
            self.canv.bookmarkPage(key);self.canv.addOutlineEntry(name,key,flowable.toc_level,False)
            self.notify('TOCEntry',(flowable.toc_level,name,self.page,key))
            self.headings.append({'title':name,'page':self.page})
    def beforeDocument(self):self.headings=[]

def pdf_report(sections):
    fonts=Path(matplotlib.get_data_path())/'fonts/ttf'
    for name,file in [('PaperSerif','DejaVuSerif.ttf'),('PaperBold','DejaVuSerif-Bold.ttf'),('PaperSans','DejaVuSans.ttf')]:
        pdfmetrics.registerFont(TTFont(name,str(fonts/file)))
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle('PaperBody',fontName='PaperSerif',fontSize=10.2,leading=14.5,spaceAfter=7,alignment=4))
    styles.add(ParagraphStyle('PaperHeading',fontName='PaperBold',fontSize=14,leading=18,spaceBefore=16,spaceAfter=9,keepWithNext=True))
    styles.add(ParagraphStyle('PaperCaption',fontName='PaperSerif',fontSize=9,leading=12,spaceAfter=5,spaceBefore=5,keepWithNext=True))
    styles.add(ParagraphStyle('PaperCell',fontName='PaperSerif',fontSize=8,leading=10))
    styles.add(ParagraphStyle('PaperNote',fontName='PaperSerif',fontSize=8.5,leading=11,spaceAfter=9))
    styles.add(ParagraphStyle('PaperTOC',fontName='PaperSerif',fontSize=10,leading=16,spaceBefore=5,leftIndent=0,firstLineIndent=0))
    width=A4[0]-96;story=[];dest=_math_images();first_appendix=True
    def para(text,sty='PaperBody'):return Paragraph(html.escape(text),styles[sty])
    def addimage(path,maxheight):
        im=Image(str(path));ratio=min(width/im.imageWidth,maxheight/im.imageHeight)
        im.drawWidth=im.imageWidth*ratio;im.drawHeight=im.imageHeight*ratio;return im
    for i,sec in enumerate(sections):
        summary=sec['title']=='Summary Sheet';toc=sec['title']=='Contents';ai=sec['title']=='Report on Use of AI Tools';app=sec['title'].startswith('Appendix:')
        if i in (1,2) or ai or app:story.append(PageBreak())
        elif i:story.append(CondPageBreak(120))
        heading=para(sec['title'],'PaperHeading')
        if not summary and not toc:
            heading.toc_level=0;heading.anchor_key='sec_'+str(i)
        story.append(heading)
        if summary:
            story.append(para(TITLE,'PaperHeading'))
        if toc:
            contents=TableOfContents();contents.levelStyles=[styles['PaperTOC']];contents.dotsMinLevel=0;story.append(contents);continue
        for j,text in enumerate(sec['paragraphs']):
            if summary and j==0:continue
            story.append(para(text))
        eqflows=[]
        for no,(eq,gloss) in enumerate(sec.get('equations',[]),1):
            row=Table([[addimage(dest/f'eq_{no}.png',42),para(f'({no})','PaperNote')]],colWidths=[width-30,30])
            row.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'MIDDLE'),('ALIGN',(1,0),(1,0),'RIGHT')]))
            eqflows.append([row,para(gloss,'PaperNote')])
        for k,pair in enumerate(eqflows):
            if k==len(eqflows)-2:
                story.append(KeepTogether(pair+eqflows[k+1]));break
            story.append(KeepTogether(pair))
        if sec['table']:
            tablecap=para(f'Table {sec["table_no"]}. '+sec['table_caption'],'PaperCaption')
            # Caption and table are kept together for compact main-text tables.
            data=[]
            for j,row in enumerate(sec['table']):
                cells=[]
                for k,value in enumerate(row):
                    numeric=sum(bool(re.match(r'^[-+\d\[]',str(rr[k]))) for rr in sec['table'][1:])>len(sec['table'][1:])/2
                    st=ParagraphStyle('cell',parent=styles['PaperCell'],alignment=2 if numeric else 0,fontSize=8 if app else 8.7,leading=10 if app else 11.5,fontName='PaperBold' if j==0 else 'PaperSerif')
                    cells.append(Paragraph(html.escape(str(value)),st))
                data.append(cells)
            table=Table(data,colWidths=[width*x for x in _fractions(len(data[0]))],repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('LINEABOVE',(0,0),(-1,0),.8,colors.black),('LINEBELOW',(0,0),(-1,0),.45,colors.black),('LINEBELOW',(0,-1),(-1,-1),.8,colors.black),('TOPPADDING',(0,0),(-1,-1),2),('BOTTOMPADDING',(0,0),(-1,-1),2),('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4)]))
            story.append(KeepTogether([tablecap,table,Spacer(1,7)]) if len(data)<30 else tablecap)
            if len(data)>=30:story.extend([table,Spacer(1,7)])
            if sec.get('table_note'):story.append(para(sec['table_note'],'PaperNote'))
        if sec['figure']:
            story.append(KeepTogether([addimage(F/(sec['figure']+'.png'),250),para(f'Figure {sec["figure_no"]}. '+sec['figure_caption'],'PaperCaption')]))
        if sec.get('references'):
            story.append(para('References','PaperHeading'))
            for text in sec['references']:story.append(para(text,'PaperNote'))
    def footer(canvas,doc):
        canvas.saveState();canvas.setFont('PaperSans',8);canvas.setFillColor(colors.HexColor('#555555'))
        canvas.drawString(48,25,'2025 MCM C | Training reconstruction');canvas.drawRightString(A4[0]-48,25,str(doc.page));canvas.restoreState()
    doc=AcademicDoc(str(R/'solution.pdf'),pagesize=A4,leftMargin=48,rightMargin=48,topMargin=42,bottomMargin=45,title=TITLE,author='AI-assisted training reconstruction')
    doc.multiBuild(story,onFirstPage=footer,onLaterPages=footer)
    (V/'paper_section_pages.json').write_text(json.dumps(doc.headings,indent=2),encoding='utf-8')

def escape_tex(text):
    substitutions={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
    return ''.join(substitutions.get(c,c) for c in str(text))

def _tex_prose(text):
    text=escape_tex(text)
    keys={1:'comap-problem',2:'comap-data',3:'nb2-doc',4:'codex-use',5:'comap-ai'}
    text=re.sub(r'\[(\d(?:,\d)*)\]',lambda m:r'\cite{'+','.join(keys[int(n)] for n in m[1].split(','))+'}',text)
    text=re.sub(r'Figure (\d+)',lambda m:r'Figure~\ref{fig:'+m[1]+'}',text)
    text=re.sub(r'Table (\d+)',lambda m:r'Table~\ref{tab:'+m[1]+'}',text)
    return text.replace('–','--').replace('—','---')

def _inline_heatmap():
    shares=pd.read_csv(ROOT/'outputs/tables/sport_strength_shares.csv').set_index('NOC')
    top=pd.read_csv(ROOT/'outputs/tables/predictions_2028.csv').sort_values('total_point',ascending=False).head(10).NOC.tolist()
    cols=shares.loc[top].sum().sort_values(ascending=False).head(10).index.tolist()
    parts=[r'\begin{tikzpicture}[x=0.82cm,y=0.40cm]']
    for j,c in enumerate(cols):parts.append(r'\node[rotate=50,anchor=east,font=\scriptsize] at ('+str(j+.5)+',-.2) {'+escape_tex(c)+'};')
    for i,noc in enumerate(top):
        y=len(top)-i-1;parts.append(r'\node[anchor=east,font=\scriptsize] at (-.15,'+str(y+.5)+') {'+noc+'};')
        for j,c in enumerate(cols):
            a=float(shares.loc[noc,c]);intensity=min(90,round(a/.4*90))
            parts.append(r'\filldraw[fill=blue!'+str(intensity)+r'!white,draw=white] ('+f'{j},{y}'+') rectangle ('+f'{j+1},{y+1}'+');')
    parts.append(r'\node[anchor=west,font=\scriptsize] at (0,11) {Proxy share: white = 0; darkest blue = 0.4 or greater};')
    parts.append(r'\end{tikzpicture}');return '\n'.join(parts)

def latex_report(sections):
    # Charts embed numerical coordinates; the native editor needs no external project files.
    from build_report import tex_inline_chart
    parts=[r'\documentclass[11pt,a4paper]{article}',r'\usepackage[margin=1in]{geometry}',r'\usepackage[T1]{fontenc}',r'\usepackage{lmodern}',r'\usepackage{microtype}',r'\usepackage{booktabs}',r'\usepackage{longtable,array,amsmath,amssymb}',r'\usepackage{caption}',r'\captionsetup{font=small,labelfont=bf}',r'\usepackage{tikz,pgfplots}',r'\pgfplotsset{compat=1.18}',r'\usepackage{xurl,hyperref}',r'\hypersetup{hidelinks}',r'\setlength{\parindent}{0pt}',r'\setlength{\parskip}{5pt}',r'\setlength{\tabcolsep}{3pt}',r'\renewcommand{\arraystretch}{1.1}',r'\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}',r'\newcolumntype{R}[1]{>{\raggedleft\arraybackslash}p{#1}}',r'\begin{document}',
     '% FORMAT [P2]: MCM context retained; generated hierarchy, captions and cross-references.',
     '% Protected: numerical outputs and input data. Meaning-Check: PRESERVED. Risk-Flags: none.']
    appendix_started=False;refs=[]
    for sec in sections:
        title=sec['title']
        if title=='Summary Sheet':
            parts += [r'\begin{center}',r'{\Large\bfseries '+TITLE+r'\par}',r'\medskip Summary Sheet | 2025 MCM Problem C | Training reconstruction',r'\end{center}']
        elif title=='Contents':parts += [r'\clearpage',r'\tableofcontents',r'\clearpage'];continue
        elif title=='Report on Use of AI Tools':parts += [r'\clearpage',r'\section*{Report on Use of AI Tools}',r'\addcontentsline{toc}{section}{Report on Use of AI Tools}']
        elif title.startswith('Appendix:'):
            if not appendix_started:parts += [r'\clearpage',r'\appendix',r'\section{Complete delegation forecasts}'];appendix_started=True
            else:parts.append(r'\clearpage')
            parts.append(r'\subsection*{'+escape_tex(title.split('(',1)[1].rstrip(')'))+'}')
        else:parts.append(r'\section{'+escape_tex(re.sub(r'^\d+\.\s*','',title))+'}')
        for j,text in enumerate(sec['paragraphs']):
            if title=='Summary Sheet' and j==0:continue
            parts.append(_tex_prose(text)+'\n')
        for i,(eq,gloss) in enumerate(sec.get('equations',[]),1):
            parts.extend([r'\begin{equation}',eq+r'\label{eq:'+str(i)+'}',r'\end{equation}',_tex_prose(gloss)])
        if sec['table']:
            rows=sec['table'];n=len(rows[0]);fractions=_fractions(n)
            # Width accounts for 2*tabcolsep*n within the available 15.92 cm.
            inner=15.90-(6*n*2.54/72.27)
            spec='@{}'+' '.join(('L' if k<2 and n==6 or k==0 else 'R')+'{'+f'{inner*f:.3f}'+'cm}' for k,f in enumerate(fractions))+'@{}'
            long=title.startswith('Appendix:')
            if not long:parts += [r'\begin{table}[htbp]',r'\centering',r'\caption{'+escape_tex(sec['table_caption'])+'}',r'\label{tab:'+str(sec['table_no'])+'}',r'\small',r'\begin{tabular}{'+spec+'}']
            else:parts += [r'{\small\begin{longtable}{'+spec+'}',r'\caption{'+escape_tex(sec['table_caption'])+r'}\label{tab:'+str(sec['table_no'])+r'}\\']
            parts += [r'\toprule',' & '.join(escape_tex(c) for c in rows[0])+r'\\',r'\midrule']
            if long:parts.append(r'\endhead')
            parts += [' & '.join(escape_tex(c) for c in row)+r'\\' for row in rows[1:]]
            parts += [r'\bottomrule',r'\end{longtable}}' if long else r'\end{tabular}']
            if sec.get('table_note'):parts.append(r'\par\smallskip{\footnotesize '+escape_tex(sec['table_note'])+'}')
            if not long:parts.append(r'\end{table}')
        if sec['figure']:
            name=sec['figure'];parts += [r'\begin{figure}[htbp]',r'\centering']
            if name=='07_sport_heatmap':parts.append(_inline_heatmap())
            elif name=='03_2028_total_intervals':
                df=pd.read_csv(ROOT/'outputs/tables/predictions_2028.csv').sort_values('total_point',ascending=False).head(10).iloc[::-1]
                parts.append(r'\begin{tikzpicture}\begin{axis}[width=.9\linewidth,height=6.7cm,xlabel={Total medals},symbolic y coords={'+','.join(df.NOC)+r'},ytick=data,xmin=0,error bars/x dir=both,error bars/x explicit]')
                parts.append(r'\addplot+[only marks] coordinates {'+' '.join(f'({x.total_point:.4f},{x.NOC}) += ({x.total_hi95-x.total_point:.4f},0) -= ({x.total_point-x.total_lo95:.4f},0)' for x in df.itertuples())+'};')
                parts.append(r'\end{axis}\end{tikzpicture}')
            else:parts.append(tex_inline_chart(name))
            parts += [r'\caption{'+escape_tex(sec['figure_caption']).replace('–','--')+'}',r'\label{fig:'+str(sec['figure_no'])+'}',r'\end{figure}']
        refs.extend(sec.get('references',[]))
        if sec.get('references'):
            keys=['comap-problem','comap-data','nb2-doc','codex-use','comap-ai']
            parts.append(r'\begin{thebibliography}{9}')
            for key,text in zip(keys,refs):
                text=re.sub(r'^\[\d\]\s*','',text)
                segments=re.split(r'(https?://[^\s]+)',text)
                value=''.join(r'\url{'+x+'}' if x.startswith('http') else escape_tex(x) for x in segments)
                parts.append(r'\bibitem{'+key+'}'+value)
            parts.append(r'\end{thebibliography}')
    parts.append(r'\end{document}');(R/'solution.tex').write_text('\n'.join(parts),encoding='utf-8')
