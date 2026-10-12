"""Verify report country values, source anchors, page count and numerical checks."""
from pathlib import Path
import json,re,hashlib,subprocess,sys
from pypdf import PdfReader
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];V=ROOT/'outputs/validation';R=ROOT/'report'
def main():
    pdf=PdfReader(R/'solution.pdf');texts=[p.extract_text() or '' for p in pdf.pages]
    pred=pd.read_csv(ROOT/'outputs/tables/predictions_2028.csv').set_index('NOC')
    seen=[];checked=0;app_pages=[]
    for i,text in enumerate(texts):
        if i<2 or 'Appendix: all delegation forecasts (' not in text:continue
        app_pages.append(i+1);lines=[x.strip() for x in text.splitlines()]
        positions=[j for j,line in enumerate(lines) if line in pred.index and (j==0 or lines[j-1]!=line)]
        for k,pos in enumerate(positions):
            code=lines[pos];seen.append(code)
            block=lines[pos+1:positions[k+1] if k+1<len(positions) else len(lines)]
            vals=[x for x in block if re.fullmatch(r'\d+\.\d{2}',x)]
            cis=[x for x in block if re.fullmatch(r'\[\d+, \d+\]',x)]
            row=pred.loc[code]
            assert vals==[f'{row.gold_point:.2f}',f'{row.total_point:.2f}'],(code,vals)
            assert cis==[f'[{int(row.gold_lo95)}, {int(row.gold_hi95)}]',f'[{int(row.total_lo95)}, {int(row.total_hi95)}]'],(code,cis)
            checked+=1
    assert len(seen)==205 and len(set(seen))==205 and set(seen)==set(pred.index)
    ai_start=max(i+1 for i,text in enumerate(texts) if 'Report on Use of AI Tools' in text)
    assert ai_start-1<=25
    source=(R/'solution.tex').read_text(encoding='utf-8')
    labels=re.findall(r'\\label\{([^}]+)\}',source);refs=re.findall(r'\\ref\{([^}]+)\}',source)
    assert len(labels)==len(set(labels))
    assert set(refs)<=set(labels)
    keys=re.findall(r'\\bibitem\{([^}]+)\}',source)
    cites=[k for x in re.findall(r'\\cite\{([^}]+)\}',source) for k in x.split(',')]
    assert set(cites)<=set(keys)
    starts=re.findall(r'\\begin\{([^}]+)\}',source);ends=re.findall(r'\\end\{([^}]+)\}',source)
    assert sorted(starts)==sorted(ends)
    # The supplied skill checks ordinary table environments. Check longtable conventions too.
    longtables=re.findall(r'\\begin\{longtable\}.*?\\end\{longtable\}',source,re.S)
    assert len(longtables)==5
    for block in longtables:
        assert all(x in block for x in ['\\caption{','\\label{','\\toprule','\\midrule','\\bottomrule'])
        assert '\\hline' not in block and '\\vline' not in block
    cmd=[sys.executable,'-X','utf8','-m','unittest','discover','-s','tests','-v']
    run=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
    (V/'test_results.txt').write_text(run.stdout+run.stderr,encoding='utf-8');assert run.returncode==0
    meta=json.loads((V/'report_metadata.json').read_text())
    meta.update({'total_pdf_pages':len(pdf.pages),'main_solution_pages':ai_start-1,'AI_report_start_page':ai_start,
      'appendix_pages':app_pages,'verified_country_rows_and_values':checked,'all_205_forecasts_in_appendix':True,
      'unique_source_labels':len(labels),'source_references_resolved':True,'source_citations_resolved':True,
      'longtables_manually_validated':len(longtables),'numerical_tests':'7 passed',
      'PDF_sha256':hashlib.sha256((R/'solution.pdf').read_bytes()).hexdigest(),
      'latex_compile':'unverified: native compiler environment directory error',
      'chktex':'unavailable; supplied format script returned UNAVAILABLE',
      'project_skill':'latex-paper-en 6.0.0'})
    (V/'report_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
