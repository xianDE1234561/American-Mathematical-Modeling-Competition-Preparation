"""Rebuild verified inputs, audits, models and analysis products."""
import sys,json,time,subprocess,platform
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from data_pipeline import prepare
from modeling import run

def main():
    start=time.time()
    panel,sport,program,hosts,events=prepare()
    result=run(panel,hosts)
    from analysis_outputs import generate
    generate(panel,sport,program,hosts,events,*result)
    versions=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
    (ROOT/'requirements-lock.txt').write_text(versions,encoding='utf-8')
    (ROOT/'logs/last_run.json').write_text(json.dumps({'python':sys.version,'platform':platform.platform(),'elapsed_seconds':time.time()-start,'seed':20251009,'completed':True},indent=2),encoding='utf-8')
    print('ALL_NUMERICAL_PRODUCTS_COMPLETE')

if __name__=='__main__':main()
