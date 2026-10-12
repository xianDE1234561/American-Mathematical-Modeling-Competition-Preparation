"""Audit and construct country-edition and sport-edition panels from supplied files only."""
from __future__ import annotations
import csv, hashlib, json, re, unicodedata
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw'
PROCESSED = ROOT / 'data/processed'
VALIDATION = ROOT / 'outputs/validation'

def norm(s):
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode()
    return re.sub(r'[^a-z0-9]', '', s.lower())

# Renamings are retained within the recorded delegation, not allocations of predecessor medals.
ALIASES = {
    'cabo verde':'CPV', 'ivory coast':'CIV', 'formosa':'TPE', 'taiwan':'TPE',
    'virgin islands':'ISV', 'british west indies':'WIF', 'ceylon':'SRI',
    'fr yugoslavia':'SCG', 'yugoslavia':'YUG', 'russian empire':'RUS',
    'united team of germany':'GER', 'independent olympic participants':'IOA',
    'independent olympic athletes':'IOA', 'roc':'ROC', 'refugee olympic team':'EOR',
    'north macedonia':'MKD', 'lebanon':'LIB', 'mixed team':'MIX',
    'ireland':'IRL', 'russia':'RUS', 'germany':'GER',
}
ALIASES = {norm(k):v for k,v in ALIASES.items()}

def sport_category(s):
    cats=[]
    for part in str(s).split(','):
        part=part.strip()
        if 'Swimming' in part or part in ['Diving','Water Polo']: cat='Aquatics'
        elif part.startswith('Cycling'): cat='Cycling'
        elif 'Gymnastics' in part or part=='Trampolining': cat='Gymnastics'
        elif part.startswith('Canoe'): cat='Canoeing'
        elif part in ['Baseball','Softball','Baseball/Softball']: cat='Baseball and Softball'
        elif 'Basketball' in part: cat='Basketball'
        elif part in ['Volleyball','Beach Volleyball']: cat='Volleyball'
        elif part in ['Hockey','Field Hockey']: cat='Field hockey'
        elif part=='Rugby Sevens': cat='Rugby'
        elif part=='Equestrianism': cat='Equestrian'
        else: cat=part
        cats.append(cat)
    return cats[0] if len(set(cats))==1 else 'Mixed disciplines'

def verify_inputs():
    m = json.loads((ROOT/'source_manifest.json').read_text(encoding='utf-8-sig'))
    checked = []
    for item in m['files']:
        p = ROOT/item['relative_path']
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if digest != item['sha256']:
            raise ValueError(f'Changed raw input: {p.name}')
        checked.append({'file':item['relative_path'], 'sha256':digest, 'verified':True})
    return checked

def prepare():
    checked = verify_inputs()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    VALIDATION.mkdir(parents=True, exist_ok=True)
    a = pd.read_csv(RAW/'summerOly_athletes.csv')
    m = pd.read_csv(RAW/'summerOly_medal_counts.csv')
    h = pd.read_csv(RAW/'summerOly_hosts.csv')
    p = pd.read_csv(RAW/'summerOly_programs.csv', encoding='cp1252')
    p['Sport']=p.Sport.str.replace('\u00a0',' ',regex=False).str.strip()
    raw_sizes = {f:len(x) for f,x in [('athletes',a),('medals',m),('hosts',h),('programs',p)]}
    exact_duplicates = int(a.duplicated().sum())
    a = a.drop_duplicates().copy()
    for col in ['Team','NOC','Sport','Event','Medal','Name']:
        a[col] = a[col].astype(str).str.strip()
    # The supplied file changes Lebanon's code in 2024; same Team and no predecessor split.
    a['NOC'] = a.NOC.replace({'LBN':'LIB'})
    m['NOC'] = m.NOC.str.replace('\u00a0',' ',regex=False).str.strip()
    names = {}
    map_records = []
    for noc, g in a.groupby('NOC'):
        teams = g.Team.str.replace(r'-\d+$','',regex=True)
        common = teams.value_counts()
        names[noc] = common.index[0]
        for team in common.index:
            map_records.append({'name':team,'normalized':norm(team),'code':noc,'method':'supplied athlete Team'})
    lookup = {}
    for rec in map_records:
        lookup.setdefault(rec['normalized'],set()).add(rec['code'])
    mappings=[]
    for name in m.NOC.unique():
        key=norm(name)
        if key in ALIASES:
            code=ALIASES[key]; method='documented naming alias'
        elif len(lookup.get(key,set()))==1:
            code=next(iter(lookup[key])); method='exact normalized supplied Team match'
        else:
            raise ValueError(f'Unresolved medal delegation: {name}: {lookup.get(key)}')
        mappings.append({'medal_name':name,'NOC':code,'mapping_method':method})
    mapping = pd.DataFrame(mappings)
    m = m.rename(columns={'NOC':'medal_name'}).merge(mapping,on='medal_name',validate='many_to_one')
    assert (m.Gold+m.Silver+m.Bronze==m.Total).all()
    assert (m[['Gold','Silver','Bronze','Total']].to_numpy()>=0).all()
    collisions=m[m.duplicated(['NOC','Year'],keep=False)]
    if len(collisions): raise ValueError('Duplicate official code-edition counts')
    # Country aggregates are official medal-table targets. No athlete-count inflation.
    participation=a.groupby(['NOC','Year']).agg(
        entries=('Name','size'), athletes=('Name',lambda x:x.nunique()),
        sports=('Sport','nunique'), medal_entries=('Medal',lambda x:int((x!='No medal').sum()))).reset_index()
    panel=participation.merge(m[['NOC','Year','Gold','Total']],on=['NOC','Year'],how='outer',indicator=True,validate='one_to_one')
    panel['aggregate_status']=np.where(panel.Total.notna(),'official',np.where(panel.medal_entries.fillna(0)==0,'zero_nonmedalist','unresolved_medalist'))
    unresolved=panel[panel.aggregate_status=='unresolved_medalist'].copy()
    missing_participation=panel[panel['_merge']=='right_only'].copy()
    unresolved.to_csv(VALIDATION/'unresolved_medal_aggregates.csv',index=False)
    missing_participation.to_csv(VALIDATION/'medal_without_participation.csv',index=False)
    panel.loc[panel.aggregate_status=='zero_nonmedalist',['Gold','Total']]=0
    # Preserve unresolved records for audit; they are not labelled zero in supervised learning.
    panel['name']=panel.NOC.map(names).fillna(panel.NOC)
    host_map={}
    for row in h.itertuples():
        txt=str(row.Host).replace('\u00a0',' ').strip()
        if 'Cancelled' in txt: continue
        country=txt.split(',')[-1].split(' (')[0].strip()
        if country=='United Kingdom': code='GBR'
        else:
            key=norm(country)
            codes=lookup.get(key,set())
            code=ALIASES.get(key) or (next(iter(codes)) if len(codes)==1 else None)
        if code is None: raise ValueError(f'Unresolved host: {txt}')
        host_map[int(row.Year)]=code
    panel['host']=panel.apply(lambda r:int(host_map.get(int(r.Year))==r.NOC),axis=1)
    year_cols=[c for c in p if re.fullmatch(r'\d{4}',c)]
    total_events={int(y):int(float(p.loc[p.Sport=='Total events',y].iloc[0])) for y in year_cols}
    panel['events']=panel.Year.map(total_events)
    panel=panel.drop(columns='_merge').sort_values(['NOC','Year'])
    # Keep project opportunity categories consistent with athlete Sport labels.
    prog=p[~p.Sport.str.startswith('Total')].copy()
    for col in year_cols:
        prog[col]=pd.to_numeric(prog[col],errors='coerce').fillna(0)
    program_long=prog.groupby('Sport')[year_cols].sum().reset_index().melt(id_vars='Sport',var_name='Year',value_name='events')
    program_long.Year=program_long.Year.astype(int)
    a['opportunity_sport']=a.Sport.map(sport_category)
    winners=a[a.Medal!='No medal'].copy()
    units=winners.drop_duplicates(['NOC','Year','Sport','Event','Team','Medal'])
    sport=a.groupby(['NOC','Year','opportunity_sport']).agg(entries=('Name','size')).reset_index().rename(columns={'opportunity_sport':'Sport'})
    unit_count=units.groupby(['NOC','Year','opportunity_sport']).size().reset_index(name='medal_units').rename(columns={'opportunity_sport':'Sport'})
    sport=sport.merge(unit_count,on=['NOC','Year','Sport'],how='left').fillna({'medal_units':0})
    compare=units.groupby(['NOC','Year']).size().reset_index(name='reconstructed')
    compare=compare.merge(m[['NOC','Year','Total']],on=['NOC','Year'],how='outer')
    compare['difference']=compare.reconstructed-compare.Total
    compare.to_csv(VALIDATION/'sport_medal_reconciliation.csv',index=False)
    # Consistent proxy for sport analysis; exact national totals are never replaced by units.
    sport=sport.merge(program_long,on=['Sport','Year'],how='left',validate='many_to_one')
    panel.to_csv(PROCESSED/'country_year_panel.csv',index=False)
    mapping.to_csv(PROCESSED/'country_mapping.csv',index=False)
    sport.to_csv(PROCESSED/'country_sport_year.csv',index=False)
    program_long.to_csv(PROCESSED/'sport_programs.csv',index=False)
    m.to_csv(PROCESSED/'official_medals_mapped.csv',index=False)
    hosts={'hosts':host_map,'events':total_events}
    (PROCESSED/'edition_metadata.json').write_text(json.dumps(hosts,indent=2),encoding='utf-8')
    recent=compare[compare.Year>=1996].dropna(subset=['Total','reconstructed'])
    audit={'raw_rows':raw_sizes,'exact_duplicate_athlete_rows_removed':exact_duplicates,
           'mapped_medal_names':len(mapping),'panel_rows':len(panel),'unresolved_aggregate_rows':len(unresolved),
           'recent_unresolved':unresolved[unresolved.Year>=1996][['NOC','Year','medal_entries']].to_dict('records'),
           'official_totals_without_participation':len(missing_participation),
           'recent_sport_reconciliation_exact':int((recent.difference==0).sum()),
           'recent_sport_reconciliation_rows':len(recent),'program_last_year':max(map(int,year_cols)),
           'sport_missing_opportunity_categories_recent':sorted(sport[(sport.events.isna())&(sport.Year>=1996)].Sport.unique().tolist()),
           'inputs':checked,'notes':['Historical predecessor medals are not redistributed.',
               'Lebanon LBN is normalized to LIB based on the same supplied Team label.',
               'Sport medal_units is a deduplicated proxy; country targets use official medal table.',
               'Reported absent aggregates with athlete medals are excluded, never assigned zero.',
               'Athlete counts are approximate name counts: no unique athlete ID is supplied.']}
    (VALIDATION/'data_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in audit.items() if k!='inputs'},ensure_ascii=False,indent=2))
    return panel,sport,program_long,host_map,total_events

if __name__=='__main__': prepare()
