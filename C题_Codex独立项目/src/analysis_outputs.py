"""Sport and coaching scenarios, descriptive insight, sensitivity and scientific figures."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_pipeline import ROOT,PROCESSED,VALIDATION
from modeling import CFG,TABLES,fit_model,predict_params,build_features

FIGURES=ROOT/'outputs/figures'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})

def save(name):
    plt.tight_layout();plt.savefig(FIGURES/(name+'.png'),bbox_inches='tight');plt.savefig(FIGURES/(name+'.pdf'),bbox_inches='tight');plt.close()

def sport_outputs(panel,sport,program,out):
    recent=sport[sport.Year.isin([2016,2020,2024])].copy()
    recent['weight']=recent.Year.map({2016:.1,2020:.3,2024:.6})
    recent['weighted_units']=recent.medal_units*recent.weight
    mix=recent.groupby(['NOC','Sport']).weighted_units.sum().unstack(fill_value=0)
    mix=mix.reindex(out.NOC,fill_value=0)
    sum_units=mix.sum(axis=1)
    entrymix=recent.groupby(['NOC','Sport']).entries.sum().unstack(fill_value=0).reindex(out.NOC,fill_value=0)
    for code in mix.index[sum_units==0]:mix.loc[code]=entrymix.loc[code].reindex(mix.columns,fill_value=0)
    share=mix.div(mix.sum(axis=1).replace(0,1),axis=0)
    share.to_csv(TABLES/'sport_strength_shares.csv')
    events=program[program.Year==2024].set_index('Sport').events.reindex(share.columns,fill_value=0)
    # Hold total opportunity count fixed for the host-oriented reallocation.
    us_strength=share.loc['USA'].sort_values(ascending=False)
    top=us_strength.index[us_strength.index!='Mixed disciplines'][:3].tolist()
    scenarios={'2024 retained':np.ones(len(events)), 'Aquatics +10%':np.where(events.index=='Aquatics',1.1,1),
               'Athletics +10%':np.where(events.index=='Athletics',1.1,1),'Gymnastics +10%':np.where(events.index=='Gymnastics',1.1,1)}
    host_mult=np.where(events.index.isin(top),1.1,1.)
    gained=float((events*(host_mult-1)).sum());others=~events.index.isin(top)
    denom=float(events[others].sum())
    if denom>0:host_mult[others]=1-gained/denom
    scenarios['USA top-3 +10%, fixed opportunity total']=host_mult
    rows=[]
    for name,mult in scenarios.items():
        factor=share.to_numpy()@mult
        raw=out.total_point.to_numpy()*factor
        new_budget=float(out.total_point.sum())*float((events*mult).sum()/events.sum())
        adjusted=raw*new_budget/raw.sum()
        for i,r in enumerate(out.itertuples()):
            rows.append({'scenario':name,'NOC':r.NOC,'name':r.name,'base_total':r.total_point,
                         'opportunity_multiplier':factor[i],'scenario_total':adjusted[i],
                         'change':adjusted[i]-r.total_point,'scenario_total_budget':new_budget,
                         'interpretation':'hypothetical elasticity=1 and budget proportional to event count; not actual 2028 programme'})
    pd.DataFrame(rows).to_csv(TABLES/'sport_scenarios.csv',index=False)
    meta={'USA_top3_proxy_categories':top,'event_budget_2024':float(events.sum()),
          'fixed_budget_reallocation_sum':float((events*host_mult).sum()),
          'limitations':['Shares use reconstructed medal-unit proxies, not fully reconciled national medals.',
                        'Opportunity elasticity 1 is a scenario assumption, not a causal estimate.',
                        'Multi-category athlete records are marked Mixed disciplines.']}
    (VALIDATION/'sport_scenario_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    return share

def coach_outputs(panel,sport,out,share):
    codes=set(out.NOC)
    rows=[]
    for (code,category),g in sport[(sport.Year>=1996)&sport.NOC.isin(codes)&sport.events.notna()&(sport.events>0)].groupby(['NOC','Sport']):
        # Only count observed participation; absent sport gives zero reconstructed units.
        series=g.set_index('Year').medal_units
        exposures=g.set_index('Year').events
        for split in [2008,2012,2016,2020]:
            pre=[split-8,split-4];post=[split,split+4]
            if max(post)>2024 or not set(pre+post).issubset(set(series.index)):continue
            pre_rate=float(sum(series[y] for y in pre)/sum(exposures[y] for y in pre))
            # Control for country-wide official medal trajectory (not a causal adjustment).
            national=panel[(panel.NOC==code)&panel.Year.isin(pre+post)].set_index('Year').Total
            if national.isna().any() or len(national)<4:continue
            strength=float((national.loc[post].sum()+2)/(national.loc[pre].sum()+2))
            expected=float(pre_rate*sum(exposures[y] for y in post)*strength)
            observed=float(sum(series[y] for y in post));excess=observed-expected
            if observed>=3 and excess>1:
                rows.append({'NOC':code,'Sport':category,'split_year':split,'pre_years':str(pre),'post_years':str(post),
                             'post_proxy_units':observed,'baseline_expected_proxy_units':expected,
                             'positive_excess_proxy_units':excess,'country_strength_multiplier':strength,
                             'scenario_no_attribution_proxy_units':0.,'scenario_full_attribution_proxy_units':excess,
                             'evidence':'retrospective proxy change; no coach appointment observed'})
    evidence=pd.DataFrame(rows).sort_values('positive_excess_proxy_units',ascending=False)
    evidence.to_csv(TABLES/'coach_evidence.csv',index=False)
    recent=sport[sport.Year.isin([2012,2016,2020,2024])].copy()
    piv=recent.pivot_table(index=['NOC','Sport'],columns='Year',values='medal_units',fill_value=0)
    candidates=[]
    for (code,category),r in piv.iterrows():
        if code not in codes or category not in share.columns or category=='Mixed disciplines':continue
        current=recent[(recent.NOC==code)&(recent.Sport==category)&(recent.Year==2024)]
        if current.empty or current.entries.sum()<=0:continue
        official=out[out.NOC==code].iloc[0]
        if not 3<=official.total_2024<=45:continue
        history=float(np.mean([r.get(y,0) for y in [2012,2016,2020]]))
        now=float(r.get(2024,0));gap=history-now
        if history>=2 and gap>0:
            candidates.append({'NOC':code,'name':official['name'],'Sport':category,'historical_proxy_mean':history,
                               'proxy_2024':now,'proxy_gap':gap,'share':float(share.loc[code,category]),
                               'country_base_total':float(official.total_point)})
    cand=pd.DataFrame(candidates).sort_values('proxy_gap',ascending=False).drop_duplicates('NOC').head(3)
    if len(cand)!=3:raise ValueError('Insufficient data-backed coaching scenarios')
    scenarios=[]
    for r in cand.to_dict('records'):
        for lift in CFG['coaching_scenario_uplift']:
            rr=dict(r);rr['assumed_sport_gain']=lift
            raw_gain=r['country_base_total']*r['share']*lift
            budget=float(out.total_point.sum())
            rr['scenario_country_total']=budget*(r['country_base_total']+raw_gain)/(budget+raw_gain)
            rr['country_expected_gain']=rr['scenario_country_total']-r['country_base_total']
            rr['interpretation']='assumed 10-30% sport-strength uplift and fixed medal-budget redistribution; not estimated coach causality or budget ROI'
            scenarios.append(rr)
    pd.DataFrame(scenarios).to_csv(TABLES/'coach_investment_scenarios.csv',index=False)
    return evidence,pd.DataFrame(scenarios)

def insight_outputs(panel,share,out):
    rows=[]
    for code in share.index:
        g=panel[(panel.NOC==code)&panel.Year.isin([2012,2016,2020,2024])].dropna(subset=['Total'])
        if len(g)<4 or g.Total.mean()<2:continue
        weights=share.loc[code].to_numpy()
        rows.append({'NOC':code,'concentration_HHI':float(np.sum(weights**2)),
                     'mean_total':float(g.Total.mean()),'medal_cv':float(g.Total.std(ddof=1)/g.Total.mean()),
                     'source':'descriptive: 2016-2024 proxy specialization vs 2012-2024 official volatility'})
    df=pd.DataFrame(rows)
    df.to_csv(TABLES/'specialization_volatility.csv',index=False)
    rho,p=spearmanr(df.concentration_HHI,df.medal_cv)
    insight={'n':len(df),'spearman_rho':float(rho),'p_value':float(p),
             'limitations':'Exploratory association, overlapping windows and medal-unit measurement error; not causal.'}
    (VALIDATION/'original_insight.json').write_text(json.dumps(insight,indent=2),encoding='utf-8')
    return df,insight

def sensitivity_outputs(panel,hosts,features,base,f,out):
    records=[]
    selected=base['kind']
    def budget_point(mean,p):
        g=mean*p; ng=mean*(1-p)
        g=g*(out.gold_point.sum()/g.sum());ng=ng*((out.total_point.sum()-out.gold_point.sum())/ng.sum())
        return g+ng,g
    for label,start in [('training 1996',1996),('training 2000',2000),('training 2004',2004)]:
        mod=fit_model(features[features.Year>=start].dropna(subset=['Total','Gold']),selected)
        mu,p,alpha,pf,mean=predict_params(mod,f)
        bt,bg=budget_point(mean,p)
        records.extend([{'scenario':label,'NOC':f.NOC.iloc[i],'total_point':float(bt[i]),'gold_point':float(bg[i]),'method':'gold/non-gold budget-normalized plug-in means, not bootstrap predictive means'} for i in range(len(f))])
    nohost=f.copy();nohost['host']=0
    mu,p,alpha,pf,mean=predict_params(base,nohost)
    bt,bg=budget_point(mean,p)
    records.extend([{'scenario':'no USA host counterfactual','NOC':f.NOC.iloc[i],'total_point':float(bt[i]),'gold_point':float(bg[i]),'method':'gold/non-gold budget-normalized plug-in means, not bootstrap predictive means'} for i in range(len(f))])
    pd.DataFrame(records).to_csv(VALIDATION/'sensitivity_predictions.csv',index=False)
    # Conditional repeat participation of historically observed never-medal delegations.
    past=panel[(panel.Year>=2012)&panel.Total.notna()&panel.entries.notna()]
    rare_codes=[]
    for code in sorted(past.NOC.unique()):
        if code in set(f.NOC) or code in ['AIN','IOA','ROT','ROC']:continue
        if not (panel[(panel.NOC==code)&panel.Total.notna()].Total>0).any():rare_codes.append(code)
    if rare_codes:
        extra=build_features(panel,2028,hosts,rare_codes)
        for i,r in extra.iterrows():
            prev=panel[(panel.NOC==r.NOC)&panel.entries.notna()].sort_values('Year').iloc[-1]
            extra.loc[i,['prior_athletes','prior_entries','prior_sports']]=prev[['athletes','entries','sports']].to_numpy()
        mu,p,alpha,pf,mean=predict_params(base,extra)
        pd.DataFrame({'NOC':extra.NOC,'name':extra['name'],'conditional_first_probability':pf,'assumption':'returns in 2028 with last-observed participation scale'}).to_csv(TABLES/'extended_first_medal_candidates.csv',index=False)
    else:
        pd.DataFrame(columns=['NOC','name','conditional_first_probability','assumption']).to_csv(TABLES/'extended_first_medal_candidates.csv',index=False)
    # Sensitivity to output competition budget, separate from unconstrained primary model.
    actual_sum=float(panel[(panel.Year==2024)&panel.Total.notna()].Total.sum())
    predicted_sum=float(out.total_point.sum())
    factor=actual_sum/predicted_sum
    pd.DataFrame({'NOC':out.NOC,'unconstrained_point':out.total_point,
                  '2024_budget_normalized_point':out.total_point*factor,
                  'note':'point-only budget sensitivity, not primary forecast or a replacement predictive interval'}).to_csv(VALIDATION/'aggregate_budget_sensitivity.csv',index=False)
    return {'observed_2024_total':actual_sum,'forecast_total':predicted_sum,'budget_ratio':predicted_sum/actual_sum,'additional_never_medal_candidates':rare_codes}

def plots(out,metrics,pred,share,evidence,coaches,insight_df):
    fig,ax=plt.subplots(figsize=(8,4.8))
    m=metrics[metrics.year<=2020].groupby('model')[['total_mae','gold_mae']].mean().sort_values('total_mae')
    m.plot.bar(ax=ax,color=['#207A9A','#EAA54A']);ax.set_ylabel('Mean absolute error (medals)');ax.set_title('Rolling validation, 2008-2020');ax.tick_params(axis='x',rotation=25);save('01_model_comparison')
    selected=json.loads((VALIDATION/'model_selection.json').read_text())['selected_model']
    test=pred[(pred.Year==2024)&(pred.model==selected)]
    fig,ax=plt.subplots(figsize=(7,5));ax.scatter(test.total_actual,test.total_point,s=20,alpha=.65,color='#207A9A');lim=max(test.total_actual.max(),test.total_point.max())+5;ax.plot([0,lim],[0,lim],color='#777',ls='--');ax.set(xlabel='Observed 2024 total',ylabel='Forecast total',title='Retrospective 2024 check');save('02_2024_observed_predicted')
    top=out.sort_values('gold_point',ascending=False).head(15).iloc[::-1]
    fig,ax=plt.subplots(figsize=(8,6));mid=top.total_mean.to_numpy();err=np.vstack([mid-top.total_lo95,top.total_hi95-mid]);ax.errorbar(mid,top.name,xerr=err,fmt='o',capsize=3,color='#207A9A');ax.set(xlabel='Total medals: predictive mean and 95% interval',title='2028 conditional scenario; 2024 roster and programme');save('03_2028_total_intervals')
    fig,ax=plt.subplots(figsize=(8,6));mid=top.gold_mean.to_numpy();err=np.vstack([mid-top.gold_lo95,top.gold_hi95-mid]);ax.errorbar(mid,top.name,xerr=err,fmt='o',capsize=3,color='#C89523');ax.set(xlabel='Gold medals: predictive mean and 95% interval',title='2028 gold: joint binomial-with-total model');save('04_2028_gold_intervals')
    fp=pd.read_csv(TABLES/'first_medal_probabilities.csv').head(15).iloc[::-1]
    fig,ax=plt.subplots(figsize=(8,5));ax.barh(fp.name,fp.first_medal_probability,color='#207A9A');ax.set(xlabel='Conditional probability of first summer medal',title='First-medal candidates assuming 2024 roster returns');save('05_first_medal_probabilities')
    dist=pd.read_csv(TABLES/'first_medal_count_distribution.csv')
    fig,ax=plt.subplots(figsize=(7,4));ax.bar(dist.first_medal_count,dist.probability,color='#207A9A');ax.set(xlabel='Number of first-medal delegations',ylabel='Probability',title='2028 first-medal count distribution');save('06_first_medal_count')
    codes=out.sort_values('total_point',ascending=False).head(15).NOC
    ss=share.loc[codes]; cats=ss.sum().sort_values(ascending=False).head(12).index
    fig,ax=plt.subplots(figsize=(10,6));im=ax.imshow(ss[cats],aspect='auto',cmap='Blues',vmin=0);ax.set_xticks(range(len(cats)),cats,rotation=40,ha='right');ax.set_yticks(range(len(codes)),codes);fig.colorbar(im,ax=ax,label='Share of reconstructed recent medal-unit proxy');ax.set_title('Sport specialization (proxy, not exact official medal allocation)');save('07_sport_heatmap')
    fig,ax=plt.subplots(figsize=(7,5));ax.scatter(insight_df.concentration_HHI,insight_df.medal_cv,s=22,color='#207A9A',alpha=.7);ax.set(xlabel='Sport specialization HHI (proxy)',ylabel='Official total-medal coefficient of variation',title='Exploratory specialization and volatility');save('08_specialization_volatility')
    c=coaches[coaches.assumed_sport_gain==.2].sort_values('country_expected_gain')
    fig,ax=plt.subplots(figsize=(8,4));ax.barh(c['name']+' / '+c.Sport,c.country_expected_gain,color='#EAA54A');ax.set(xlabel='Expected medal gain under assumed 20% sport-strength uplift',title='Coaching investment scenarios: assumed effects, not causal estimates');save('09_coaching_scenarios')
    sm=metrics[metrics.model==selected]
    fig,ax=plt.subplots(figsize=(7,4));ax.plot(sm.year,sm.total_coverage95,'o-',label='Total');ax.plot(sm.year,sm.gold_coverage95,'s-',label='Gold');ax.axhline(.95,color='#777',ls='--');ax.set(xlabel='Target edition',ylabel='Empirical 95% interval coverage',ylim=(.7,1.02));ax.legend();ax.set_title('Process-only backtest intervals; future adds cluster bootstrap');save('10_interval_coverage')
    sens=pd.read_csv(VALIDATION/'sensitivity_predictions.csv');topcodes=out.sort_values('total_point',ascending=False).head(8).NOC
    sp=sens[sens.NOC.isin(topcodes)].pivot(index='NOC',columns='scenario',values='total_point')
    fig,ax=plt.subplots(figsize=(10,5));sp.plot.bar(ax=ax);ax.set_ylabel('Expected total medals');ax.tick_params(axis='x',rotation=0);ax.set_title('Training-window and host sensitivity');ax.legend(fontsize=8);save('11_sensitivity')

def generate(panel,sport,program,hosts,events,out,base,f,features,metrics,pred):
    FIGURES.mkdir(parents=True,exist_ok=True)
    share=sport_outputs(panel,sport,program,out)
    evidence,coaches=coach_outputs(panel,sport,out,share)
    idf,insight=insight_outputs(panel,share,out)
    sensitivity=sensitivity_outputs(panel,hosts,features,base,f,out)
    (VALIDATION/'sensitivity_summary.json').write_text(json.dumps(sensitivity,indent=2),encoding='utf-8')
    plots(out,metrics,pred,share,evidence,coaches,idf)
    print('SUPPLEMENTARY_ANALYSIS',json.dumps({'coaching_cases':coaches.NOC.unique().tolist(),'insight':insight,'budget':sensitivity},ensure_ascii=False,indent=2))

if __name__=='__main__':
    panel=pd.read_csv(PROCESSED/'country_year_panel.csv');sport=pd.read_csv(PROCESSED/'country_sport_year.csv');prog=pd.read_csv(PROCESSED/'sport_programs.csv')
    meta=json.loads((PROCESSED/'edition_metadata.json').read_text());hosts={int(k):v for k,v in meta['hosts'].items()};events={int(k):v for k,v in meta['events'].items()}
    out=pd.read_csv(TABLES/'predictions_2028.csv').sort_values('NOC').reset_index(drop=True)
    base=json.loads((VALIDATION/'fitted_model.json').read_text());f=pd.read_csv(PROCESSED/'forecast_features_2028.csv')
    features=pd.read_csv(PROCESSED/'forecast_feature_panel.csv');metrics=pd.read_csv(VALIDATION/'backtest_metrics.csv');pred=pd.read_csv(VALIDATION/'backtest_predictions.csv')
    generate(panel,sport,prog,hosts,events,out,base,f,features,metrics,pred)
