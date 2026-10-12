"""Leakage-aware edition forecasts, count models and a first-medal hurdle."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit, gammaln
from scipy.stats import nbinom, poisson
from data_pipeline import ROOT, PROCESSED, VALIDATION

CFG=json.loads((ROOT/'config/analysis.json').read_text(encoding='utf-8'))
TABLES=ROOT/'outputs/tables'
FEATURES=['intercept','log_weighted_total','log_prior_athletes','log_prior_sports','host','prior_host','trend']
MODELS=['last','weighted','poisson','negative_binomial','nb_first_hurdle','anchored_hurdle']

def build_features(panel, year, hosts, roster=None):
    """Roster and all non-host features use editions strictly earlier than target."""
    past=panel[panel.Year<year].copy()
    prev_year=int(past.Year.max())
    if roster is None:
        roster=sorted(past[(past.Year==prev_year)&past.entries.notna()&past.Total.notna()].NOC.unique())
    rows=[]
    for noc in roster:
        g=past[past.NOC==noc].sort_values('Year')
        known=g.dropna(subset=['Total'])
        prev=g[g.Year==prev_year]
        old=g[g.Year==prev_year-4]
        def value(ed,col):
            z=g[g.Year==ed]
            return float(z[col].iloc[0]) if len(z) and pd.notna(z[col].iloc[0]) else 0.
        totals=[value(prev_year-4*k,'Total') for k in range(3)]
        golds=[value(prev_year-4*k,'Gold') for k in range(3)]
        weighted=float(np.dot([.6,.3,.1],totals)); wg=float(np.dot([.6,.3,.1],golds))
        last=g.iloc[-1] if len(g) else None
        athletes=float(prev.athletes.iloc[0]) if len(prev) and pd.notna(prev.athletes.iloc[0]) else 0.
        entries=float(prev.entries.iloc[0]) if len(prev) and pd.notna(prev.entries.iloc[0]) else 0.
        sports=float(prev.sports.iloc[0]) if len(prev) and pd.notna(prev.sports.iloc[0]) else 0.
        prior_host=int(hosts.get(prev_year)==noc)
        ever=bool((known.Total>0).any())
        name=str(last['name']) if last is not None else noc
        rows.append({'NOC':noc,'name':name,'Year':year,'source_max_year':prev_year,
                     'last_total':totals[0],'last_gold':golds[0],'weighted_total':weighted,'weighted_gold':wg,
                     'prior_athletes':athletes,'prior_entries':entries,'prior_sports':sports,
                     'host':int(hosts.get(year)==noc),'prior_host':prior_host,
                     'trend':float(np.clip(np.log1p(totals[0])-np.log1p(totals[1]),-1.5,1.5)),
                     'prior_athlete_trend':float(np.log1p(athletes)-np.log1p(value(prev_year-4,'athletes'))),
                     'past_games':int(g.entries.notna().sum()),'ever_medal':ever,
                     'gold_share':(sum(golds)+1)/(sum(totals)+3),
                     'Total':np.nan,'Gold':np.nan})
    out=pd.DataFrame(rows)
    actual=panel[panel.Year==year][['NOC','Total','Gold','entries','aggregate_status']].rename(columns={'entries':'target_entries'})
    out=out.drop(columns=['Total','Gold']).merge(actual,on='NOC',how='left',validate='one_to_one')
    out['target_participated']=out.target_entries.notna()
    # Absent delegates receive zero outcome in historical roster forecasts, not zero ability.
    absent=out.aggregate_status.isna()
    out.loc[absent,['Total','Gold']]=0
    return out

def design(f, ablation=None):
    cols=[np.ones(len(f)),np.log1p(f.weighted_total),np.log1p(f.prior_athletes),np.log1p(f.prior_sports),f.host.astype(float),f.prior_host.astype(float),f.trend]
    X=np.column_stack(cols).astype(float)
    if ablation=='host': X[:,4:6]=0
    if ablation=='athletes': X[:,2]=0
    if ablation=='sports': X[:,3]=0
    return X

def gold_design(f):
    return np.column_stack([np.ones(len(f)),logit(np.clip(f.gold_share.to_numpy(),.02,.98)),f.host,f.prior_host])

def first_design(f):
    return np.column_stack([np.ones(len(f)),np.log1p(f.prior_athletes),np.clip(f.prior_athlete_trend,-2,2),np.log1p(f.past_games)])

def fit_count(train, kind, ablation=None):
    train=train.dropna(subset=['Total'])
    X=design(train,ablation); y=train.Total.to_numpy(float); n=len(y)
    init=np.zeros(X.shape[1]);init[0]=-2;init[1]=.8;init[2]=.4
    nb=kind in ['negative_binomial','nb_first_hurdle']
    start=np.r_[init,np.log(.3)] if nb else init
    def obj(theta):
        beta=theta[:X.shape[1]]; mu=np.exp(np.clip(X@beta,-12,8))
        if nb:
            alpha=np.exp(theta[-1]); r=1/alpha
            ll=nbinom.logpmf(y,r,r/(r+mu))
        else: ll=poisson.logpmf(y,mu)
        return -float(ll.mean())+.002*np.sum(beta[1:]**2)
    bounds=[(-8,8)]*len(init)+([(-8,2)] if nb else [])
    opt=minimize(obj,start,method='L-BFGS-B',bounds=bounds,options={'maxiter':800,'ftol':1e-10})
    if not opt.success: raise RuntimeError(f'Count fit failed: {kind}: {opt.message}')
    return {'beta':opt.x[:len(init)].tolist(),'alpha':float(np.exp(opt.x[-1])) if nb else 0.,
            'objective':float(opt.fun),'converged':True,'kind':kind,'ablation':ablation}

def fit_gold(train):
    tr=train.dropna(subset=['Total','Gold']); tr=tr[tr.Total>0]
    X=gold_design(tr); success=tr.Gold.to_numpy(float); trials=tr.Total.to_numpy(float)
    def obj(b):
        z=X@b
        return float(np.mean(trials*np.logaddexp(0,z)-success*z))+.01*float(np.sum(b[1:]**2))
    fit=minimize(obj,[0,1,0,0],method='L-BFGS-B',bounds=[(-5,5)]*4)
    if not fit.success: raise RuntimeError('Gold-share fit failed')
    return fit.x.tolist()

def fit_first(train):
    tr=train[(~train.ever_medal)&train.Total.notna()]
    X=first_design(tr); y=(tr.Total>0).to_numpy(float)
    if y.sum()==0: return {'beta':[-8,0,0,0],'n':len(y),'positive':0,'converged':True}
    def obj(b):
        z=X@b
        return float(np.mean(np.logaddexp(0,z)-y*z))+.006*float(np.sum(b[1:]**2))
    start=[float(logit(np.clip(y.mean(),.001,.999))),.2,0,0]
    fit=minimize(obj,start,method='L-BFGS-B',bounds=[(-12,6)]*4)
    if not fit.success: raise RuntimeError('First-medal fit failed')
    return {'beta':fit.x.tolist(),'n':len(y),'positive':int(y.sum()),'converged':True}

def fit_model(train,kind,ablation=None):
    result={'kind':kind,'gold_beta':fit_gold(train),'first':fit_first(train),'ablation':ablation}
    if kind=='anchored_hurdle':
        tr=train.dropna(subset=['Total']); tr=tr[tr.ever_medal]
        X=np.column_stack([np.ones(len(tr)),tr.host,tr.prior_host]).astype(float)
        if ablation=='host': X[:,1:]=0
        offset=np.log(tr.weighted_total.to_numpy()+.05); y=tr.Total.to_numpy()
        def objective(theta):
            mu=np.exp(np.clip(offset+X@theta[:3],-12,8)); alpha=np.exp(theta[3]); r=1/alpha
            return -float(nbinom.logpmf(y,r,r/(r+mu)).mean())+.02*float(np.sum(theta[:3]**2))
        fit=minimize(objective,[0,.2,-.1,np.log(.2)],method='L-BFGS-B',bounds=[(-1,1),(-1,1),(-1,1),(-8,2)],options={'maxiter':800})
        if not fit.success:raise RuntimeError('Anchored fit failed')
        result.update({'anchor_beta':fit.x[:3].tolist(),'alpha':float(np.exp(fit.x[3])), 'objective':float(fit.fun),'converged':True})
        result['rare_count']=fit_count(train,'negative_binomial',ablation)
    elif kind not in ['last','weighted']: result.update(fit_count(train,kind,ablation))
    return result

def predict_params(model,f):
    kind=model['kind']
    if kind in ['last','weighted']:
        mu=f[f'{kind}_total'].to_numpy(float)
        gm=f[f'{kind}_gold'].to_numpy(float)
        p=np.divide(gm,mu,out=np.full_like(mu,1/3),where=mu>0)
    elif kind=='anchored_hurdle':
        X=np.column_stack([np.ones(len(f)),f.host,f.prior_host]).astype(float)
        if model.get('ablation')=='host':X[:,1:]=0
        mu=(f.weighted_total.to_numpy()+.05)*np.exp(X@np.array(model['anchor_beta']))
        rare=~f.ever_medal.to_numpy(); rc=model['rare_count']
        mu[rare]=np.exp(np.clip(design(f,model.get('ablation'))[rare]@np.array(rc['beta']),-12,8))
        p=np.divide(f.weighted_gold, f.weighted_total, out=np.full(len(f),1/3),where=f.weighted_total.to_numpy()>0)
    else:
        mu=np.exp(np.clip(design(f,model.get('ablation'))@np.array(model['beta']),-12,8))
        p=expit(gold_design(f)@np.array(model['gold_beta']))
    alpha=model.get('alpha',0.)
    p0=np.exp(-mu) if alpha<=0 else (1+alpha*mu)**(-1/alpha)
    pf=expit(first_design(f)@np.array(model['first']['beta']))
    mean=mu.copy()
    if kind in ['nb_first_hurdle','anchored_hurdle']:
        zero=~f.ever_medal.to_numpy()
        mean[zero]=pf[zero]*mu[zero]/np.maximum(1-p0[zero],1e-10)
    return mu,np.clip(p,0.001,.999),alpha,pf,mean

def simulate(model,f,draws,rng,scale=None):
    mu,p,alpha,pf,mean=predict_params(model,f)
    if scale is not None: mu=mu*np.asarray(scale)
    if alpha<=0: t=rng.poisson(mu,size=(draws,len(f)))
    else: t=rng.negative_binomial(1/alpha,1/(1+alpha*mu),size=(draws,len(f)))
    if model['kind'] in ['nb_first_hurdle','anchored_hurdle']:
        rare=~f.ever_medal.to_numpy(); rm=mu[rare]
        p0=(1+alpha*rm)**(-1/alpha)
        u=p0+rng.random((draws,int(rare.sum())))*(1-p0)
        positive=nbinom.ppf(np.minimum(u,1-1e-12),1/alpha,1/(1+alpha*rm)).astype(int)
        t[:,rare]=positive*(rng.random(positive.shape)<pf[rare])
    g=rng.binomial(t,p)
    assert np.all(g<=t) and np.all(t>=0)
    return t,g

def summarize(draw_t,draw_g,f):
    out=f[['NOC','name']].copy()
    for label,d in [('total',draw_t),('gold',draw_g)]:
        out[label+'_mean']=d.mean(axis=0)
        out[label+'_median']=np.median(d,axis=0)
        out[label+'_lo95']=np.quantile(d,.025,axis=0)
        out[label+'_hi95']=np.quantile(d,.975,axis=0)
    out['p_any_medal']=(draw_t>0).mean(axis=0)
    return out

def allocate_budget(weights,budget,rng):
    """Hamilton integer allocation preserving zero support and global medal budgets."""
    totals=weights.sum(axis=1,keepdims=True)
    if np.any(totals<=0):raise ValueError('No positive support for allocation')
    fractional=weights/totals*int(budget)
    result=np.floor(fractional).astype(int)
    remaining=int(budget)-result.sum(axis=1)
    fractions=fractional-result+rng.uniform(0,1e-10,weights.shape)
    fractions[weights==0]=-1
    order=np.argsort(-fractions,axis=1)
    for i,k in enumerate(remaining):result[i,order[i,:int(k)]]+=1
    assert np.all(result.sum(axis=1)==budget)
    return result

def build_training(panel, hosts):
    frames=[]
    for year in sorted(panel.Year.unique()):
        if year>=CFG['training_start'] and year<=2024:
            frames.append(build_features(panel,int(year),hosts))
    features=pd.concat(frames,ignore_index=True)
    features.to_csv(PROCESSED/'forecast_feature_panel.csv',index=False)
    return features

def backtest(features):
    metrics=[];predictions=[]; fitted={};rng=np.random.default_rng(CFG['seed'])
    years=CFG['validation_years']+[CFG['holdout_year']]
    for year in years:
        train=features[features.Year<year].dropna(subset=['Total','Gold'])
        test=features[(features.Year==year)&features.Total.notna()].reset_index(drop=True)
        assert (train.Year<year).all() and (test.source_max_year<year).all()
        for kind in MODELS:
            model=fit_model(train,kind); fitted[(year,kind)]=model
            mu,p,alpha,pf,mean=predict_params(model,test)
            t,g=simulate(model,test,4000,rng)
            s=summarize(t,g,test);s['Year']=year;s['model']=kind
            s['total_actual']=test.Total;s['gold_actual']=test.Gold
            s['total_point']=mean;s['gold_point']=mean*p
            s['ever_medal']=test.ever_medal;s['target_participated']=test.target_participated
            predictions.append(s)
            rare=~test.ever_medal.to_numpy(); yr=(test.Total.to_numpy()[rare]>0).astype(float)
            prob=pf[rare] if kind in ['nb_first_hurdle','anchored_hurdle'] else (1-(np.exp(-mu[rare]) if alpha<=0 else (1+alpha*mu[rare])**(-1/alpha)))
            metrics.append({'year':year,'model':kind,'n':len(test),'train_n':len(train),
                'total_mae':float(np.mean(abs(test.Total-mean))),
                'gold_mae':float(np.mean(abs(test.Gold-mean*p))),
                'total_rmse':float(np.sqrt(np.mean((test.Total-mean)**2))),
                'gold_rmse':float(np.sqrt(np.mean((test.Gold-mean*p)**2))),
                'total_coverage95':float(np.mean((test.Total>=s.total_lo95)&(test.Total<=s.total_hi95))),
                'gold_coverage95':float(np.mean((test.Gold>=s.gold_lo95)&(test.Gold<=s.gold_hi95))),
                'total_width95':float((s.total_hi95-s.total_lo95).mean()),
                'first_medal_brier':float(np.mean((prob-yr)**2)),
                'first_candidates':int(rare.sum()),'actual_first_medal_count':int(yr.sum()),
                'predicted_first_medal_count':float(prob.sum()),'nonparticipating_from_prior_roster':int((~test.target_participated).sum())})
        print(f'Backtest {year}: {len(test)} delegations')
    metrics=pd.DataFrame(metrics);allpred=pd.concat(predictions,ignore_index=True)
    metrics.to_csv(VALIDATION/'backtest_metrics.csv',index=False)
    allpred.to_csv(VALIDATION/'backtest_predictions.csv',index=False)
    val=metrics[metrics.year.isin(CFG['validation_years'])].groupby('model')[['total_mae','gold_mae']].mean()
    val['selection_score']=val.total_mae+val.gold_mae
    # Keep best point benchmark distinct from the model covering first medals + host + intervals.
    selected_point=str(val.selection_score.idxmin())
    eligible=metrics[(metrics.year.isin(CFG['validation_years']))&metrics.model.isin(['nb_first_hurdle','anchored_hurdle'])].groupby('model').agg(total_mae=('total_mae','mean'),gold_mae=('gold_mae','mean'),first_medal_brier=('first_medal_brier','mean'))
    eligible['joint_score']=eligible.total_mae+eligible.gold_mae+eligible.first_medal_brier
    selected=str(eligible.joint_score.idxmin())
    val.sort_values('selection_score').to_csv(VALIDATION/'model_selection.csv')
    selected_note={'selection_years':CFG['validation_years'],'holdout_year':2024,'selected_model':selected,'best_point_benchmark':selected_point,
                   'selection_rule':'joint total+gold MAE and first-medal Brier among host-aware first-medal hurdle models; best unconstrained point benchmark retained',
                   'development_disclosure':'Anchored hurdle added after initial benchmark comparison; 2024 benchmark results had already been inspected, so 2024 is retrospective rather than a pristine untouched holdout.',
                   'interval_backtests':'conditional process uncertainty; no bootstrap in backtests',
                   'future_intervals':'edition-cluster parameter bootstrap + process uncertainty',
                   'roster_rule':'preceding edition roster, absent target delegations scored as zero; new entrants are not forecast'}
    (VALIDATION/'model_selection.json').write_text(json.dumps(selected_note,indent=2),encoding='utf-8')
    print('MODEL_SELECTION',val.to_string(), '\nSELECTED',selected)
    return selected,metrics,allpred

def final_forecast(panel,hosts,features,selected):
    TABLES.mkdir(parents=True,exist_ok=True)
    training=features.dropna(subset=['Total','Gold'])
    roster=sorted(panel[(panel.Year==2024)&panel.Total.notna()&panel.entries.notna()].NOC.unique())
    f=build_features(panel,2028,hosts,roster).reset_index(drop=True)
    base=fit_model(training,selected)
    # Freeze coefficients after final fit; predictive intervals use edition-cluster refits.
    rng=np.random.default_rng(CFG['seed']+28)
    boot=[];errors=[];years=sorted(training.Year.unique())
    for b in range(80):
        chosen=rng.choice(years,size=len(years),replace=True)
        sample=pd.concat([training[training.Year==y] for y in chosen],ignore_index=True)
        try: boot.append(fit_model(sample,selected))
        except RuntimeError as e: errors.append(str(e))
    if len(boot)<50: raise RuntimeError('Too few converged edition bootstrap fits')
    (VALIDATION/'bootstrap_models.json').write_text(json.dumps(boot,indent=2),encoding='utf-8')
    per=CFG['simulation_draws']//len(boot)
    t_all=[];g_all=[]
    for mod in boot:
        t,g=simulate(mod,f,per,rng);t_all.append(t);g_all.append(g)
    t=np.vstack(t_all);g=np.vstack(g_all)
    summarize(t,g,f).to_csv(TABLES/'predictions_2028_unconstrained.csv',index=False)
    budget_panel=panel[(panel.Year==2024)&panel.Total.notna()]
    total_budget=int(budget_panel.Total.sum());gold_budget=int(budget_panel.Gold.sum())
    gold_alloc=allocate_budget(g,gold_budget,rng)
    nongold_alloc=allocate_budget(t-g,total_budget-gold_budget,rng)
    g=gold_alloc;t=gold_alloc+nongold_alloc
    out=summarize(t,g,f)
    mu,p,alpha,pf,mean=predict_params(base,f)
    out['total_point']=out.total_mean;out['gold_point']=out.gold_mean
    out['unconstrained_fitted_total']=mean;out['unconstrained_fitted_gold']=mean*p
    prev=panel[panel.Year==2024].set_index('NOC')
    out['total_2024']=f.NOC.map(prev.Total).to_numpy()
    out['gold_2024']=f.NOC.map(prev.Gold).to_numpy()
    out['p_total_improve']=(t>out.total_2024.to_numpy()).mean(axis=0)
    out['p_total_decline']=(t<out.total_2024.to_numpy()).mean(axis=0)
    out['p_gold_improve']=(g>out.gold_2024.to_numpy()).mean(axis=0)
    out['p_gold_decline']=(g<out.gold_2024.to_numpy()).mean(axis=0)
    out['total_change']=out.total_point-out.total_2024
    out['gold_change']=out.gold_point-out.gold_2024
    out['model']=selected;out['scenario']='2024 roster/programme/official budgets retained; USA host; joint budget allocation'
    out.sort_values(['gold_point','total_point'],ascending=False).to_csv(TABLES/'predictions_2028.csv',index=False)
    out.sort_values('total_change',ascending=False).to_csv(TABLES/'change_vs_2024.csv',index=False)
    np.savez_compressed(VALIDATION/'forecast_draws.npz',total=t,gold=g,codes=f.NOC.to_numpy(dtype='U8'))
    rare=~f.ever_medal.to_numpy()
    first=out[rare].copy();first['first_medal_probability']=(t[:,rare]>0).mean(axis=0)
    first[['NOC','name','first_medal_probability','total_point','total_lo95','total_hi95']].sort_values('first_medal_probability',ascending=False).to_csv(TABLES/'first_medal_probabilities.csv',index=False)
    counts=(t[:,rare]>0).sum(axis=1)
    unique,num=np.unique(counts,return_counts=True)
    pd.DataFrame({'first_medal_count':unique,'probability':num/len(counts)}).to_csv(TABLES/'first_medal_count_distribution.csv',index=False)
    boot_info={'requested':80,'converged':len(boot),'failures':errors,'draws':len(t),
               'target_delegations':len(f),'first_candidates':int(rare.sum()),
               'expected_first_count':float(counts.mean()),'first_count_lo95':float(np.quantile(counts,.025)),
               'first_count_hi95':float(np.quantile(counts,.975)),
               'mode_first_count':int(unique[np.argmax(num)]),'mode_probability':float(np.max(num/len(counts))),
               'expected_total_medals':float(out.total_point.sum()),'expected_gold_medals':float(out.gold_point.sum()),
               'total_budget':total_budget,'gold_budget':gold_budget,
               'aggregate_constraint':'each draw preserves 2024 provided gold and total budgets; unreported AIN aggregate excluded',
               'budget_uncertainty':'conditional fixed-budget scenario; alternative opportunity scenarios separate'}
    (VALIDATION/'forecast_summary.json').write_text(json.dumps(boot_info,indent=2),encoding='utf-8')
    (VALIDATION/'fitted_model.json').write_text(json.dumps(base,indent=2),encoding='utf-8')
    f.to_csv(PROCESSED/'forecast_features_2028.csv',index=False)
    print('FORECAST_SUMMARY',json.dumps(boot_info,indent=2))
    print(out.sort_values('gold_point',ascending=False).head(10)[['NOC','name','gold_point','total_point','total_lo95','total_hi95']].to_string(index=False))
    return out,base,f

def ablations(features, selected):
    if selected in ['last','weighted']: return
    records=[]
    for feature in ['host','athletes','sports']:
        for year in CFG['validation_years']:
            train=features[features.Year<year].dropna(subset=['Total','Gold'])
            test=features[(features.Year==year)&features.Total.notna()]
            mod=fit_model(train,selected,feature)
            mu,p,alpha,pf,mean=predict_params(mod,test)
            records.append({'removed':feature,'year':year,'total_mae':float(abs(test.Total-mean).mean()),'gold_mae':float(abs(test.Gold-mean*p).mean())})
    pd.DataFrame(records).to_csv(VALIDATION/'ablation_metrics.csv',index=False)

def joint_backtest(panel,features,selected):
    """Retrospective joint scenario retains preceding edition's supplied medal budget."""
    rng=np.random.default_rng(CFG['seed']+112); rows=[]
    for year in CFG['validation_years']+[2024]:
        tr=features[features.Year<year].dropna(subset=['Total','Gold'])
        te=features[(features.Year==year)&features.Total.notna()].reset_index(drop=True)
        mod=fit_model(tr,selected);t,g=simulate(mod,te,4000,rng)
        prev=panel[(panel.Year==year-4)&panel.Total.notna()]
        gb=int(prev.Gold.sum());tb=int(prev.Total.sum())
        ga=allocate_budget(g,gb,rng);ng=allocate_budget(t-g,tb-gb,rng);t=ga+ng;g=ga
        s=summarize(t,g,te);s['Year']=year;s['total_actual']=te.Total;s['gold_actual']=te.Gold
        rows.append({'year':year,'n':len(te),'prior_budget_total':tb,'prior_budget_gold':gb,
                     'total_mae':float(abs(te.Total-s.total_mean).mean()),'gold_mae':float(abs(te.Gold-s.gold_mean).mean()),
                     'total_coverage95':float(((te.Total>=s.total_lo95)&(te.Total<=s.total_hi95)).mean()),
                     'gold_coverage95':float(((te.Gold>=s.gold_lo95)&(te.Gold<=s.gold_hi95)).mean()),
                     'scenario':'preceding edition medal budgets retained; process-only uncertainty'})
        s.to_csv(VALIDATION/f'joint_predictions_{year}.csv',index=False)
    pd.DataFrame(rows).to_csv(VALIDATION/'joint_backtest_metrics.csv',index=False)

def run(panel,hosts):
    TABLES.mkdir(parents=True,exist_ok=True)
    features=build_training(panel,hosts)
    selected,metrics,pred=backtest(features)
    out,base,f=final_forecast(panel,hosts,features,selected)
    ablations(features,selected)
    joint_backtest(panel,features,selected)
    return out,base,f,features,metrics,pred

if __name__=='__main__':
    panel=pd.read_csv(PROCESSED/'country_year_panel.csv')
    meta=json.loads((PROCESSED/'edition_metadata.json').read_text())
    run(panel,{int(k):v for k,v in meta['hosts'].items()})
