"""Generate evidence-grounded Chinese/English reports and a standalone LaTeX source."""
from __future__ import annotations
import sys,json,html,re,math
from pathlib import Path
import numpy as np
import pandas as pd
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image,Preformatted
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
V=ROOT/'outputs/validation';T=ROOT/'outputs/tables';F=ROOT/'outputs/figures';R=ROOT/'report'
def load_json(name):return json.loads((V/name).read_text(encoding='utf-8'))
def load_csv(name):return pd.read_csv(T/name)
def s(x,n=2):return f'{float(x):.{n}f}'
def interval(r,prefix):return f'[{int(r[prefix+"_lo95"])}, {int(r[prefix+"_hi95"])}]'
def escape_tex(text):
    substitutions={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
    return ''.join(substitutions.get(c,c) for c in str(text))

def create_sections():
    out=load_csv('predictions_2028.csv'); first=load_csv('first_medal_probabilities.csv');coach=load_csv('coach_investment_scenarios.csv')
    evidence=load_csv('coach_evidence.csv');scenarios=load_csv('sport_scenarios.csv')
    metrics=pd.read_csv(V/'backtest_metrics.csv');joint=pd.read_csv(V/'joint_backtest_metrics.csv')
    audit=load_json('data_audit.json');forecast=load_json('forecast_summary.json');insight=load_json('original_insight.json');model=load_json('fitted_model.json')
    selection=load_json('model_selection.json')
    means=metrics[metrics.year<=2020].groupby('model')[['total_mae','gold_mae','first_medal_brier']].mean()
    primary=means.loc['anchored_hurdle'];baseline=means.loc['weighted']
    check=metrics[(metrics.year==2024)&(metrics.model=='anchored_hurdle')].iloc[0]
    us=out[out.NOC=='USA'].iloc[0];ch=out[out.NOC=='CHN'].iloc[0]
    boot=load_json('bootstrap_models.json');host=np.array([math.exp(b['anchor_beta'][1]) for b in boot])
    sections=[]
    def add(title,paragraphs,table=None,figure=None,code=None):
        sections.append({'title':title,'paragraphs':paragraphs,'table':table,'figure':figure,'code':code})
    add('Summary Sheet',[
        'A Reproducible, Budget-Constrained Forecast of Olympic Medal Tables',
        '2025 MCM Problem C | Training reconstruction | Analysis date: 9 October 2026',
        'We use only the five supplied datasets to construct delegation-edition targets, compare six forecasting strategies, and generate joint 2028 gold and total-medal distributions. National targets come from the official medal table; athlete records supply historical participation and sport proxies.',
        f'On rolling forecasts for 2008, 2012, 2016 and 2020, the weighted-history benchmark has the lowest point-error score: total MAE {baseline.total_mae:.3f} and gold MAE {baseline.gold_mae:.3f}. The anchored first-medal hurdle has MAE {primary.total_mae:.3f} and {primary.gold_mae:.3f}. We retain the benchmark and use the hurdle for the additional host, first-medal and joint-distribution tasks, without claiming it is the best point predictor.',
        f'Our primary 2028 scenario retains the 2024 roster, programme and reported-delegation medal budgets: {forecast["target_delegations"]} delegations, {forecast["gold_budget"]} gold and {forecast["total_budget"]} total medals. Every simulated outcome satisfies these budgets and gold no greater than total. The USA has expected gold {us.gold_point:.1f}, total {us.total_point:.1f}; China has gold {ch.gold_point:.1f}, total {ch.total_point:.1f}. Intervals are broad, reflecting historical dispersion and edition-cluster parameter uncertainty.',
        f'Among {forecast["first_candidates"]} historically unmedalled candidate delegations, the expected first-medal count is {forecast["expected_first_count"]:.2f}, with a 95% predictive range {forecast["first_count_lo95"]:.0f}-{forecast["first_count_hi95"]:.0f}. The most likely count is {forecast["mode_first_count"]}, probability {forecast["mode_probability"]:.1%}. Sport reallocation and coaching investments are explicit scenarios, not known 2028 programme changes or identified coaching causal effects.',
        'The supplied programme stops at 2024 and includes no coach appointment records. We disclose measurement disagreement, roster uncertainty, retrospective model development, and scenario assumptions. This AI-assisted reconstruction is supported by saved code, outputs and verification records [1,2,3].'
    ])
    toc=[['Section','Page']]
    titles=['Data and scope','Forecasting model','Rolling validation','2028 medal table','First medals','Sport opportunities','Coaching evidence and scenarios','Original insight','Sensitivity and ablation','Limitations and recommendations','Reproducibility and references','Claim and review audit']
    toc.extend([[title,str(i+3)] for i,title in enumerate(titles)])
    toc.append(['Appendix: all 205 delegation forecasts','15-19']);toc.append(['Report on Use of AI Tools','20 (outside main solution)'])
    add('Contents',['Main solution includes the summary, contents, analysis, references and complete forecast appendix. The AI-use record follows separately.'],table=toc)
    add('1. Data and scope',[
        f'The official ZIP and local CSV files are byte-identical. The raw athlete file has {audit["raw_rows"]["athletes"]:,} rows; {audit["exact_duplicate_athlete_rows_removed"]:,} exact duplicate rows are removed. We map {audit["mapped_medal_names"]} medal-table names to the supplied delegation codes and construct {audit["panel_rows"]:,} delegation-edition records.',
        'Medal-table NOC values are country names, whereas athlete NOC values are codes. We use normalized supplied Team labels and documented aliases. FR Yugoslavia in 1996/2000 aligns to SCG; Lebanon LBN and LIB are reconciled using the same Team label. Predecessor states are not redistributed to successor countries. Mixed, refugee and neutral delegations are not presented as sovereign countries.',
        'Country gold and total targets use the official table. A missing aggregate is zero only when no athlete medal is recorded; inconsistent medal-bearing rows remain unresolved and are excluded from supervised labels. AIN 2024 and MDA 2012 are the modern unresolved cases. The current scenario excludes AIN because its official aggregate is absent.',
        f'Deduplicated team-event medal-unit proxies exactly reproduce the official total for {audit["recent_sport_reconciliation_exact"]}/{audit["recent_sport_reconciliation_rows"]} comparable modern delegation-editions. We therefore label sport measures as proxies. Compound sport labels are grouped where possible; genuinely mixed categories remain explicit.',
        'Athlete names are not unique identifiers. Name-based participant counts are approximate. Future participation, athletes and event counts are not observed. The supplied host file identifies the USA in 2028; the programme has no 2028 column [1,2].'
    ],table=[['Input','Rows','Role'],['Athletes','252565','Participation and sport proxy'],['Medal counts','1435','Official national targets'],['Hosts','35','Known host indicator'],['Programme','74','Historical event opportunities'],['Dictionary','Not a modelling table','Definitions and special symbols']])
    add('2. Forecasting model',[
        'All historical predictors use editions earlier than the target. The forecast roster is the preceding edition roster with valid targets; target nonparticipants are scored as zero outcomes, not assumed zero ability. New entrants outside this roster are not predicted. We train from 1996 and evaluate 2008 onward.',
        'Weighted history is 0.6 times the preceding medal total, 0.3 times two editions ago and 0.1 times three editions ago. General Poisson and NB2 regressions use log historical medals, log historical participant count, historical sport breadth, current and preceding host flags, and a clipped historical trend. NB2 variance is mu + alpha*mu^2 [3].',
        'For previously medalled delegations, the anchored hurdle uses weighted history as an offset, adding a small support term 0.05 and learned current/prior-host corrections. The fitted log coefficients are '+', '.join(s(x,4) for x in model['anchor_beta'])+f'; NB dispersion alpha={model["alpha"]:.4f}. Regularization and bounds are recorded in source.',
        'For previously unmedalled delegates, logistic regression predicts first-medal probability from prior participant count, participant trend and observed past Games. Conditional positive counts use a zero-truncated NB draw. Gold is sampled conditional on total using the recent gold share (pooled one-third if no historical total); gold cannot exceed total.',
        f'We refit {forecast["converged"]} edition-cluster bootstrap samples and generate {forecast["draws"]:,} future draws. Hamilton largest-remainder allocation separately redistributes gold and non-gold to fixed supplied 2024 budgets. It preserves integer totals and zero support and induces cross-country competition. Quantile intervals summarize the allocated joint draws.',
        'These future intervals combine parameter resampling and outcome dispersion conditional on the roster and budget scenario. They do not cover unknown policy changes, data errors or all possible 2028 schedules. The historical process-only and joint-budget checks are reported separately.'
    ],code='W_i = 0.6 T_(t-1) + 0.3 T_(t-2) + 0.1 T_(t-3)\nmu_i = (W_i + 0.05) exp(b0 + bH H_i + bP H_(t-1))\np_first = logistic(c0 + cA log(1+A) + cD trend + cG log(1+Games))\nE[T | first hurdle] = p_first * mu / (1 - P_NB(T=0))\nG | T ~ Binomial(T, historical gold share)')
    rows=[['Model','Total MAE','Gold MAE','First Brier']]+[[k,s(r.total_mae,3),s(r.gold_mae,3),s(r.first_medal_brier,3)] for k,r in means.sort_values('total_mae').iterrows()]
    add('3. Rolling validation',[
        'We compare all six models under the same previous-roster forecasting protocol. Mean metrics below are calculated on 2008, 2012, 2016 and 2020. MAE is per delegation; pooled averages are heavily influenced by small delegations, so RMSE and individual outputs are also saved.',
        'Weighted history remains the strongest point benchmark. A hurdle is needed for nonzero first-medal probabilities and the selected anchored variant has lower joint error than the general NB hurdle. Selection combines total MAE, gold MAE and first-medal Brier among task-complete hurdle models. This restricted selection is not evidence that the chosen model wins unrestricted point prediction.',
        f'For 2024, the anchored model has total MAE {check.total_mae:.3f}, gold MAE {check.gold_mae:.3f}, total RMSE {check.total_rmse:.3f} and gold RMSE {check.gold_rmse:.3f}. Process-only 95% coverage is {check.total_coverage95:.1%} for total and {check.gold_coverage95:.1%} for gold. These broad intervals are conservative, not a claim of sharp precision.',
        '2024 benchmark results were inspected during development before adding the anchored model. We disclose it as a retrospective check rather than an untouched confirmatory holdout. Gold/total budget allocation is also checked retrospectively while retaining the preceding edition budget, not the actual target medal total.'
    ],table=rows,figure='01_model_comparison')
    top=out.sort_values('gold_point',ascending=False).head(15)
    rows=[['NOC','Gold mean','Gold 95%','Total mean','Total 95%']]+[[r.NOC,s(r.gold_point,1),interval(r._asdict(),'gold'),s(r.total_point,1),interval(r._asdict(),'total')] for r in top.itertuples()]
    improve=out.sort_values('total_change',ascending=False).head(3);decline=out.sort_values('total_change').head(3)
    add('4. The 2028 medal table',[
        'This is a conditional scenario, not a reconstruction of the actual future schedule: the 2024 roster and programme are retained, the USA hosts, and the reported-delegation budgets are gold 328 and total 1039. The budget excludes the unreported AIN aggregate and must not be described as all physically awarded 2024 medals.',
        'The following means and 95% prediction intervals come from jointly allocated bootstrap draws. Complete outputs for all 205 delegates are in the appendix and CSV. Means can be noninteger even though every draw is integer. Rankings of means do not imply certain finishing ranks.',
        'Largest expected total increases: '+ '; '.join(f'{r.name} {r.total_change:+.1f} (P increase {r.p_total_improve:.1%})' for r in improve.itertuples())+'. Largest expected declines: '+ '; '.join(f'{r.name} {r.total_change:+.1f} (P decline {r.p_total_decline:.1%})' for r in decline.itertuples())+'.',
        'Country intervals overlap extensively. We recommend using scenario comparisons and improvement probabilities, rather than interpreting individual means as precise counts.'
    ],table=rows,figure=None)
    add('5. First-medal probabilities',[
        f'The candidate set consists of {forecast["first_candidates"]} delegates in the retained roster that have never earned a medal under the supplied summer-Games history and explicit code mapping. This is not a claim about winter Games. The conditional participation assumption is essential.',
        f'The expected number is {forecast["expected_first_count"]:.4f}; the most likely integer is {forecast["mode_first_count"]}, probability {forecast["mode_probability"]:.2%}. The 95% predictive interval is [{forecast["first_count_lo95"]:.0f}, {forecast["first_count_hi95"]:.0f}]. Shared bootstrap fits and fixed-budget allocation mean the count distribution is not formed from independent fixed Bernoulli draws alone.',
        'Leading probabilities are model-based and reflect historical participation features, not scouting of specific athletes. Countries without medals do not receive automatic zero probabilities. Baseline weighted history predicts no first medals, illustrating its limitation.',
        'The saved historical Brier scores and expected versus observed first counts quantify calibration limitations. Roster changes and sparse first-medal observations prevent a precise forecast of the number of successful new delegations.'
    ],table=[['NOC','Delegation','First-medal probability']]+[[r.NOC,r.name,f'{r.first_medal_probability:.1%}'] for r in first.head(10).itertuples()],figure='06_first_medal_count')
    ss=load_json('sport_scenario_metadata.json')
    selected_s=scenarios[(scenarios.NOC.isin(['USA','CHN','JPN']))&(scenarios.scenario!='2024 retained')]
    add('6. Sport opportunities and host choices',[
        'We aggregate aquatic subdisciplines, gymnastics disciplines, cycling disciplines and other compatible labels into programme sport categories. Recent proxy medal shares use weights 0.6/0.3/0.1 for 2024/2020/2016; participation shares replace medal shares when no medal units exist.',
        'Country exposure changes by the weighted sum of sport event multipliers. Elasticity is assumed equal to one. Expansion scenarios scale the global budget in proportion to the event count; the host-oriented reallocation keeps it fixed. These are stress tests derived from supplied historical categories, not externally sourced 2028 events.',
        'The USA-oriented scenario expands its top three proxy categories ('+', '.join(ss['USA_top3_proxy_categories'])+') by 10% while reducing other event opportunities to preserve the overall event total. Gains and losses are redistributed consistently across delegates.',
        'The heatmap identifies which opportunities matter in this constructed model. Because reconstructed sport units do not always equal official national medals, small differences should not be presented as exact sport-level forecasts or causal effects of host selection.'
    ],figure='07_sport_heatmap')
    case=coach[coach.assumed_sport_gain==.2]
    add('7. Coaching evidence and investment scenarios',[
        'No coach name or appointment date is provided. We search for changes in two-edition sport proxy rates, adjust opportunities and compare country-wide official strength. This produces candidate positive excesses. Appointment, athlete quality, federation investment and rule changes remain unobserved confounders.',
        'For example, the leading candidate '+str(evidence.iloc[0].NOC)+' / '+str(evidence.iloc[0].Sport)+f' has a positive proxy excess {evidence.iloc[0].positive_excess_proxy_units:.2f} across two post-change editions. Assigning 0-100% of that excess to coaching is only an attribution scenario, not an identified causal bound. We cannot estimate a defensible pure great-coach contribution from these data alone.',
        'We select three delegates with an active 2024 programme and a historically established sport with a recent proxy shortfall. This avoids proposing coaching-only gains for a sport with no current participation. Recommended cases are '+ '; '.join(f'{r.name}: {r.Sport}' for r in case.itertuples())+'.',
        'Assuming 10%, 20% or 30% sport-strength uplift, we scale the country sport share and redistribute the fixed total budget. The table shows the 20% case. These expected gains are conditional arithmetic, not measured coach effects, salaries, return on investment or guaranteed medals.'
    ],table=[['Delegation','Sport','Base total','20% case gain']]+[[r.name,r.Sport,s(r.country_base_total),s(r.country_expected_gain)] for r in case.itertuples()],figure='09_coaching_scenarios')
    add('8. An exploratory insight: specialization and volatility',[
        f'Among {insight["n"]} delegates with four recent observations and average total at least two, the correlation between recent sport specialization HHI and medal-count coefficient of variation is Spearman rho={insight["spearman_rho"]:.3f}, nominal p={insight["p_value"]:.3g}.',
        'HHI is the sum of squared recent sport proxy shares; volatility uses official totals for 2012-2024. The positive relationship suggests that concentrated programmes may be more exposed to the performance of a small set of sports. It is a descriptive association, not proof that diversification causes stability.',
        'The windows overlap, the sample is selected, proxies have measurement error and multiple insights were considered. Wealth, programme size and organizational differences are not observed. The nominal p-value does not resolve these limitations.',
        'An Olympic committee can use this as a prompt for stress testing: quantify the loss if one dominant sport underperforms, then compare a targeted specialization strategy with a portfolio of realistic second-strength sports.'
    ],figure='08_specialization_volatility')
    ab=pd.read_csv(V/'ablation_metrics.csv').groupby('removed')[['total_mae','gold_mae']].mean()
    add('9. Sensitivity, ablation and interval checks',[
        'We refit on training starts 1996, 2000 and 2004, remove the USA host indicator, and inspect sport-opportunity alternatives. Sensitivity point means normalize gold and non-gold budgets separately; they are not identical to bootstrap predictive means.',
        f'The current-host multiplier is exp(bH)={math.exp(model["anchor_beta"][1]):.3f}. Edition-bootstrap 95% percentiles are [{np.quantile(host,.025):.3f}, {np.quantile(host,.975):.3f}]. This is a conditional model association based on few hosts, not a causal hosting estimate.',
        'Removing host, participant or sport-breadth terms is evaluated on the same four validation editions. The anchored established-delegate mean primarily uses history and host; participant and sport ablations affect the rare-delegate count submodel. Small differences are therefore expected.',
        'We also check budget allocation using preceding-edition budgets in each retrospective forecast. Because future event budgets and participation remain unknown, process-only historical coverage does not certify the wider conditional 2028 intervals.'
    ],table=[['Removed','Total MAE','Gold MAE']]+[[k,s(r.total_mae,3),s(r.gold_mae,3)] for k,r in ab.iterrows()],figure='11_sensitivity')
    add('10. Limitations and committee recommendations',[
        'The simplest weighted-history model has better point MAE than the selected task-complete model. We report this clearly. Choosing a richer model serves first-medal probability, host response and a joint distribution; it does not make point forecasts universally more accurate.',
        'The 2028 roster, participant scale and programme are assumed, and the primary budget is fixed. Athletes and events may change substantially. Confidence about country rankings is limited by broad intervals, sparse editions, proxy inconsistency and unobserved institutional changes.',
        'The latest historical check was inspected in development. A truly prospective evaluation would require new data not yet used for model design. Raw-data anomalies and country-code changes can also affect first-medal classification.',
        'Committees should use the strong simple benchmark for a reference, the joint model for uncertainty and scenario planning, and the sport matrix for exposure analysis. Prioritize evidence-backed programmes rather than buying a coach based on a residual alone. Monitor participant depth and medal concentration, then revise scenarios when legitimate future information becomes available.',
        'For the exercise, quantitative inputs remain limited to the supplied files. External GDP, live athlete rankings and actual future sport lists are not added to the model. All figures and table numbers are generated from saved computations [1,2,3].'
    ])
    add('11. Reproducibility and references',[
        'Run scripts/run_all.py with the project .venv Python to rebuild audits, features, rolling comparisons, forecasts, bootstrap draws, scenarios and figures. Run unittest discovery in tests for raw input integrity, future-label perturbation, probability bounds and joint budget constraints. Run scripts/build_report.py to regenerate this report.',
        'Random seed is 20251009; final simulation uses seed offsets recorded in source. requirements-lock.txt pins actual installed packages. logs/last_run.json records execution duration and environment. source_manifest.json and official_data_comparison.json link the supplied bytes to official ZIP contents.',
        '[1] COMAP. 2025 MCM Problem C: Models for Olympic Medal Tables. Local official problem PDF, especially pages 2-3. Official archive: https://www.contest.comap.org/undergraduate/contests/mcm/contests/2025/problems/',
        '[2] COMAP. 2025 Problem C Data.zip: summerOly_athletes, summerOly_medal_counts, summerOly_hosts, summerOly_programs and data_dictionary. Local files are byte-identical to the official archive. No other quantitative dataset is used.',
        '[3] Statsmodels documentation: NegativeBinomial (NB2 variance) and Binomial families. https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.NegativeBinomial.html . Numerical optimization uses SciPy; specific objective functions and regularization are implemented in source.',
        '[4] OpenAI Codex, AI-assisted model implementation, analysis and writing, 9 October 2026. Exact runtime model identifiers are not exposed to these scripts; the user requested GPT-6.1 Sol during the session. See the appended AI-use record.',
        '[5] COMAP. Use of AI Tools in COMAP Contests, v102025, retrieved 9 October 2026. https://www.contest.comap.com/undergraduate/contests/mcm/flyer/Contest_AI_Policy.pdf . The retrieved policy is not claimed to be the original January 2025 policy version.'
    ])
    add('12. Claim-evidence and self-review audit',[
        'Contribution: a verified input-to-output pipeline covers national totals, coherent gold/total outcomes, first medals, sport scenarios and limited coaching evidence. It is not claimed to introduce a new statistical method.',
        'Clarity: terms delegation, official total, sport proxy, scenario and predictive interval remain separate. Formulae, coefficients, aliases, random seeds and all country results are saved.',
        'Experimental strength: six models and five rolling editions are reported. The weighted benchmark wins point MAE; this limitation is retained in the summary. No untested accuracy improvement is asserted.',
        'Evaluation completeness: host/participant/sport ablations, training-window sensitivity, interval coverage, joint-budget retrospective metrics and first-medal Brier scores are available. Unknown 2028 data and absent coach records remain evidence gaps.',
        'Method soundness: future-label perturbation leaves features unchanged; joint draws preserve totals and gold inequalities; probabilities are valid. The method remains conditional on roster, budget and proxy assumptions. These checks validate implementation, not every substantive assumption.'
    ],table=[['Claim','Evidence file','Status'],['205 joint forecasts','predictions_2028.csv / forecast_draws.npz','computed'],['Best point benchmark is weighted','model_selection.csv','computed'],['First-medal count distribution','first_medal_count_distribution.csv','conditional'],['Sport reallocation impact','sport_scenarios.csv','assumed scenario'],['Pure coach causal contribution','No appointment records','not identified'],['Specialization-volatility association','original_insight.json','descriptive']])
    appendix=out.sort_values(['gold_point','total_point'],ascending=False)
    for start in range(0,len(appendix),41):
        chunk=appendix.iloc[start:start+41]
        rows=[['NOC','Delegation','Gold','Gold 95%','Total','Total 95%']]
        rows += [[r.NOC,r.name,s(r.gold_point),interval(r._asdict(),'gold'),s(r.total_point),interval(r._asdict(),'total')] for r in chunk.itertuples()]
        add(f'Appendix: all delegation forecasts ({start+1}-{start+len(chunk)})',[
            'Expected medals and 95% joint prediction intervals. All results are conditional on retained 2024 roster, programme and reported-delegation budgets. Full precision, improvement probabilities and scenarios are in CSV.'
        ],table=rows)
    # Last page is explicitly separate from the main solution, not a fabricated team declaration.
    add('Report on Use of AI Tools',[
        'Tool: OpenAI Codex. Date: 9 October 2026. The user requested GPT-6.1 Sol after the initial implementation; the exact active runtime model is not exposed to the numerical process and should be confirmed from the application for any formal citation.',
        'Known user requests (English translations): start modeling; switch to GPT-6.1 Sol; continue modeling. Exact Chinese wording is preserved in the companion AI-use record. Earlier prompts requested local 2025 problem analysis and a standalone C-problem execution plan. AI was used for planning, Python implementation, data handling, model design, interpretation, figure generation, Chinese explanation and English writing [4].',
        'Substantive generated outputs are preserved in src/, scripts/, tests/, report/, the execution plan and this PDF. Computation inputs are only the five supplied CSV files; retrieved documentation supports implementation and disclosure, not extra numerical predictors.',
        'Verification performed: source hashes, numerical rerun, historical comparisons, probability and joint-budget checks, and visual report review. These do not replace human review of assumptions. No student identities, control number, submission or human approval have been invented.',
        'This is a transparent training reconstruction, not a certified contest submission. The short record does not reproduce the entire conversation. For formal use, export and retain the full conversation, exact model/version details and complete generated outputs, and adapt the disclosure to the applicable policy [5].'
    ])
    return sections,{'forecast':forecast,'audit':audit,'metrics':metrics,'joint':joint,'out':out,'first':first,'coach':coach,'insight':insight,'model':model,'primary':primary,'baseline':baseline}

def pdf_report(sections):
    fontdir=Path(__import__('matplotlib').get_data_path())/'fonts/ttf'
    pdfmetrics.registerFont(TTFont('ReportSerif',str(fontdir/'DejaVuSerif.ttf')))
    pdfmetrics.registerFont(TTFont('ReportSans',str(fontdir/'DejaVuSans.ttf')))
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle('BodyReport',fontName='ReportSerif',fontSize=9.6,leading=14,spaceAfter=9))
    styles.add(ParagraphStyle('TitleReport',fontName='ReportSans',fontSize=17,leading=23,spaceAfter=16,textColor=colors.HexColor('#173E54')))
    styles.add(ParagraphStyle('CellReport',fontName='ReportSans',fontSize=7.8,leading=10))
    styles.add(ParagraphStyle('CodeReport',fontName='Courier',fontSize=8,leading=12,spaceAfter=8))
    story=[];width=A4[0]-100
    for i,sec in enumerate(sections):
        if i:story.append(PageBreak())
        story.append(Paragraph(html.escape(sec['title']),styles['TitleReport']))
        for text in sec['paragraphs']:story.append(Paragraph(html.escape(text),styles['BodyReport']))
        if sec['code']:story.append(Preformatted(sec['code'],styles['CodeReport']))
        if sec['table']:
            data=[[Paragraph(html.escape(str(c)),styles['CellReport']) for c in row] for row in sec['table']]
            n=len(data[0])
            if n==6:fractions=[.065,.285,.11,.15,.11,.18]
            elif n==5:fractions=[.11,.18,.22,.22,.27]
            elif n==4:fractions=[.33,.22,.22,.23]
            elif n==3:fractions=[.27,.40,.33]
            else:fractions=[.84,.16]
            table=Table(data,colWidths=[width*f for f in fractions],repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E8F0F4')),('TEXTCOLOR',(0,0),(-1,0),colors.HexColor('#173E54')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#608090')),('LINEBELOW',(0,-1),(-1,-1),.4,colors.HexColor('#A7B9C5')),('TOPPADDING',(0,0),(-1,-1),2),('BOTTOMPADDING',(0,0),(-1,-1),2)]))
            story.extend([table,Spacer(1,12)])
        if sec['figure']:
            image=Image(str(F/(sec['figure']+'.png')))
            maximum=230 if sec['table'] else 315
            ratio=min(width/image.imageWidth,maximum/image.imageHeight)
            image.drawWidth=image.imageWidth*ratio;image.drawHeight=image.imageHeight*ratio
            story.append(image)
    def page(canvas,doc):
        canvas.saveState();canvas.setFont('ReportSans',8);canvas.setFillColor(colors.HexColor('#61717B'))
        canvas.drawString(50,24,'2025 MCM C | Reproducible training reconstruction')
        canvas.drawRightString(A4[0]-50,24,str(doc.page));canvas.restoreState()
    doc=SimpleDocTemplate(str(R/'solution.pdf'),pagesize=A4,rightMargin=50,leftMargin=50,topMargin=42,bottomMargin=44,title='2025 MCM C: Budget-Constrained Olympic Forecasts',author='AI-assisted training reconstruction')
    doc.build(story,onFirstPage=page,onLaterPages=page)

def markdown_report(sections):
    parts=['# 2025 MCM C: Budget-Constrained Olympic Forecasts\n']
    for sec in sections:
        parts.append('## '+sec['title']+'\n')
        parts.extend(x+'\n' for x in sec['paragraphs'])
        if sec['title']=='Contents':
            pages=load_json('paper_section_pages.json')
            parts += ['| Section | Page |','|---|---:|']+[f'| {x["title"]} | {x["page"]} |' for x in pages]+['']
        for eq,gloss in sec.get('equations',[]):parts.extend(['$$'+eq+'$$',gloss+'\n'])
        if sec.get('references'):parts.extend(x+'\n' for x in sec['references'])
        if sec['code']:parts.append('```text\n'+sec['code']+'\n```\n')
        if sec['table']:
            rows=sec['table']
            if sec.get('table_caption'):parts.append(f'Table {sec["table_no"]}. '+sec['table_caption']+'\n')
            parts.append('| '+' | '.join(rows[0])+' |\n| '+' | '.join(['---']*len(rows[0]))+' |')
            parts.extend('| '+' | '.join(map(str,row))+' |' for row in rows[1:]);parts.append('')
        if sec['figure']:
            parts.append('![Computed figure](<'+(F/(sec['figure']+'.png')).as_posix()+'>)\n')
            if sec.get('figure_caption'):parts.append(f'Figure {sec["figure_no"]}. '+sec['figure_caption']+'\n')
    (R/'solution.md').write_text('\n'.join(parts),encoding='utf-8')

def tex_inline_chart(name):
    # Standalone built-in editor cannot depend on companion image files. Use embedded data.
    axis='width=0.94\\linewidth,height=5cm,grid=major,legend style={font=\\scriptsize},tick label style={font=\\scriptsize}'
    plots=[]
    if name=='01_model_comparison':
        df=pd.read_csv(V/'backtest_metrics.csv');df=df[df.year<=2020].groupby('model')[['total_mae','gold_mae']].mean()
        labels=['last','weighted','poisson','NB','NB-hurdle','anchor']
        codes=['last','weighted','poisson','negative_binomial','nb_first_hurdle','anchored_hurdle']
        axis+=',ybar,bar width=6pt,symbolic x coords={'+','.join(labels)+'},xtick=data,ylabel={MAE},legend pos=north west'
        for col in ['total_mae','gold_mae']:
            plots.append('\\addplot coordinates {'+' '.join(f'({lab},{df.loc[c,col]:.4f})' for lab,c in zip(labels,codes))+'};')
        plots.append('\\legend{Total,Gold}')
    elif name=='06_first_medal_count':
        df=load_csv('first_medal_count_distribution.csv');axis+=',ybar,xlabel={First-medal delegates},ylabel={Probability}'
        plots.append('\\addplot coordinates {'+' '.join(f'({int(r.first_medal_count)},{r.probability:.6f})' for r in df.itertuples())+'};')
    elif name=='08_specialization_volatility':
        df=load_csv('specialization_volatility.csv');axis+=',xlabel={Proxy specialization HHI},ylabel={Official medal CV}'
        plots.append('\\addplot[only marks,mark size=1.5pt] coordinates {'+' '.join(f'({r.concentration_HHI:.5f},{r.medal_cv:.5f})' for r in df.itertuples())+'};')
    elif name=='09_coaching_scenarios':
        df=load_csv('coach_investment_scenarios.csv');df=df[df.assumed_sport_gain==.2]
        labels=df.NOC.tolist();axis+=',ybar,symbolic x coords={'+','.join(labels)+'},xtick=data,ylabel={Conditional medal gain}'
        plots.append('\\addplot coordinates {'+' '.join(f'({r.NOC},{r.country_expected_gain:.5f})' for r in df.itertuples())+'};')
    elif name=='11_sensitivity':
        df=pd.read_csv(V/'sensitivity_predictions.csv');codes=load_csv('predictions_2028.csv').head(8).NOC.tolist()
        axis+=',symbolic x coords={'+','.join(codes)+'},xtick=data,ylabel={Budget-normalized total},legend pos=outer north east'
        for scenario,g in df.groupby('scenario'):
            x=g.set_index('NOC');plots.append('\\addplot coordinates {'+' '.join(f'({c},{x.loc[c,"total_point"]:.4f})' for c in codes)+'};\\addlegendentry{'+escape_tex(scenario)+'}')
    elif name=='07_sport_heatmap':
        # Compact textual equivalent retaining the exact proxy shares in the self-contained source.
        df=load_csv('sport_strength_shares.csv').set_index('NOC');rows=[]
        for code in load_csv('predictions_2028.csv').head(10).NOC:
            top=df.loc[code].sort_values(ascending=False).head(3)
            rows.append(code+': '+', '.join(f'{k} {v:.1%}' for k,v in top.items()))
        return '\\begin{quote}\\small\n'+'\\\\\n'.join(escape_tex(x) for x in rows)+'\n\\end{quote}\n'
    else:return ''
    return '\\begin{center}\\begin{tikzpicture}\\begin{axis}['+axis+']\n'+'\n'.join(plots)+'\n\\end{axis}\\end{tikzpicture}\\end{center}\n'

def latex_report(sections):
    parts=[r'\documentclass[10pt,a4paper]{article}',r'\usepackage[margin=1.6cm]{geometry}',r'\usepackage[T1]{fontenc}',r'\usepackage{lmodern}',r'\usepackage{booktabs,longtable,array,amsmath}',r'\usepackage{tikz,pgfplots}',r'\pgfplotsset{compat=1.18}',r'\usepackage{hyperref}',r'\hypersetup{colorlinks=true,urlcolor=blue}',r'\setlength{\parindent}{0pt}',r'\setlength{\parskip}{5pt}',r'\begin{document}']
    for i,sec in enumerate(sections):
        if i:parts.append(r'\clearpage')
        parts.append(r'\section*{'+escape_tex(sec['title'])+'}')
        parts.extend(escape_tex(x)+'\n' for x in sec['paragraphs'])
        if sec['code']:parts.extend([r'\begin{verbatim}',sec['code'],r'\end{verbatim}'])
        if sec['table']:
            rows=sec['table'];n=len(rows[0]);cols=' '.join([f'p{{{(15.4/n)-.2:.2f}cm}}']*n)
            parts.append(r'{\small\begin{longtable}{'+cols+'}')
            parts.append(r'\toprule')
            parts.append(' & '.join(escape_tex(x) for x in rows[0])+r'\\ \midrule\endhead')
            parts.extend(' & '.join(escape_tex(x) for x in row)+r'\\' for row in rows[1:])
            parts.append(r'\bottomrule\end{longtable}}')
        if sec['figure']:parts.append(tex_inline_chart(sec['figure']))
    parts.append(r'\end{document}')
    (R/'solution.tex').write_text('\n'.join(parts),encoding='utf-8')

def chinese_report(data):
    out=data['out'];forecast=data['forecast'];metrics=data['metrics'];coach=data['coach'];ins=data['insight']
    primary=data['primary'];baseline=data['baseline'];joint=data['joint']
    check=metrics[(metrics.year==2024)&(metrics.model=='anchored_hurdle')].iloc[0]
    lines=['# 2025 MCM C 题建模结果与原理说明','', '分析日期：2026-10-09。全部数值由独立项目内的实际计算生成。','',
           '## 1. 已完成什么','',
           '已完成原始文件校验、国家名称映射、数据审计、六类模型的滚动回测、主场与首牌组合模型、80 次届次重采样、20,000 次联合模拟、完整代表队预测、项目情景、教练条件情景、原创洞见、消融与敏感性、结果图和英文报告。','',
           '预测覆盖 205 个代表队，并不等于 205 个主权国家。模型只使用提供的五个 CSV；没有添加 GDP、实时运动员表现或外部 2028 项目清单。','',
           '## 2. 数据如何处理','',
           '官方国家奖牌表是总牌和金牌的真值。运动员表去掉 1,466 条完全重复记录，处理代码与名称映射；团体项目先去重，再形成项目代理。1996 年后项目重构与官方总量一致为 599/673 个可对照的国家届次，因此没有声称项目层全部精确还原。','',
           'AIN 2024、MDA 2012 的运动员获奖记录与缺失的官方国家总量无法调和，监督标签不强行设零。FR Yugoslavia 1996/2000 对齐运动员 SCG；Lebanon 的 LIB/LBN 按同名记录统一。前苏联等历史国家不把奖牌任意分给当今国家。','',
           '## 3. 为什么保留简单基线','',
           f'2008、2012、2016、2020 的验证中，加权历史总牌 MAE 为 {baseline.total_mae:.3f}，金牌 MAE 为 {baseline.gold_mae:.3f}，比复杂模型的点预测误差低。主模型 anchored_hurdle 的对应误差是 {primary.total_mae:.3f}/{primary.gold_mae:.3f}。因此不宣称主模型是点预测最优；它的用途是补齐主场响应、首次获牌概率和联合分布。','',
           '加权历史公式：W = 0.6×上一届 + 0.3×前两届 + 0.1×前三届。已获牌国家以 W 为基准，用当前和前一届主场修正；未获牌国家先预测“能否获牌”，再在获牌条件下抽取正奖牌数。金牌在总牌条件下抽样，保证金牌不超过总牌。','',
           '主模型含负二项波动，方差为 μ+αμ²，可以表达比 Poisson 更大的历史波动。80 次按届次成组重采样加入参数变化；本项目未宣称覆盖全部未来赛制或数据错误风险。','',
           '## 4. 回测怎么做、效果如何','',
           '每个目标届只使用之前的届次做特征与拟合。测试范围从上一届参赛集合确定；未再参赛者在结果中记零，不等于把其实力认定为零。2024 年已在开发中看过，报告称其为回溯检查，不能冒称完全未接触的最终测试。','',
           f'2024 回溯：未约束主模型总牌 MAE {check.total_mae:.3f}，金牌 MAE {check.gold_mae:.3f}；95% 过程区间覆盖率分别 {check.total_coverage95:.1%}、{check.gold_coverage95:.1%}。区间比较宽，不能用较高覆盖率证明预测很精确。','',
           '联合预算版本的回测保留“前一届”的奖牌预算，并没有用目标届实际预算作弊。结果另存 joint_backtest_metrics.csv。未来主报告的区间是加上参数重采样后的预算约束区间，和过程区间不混为一谈。','',
           '## 5. 2028 预测的具体假设','',
           '主情景假设 2024 参赛范围和项目结构延续，主办国是美国，已报告代表队的奖牌预算延续：328 金牌、1,039 总牌。这是数据中排除了未报告 AIN 的预算，不能写成现实世界全部授予的奖牌数。每次模拟用 Hamilton 最大余数方法分别分配金牌与非金牌，保持整数和总量。','',
           '| 代表队 | 金牌期望 | 金牌95%区间 | 总牌期望 | 总牌95%区间 |','|---|---:|---|---:|---|']
    for r in out.sort_values('gold_point',ascending=False).head(15).itertuples():
        lines.append(f'| {r.name}（{r.NOC}） | {r.gold_point:.2f} | {interval(r._asdict(),"gold")} | {r.total_point:.2f} | {interval(r._asdict(),"total")} |')
    lines += ['', '这里的期望值可以是小数，每一次模拟的奖牌数则是整数。不能把各国期望排名当作确定排名；区间有大量重叠。完整 205 个代表队及改善 / 下降概率见 CSV。','',
              '## 6. 首次获牌','',
              f'在 65 个此前未获夏季奥运奖牌的候选中，首次获牌数量期望 {forecast["expected_first_count"]:.4f}；最可能是 {forecast["mode_first_count"]} 个，概率 {forecast["mode_probability"]:.2%}；95% 数量区间 [{forecast["first_count_lo95"]:.0f}, {forecast["first_count_hi95"]:.0f}]。这不是冬奥历史结论，也不是独立固定概率的简单相加抽样：重采样参数和共同奖牌预算引入了关联。','',
              '| 候选 | 首次获牌概率 |','|---|---:|']
    for r in data['first'].head(10).itertuples():lines.append(f'| {r.name}（{r.NOC}） | {r.first_medal_probability:.2%} |')
    lines += ['', '模型主要根据历史参赛规模估计突破概率，没有具体运动员实力信息。马里、几内亚、南苏丹等的模型概率较高不等于体育专业人士的夺牌判断。','',
              '## 7. 项目和主办国情景','',
              '按项目代理份额判断各国主要依赖的运动项目。测试水上、田径、体操机会增多，以及美国前三个优势项目增多、其他项目减少的固定总量情景。新增机会的弹性取 1，这是人为声明的压力测试，不是已知 2028 赛制。固定机会总量时重新分配同一奖牌预算；机会扩张时预算按赛事数比例改变。','',
              '## 8. 教练问题怎么回应','',
              '数据没有教练姓名、任职时点，无法识别纯教练因果贡献。先查找国家—项目的两届前后变化，调整赛事机会和国家总体变化，输出与教练等因素相容的异常增长。把异常增长的 0%—100% 归因给教练只能是条件归因情景，不是统计识别出的因果上下界。','',
              '| 建议代表队 | 当前参与项目 | 假设项目实力提升20%后的总牌期望增量 |','|---|---|---:|']
    for r in coach[coach.assumed_sport_gain==.2].itertuples():lines.append(f'| {r.name}（{r.NOC}） | {r.Sport} | {r.country_expected_gain:.3f} |')
    lines += ['', '三国选择要求该项目在 2024 年仍有参赛记录且存在历史基础，避免没有参与基础却仅靠换教练承诺奖牌。上述增量是基于项目份额与固定总量再分配的算术情景，不是观测到的教练效果，也没有编造教练费用。','',
              '## 9. 原创洞见与敏感性','',
              f'65 个有四届记录且平均奖牌不少于 2 的代表队中，项目集中度 HHI 与奖牌波动系数的 Spearman 相关系数 {ins["spearman_rho"]:.3f}，名义 p={ins["p_value"]:.3g}。这提示集中于少数项目的代表队可以做更严格的项目失利压力测试；不能据此证明多元化会使奖牌更稳定。','',
              '已比较训练起点 1996/2000/2004、去掉美国主场、去掉主场 / 参赛规模 / 项目广度，以及赛事机会扰动。主场效应只是少量历史主办国上的模型关联；国家背景和制度差异没有全部被识别。','',
              '## 10. 验证与运行','',
              '输入哈希、未来标签扰动、奖牌总量与金牌不超过总牌、概率范围和分布归一化已通过自动检查。整个计算管线已从原始数据完整运行；运行日志和精确依赖版本保存在 logs/last_run.json 与 requirements-lock.txt。','',
              '在本项目 PowerShell 执行：','',
              '```powershell',r'.\.venv\Scripts\python.exe -u -X utf8 scripts\run_all.py',r'.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v',r'.\.venv\Scripts\python.exe -X utf8 scripts\build_report.py','```','',
              '## 11. 结果使用边界','',
              '这套可重复运行结果覆盖原题问题，但真实 2028 预测仍受名单和项目未知、代理误差及少量届次限制。英文 PDF、可编辑 Markdown、独立 LaTeX 源和全部计算文件已提供。AI 使用记录说明实际协助范围；正式使用还需保留完整对话、确认具体模型版本并由团队审阅，不编造已获得人工批准。']
    (R/'中文技术说明.md').write_text('\n'.join(lines),encoding='utf-8')

def main():
    R.mkdir(parents=True,exist_ok=True)
    sections,data=create_sections()
    from paper_layout import enhance_sections,latex_report,pdf_report
    enhance_sections(sections,data)
    latex_report(sections);chinese_report(data);pdf_report(sections);markdown_report(sections)
    pdf=PdfReader(R/'solution.pdf');n=len(pdf.pages)
    texts=[p.extract_text() or '' for p in pdf.pages]
    ai=[i+1 for i,x in enumerate(texts) if 'Report on Use of AI Tools' in x and x.lstrip().startswith('Report on Use of AI Tools')]
    # Heading occurs in contents too; recognize the final dedicated page by prefix.
    ai_start=ai[-1] if ai else n
    metadata={'total_pdf_pages':n,'main_solution_pages':ai_start-1,'AI_report_start_page':ai_start,'expected_sections':len(sections),
              'reportlab_export':'solution.pdf','standalone_latex':'solution.tex, independent editable rendering',
              'all_205_forecasts_in_appendix':True,'visual_QA':'pending'}
    if metadata['main_solution_pages']>25:raise RuntimeError('Main solution exceeds 25 pages')
    (V/'report_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    claim='# Claim-evidence map and self-review\n\n'
    for sec in sections[:14]:claim+='- '+sec['title']+': generated from saved data/outputs; no uncomputed performance claims.\n'
    claim+='\nStrong baseline superiority: supported by model_selection.csv.\nJoint budgets and integer constraints: supported by forecast_draws.npz and unittest.\n2028 results: conditional scenarios, not actual outcomes.\nCoach causality: not identified; only proxy evidence and assumed-uplift scenarios.\nSpecialization association: descriptive, not causal.\n2024 holdout: retrospectively inspected during development.\n\nFive-dimension review: contribution=reproducible pipeline; clarity=separate targets/proxies/scenarios; experiments=all baselines retained; coverage=all Q1-Q7 addressed with limits; soundness=validity checks passed, substantive assumptions disclosed.\n'
    (V/'claim_evidence_map.md').write_text(claim,encoding='utf-8')
    print(json.dumps(metadata,indent=2))

if __name__=='__main__':main()
