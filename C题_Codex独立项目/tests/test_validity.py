"""Counterexample-based checks for leakage, input integrity and predictive constraints."""
import sys,unittest,json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from data_pipeline import verify_inputs
from modeling import build_features,design,first_design,allocate_budget

class ValidityTests(unittest.TestCase):
    def test_original_inputs_unchanged(self):
        self.assertEqual(len(verify_inputs()),6)

    def test_future_outcomes_do_not_enter_features(self):
        panel=pd.read_csv(ROOT/'data/processed/country_year_panel.csv')
        hosts=json.loads((ROOT/'data/processed/edition_metadata.json').read_text())['hosts']
        hosts={int(k):v for k,v in hosts.items()}
        original=build_features(panel,2024,hosts)
        altered=panel.copy()
        target=altered.Year>=2024
        altered.loc[target,'Total']=999999
        altered.loc[target,'Gold']=99999
        altered.loc[target,'athletes']=777777
        changed=build_features(altered,2024,hosts)
        np.testing.assert_array_equal(design(original),design(changed))
        np.testing.assert_array_equal(first_design(original),first_design(changed))
        self.assertTrue((original.source_max_year<2024).all())

    def test_integer_allocation_and_zero_support(self):
        rng=np.random.default_rng(7)
        w=np.array([[0,1,4,9],[8,0,1,5],[2,7,0,1]])
        allocated=allocate_budget(w,17,rng)
        self.assertTrue(np.all(allocated.sum(axis=1)==17))
        self.assertTrue(np.all(allocated[w==0]==0))
        self.assertTrue(np.all(allocated>=0))
        with self.assertRaises(ValueError):allocate_budget(np.zeros((2,3)),10,rng)

    def test_future_draws_are_jointly_feasible(self):
        d=np.load(ROOT/'outputs/validation/forecast_draws.npz')
        meta=json.loads((ROOT/'outputs/validation/forecast_summary.json').read_text())
        t,g=d['total'],d['gold']
        self.assertTrue(np.all((g>=0)&(t>=g)))
        self.assertTrue(np.all(t.sum(axis=1)==meta['total_budget']))
        self.assertTrue(np.all(g.sum(axis=1)==meta['gold_budget']))
        self.assertEqual(len(set(d['codes'])),len(d['codes']))

    def test_first_medal_and_interval_outputs(self):
        f=pd.read_csv(ROOT/'outputs/tables/first_medal_probabilities.csv')
        dist=pd.read_csv(ROOT/'outputs/tables/first_medal_count_distribution.csv')
        pred=pd.read_csv(ROOT/'outputs/tables/predictions_2028.csv')
        self.assertTrue(f.first_medal_probability.between(0,1).all())
        self.assertAlmostEqual(float(dist.probability.sum()),1,places=9)
        self.assertTrue((pred.gold_point<=pred.total_point).all())
        for kind in ['gold','total']:
            self.assertTrue((pred[kind+'_lo95']<=pred[kind+'_hi95']).all())
        self.assertFalse(pred.isna().any().any())

    def test_sport_scenario_totals_match_declared_budgets(self):
        scenarios=pd.read_csv(ROOT/'outputs/tables/sport_scenarios.csv')
        for name,g in scenarios.groupby('scenario'):
            self.assertEqual(g.NOC.nunique(),205)
            self.assertAlmostEqual(g.scenario_total.sum(),g.scenario_total_budget.iloc[0],places=7)
            self.assertTrue((g.scenario_total>=0).all())
            if 'fixed opportunity total' in name:
                self.assertAlmostEqual(g.scenario_total.sum(),1039,places=7)

    def test_coaching_recommendations_have_current_participation(self):
        cases=pd.read_csv(ROOT/'outputs/tables/coach_investment_scenarios.csv')
        sport=pd.read_csv(ROOT/'data/processed/country_sport_year.csv')
        self.assertEqual(cases.NOC.nunique(),3)
        for r in cases[['NOC','Sport']].drop_duplicates().itertuples():
            participation=sport[(sport.NOC==r.NOC)&(sport.Sport==r.Sport)&(sport.Year==2024)]
            self.assertGreater(participation.entries.sum(),0)
        for _,g in cases.groupby('NOC'):
            self.assertTrue((g.sort_values('assumed_sport_gain').country_expected_gain.diff().dropna()>0).all())

if __name__=='__main__':unittest.main(verbosity=2)
