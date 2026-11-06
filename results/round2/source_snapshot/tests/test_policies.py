"""Check budget constraints and DP optimality independently by enumeration."""
import itertools
import unittest
import numpy as np
from src.cache_policy import calibrated_schedule, feature_interval_cost, paper_nonuniform_schedule


class PolicyTests(unittest.TestCase):
    def test_dp_matches_exhaustive_search(self):
        rng = np.random.default_rng(3)
        trace = rng.normal(size=(7, 2, 5))
        cost, _ = feature_interval_cost([trace])
        result = calibrated_schedule(cost, 3, max_gap=4)
        def objective(s):
            return sum(cost[a, b] for a, b in zip(s, s[1:] + [7]))
        possible = [[0] + list(c) for c in itertools.combinations(range(1, 7), 2)]
        possible = [s for s in possible if max(np.diff(s + [7])) <= 4]
        self.assertAlmostEqual(objective(result), min(map(objective, possible)))

    def test_integer_nonuniform_budget(self):
        for budget in [10, 17, 25]:
            for center in [10, 15, 20]:
                for power in [1.2, 1.4, 1.6]:
                    s = paper_nonuniform_schedule(50, budget, center, power)
                    self.assertEqual(len(s), budget)
                    self.assertEqual(s[0], 0)
                    self.assertEqual(s, sorted(set(s)))
                    self.assertLess(s[-1], 50)

    def test_constant_features_have_no_drift(self):
        cost, drift = feature_interval_cost([np.ones((6, 2, 10))])
        np.testing.assert_array_equal(cost, 0)
        np.testing.assert_array_equal(drift, 0)


if __name__ == '__main__':
    unittest.main()
