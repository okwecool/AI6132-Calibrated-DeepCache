"""Verify inference weights independent prompts, rather than repeated seeds."""
import unittest
from scripts.evaluate_pilot import prompt_cluster_ci


def pair(image_id,seed,delta):
    return [
        {'image_id':image_id,'seed':seed,'method':'uniform','lpips_to_ddim50':2.0},
        {'image_id':image_id,'seed':seed,'method':'calibrated','lpips_to_ddim50':2.0+delta},
    ]


class PairedAnalysisTests(unittest.TestCase):
    def test_constant_difference_has_exact_point_interval(self):
        rows=[r for image_id in range(3) for seed in [7,11] for r in pair(image_id,seed,.25)]
        result=prompt_cluster_ci(rows,'lpips_to_ddim50')
        self.assertEqual(result['clusters'],3)
        self.assertEqual(result['mean_difference'],.25)
        self.assertEqual(result['cluster_bootstrap_95pct_ci'],[.25,.25])

    def test_extra_seeds_do_not_give_one_prompt_more_weight(self):
        rows=[r for seed in range(10) for r in pair(1,seed,1)]
        rows+=pair(2,0,-1)
        result=prompt_cluster_ci(rows,'lpips_to_ddim50')
        self.assertEqual(result['clusters'],2)
        self.assertEqual(result['mean_difference'],0)


if __name__=='__main__':unittest.main()
