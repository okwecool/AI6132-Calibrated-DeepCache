import unittest
import numpy as np
import torch
from diffusers import DDIMScheduler
from src.error_calibration import ddim_epsilon_coefficient,interval_cost_from_errors


class ErrorCalibrationTests(unittest.TestCase):
    def test_sensitivity_matches_actual_scheduler_update(self):
        scheduler=DDIMScheduler(clip_sample=False,set_alpha_to_one=False,steps_offset=1)
        scheduler.set_timesteps(50)
        generator=torch.Generator().manual_seed(17)
        x=torch.randn((1,4,8,8),generator=generator)
        e=torch.randn(x.shape,generator=generator)
        delta=torch.randn(x.shape,generator=generator)*.1
        for i in [0,10,25,48,49]:
            t=scheduler.timesteps[i]
            actual=scheduler.step(e+delta,t,x,eta=0).prev_sample-scheduler.step(e,t,x,eta=0).prev_sample
            expected=ddim_epsilon_coefficient(scheduler,t)*delta
            torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-4)

    def test_cost_uses_only_allowed_counterfactual_probes(self):
        errors=np.array([[0,2,3,np.nan],[np.nan,0,7,8],[np.nan,np.nan,0,9],[np.nan,np.nan,np.nan,0]])
        cost=interval_cost_from_errors(errors,max_gap=3)
        self.assertEqual(cost[0,3],5)
        self.assertEqual(cost[1,4],15)
        self.assertTrue(np.isinf(cost[0,4]))
        self.assertEqual(cost[3,4],0)


if __name__=='__main__': unittest.main()
