"""Fixed-budget cache schedules and feature-drift calibration (our contribution)."""
import numpy as np
from DeepCache import DeepCacheSDHelper


class ScheduledDeepCache(DeepCacheSDHelper):
    """Keep the official cached module boundaries; change only refresh decisions.

    Call index, rather than timestep value, defines a step. Use a fresh context for
    each generation. All blocks share the same per-forward refresh decision.
    """
    def __init__(self, pipe, refresh_steps, branch=0):
        super().__init__(pipe)
        self.refresh_steps = frozenset(int(i) for i in refresh_steps)
        if 0 not in self.refresh_steps:
            raise ValueError('The first U-Net call must refresh the cache')
        self.set_params(cache_interval=1, cache_branch_id=branch)
        self.calls = 0
        self.refresh_log = []

    def wrap_unet_forward(self):
        self.function_dict['unet_forward'] = self.pipe.unet.forward
        def wrapped(*args, **kwargs):
            self.cur_timestep = self.calls
            self.calls += 1
            if self.cur_timestep in self.refresh_steps:
                self.refresh_log.append(self.cur_timestep)
            return self.function_dict['unet_forward'](*args, **kwargs)
        self.pipe.unet.forward = wrapped

    def is_skip_step(self, block_i, layer_i, blocktype='down'):
        if self.cur_timestep in self.refresh_steps:
            return False
        p = self.params
        if block_i > p['cache_block_id'] or blocktype == 'mid':
            return True
        if block_i < p['cache_block_id']:
            return False
        return layer_i >= p['cache_layer_id'] if blocktype == 'down' else layer_i > p['cache_layer_id']

    def __enter__(self):
        self.calls = 0
        self.refresh_log = []
        self.enable()
        return self

    def __exit__(self, *exc):
        self.disable()


def uniform_schedule(steps, interval):
    return list(range(0, steps, interval))


def paper_nonuniform_schedule(steps, budget, center, power):
    """Eq. 7 signed-real-power adaptation, not a bit-exact legacy pipeline port.

    Signed roots avoid fractional powers of negative numbers. Reduce power when
    integer rounding would violate the requested budget, as in the legacy code.
    """
    p = power
    while p >= 1.0 - 1e-8:
        lo = -float(center) ** (1 / p)
        hi = float(steps - center) ** (1 / p)
        x = np.linspace(lo, hi, budget + 1)
        y = np.sign(x) * np.abs(x) ** p + center
        schedule = sorted(set([0] + np.floor(y[:-1] + 1e-7).astype(int).clip(0, steps - 1).tolist()))
        if len(schedule) == budget:
            return schedule
        p -= .02
    raise ValueError('Could not construct the requested nonuniform budget')


def feature_interval_cost(traces):
    """Mean normalized L1 drift; normalize CFG branches separately.

    Each trace is [steps, cfg_branches, pooled_features]. C[a,b] estimates
    accumulated error when reusing feature a until just before b.
    """
    steps = traces[0].shape[0]
    cost = np.zeros((steps, steps + 1), dtype=np.float64)
    drift = np.zeros((steps, steps), dtype=np.float64)
    for g in traces:
        norm = np.abs(g).mean(axis=-1) + 1e-6
        for a in range(steps):
            relative = (np.abs(g[a:] - g[a]).mean(axis=-1) / norm[a:]).mean(axis=-1)
            drift[a, a:] += relative / len(traces)
    for a in range(steps):
        cost[a, a + 1:] = np.cumsum(drift[a, a:])
    return cost, drift


def calibrated_schedule(cost, budget, max_gap=6):
    """Exact DP for the surrogate additive interval cost at fixed refresh count."""
    steps = cost.shape[0]
    dp = np.full((budget + 1, steps + 1), np.inf)
    prev = np.full((budget + 1, steps + 1), -1, dtype=int)
    dp[0, 0] = 0
    for k in range(1, budget + 1):
        for end in range(k, steps + 1):
            for start in range(max(k - 1, end - max_gap), end):
                val = dp[k - 1, start] + cost[start, end]
                if val < dp[k, end]:
                    dp[k, end], prev[k, end] = val, start
    if not np.isfinite(dp[budget, steps]):
        raise ValueError('Infeasible budget/max_gap')
    result, end = [], steps
    for k in range(budget, 0, -1):
        start = int(prev[k, end])
        result.append(start)
        end = start
    return sorted(result)
