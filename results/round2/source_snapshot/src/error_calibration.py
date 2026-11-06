"""Local counterfactual error costs for deterministic DDIM cache schedules.

Only offline calibration executes extra full U-Net calls. Test-time policies
remain fixed lists with the same refresh count as official uniform DeepCache.
"""
import numpy as np
import torch


def guided_prediction(prediction, scale=7.5):
    unconditional, conditional = prediction.float().chunk(2)
    return unconditional + scale * (conditional - unconditional)


def ddim_epsilon_coefficient(scheduler, timestep):
    """d(prev_sample)/d(epsilon) for eta=0, clipping disabled, epsilon prediction."""
    if scheduler.config.prediction_type != 'epsilon' or scheduler.config.clip_sample or scheduler.config.thresholding:
        raise ValueError('Analytical coefficient requires unclipped epsilon-prediction DDIM')
    t = int(timestep)
    prev_t = t - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    alpha = float(scheduler.alphas_cumprod[t])
    alpha_prev = float(scheduler.alphas_cumprod[prev_t]) if prev_t >= 0 else float(scheduler.final_alpha_cumprod)
    return np.sqrt(1-alpha_prev) - np.sqrt(alpha_prev/alpha)*np.sqrt(1-alpha)


def interval_cost_from_errors(errors, max_gap=6):
    """C[a,b] = sum error[a,i], i in [a,b), with unsupported intervals infinite."""
    steps = errors.shape[0]
    cost = np.full((steps,steps+1),np.inf,dtype=np.float64)
    for a in range(steps):
        limit=min(steps,a+max_gap)
        values=errors[a,a:limit]
        if not np.isfinite(values).all():
            raise ValueError(f'Missing probe in interval starting at {a}')
        cost[a,a+1:limit+1]=np.cumsum(values)
    return cost


def residualized_rank_correlation(x,y,target_steps,ages):
    """Exploratory association after removing target-step and cache-age fixed effects."""
    from scipy.stats import rankdata,pearsonr
    x,y=rankdata(x),rankdata(y)
    steps=np.unique(target_steps);unique_ages=np.unique(ages)
    design=np.column_stack([np.ones(len(x))]+[(target_steps==i).astype(float) for i in steps[1:]]+[(ages==a).astype(float) for a in unique_ages[1:]])
    rx=x-design@np.linalg.lstsq(design,x,rcond=None)[0]
    ry=y-design@np.linalg.lstsq(design,y,rcond=None)[0]
    return float(pearsonr(rx,ry).statistic)
