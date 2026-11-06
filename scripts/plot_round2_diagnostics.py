"""Render calibration diagnostics without running or loading the generator."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/round2'


def main():
    with (OUT/'diagnostic_probes.csv').open(encoding='utf-8-sig') as f:
        rows=list(csv.DictReader(f))
    x=np.array([float(r['feature_drift']) for r in rows])
    noise=np.array([float(r['noise_mse']) for r in rows])
    step=np.array([float(r['step_mse']) for r in rows])
    age=np.array([int(r['age']) for r in rows])
    target=np.array([int(r['target_step']) for r in rows])
    diagnosis=json.loads((OUT/'diagnostic_summary.json').read_text(encoding='utf-8'))
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,y,name,key in [(axes[0,0],noise,'Guided noise prediction MSE','noise_mse'),(axes[0,1],step,'DDIM one-step update MSE','step_mse')]:
        points=ax.scatter(x,np.maximum(y,1e-12),c=age,cmap='viridis',s=7,alpha=.4)
        ax.set_yscale('log')
        rho=diagnosis['associations'][key]['overall_spearman']
        ax.set(xlabel='Old pooled-feature relative L1 drift',ylabel=name,title=f'Pooled association: Spearman r={rho:.3f}')
        fig.colorbar(points,ax=ax,label='Cache age (calls)')
    for values,label in [(x,'Old feature drift'),(noise,'Noise MSE'),(step,'DDIM update MSE')]:
        series=np.array([np.mean(values[(age==3)&(target==i)]) for i in range(3,50)])
        axes[1,0].plot(range(3,50),series/(series.max()+1e-12),label=label)
    axes[1,0].set(xlabel='Target denoising call index',ylabel='Each curve / its maximum',title='Phase dependence at fixed cache age=3')
    axes[1,0].legend(fontsize=8)
    for key,label in [('noise_mse','Feature vs noise MSE'),('step_mse','Feature vs DDIM update MSE')]:
        values=diagnosis['associations'][key]['per_age_spearman']
        axes[1,1].plot(range(1,6),[values[str(i)] for i in range(1,6)],marker='o',label=label)
    axes[1,1].set(xlabel='Cache age',ylabel='Within-age Spearman correlation',title='Cache age explains part of the pooled association',xticks=range(1,6),ylim=(0,1))
    axes[1,1].legend(fontsize=8)
    for ax in axes.flat:ax.grid(alpha=.2)
    fig.suptitle('Round 2 calibration only: 8 prompts, 1880 local cache probes (not independent images)',fontsize=11)
    fig.savefig(OUT/'proxy_diagnostics.png',dpi=170)
    plt.close(fig)


if __name__=='__main__':main()
