"""Frozen confirmation: primary inference, paired budget ablation and report."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.evaluate_pilot import prompt_cluster_ci
from scripts.run_pilot import write_json

OUT=ROOT/'results/confirmation'


def interval(d):
    lo,hi=d['cluster_bootstrap_95pct_ci']
    return f"{d['mean_difference']:+.5f} [{lo:+.5f}, {hi:+.5f}]"


def budget_analysis(test,extra,manifest):
    subset=set(manifest['ablation_image_ids'])
    combined={}
    for budget in [10,17,25]:
        if budget==17:
            rows=[dict(r) for r in test if r['image_id'] in subset and r['method'] in ['uniform','calibrated']]
        else:
            rows=[dict(r,method=r['method'].split('_b')[0]) for r in extra if r['full_refresh_count']==budget]
        assert len(rows)==32 and len({r['image_id'] for r in rows})==16
        stats={}
        for method in ['uniform','calibrated']:
            selected=[r for r in rows if r['method']==method]
            assert len(selected)==16
            stats[method]={'n':16,'median_seconds':float(np.median([r['seconds'] for r in selected])),'mean_lpips':float(np.mean([r['lpips_to_ddim50'] for r in selected])),'mean_clip':float(np.mean([r['clip_cosine'] for r in selected])),'max_peak_allocated_mib':max(r['peak_allocated_mib'] for r in selected)}
        combined[str(budget)]={'methods':stats,'paired_differences':{field:prompt_cluster_ci(rows,field) for field in ['lpips_to_ddim50','clip_cosine','seconds']}}
    write_json(OUT/'budget_summary.json',{'subset_image_ids':manifest['ablation_image_ids'],'sample_size':16,'budgets':combined,'note':'All three budgets use the same predefined 16 prompts; budget17/reference reuse main test images. Secondary exploratory ablation.'})
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for method,label in [('uniform','Uniform'),('calibrated','Frozen error calibration')]:
        for ax,key,title in [(axes[0],'mean_lpips','LPIPS to DDIM50 (lower = closer)'),(axes[1],'mean_clip','CLIP text-image cosine (higher)'),(axes[2],'median_seconds','Median generation time (s)')]:
            ax.plot([10,17,25],[combined[str(b)]['methods'][method][key] for b in [10,17,25]],marker='o',label=label)
            ax.set(xlabel='Full-refresh budget / 50 DDIM calls',ylabel=title,xticks=[10,17,25]);ax.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    fig.suptitle('Budget ablation: same 16 predefined held-out prompts, 1 seed each',fontsize=11)
    fig.savefig(OUT/'budget_ablation.png',dpi=170);plt.close(fig)
    return combined


def main():
    s=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    test=json.loads((OUT/'evaluated_rows.json').read_text(encoding='utf-8'))
    extra=json.loads((OUT/'evaluated_ablation_rows.json').read_text(encoding='utf-8'))
    manifest=json.loads((ROOT/'manifests/confirmation.json').read_text(encoding='utf-8'))
    assert len(test)==500 and len(extra)==64
    budgets=budget_analysis(test,extra,manifest)
    m=s['methods'];a=m['calibrated'];u=m['uniform']
    primary=s['paired_differences']['calibrated_minus_uniform']['lpips_to_ddim50']
    lo,hi=primary['cluster_bootstrap_95pct_ci']
    if hi<0:verdict='在这批冻结策略、全新提示词的测试中，主指标支持新方案降低参考偏差。'
    elif lo>0:verdict='在这批冻结策略、全新提示词的测试中，新方案的参考偏差更大，主假设未获支持。'
    else:verdict='主指标差异区间仍包含零，暂不能确认相对均匀刷新的稳定优势。'
    relative=100*(u['mean_lpips_to_ddim50']-a['mean_lpips_to_ddim50'])/u['mean_lpips_to_ddim50']
    change_description=f'相对下降 {relative:.2f}%' if relative>=0 else f'相对升高 {-relative:.2f}%'
    by={}
    for r in test:by.setdefault(r['image_id'],{})[r['method']]=r
    deltas=np.array([p['calibrated']['lpips_to_ddim50']-p['uniform']['lpips_to_ddim50'] for p in by.values()])
    wins=int((deltas<0).sum())
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    ax.hist(deltas,bins=20,color='tab:blue',alpha=.8);ax.axvline(0,color='gray',linestyle='--')
    ax.axvline(deltas.mean(),color='tab:red',label=f'Mean difference: {deltas.mean():+.4f}')
    ax.set(xlabel='LPIPS: calibrated minus uniform (negative = closer to DDIM50)',ylabel='Number of independent prompts',title=f'Frozen confirmation: {wins}/100 prompts have lower reference deviation')
    ax.legend();fig.savefig(OUT/'per_prompt_differences.png',dpi=170);plt.close(fig)
    write_json(OUT/'primary_result.json',{'predefined_primary_comparison':'calibrated minus uniform LPIPS','verdict':verdict,'mean_difference':primary['mean_difference'],'paired_bootstrap_95pct_ci':[lo,hi],'relative_lpips_reduction_pct':relative,'prompts_with_lower_lpips':wins,'total_independent_prompts':100,'no_test_parameter_selection':True,'scope':'SD1.5 FP16 512px CFG7.5 DDIM50 branch0 budget17; does not establish universal visual quality improvement'})
    names={'ddim50':'DDIM50，无缓存','uniform':'DeepCache 均匀刷新','paper_nonuniform':'冻结的论文式非均匀刷新','calibrated':'冻结的新误差校准策略','dpmpp_matched':'冻结的 DPM-Solver++ 20 步'}
    lines=['# DeepCache 冻结策略确认与预算消融','', '日期：2026-11-06。RTX 5060 Laptop 8GB 本机实测。','', '## 主结论','',verdict,'',f'- 新方案 LPIPS 均值 {a["mean_lpips_to_ddim50"]:.5f}，均匀刷新 {u["mean_lpips_to_ddim50"]:.5f}；相对下降 {relative:.2f}%（负值表示变差）。',f'- 配对差值及 95% bootstrap 区间：{interval(primary)}；100 个提示词中 {wins} 个参考偏差更小。',f'- 新方案单图耗时中位数 {a["median_seconds"]:.3f} 秒，对无缓存 DDIM50 的加速比 {a["speedup_vs_ddim50"]:.2f}×。',f'- 新/均匀 CLIP 余弦分别为 {a["mean_clip_cosine"]:.5f}/{u["mean_clip_cosine"]:.5f}。LPIPS 只是与参考输出的距离，不等于真实感或整体生成质量。','', '## 预先固定的确认协议','', '- 100 条新 COCO 2014 caption、固定种子 2026，100 个独立提示词；五种方法共 500 张主测试图。', '- 图像 ID 与前两轮所有 48 个校准/验证/测试 ID 完全不重叠。', '- 直接复用第二轮选中的 DDIM 更新误差代价与 17 次刷新列表；论文式非均匀参数、DPM-Solver++ 步数也冻结。没有使用本轮结果重新选参。', '- 主比较预先指定为新方案减均匀方案的 LPIPS 均值差；其他比较和预算消融为次要分析。', '- 校准/验证数据来自第二轮的既有 8/8 个提示词，本轮未新增拟合。更早的约 109 秒校准开销仍需计入方法准备成本。', '- SD1.5、512×512、FP16、CFG=7.5、batch=1、DDIM50、branch=0；主缓存组均为 17 次完整刷新。', '- 所有方法使用相同提示词和初始 latent，测量顺序随机。每案例测量一次，采用全部测量的中位数，未作重复计时性能确认。','', '## 100 提示词的主测试结果','', '| 方法 | 耗时中位数/秒 ↓ | 加速比 ↑ | 峰值分配/MiB | CLIP 余弦 ↑ | 对 DDIM50 的 LPIPS ↓ |','|---|---:|---:|---:|---:|---:|']
    for method,d in m.items():lines.append(f'| {names[method]} | {d["median_seconds"]:.3f} | {d["speedup_vs_ddim50"]:.2f}× | {d["max_peak_allocated_mib"]:.0f} | {d["mean_clip_cosine"]:.5f} | {d["mean_lpips_to_ddim50"]:.5f} |')
    lines += ['', '## 配对差异及区间','', '| 新方案减比较对象 | LPIPS 差值 [95% 区间] | CLIP 差值 [95% 区间] |','|---|---|---|']
    for method in ['uniform','paper_nonuniform','dpmpp_matched']:
        d=s['paired_differences'][f'calibrated_minus_{method}']
        lines.append(f'| {names[method]} | {interval(d["lpips_to_ddim50"])} | {interval(d["clip_cosine"])} |')
    lines += ['', '区间按提示词配对 bootstrap。仅主比较用于此次确认判断；次要比较未做多重检验校正。未预设 CLIP 非劣界值，不能把其区间跨零解释为等价。','', '## 刷新预算消融','', '预先指定主测试清单的前 16 条提示词，固定一个种子，在同一子集比较 10/17/25 次完整刷新。17 次及无缓存图复用主测试；新增 64 张图，合计 564 张唯一生成结果。预算列表只从第二轮既有代价推导，最大间隔仍为 6，没有用本轮子集选预算。','', '| 预算 | 均匀 LPIPS | 新策略 LPIPS | 配对差值 [95% 区间] | 均匀/新耗时中位数/秒 |','|---:|---:|---:|---|---|']
    for budget,d in budgets.items():
        um=d['methods']['uniform'];am=d['methods']['calibrated']
        lines.append(f'| {budget} | {um["mean_lpips"]:.5f} | {am["mean_lpips"]:.5f} | {interval(d["paired_differences"]["lpips_to_ddim50"])} | {um["median_seconds"]:.3f}/{am["median_seconds"]:.3f} |')
    lines += ['', '三个预算的样本一致，避免把数据集差异解释为预算效果。16 提示词消融仍属小规模次要分析，不能据此事后更换主假设。','', '## 边界与可复现文件','', '- 主指标衡量冻结算法对同一 DDIM 参考输出的逼近；没有直接评价真实照片重建、人工偏好或整体真实感。', '- 一种底模、CFG、分辨率、branch 和一个种子，不足以证明跨配置泛化。', '- 论文式非均匀仍是原论文 Eq.7 的有符号实数幂改写，并非逐位复现旧 pipeline。', '- 缓存仍执行 50 次 U-Net 调用，完整与局部调用成本不同；加速结论使用实际时间，不只比较刷新次数。', '', '- [质量与延迟](quality_latency.png)', '- [配对差异区间](paired_differences.png)', '- [逐提示词差异分布](per_prompt_differences.png)', '- [预算消融](budget_ablation.png)', '- [固定前四条提示词样例](comparison_1.jpg)', '- [主测试逐图 CSV](evaluated_rows.csv)', '- [预算新增图逐图 CSV](evaluated_ablation_rows.csv)', '- [完整统计](summary.json)', '- [预算统计](budget_summary.json)', '- [主比较判断](primary_result.json)', '']
    audit=json.loads((OUT/'timing_repairs.json').read_text(encoding='utf-8'))
    nrepairs=len(audit['repairs'])
    lines += ['', '## 计时中断处理','', f'本轮检测到 {nrepairs} 条长时间暂停记录。规则为墙钟耗时大于 60 秒，且墙钟与 GPU-event U-Net 耗时差大于 60 秒。原始记录与原始质量图像保留；受影响案例按完全相同设置补测一次，只替换统计中的计时字段。补测图与原图逐像素一致。','', '564 张用于质量评价的图片之外，计时补测不增加独立质量样本。详见 [计时审计](timing_repairs.json)，CSV 同时保留 original_seconds 和 timing_repaired 字段。','']
    lines += ['## 固定展示样例的观察','', '第一组对比图按 manifest 顺序固定展示，并未按分数挑选。COCO 559778 的 caption 提到 German shepherd，但无缓存参考及缓存结果主要生成了人物与玩具熊。新策略更接近该参考，也不能据此说语义错误被修复。这说明主指标确认的是对原模型行为的保真加速，而不是原模型所有生成能力的提升。','', 'COCO 215826 的公园长椅样例中，新策略比均匀缓存更接近参考；其他样例仍有局部退步。完整 100 提示词结果和逐图表格应与固定样例一起阅读，不能仅凭少数图片概括全部效果。','']
    text='\n'.join(lines).replace(f'相对下降 {relative:.2f}%（负值表示变差）',change_description)
    (OUT/'report.md').write_text(text,encoding='utf-8')


if __name__=='__main__':main()
