"""Write the Chinese first-round report from measured results, not placeholders."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/pilot'


def main():
    summary=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    methods=summary['methods']
    policies=json.loads((OUT/'policies.json').read_text(encoding='utf-8'))
    selection=json.loads((OUT/'selection.json').read_text(encoding='utf-8'))
    labels={'ddim50':'DDIM 50 步，无缓存','uniform':'DeepCache 均匀刷新','paper_nonuniform':'DeepCache 论文式非均匀刷新','calibrated':'校准刷新（本项目）','dpmpp_matched':f'DPM-Solver++ {selection["dpmpp_steps"]} 步'}
    lines=['# DeepCache 第一轮实测报告','', '日期：2026-11-06。结果来自本机 RTX 5060 Laptop GPU 8GB 的实际推理。', '', '## 实验设置', '', '- SD1.5、FP16、512×512、batch=1、CFG=7.5、PyTorch SDPA、无 CPU offload。', '- COCO 2014 官方 caption 评价标注；按图像 ID 分离 4 条校准、4 条验证、8 条测试提示词。测试各用两个种子，共 16 个配对案例、80 张测试图。', '- 缓存组均为 DDIM 50 步、branch=0、17 次完整刷新，仅刷新位置不同。DPM-Solver++ 的步数由验证集耗时确定。', '- 原计划的 COCO2017 全标注下载停滞，实际使用的是 COCO 官方评价仓库的 2014 标注；源地址、文件 SHA256、ID 和提示词保存在 manifest。', '- 两项 GPU 正确性检查通过：interval=1 与无缓存逐像素一致；自定义列表复现固定间隔时与官方 helper 逐像素一致。动态规划也通过独立穷举最优性测试。', '', '## 测试集结果', '', '| 方法 | 耗时中位数/秒 ↓ | 对 DDIM50 加速比 ↑ | 峰值分配/MiB | CLIP 余弦 ↑ | 对 DDIM50 的 LPIPS ↓ |', '|---|---:|---:|---:|---:|---:|']
    for key,s in methods.items():
        lines.append(f'| {labels[key]} | {s["median_seconds"]:.3f} | {s["speedup_vs_ddim50"]:.2f}× | {s["max_peak_allocated_mib"]:.0f} | {s["mean_clip_cosine"]:.4f} | {s["mean_lpips_to_ddim50"]:.4f} |')
    lines += ['', '指标解释：CLIP 为 ViT-B/32 归一化文本/图像向量的原始余弦相似度；LPIPS 衡量与相同提示词、相同初始 latent 下 DDIM50 输出的差异，不直接等于真实感。表中的显存是 PyTorch 峰值 allocated，不含全部 CUDA 上下文；峰值 reserved 另见 summary.json。', '', '## 校准策略的配对差异', '', '以下差异均为“校准刷新减去比较对象”，按 8 个提示词簇 bootstrap，两个种子一起重采样。区间仅用于探索，不构成大样本确认性结论。', '', '| 比较对象 | LPIPS 差值及 95% 区间 | CLIP 差值及 95% 区间 |', '|---|---|---|']
    for key,label in [('uniform','均匀刷新'),('paper_nonuniform','论文式非均匀刷新')]:
        comp=summary['paired_differences'][f'calibrated_minus_{key}']
        cells=[]
        for field in ['lpips_to_ddim50','clip_cosine']:
            d=comp[field];lo,hi=d['cluster_bootstrap_95pct_ci']
            cells.append(f'{d["mean_difference"]:+.4f} [{lo:+.4f}, {hi:+.4f}]')
        lines.append(f'| {label} | {cells[0]} | {cells[1]} |')
    cal=methods['calibrated'];uniform=methods['uniform'];paper=methods['paper_nonuniform']
    rel=(uniform['mean_lpips_to_ddim50']-cal['mean_lpips_to_ddim50'])/uniform['mean_lpips_to_ddim50']*100
    change=f'降低 {rel:.2f}%' if rel>=0 else f'升高 {-rel:.2f}%'
    verdict='改善' if rel>0 else '改善尚未成立，当前平均参考偏差略大'
    lines += ['', f'校准方案相对均匀刷新的平均 LPIPS {change}。这不是“生成质量提升百分比”。', '', '## 本轮判断', '', f'- 原版均匀 DeepCache 达到 {uniform["speedup_vs_ddim50"]:.2f} 倍加速，实测硬件容量足够，框架可以继续用于课程项目。', f'- 本版校准策略相对均匀刷新的效果：{verdict}。其 CLIP 均值为 {cal["mean_clip_cosine"]:.4f}，均匀基线为 {uniform["mean_clip_cosine"]:.4f}；目前没有依据把它写成全面提升。', '- DPM-Solver++ 20 步在接近的延迟下保留了较高 CLIP 文本对齐，说明后续必须保留强采样器基线；它与 DDIM50 更不相似并不等于图像更差。', '- 下一轮优先检验代理特征漂移与噪声预测/输出误差的相关性，再决定是否改用更直接的代理；这只是下一轮假设，不把本轮测试集再用作调参集。', '', '## 刷新策略与校准开销', '', f'- 特征采集与求解耗时：{policies["calibration_seconds_this_invocation"]:.2f} 秒；不包含下载、验证集配置比较及质量评测。', '- 校准代理来自实际缓存边界的特征输出，空间池化到 8×8；CFG 两分支分别归一化。目标是减少参考轨迹上的特征漂移，并不保证最终图像误差最优。', '- 论文式非均匀策略是原论文 Eq.7 的有符号实数幂实现，做了明确的整数化与预算检查；不宣称逐位复现旧版 pipeline。两个 center 候选通过验证集 LPIPS 选择。', f'- 均匀刷新：`{policies["schedules"]["uniform"]}`', f'- 论文式非均匀刷新：`{policies["schedules"][selection["selected_paper"]]}`', f'- 校准刷新：`{policies["schedules"]["calibrated"]}`', '', '## 可支持的判断与局限', '', '- 本轮不能将单一 LPIPS 改善解释为全面视觉质量提升，也没有实施人工盲评或计算 FID。', '- 时间来自预热后的随机顺序单次生成；每方法 16 个测量，存在系统负载波动，不是每案例重复计时的严格性能基准。', '- 校准只有 4 个提示词；固定一个底模、分辨率、CFG、branch 和预算，不能据此声称跨模型泛化。', '- 后续应预先规定更大样本的验证方案；若修改代理或调度，使用新的测试划分，不能反复利用本轮测试结果调参后仍当独立测试。', '', '## 图表和原始结果', '', '- [质量与耗时](quality_latency.png)', '- [特征漂移和刷新位置](refresh_schedules.png)', '- [前四条测试提示词对比](comparison_1.jpg)', '- [后四条测试提示词对比](comparison_2.jpg)', '- [逐图结果 CSV](evaluated_rows.csv)', '- [完整汇总与区间](summary.json)', '- [运行环境](environment.json)', '- [原始生成记录](measurements.jsonl)', '']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__': main()
