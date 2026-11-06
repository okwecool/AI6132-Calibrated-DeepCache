"""Generate an evidence-based Chinese report for the second experiment."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/round2'


def interval(d):
    lo,hi=d['cluster_bootstrap_95pct_ci']
    return f"{d['mean_difference']:+.5f} [{lo:+.5f}, {hi:+.5f}]"


def main():
    summary=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    diagnosis=json.loads((OUT/'diagnostic_summary.json').read_text(encoding='utf-8'))
    policies=json.loads((OUT/'policies.json').read_text(encoding='utf-8'))
    selection=json.loads((OUT/'selection.json').read_text(encoding='utf-8'))
    m=summary['methods'];uniform=m['uniform'];new=m['calibrated'];old=m['feature_calibrated']
    gain=100*(uniform['mean_lpips_to_ddim50']-new['mean_lpips_to_ddim50'])/uniform['mean_lpips_to_ddim50']
    labels={'ddim50':'DDIM50，无缓存','uniform':'DeepCache 均匀刷新','paper_nonuniform':'论文式非均匀刷新','feature_calibrated':'特征漂移校准（旧目标）','calibrated':'实际误差校准（新目标）','dpmpp_matched':f'DPM-Solver++ {selection["dpmpp_steps"]} 步'}
    objective={'noise_candidate':'CFG 后的噪声预测 MSE','step_candidate':'DDIM 单步 latent 更新 MSE'}[selection['selected_new_cost']]
    lpips_diff=summary['paired_differences']['calibrated_minus_uniform']['lpips_to_ddim50']
    direction='降低' if gain>=0 else '升高'
    lines=['# DeepCache 第二轮：误差诊断与校准目标修改','', '日期：2026-11-06。全部生成与评价均在本机 RTX 5060 Laptop GPU 8GB 上执行。','', '## 结果先读','',f'- 新误差校准方案相对均匀刷新，平均 LPIPS {direction} {abs(gain):.2f}%；单图耗时中位数 {new["median_seconds"]:.3f} 秒，约 {new["speedup_vs_ddim50"]:.2f} 倍于 DDIM50 的速度。',f'- CLIP 文本图像余弦：新方案 {new["mean_clip_cosine"]:.4f}，均匀基线 {uniform["mean_clip_cosine"]:.4f}。LPIPS 是对参考输出的偏差，不能直接解释成视觉质量提升。',f'- 校准对均匀刷新的 LPIPS 配对差值及探索性区间：{interval(lpips_diff)}。', '', '## 本轮为什么修改目标','', '上一轮用池化特征的相对变化来安排刷新，但在独立测试中未优于均匀缓存。这轮先诊断代理，再比较两个预先定义的实际误差目标。', '',f'校准集为 8 个新提示词，共 {diagnosis["probe_cases"]} 个局部探测。每个探测固定同一个 latent 和 timestep，对比完整 U-Net 与使用旧缓存的输出；完整路径回放最大误差为 {diagnosis["full_replay_max_error"]}。这些探测来自同一批轨迹，不是 1880 张独立图片。', '', '| 关联 | 混合缓存时长的 Spearman | 控制目标步骤与缓存时长后的秩残差相关 |', '|---|---:|---:|']
    for key,label in [('noise_mse','特征漂移 vs CFG 噪声 MSE'),('step_mse','特征漂移 vs DDIM 更新 MSE')]:
        a=diagnosis['associations'][key]
        lines.append(f'| {label} | {a["overall_spearman"]:.3f} | {a["rank_correlation_after_step_and_age_fixed_effects"]:.3f} |')
    values=diagnosis['associations']['noise_mse']['per_age_spearman'].values()
    lines += ['',f'固定缓存时长后，特征漂移与噪声误差的相关系数约为 {min(values):.3f}–{max(values):.3f}。特征漂移有信息，但不等于噪声误差，也没有包含 scheduler 对各阶段误差的不同敏感性。这里描述的是相关性，不作因果或独立样本显著性推断。', '', '## 新的校准方式','', '1. 在无缓存参考轨迹上保存输入和完整噪声预测。', '2. 在步骤 a 完整运行并保存缓存；在后面最多 5 步的原参考输入上复用该缓存，测量 CFG 后的噪声预测 MSE。', '3. 对 eta=0、epsilon prediction、无 clipping 的 DDIM，利用实际 scheduler 的系数，将噪声误差换算为单步 latent 更新误差。该系数已用 scheduler.step 的数值输出做独立测试。', '4. 分别以噪声 MSE、单步更新 MSE 构建区间代价，动态规划选择 17 个刷新位置，最大间隔为 6。',f'5. 两个新候选仅按 8 条验证提示词的 LPIPS 选择，本轮选中 **{objective}**。之后冻结设置再生成测试集。', '', '所有额外完整计算仅用于离线校准，正式生成仍查固定刷新列表。局部探测使用参考 latent，所以仍未解决完整缓存轨迹中误差累积造成的分布偏移。', '',f'诊断、特征采集与求解耗时约 **{diagnosis["seconds_this_invocation"]:.2f} 秒**，另有验证开销；不能称为零成本优化。', '', '## 数据与公平性','', '- 8 条校准、8 条验证、16 条测试提示词；测试两个固定种子，共 32 个配对案例、6 种方法、192 张测试图。', '- 本轮所有图像 ID 与上一轮的校准/验证/测试均不重叠。新旧误差目标在同一批新校准数据上重新构造；不能直接用两轮不同数据集的均值证明算法提升。', '- SD1.5、512×512、FP16、CFG=7.5、batch=1、branch=0，缓存组均 50 个 DDIM 步、17 次完整刷新。', '- 论文式非均匀基线仍是 Eq.7 的有符号实数幂改写及整数化实现，center 在 15/20 中用验证集选取，不声称逐位复现旧代码。', '- DPM-Solver++ 步数由验证集延迟确定；测试方法顺序随机化。延迟每案例测量一次，32 个样本取中位数，未作重复计时性能确认。', '', '## 独立测试结果','', '| 方法 | 耗时中位数/秒 ↓ | 加速比 ↑ | 峰值分配/MiB | CLIP 余弦 ↑ | 对 DDIM50 的 LPIPS ↓ |','|---|---:|---:|---:|---:|---:|']
    for key,s in m.items():
        lines.append(f'| {labels[key]} | {s["median_seconds"]:.3f} | {s["speedup_vs_ddim50"]:.2f}× | {s["max_peak_allocated_mib"]:.0f} | {s["mean_clip_cosine"]:.4f} | {s["mean_lpips_to_ddim50"]:.4f} |')
    lines += ['', '显存为 PyTorch 峰值 allocated；reserved 和原始计时见 summary.json/CSV。CLIP 为 ViT-B/32 原始归一化余弦。LPIPS 对照相同提示词和初始 latent 下的 DDIM50 输出，并非对 COCO 原照片的重建误差。','', '## 配对差异','', '均为“新误差校准减去比较对象”；按 16 个提示词簇 bootstrap，保留同一提示词的两个种子。小规模区间仅供探索。','', '| 比较对象 | LPIPS 差值 [95% 区间] | CLIP 差值 [95% 区间] |','|---|---|---|']
    for key,label in [('uniform','均匀刷新'),('paper_nonuniform','论文式非均匀刷新'),('feature_calibrated','旧特征漂移目标')]:
        d=summary['paired_differences'][f'calibrated_minus_{key}']
        lines.append(f'| {label} | {interval(d["lpips_to_ddim50"])} | {interval(d["clip_cosine"])} |')
    lines += ['', '## 限制与下一步','', '- 改进目标是否有用，应同时看参考一致性、文本对齐、样例与耗时；不根据单一分数宣称全面优越。', '- 未做人工盲评、FID、跨模型和跨 CFG 泛化，16 个测试提示词仍是小规模实验。', '- 这轮测试集在选参前未使用；若继续修改策略，需要新的确认性测试集。', '- 在下一轮大规模生成前，应固定研究假设、主指标和验证流程，并检查局部误差代价能否预测完整生成误差。', '', '## 图表和可复现文件','', '- [代理诊断图](proxy_diagnostics.png)', '- [质量与延迟](quality_latency.png)', '- [刷新位置与误差曲线](refresh_schedules.png)', '- [测试样例 1](comparison_1.jpg) / [2](comparison_2.jpg) / [3](comparison_3.jpg) / [4](comparison_4.jpg)', '- [原始局部探测 CSV](diagnostic_probes.csv)', '- [逐图结果 CSV](evaluated_rows.csv)', '- [诊断汇总](diagnostic_summary.json)', '- [验证集选择](selection.json)', '- [完整测试汇总](summary.json)', '']
    lo,hi=lpips_diff['cluster_bootstrap_95pct_ci']
    if lo<=0<=hi:
        i=lines.index('## 本轮为什么修改目标')
        lines[i:i]=['相对均匀刷新的探索性差异区间仍包含零，暂不能确认稳定优势；相对旧特征目标的 LPIPS 区间不含零，为更换目标提供了初步支持。','']
    # Cases come from the fixed first comparison page, and include regressions.
    rows=json.loads((OUT/'evaluated_rows.json').read_text(encoding='utf-8'))
    lookup={(r['image_id'],r['seed'],r['method']):r for r in rows}
    examples=['## 固定样例观察','', '以下均来自预先固定展示的第一组测试提示词、种子 2026，包含改善与退步。','', '| COCO ID | 观察 |','|---|---|']
    for image_id,label in [(148077,'飞机'),(113588,'学生与电脑'),(193704,'仙人掌旅馆招牌')]:
        a=lookup[(image_id,2026,'calibrated')];u=lookup[(image_id,2026,'uniform')];old_case=lookup[(image_id,2026,'feature_calibrated')]
        examples.append(f'| {image_id}（{label}） | 新/均匀/旧目标 LPIPS：{a["lpips_to_ddim50"]:.3f}/{u["lpips_to_ddim50"]:.3f}/{old_case["lpips_to_ddim50"]:.3f}；新/均匀 CLIP：{a["clip_cosine"]:.3f}/{u["clip_cosine"]:.3f}。 |')
    examples += ['', '飞机样例相对旧目标更接近参考，但均匀方案仍更接近；学生样例的新方案偏差更大；招牌样例的新方案更偏离参考而文本对齐更高。这些例子说明参考一致性、文本对齐和最终视觉判断需要分别分析。','']
    index=lines.index('## 限制与下一步')
    lines[index:index]=examples
    lines.insert(lines.index('- [质量与延迟](quality_latency.png)')+1,'- [配对差异及探索性区间](paired_differences.png)')
    lines += ['- [GPU 功耗/频率采样](gpu_telemetry.csv)','']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':main()
