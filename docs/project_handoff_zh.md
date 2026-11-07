# Calibrated DeepCache 项目简介与交接

本文面向接手写课程报告、制作展示和整理代码包的同学。实验交接基线为本地 Git 提交 `da42ae2`。建议先看本页的证据表，再按报告或展示参考文档选图。

| 交接材料 | 用途 |
| --- | --- |
| [集中图包](handoff_figures/README.md) | 可直接插入报告或展示的图，另附完整图索引和来源 |
| [中英文 Report 参考](report_reference_bilingual.md) | 中文写作指导、英文段落示例、方法公式、结果表和图注 |
| [中英文 Slides 参考](slides_reference_bilingual.md) | 按页给出的标题、要点、配图和讲述提示 |
| [商品背景替换设计](product_background_replacement_design.md) | 原始实验方案、分阶段验收和后续消融 |
| [仓库 README](../README.md) | 环境与各阶段执行命令 |

## 项目在做什么

课程题目是 **Calibrated DeepCache for Stable Diffusion**。实验首先研究文生图加速，再把同一思想用于商品背景替换。应用的输入为商品照片、前景掩码和背景描述，生成后按掩码合成，让商品原有像素保留在图中。底模分别为 Stable Diffusion 1.5 文生图和 SD 1.5 Inpainting，均使用预训练权重。

扩散模型要多次运行 U-Net。DeepCache 在部分去噪步复用深层特征，其余步骤执行完整计算并刷新缓存。本项目在固定完整刷新次数下，从无缓存参考轨迹采集复用造成的噪声预测误差，将它换算为 DDIM 单步更新误差，再用动态规划选择刷新位置。校准在推理前完成，正式生成只读取冻结的刷新表。50 个去噪步中 17 次完整刷新，意味着其余 33 步仍执行部分 U-Net。

DeepCache 的原始缓存机制和论文式非均匀方案来自已有工作。小组自己的工作是误差校准、固定预算调度、商品背景替换适配，以及两条实验线的复现与分析。按报告主线区分三类证据：

1. **独立验证**回答固定校准表在新样本上能否比均匀缓存更贴近无缓存参考。
2. **方法消融**分析区域权重和刷新预算如何改变策略。
3. **应用案例**展示换景是否自然，以及为何商品掩码外仍可能被补画出额外部件。

## 最值得交接的结果

| 阶段 | 样本与对照 | 结果与解释 | 原始记录 |
| --- | --- | --- | --- |
| 文生图确认 | 100 条新 COCO 2014 caption，单种子，5 种方法，共 500 张主测试图 | B=17 时，平均全图 LPIPS 从 0.19376 降到 0.16728，下降 13.66%。相对无缓存 DDIM50 约 2.37 倍加速 | [确认报告](../results/confirmation/report.md) |
| 真实照片先导 | 4 件商品，两个种子，两种提示词，两种缓存配置，共 32 张 | 只写背景与点名商品的效果随样本变化。点名条纹杯时，两组都补出把手 | [先导报告](../results/product_background/photo_pilot/report.md) |
| 商品 M2 | 8 件校准、8 件验证，独立于先导 | 全图、编辑区和两种轮廓权重在 B=17 时得到同一刷新表 | [校准报告](../results/product_background/m2/report.md) |
| 商品 M3 | 24 件新商品，各一条背景提示词、两个种子、四种方法，共 192 次正式生成 | 轮廓 ROI LPIPS 从 0.03273 降至 0.02827，下降 13.6%，24 件中有 19 件改善。两种缓存约 2.40 秒/图，相对无缓存约 2.31 倍加速 | [独立测试](../results/product_background/m3/report.md) |
| B=21 消融 | 复用 M3 的商品、种子和参考图，三种缓存表，144 张新图 | 全图校准 0.01934，轮廓加权 0.02015，均匀刷新 0.02599。轮廓加权相对全图校准的配对区间跨零 | [预算消融](../results/product_background/ablation21/report.md) |
| 局部修复小试 | 4 个已知失败，两种修复、两个种子，共 16 张 | 所有结果均保持商品和标注外像素。杯把有成功去除的案例，其他缺陷或另一种子仍出现残余与新伪影 | [修复报告](../results/product_background/repair_pilot/report.md) |

文生图的 13.66% 来自**全图** LPIPS，商品任务的 13.6% 来自**轮廓裁块** LPIPS。两个值各自在同一任务内比较。LPIPS 在这里度量与同输入无缓存生成图的距离。M3 的人类偏好结果尚未收集，CLIP 的主方法比较区间跨零。商品任务中，保持掩码内像素来自硬合成，背景仍可出现长裤腿、额外杯把或纹理外溢。

## 代码与文件怎么找

| 路径 | 职责 | 接手时优先看哪里 |
| --- | --- | --- |
| [src/cache_policy.py](../src/cache_policy.py) | DeepCache 封装、均匀和论文式非均匀刷新、动态规划 | `ScheduledDeepCache` 和 `calibrated_schedule` |
| [src/error_calibration.py](../src/error_calibration.py) | CFG 合并、DDIM 更新系数、区间代价 | `ddim_epsilon_coefficient` |
| [src/region_cost.py](../src/region_cost.py) | 编辑区和外侧轮廓权重、轮廓裁块 | `normalized_region_weights` 和 `boundary_crops` |
| [src/background_task.py](../src/background_task.py) | 等比例预处理、掩码语义与前景合成 | 白色前景掩码表示保护，传入 Inpainting 的编辑掩码取反 |
| [src/background_repair.py](../src/background_repair.py) | 约束二次重绘区域、隐藏前景条件与合成 | 修复只写回标注区与背景的交集 |
| [scripts/run_background.py](../scripts/run_background.py) | 商品背景替换单图入口、M1 GPU 验证和计时 | `generate` 子命令与 `BackgroundRunner` |
| [scripts/run_background_m2.py](../scripts/run_background_m2.py) | 收集参考轨迹、构造代价、验证和选择策略 | 235 次复用探测/商品，8 件共 1,880 次 |
| [scripts/run_background_m3.py](../scripts/run_background_m3.py) 与 [评分脚本](../scripts/evaluate_background_m3.py) | 固定策略、独立生成、评价、图集和离线演示 | 先读清单，再查看 `setup.json` 和 `summary.json` |
| [scripts/run_background_ablation21.py](../scripts/run_background_ablation21.py) 与 [评分脚本](../scripts/evaluate_background_ablation21.py) | B=21 配对消融 | 复用 M2 代价和 M3 原图 |
| [scripts/run_background_repair_pilot.py](../scripts/run_background_repair_pilot.py) | 手工标注错误区域的二次重绘 | 四件商品及坐标见对应清单 |
| [scripts/background_blind_review.py](../scripts/background_blind_review.py) | 离线匿名 A/B 表单和评分汇总 | [组织者说明](../results/product_background/blind_review/README.md) |
| [scripts/run_round2.py](../scripts/run_round2.py) 与 [scripts/run_confirmation.py](../scripts/run_confirmation.py) | 文生图误差校准、冻结策略确认 | 原始证据在 `results/round2/` 和 `results/confirmation/` |
| [manifests/](../manifests/) | 数据来源、图像身份、划分、提示词和种子 | 结果复核以实际清单为准 |
| [tests/](../tests/) | 刷新表、DDIM 系数、配对统计、空间权重和像素保护 | 交接基线通过 11 项测试 |

## 如何读取结果

- **写数值时**优先读 `summary.json`、`scores.csv` 或早期的 `evaluated_rows.csv`，从 `report.md` 了解协议、计算口径和局限。
- **核对结果来源时**查看 `setup.json`、`protocol.json`、`source_hashes.json`、刷新日志和运行配置。M2 的 `aggregate_costs.npz` 是复用冻结策略的紧凑产物。
- **找插图时**进入[集中图包](handoff_figures/README.md)。固定顺序图集可以展示覆盖情况，极值图和人工选例要在图注中标明选择规则。
- **查看所有生成图时**进入原实验的 `images/` 目录。原始图、模型、原始数据和本地环境通常由 Git 忽略。克隆后可以直接查看已版本化的统计和汇总图，单张原图和离线演示页所需素材需要从本机实验目录另行取得或重建。

## 接手后的最低运行步骤

本机用 Windows、Python 3.10.20、CUDA PyTorch 2.10.0+cu128、RTX 5060 Laptop 8GB。仓库原有 `.venv` 依赖创建它的机器环境。换机器时先准备可用的 CUDA PyTorch，再安装 [requirements.txt](../requirements.txt)。需要重新提取 Finegrain Parquet 时安装 PyArrow。首次准备资产需要联网，正式推理用本地权重和数据。

在项目根目录用 PowerShell 验证：

```powershell
& '.\.venv\Scripts\python.exe' -m unittest discover -s tests -v
& '.\.venv\Scripts\python.exe' scripts/prepare_background_assets.py
& '.\.venv\Scripts\python.exe' scripts/run_background.py smoke --output results/product_background/handoff_smoke
```

复现实验时先选择目标阶段，再依 [README](../README.md) 的命令执行。商品 M3 复用已版本化的 M2 冻结产物，从原始照片开始生成时先跑 `run_background_m3.py --prepare`，然后向生成与评价命令传入同一新的 `--output` 目录。B=21 和修复小试需要原阶段保存的无缓存输出及掩码。另开输出目录可以让已有指标、图集和报告保持可追溯。评估脚本需要的 CLIP 和 LPIPS/AlexNet 资产由文本实验资产准备流程建立。

## 报告接手清单

1. 以 M3 为应用主结果，文生图 100 条确认作为辅助证据。先解释刷新次数、延迟、参考误差三个概念，再写实验。
2. 用 B=21 和修复小试呈现方法边界。人工评分仍是待收集状态，可以把盲评作为后续工作。
3. 从 [Report 参考](report_reference_bilingual.md) 取英文正文材料，从 [Slides 参考](slides_reference_bilingual.md) 按讲述顺序拆页。图题、作者、小组号和贡献按实际情况填写。
4. 课程说明要求至少 4 页技术报告，包含 Introduction、Methods、Experiments and Analysis、Conclusion。提交文件为 `group_XX_report.pdf`、`group_XX_contribution.pdf` 和 `group_XX_code.zip`。截止日期为 2026 年 11 月 16 日，入口在 NTULearn。按要求排除数据、模型及外链替代文件。[课程说明，第 4–8 页](course%20project%20technical%20review.pdf)
5. GitHub 远程用于团队存档。新的 Git 提交按 [AGENTS.md](../AGENTS.md) 走审批，课程提交按正式附件规格整理。
