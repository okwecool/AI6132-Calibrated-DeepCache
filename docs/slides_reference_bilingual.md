# Slides 参考文档｜中文编排指导与英文页内素材

建议按“问题 → 固定预算下的策略 → 独立实验 → 实际效果 → 局限”推进。下面每个编号可以对应一页，长页可以拆开，短展示可以合并。英文要点适合改写后放到页内，中文讲述提示留在演讲者备注。组号和姓名按实际信息填写。

[集中图包](handoff_figures/README.md) · [Report 参考](report_reference_bilingual.md) · [中文交接文档](project_handoff_zh.md)

| 叙事段 | 推荐页 | 听众应带走什么 |
| --- | --- | --- |
| 任务与难点 | 01–03 | 同一商品换景，速度来自减少完整 U-Net 计算 |
| 方法和协议 | 04–06 | 误差校准选定刷新表，同预算和同输入比较 |
| 主要证据 | 07–09 | 两项独立验证、定量收益和真实案例 |
| 边界与结论 | 10–13 | 轮廓加权收益尚未证实，商品重画仍需解决 |

## 01｜封面与一句话问题

**English title**

*Calibrated DeepCache for Stable Diffusion*

**English subtitle**

*Training-free refresh scheduling for text-to-image synthesis and product background replacement*

**中文讲述**

开场报课程、小组和项目名，然后提出核心问题。给定相同的完整 U-Net 刷新次数，哪些去噪步最值得完整计算？封面可配一张高跟鞋局部图，下一页再完整展示任务。

**配图** [商品应用拼图](handoff_figures/ready_to_use/fig04_application_cases.jpg)

## 02｜任务定义与场景动机

**English title**

*Replace the background while keeping the product*

**English slide copy**

- Inputs: product photo, foreground mask, and background prompt.
- A pretrained inpainting model generates the editable region.
- Compositing preserves the prepared product pixels inside the mask.

**中文讲述**

展示源图、白色前景掩码和生成图。黑靴可直观呈现从陈列台到简洁背景的变化。白色区域表示商品保护，传给 Inpainting 的编辑掩码取反。先建立任务概念，再讲为什么要看商品边缘。

**配图** [M2 成功与失败对照](handoff_figures/existing/product_background/m2/application_good_bad.jpg)，可裁上半行作动机。

## 03｜为什么选刷新位置

**English title**

*Cache reuse and fewer sampling steps are different levers*

**English slide copy**

- Uncached DDIM50 runs the full U-Net at all 50 steps.
- DeepCache reuses deep features between full refreshes.
- At budget 17, both cache policies use 17 full and 33 partial calls.
- DPM-Solver++ uses 20 full calls as a latency-matched comparison.

**中文讲述**

这一页用一条时间线讲清“50 步”和“17 次完整刷新”。均匀方法每 3 步刷新，校准方法重新分配完整计算。DPM-Solver++ 是另一类加速，验证中选定了 20 步，实际延迟仍由实测给出。

**配图** [刷新位置原图](handoff_figures/existing/text_to_image/confirmation/refresh_schedules.png)。页内可只展示均匀与校准两行。

## 04｜离线校准与在线推理

**English title**

*Calibrate refreshes from DDIM update errors*

**English slide copy**

1. Record the uncached DDIM trajectory.
2. Probe stale-cache noise predictions at the same inputs.
3. Weight noise MSE by the squared DDIM update coefficient.
4. Optimize a fixed-budget schedule with dynamic programming.

**中文讲述**

从离线开始讲。固定同一时刻的 latent，比较完整噪声预测和复用噪声预测。DDIM 单步更新对噪声误差有不同的敏感度，因此用该系数的平方调整 MSE。动态规划选择 $B$ 次刷新。正式推理直接读取刷新列表。

**配图** [Fig. 1 方法流程](handoff_figures/ready_to_use/fig01_method_overview.png)

## 05｜数据划分和冻结协议

**English title**

*Freeze the schedule before testing on new products*

**English slide copy**

- Finegrain split: 4 pilot / 8 calibration / 8 validation / 24 test products.
- Hold out complete product identities.
- Use two seeds and one background prompt per test product.
- 24 × 2 × 4 = 192 primary generations.

**中文讲述**

先解释为什么按商品身份拆分。同件商品的两个种子是重复条件，统计时先合并成一个商品级配对。Fig. 2 可同时交代场景分布和前景比例。如果时间短，只留左图和 192 次生成的公式。

**配图** [Fig. 2 数据分布](handoff_figures/ready_to_use/fig02_data_overview.png)

## 06｜评价口径与公平对照

**English title**

*Compare matched inputs and report measured latency*

**English slide copy**

- Matched photo, mask, prompt, and initial noise.
- Boundary ROI LPIPS against the matching uncached DDIM50 output.
- Eight fixed 128 × 128 boundary crops per product.
- Product-level paired bootstrap over 24 held-out identities.

**中文讲述**

LPIPS 数值小表示更接近无缓存参考。把“参考图”和“输入的真实商品图”区分开。耗时在 RTX 5060 Laptop 8GB 上测量，512×512、FP16、batch 1。逐图耗时包含条件编码、采样、解码及合成，离线校准开销单列。

**配图** [M3 原始评价协议](../results/product_background/m3/report.md)，可据此画一个带掩码轮廓的局部裁块示意。

## 07｜文生图辅助结果

**English title**

*Text-to-image confirmation on 100 held-out captions*

**English slide copy**

- COCO 2014 captions, one fixed seed, five methods.
- Uniform vs calibrated full-image LPIPS: 0.19376 → 0.16728.
- 13.66% lower reference deviation at 17 full refreshes.
- 2.37× speedup over uncached DDIM50.

**中文讲述**

这页说明调度方法先在较通用的任务上得到验证。COCO 在这里只提供 caption。主差值为 −0.02647，95% 区间 [−0.03477, −0.01785]，100 条里有 77 条改善。报告可完整列五种方法，展示页突出均匀与校准对照。

**配图** [文生图质量与耗时](handoff_figures/existing/text_to_image/confirmation/quality_latency.png) · [配对差异区间](handoff_figures/existing/text_to_image/confirmation/paired_differences.png)

## 08｜商品背景替换主结果

**English title**

*Lower boundary deviation at the same cache budget*

**English slide copy**

- Boundary ROI LPIPS: 0.03273 → 0.02827.
- 13.6% lower reference deviation, with improvement on 19 of 24 products.
- Paired change: −0.00446, 95% CI [−0.00882, −0.00093].
- 2.397 s per image, a 2.31× speedup over uncached DDIM50.

**中文讲述**

左图讲速度。均匀 2.399 秒、校准 2.397 秒，两者接近，5.531 秒是无缓存分母。右图讲轮廓保真。DPM 20 步为 2.322 秒，略快，但在这个参考指标上偏差更高。给置信区间留空间，避免把 13.6% 表述为主观质量提升。

**配图** [Fig. 3 商品主结果](handoff_figures/ready_to_use/fig03_m3_main_results.png) · [逐商品误差图](handoff_figures/existing/product_background/m3/quality_latency.png)

## 09｜成功与失败

**English title**

*Reference fidelity and editing quality tell different stories*

**English slide copy**

- Boots and heeled shoes show readable product details after background replacement.
- The printed vest leaks texture into the generated background.
- The striped shorts acquire extra legs outside the mask in both uncached and cached outputs.

**中文讲述**

先指出上排成功图的商品与背景关系，再指出下排缺陷的位置。说明这些是目视选例，左组来自 M2 验证，右组来自 M3 独立测试。保留商品掩码内像素仍可能在外侧生成新的主体部件。

**配图** [Fig. 4 四案例拼图](handoff_figures/ready_to_use/fig04_application_cases.jpg)。需要展示均匀策略时用 [M3 完整对照](handoff_figures/existing/product_background/m3/application_good_bad.jpg)。

## 10｜空间加权与预算消融

**English title**

*Boundary weighting changes the schedule at budget 21*

**English slide copy**

- At budget 17, all four objectives produced the same schedule.
- At budget 21, global and boundary-weighted schedules differed in early steps.
- ROI LPIPS: uniform 0.02599, global 0.01934, boundary-weighted 0.02015.
- Boundary minus global: +0.00081, 95% CI [−0.00051, +0.00237].

**中文讲述**

先讲设计变化，再讲结果。B=17 的收益来自选定刷新表。B=21 可以直接比较不同目标，但轮廓加权相对全图的区间跨零。该批输入复用 M3，标为探索性消融。早期刷新位置的差异和代理代价略微变化，并未传成稳定的最终收益。

**配图** [B=21 刷新表和配对差异](handoff_figures/existing/product_background/ablation21/ablation_summary.png)

## 11｜应用修复尝试

**English title**

*A second pass can remove small artifacts, with seed-dependent results*

**English slide copy**

- Manually annotate an erroneous region outside the product.
- Compare ordinary local inpainting with a hidden product cue.
- Both preserve pixels outside the annotated background region.
- The added cup handle disappears for one seed. Larger shape errors persist.

**中文讲述**

四件已知失败商品，各两个种子，两种方法，共 16 张。普通局部修复和隐藏前景条件都只写回标注区。演示杯把成功和短裤失败。二次推理额外约 3.4–3.7 秒，修复结果属于开发诊断。

**配图** [Fig. 5 修复拼图](handoff_figures/ready_to_use/fig05_repair_cases.jpg) · [两个种子的完整小试](handoff_figures/existing/product_background/repair_pilot/)

## 12｜局限和可扩展方向

**English title**

*What the current experiments support*

**English slide copy**

- Stronger fidelity to uncached DDIM at the same refresh budget.
- A single Stable Diffusion model family and hardware setting.
- Manual product masks and limited outdoor coverage.
- Human A/B ratings and background-structure control as future work.

**中文讲述**

说明当前证据范围。M3 的 CLIP 比较区间跨零，盲评材料准备完成，人工结果尚未回收。后续可把背景结构作为约束来减少掩码外形状扩张，也可以验证局部误差与实际去噪轨迹之间的联系。

**配图** [盲评组织说明](../results/product_background/blind_review/README.md)作为备查。展示时可以继续保留短裤例图。

## 13｜总结与演示入口

**English title**

*Spend the same refresh budget more effectively*

**English slide copy**

- Offline DDIM-aware scheduling with unchanged pretrained weights.
- 13.6% lower boundary reference deviation on 24 held-out products.
- Approximately 2.40 seconds per image at 512 × 512.
- Separate the scheduling benefit from inpainting failure modes.

**中文讲述**

回扣开场，刷新的位置比单纯的刷新次数多提供一个优化变量。用一句话说方法、一句说独立证据、一句说应用限制。如果安排实时演示，提前打开本地 gallery，切换商品、种子和方法。

**配图与演示** [Fig. 3 主结果](handoff_figures/ready_to_use/fig03_m3_main_results.png) · [离线演示页](../results/product_background/m3/gallery.html)。演示页依赖原实验本地 `images/`，换机器时可用图包代替。

## 附录备选与问答

| 备份页 | 页内英文标题 | 适合回答的问题 |
| --- | --- | --- |
| A1 | *DDIM sensitivity and the dynamic program* | 为什么误差要乘 DDIM 系数，DP 精确优化的是什么 |
| A2 | *Reference fidelity across refresh budgets* | 文生图 B=10/17/25 如何变化，为什么 16 条子集与 100 条主表分开 |
| A3 | *Best and worst paired changes* | M3 最大改善与退步分别是什么，图中种子如何选 |
| A4 | *Timing and reproducibility checks* | 设备、计时范围、M3 复测、显存和校准成本 |
| A5 | *Sources and team contributions* | 已有方法归因，数据来源和各成员实际交付 |

- **DP 会训练模型吗？** 只在离线用测量误差求刷新列表，模型权重保持固定。
- **17 次刷新等于 17 个去噪步吗？** 缓存仍走 50 个 DDIM 步，17 次完整计算和 33 次部分计算。
- **LPIPS 小是否说明图更好看？** 本实验的 LPIPS 衡量相对同输入无缓存参考的接近程度。短裤失败例说明应用质量需要单独判断。
- **为什么没有采用轮廓加权作为主要贡献？** B=17 得到同一刷新表。B=21 得到了不同表，额外收益的配对区间跨零。
- **2.31 倍速度和 DPM 的关系是什么？** 2.31 倍以无缓存 DDIM50 为分母。20 步 DPM 在本实验略快，见 M3 主表。

[动态规划和公式](report_reference_bilingual.md#23-固定预算动态规划) · [文生图预算图](handoff_figures/existing/text_to_image/confirmation/budget_ablation.png) · [M3 配对极值图](handoff_figures/existing/product_background/m3/paired_extremes.jpg) · [引用来源](report_reference_bilingual.md#6-参考文献与核对)

## 拆页与分工

- 时间较短时，合并 01–02、03–04、05–06，把 07 放附录，并在 09 和 10 各用一张图。08 保持完整。
- 时间充足时，把 09 拆为成功和失败两页，把 11 拆为方案和案例两页。每页只留一个核心结论。
- 可按任务与数据、方法、定量实验、案例与局限四段组织交接。最终展示角色、作者顺序及贡献按实际分工填写。
- 图片先从 `ready_to_use/` 选。若要显示第三种方法、第二个种子或完整样本，再进 `existing/` 取原图。
