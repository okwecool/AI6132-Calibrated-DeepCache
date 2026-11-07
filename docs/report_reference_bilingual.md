# Report 参考文档｜中文写作指导与英文素材

面向后续撰写 AI6132 技术报告的同学。中文部分给出段落目标、证据和写作顺序。英文部分可按组内实际贡献、图号和篇幅改写。交接数据和命令见[中文交接文档](project_handoff_zh.md)，插图见[集中图包](handoff_figures/README.md)。

## 0. 标题与结构

推荐标题 **Calibrated DeepCache for Stable Diffusion and Product Background Replacement**。

课程结构为 Introduction、Methods、Experiments and Analysis、Conclusion，报告至少 4 页。篇幅紧张时，给方法和实验留下最多空间，把细节图、完整刷新序列和逐图指标放进附录。作者、小组号、分工和图号由同学按实际情况填写。[课程要求，第 4–8 页](course%20project%20technical%20review.pdf)

| 部分 | 要回答的问题 | 建议材料 |
| --- | --- | --- |
| Introduction | 为什么要快，为什么固定计算预算下还需要选刷新位置 | 文生图与商品换景的任务定义，已有方法与项目贡献 |
| Methods | 缓存什么，如何量化代价，如何选择刷新 | Fig. 1，单步误差和动态规划公式 |
| Experiments and Analysis | 对哪些数据、方法和指标得出什么结论 | Fig. 2–4，M3 主表、文生图确认、B=21 消融 |
| Conclusion | 哪项收益获得了支持，应用与方法还缺什么 | 主结果、形状失败、区域权重的证据边界 |

**可选 Abstract 英文素材**

> We study how to allocate full U-Net refreshes when accelerating Stable Diffusion with DeepCache. We measure stale-cache prediction errors along uncached DDIM trajectories and use a dynamic program to select a fixed refresh schedule. On 100 held-out COCO captions, the calibrated schedule reduced full-image LPIPS to the uncached reference by 13.66% relative to uniform caching at the same 17-refresh budget. On 24 held-out products with two seeds each, it reduced boundary-region LPIPS by 13.6% at approximately 2.40 seconds per image, giving a 2.31× speedup over uncached DDIM50. A 21-refresh ablation produced different global and boundary-weighted schedules, with no clear added benefit from boundary weighting. Qualitative failures include new product parts generated outside the foreground mask.

## 1. Introduction

先从重复去噪的成本切入。用商品输入、前景掩码和背景提示词解释实际任务，再提出“在同样的 17 次完整刷新下，刷新位置是否影响参考保真”。技术综述部分可以把 DeepCache 与少步 DPM-Solver++ 放在一起，分别说明降低每步计算和减少采样步数这两种加速方式。

**英文段落素材**

> Stable Diffusion repeatedly evaluates a denoising U-Net. DeepCache [1] reuses deep U-Net features across adjacent steps, while a fast solver such as DPM-Solver++ [3] reduces the number of denoising steps. Our study asks how the placement of full refreshes affects fidelity under a fixed cache budget. We first use text-to-image synthesis as a controlled experiment, then evaluate the approach in product background replacement, where visible errors near a product boundary can undermine the edit.
>
> We contribute an offline, DDIM-aware refresh calibration procedure, a controlled evaluation with disjoint calibration and test inputs, and an analysis of where reference fidelity and practical editing quality diverge. Our inpainting pipeline uses a supplied foreground mask and preserves the original product pixels through compositing.

写贡献时把 DeepCache 原机制、预训练模型和 DPM-Solver++ 归给原作者。项目实现的是刷新调度及其评估。初轮特征漂移策略未稳定优于均匀刷新，后续采用实际预测误差，可作为方法迭代动机。[第二轮原报告](../results/round2/report.md)

## 2. Methods

### 2.1 先说明三种计数

| 术语 | 报告中如何使用 |
| --- | --- |
| Denoising step | 采样器的一步。DDIM 两个缓存方案都运行 50 步 |
| Full refresh | 运行完整 U-Net 并更新缓存。主结果预算为 17，另有 33 次部分调用 |
| Pipeline latency | 实际耗时。从已预处理的输入开始，包含条件编码、采样、解码，商品任务还包括合成 |

**英文段落素材**

> On refresh steps, we run the full U-Net and store deep features. Between refreshes, we reuse the stored deep features and recompute the uncached layers. The uniform and calibrated policies use 50 DDIM steps with exactly 17 full refreshes. Both policies keep the same model weights and cache branch.

[Fig. 1 方法流程](handoff_figures/ready_to_use/fig01_method_overview.png)

### 2.2 误差探测与 DDIM 更新

用去噪迭代索引 $i$ 和对应扩散时间 $t$ 区分刷新列表的位置与噪声水平。无缓存轨迹上的同一输入分别运行完整预测与复用预测。在第 $a$ 步刷新，再探测后续第 $i$ 步。在本实验的确定性 DDIM、epsilon prediction、关闭剪裁设置下，记采样前后累计 alpha 为 $\bar\alpha_t$ 与 $\bar\alpha_{t\prime}$。

$$
x_{t'} = \sqrt{\frac{\bar\alpha_{t'}}{\bar\alpha_t}}x_t
        + c_i\,\hat\epsilon_i,\qquad
c_i = \sqrt{1-\bar\alpha_{t'}} -
      \sqrt{\frac{\bar\alpha_{t'}}{\bar\alpha_t}}\sqrt{1-\bar\alpha_t}.
$$

因此相同输入上的单步更新均方误差可写成 $e(a,i)=c_i^2\,\operatorname{MSE}(\hat\epsilon_i^{\mathrm{cached}},\hat\epsilon_i^{\mathrm{full}})$。两项预测都采用实际 CFG 合并。区间代价是 $C(a,b)=\sum_{i=a}^{b-1}e(a,i)$，在校准商品或提示词上取平均。

**英文段落素材**

> We compare cached and full classifier-free guided noise predictions at identical inputs on the uncached trajectory. Under the deterministic DDIM update [2], a change in the noise prediction changes the next latent by a known scalar coefficient. We use its squared value to weight the prediction MSE. This captures local sensitivity of the DDIM update. Error accumulation on a complete cached rollout is measured separately through generated outputs.

### 2.3 固定预算动态规划

把 $T=50$ 步划分为 $B$ 个连续区间。每段的起点执行完整刷新，段长最多为 6，第一段从第 0 步开始。用 $D[k,b]$ 表示前 $b$ 步分为 $k$ 段的最小代价。

$$
D[k,b]=\min_{\max(0,b-6)\le a < b}
\bigl(D[k-1,a]+C(a,b)\bigr),\qquad D[0,0]=0.
$$

其余不可达状态设为无穷大。回溯得到刷新位置。在线推理直接读取列表。DP 的最优性对应可加代理代价，最终参考误差由后续实验验证。

**英文段落素材**

> The dynamic program selects the lowest-cost partition of the denoising steps under an exact refresh budget and a maximum interval length of six. The first step refreshes the cache. Calibration runs offline, and test-time inference reads a frozen schedule.

### 2.4 商品编辑与区域权重

前景 $F=1$ 代表保护商品，Inpainting 的编辑掩码是 $E=1-F$。9 通道 Inpainting U-Net 接收噪声 latent、编辑掩码和被遮罩图像的 latent。合成将掩码内商品像素贴回。候选区域目标在像素网格上构造 $W=E+\lambda R$，其中 $R\subseteq E$ 是商品外侧 16 像素的轮廓带。先把权重按面积投影到 latent 网格，再归一化并乘上逐位置误差。

**英文段落素材**

> We compare a global error objective with an edit-region objective and two boundary-weighted variants. Boundary weights emphasize a 16-pixel band immediately outside the foreground mask. We average pixel weights into latent cells before normalization. The final composite restores the prepared source pixels inside the foreground mask, while the denoiser may still generate new structures in the editable region.

## 3. Experiments and Analysis

### 3.1 数据、对照和指标

写清下面这些实际运行的设定。计划中的 32 件商品没有进入最终实验，报告使用实际 24 件。数据图中的 scene 是提示词场景类别。

| 项目 | 文生图确认 | 商品 M3 |
| --- | --- | --- |
| 数据 | COCO 2014 [7] caption，只使用提示词 | Finegrain Product Masks Lite [5] 照片和前景掩码 |
| 独立测试 | 100 条新提示词，种子 2026 | 24 个商品身份，各一条背景描述，种子 2026 和 6132 |
| 校准/验证来源 | 第二轮 8/8 条提示词 | M2 的 8/8 件商品，另有 4 件真实照片先导 |
| 主方法 | DDIM50、均匀 17、论文式非均匀 17、校准 17、DPM++20 | DDIM50、均匀 17、校准 17、DPM++20 |
| 主指标 | 同输入无缓存 DDIM50 的全图 LPIPS | 同输入无缓存 DDIM50 的轮廓裁块 LPIPS |
| 配对单位 | 提示词 | 商品身份，先合并该商品的两个种子 |

主流程为 RTX 5060 Laptop 8GB、512×512、FP16、batch 1、CFG 7.5、缓存分支 0。DPM 步数在验证阶段按延迟确定为 20。M3 每件商品使用 8 个固定的 128×128 轮廓裁块。两个主比较各自使用 5,000 次按独立单位配对的 bootstrap。M3 有 192 次正式生成和 8 次额外计时复测，复测逐像素一致。CLIP [9] 图文余弦为辅助指标。

**英文段落素材**

> We freeze each schedule before held-out inference and pair methods by prompt and initial noise. Product identities are separated across pilot, calibration, validation, and test phases. For the inpainting study, 24 held-out products are evaluated using one background prompt and two seeds each. We average over seeds within each product before paired bootstrapping. The primary editing metric is LPIPS [4] on fixed boundary crops relative to the same-input uncached DDIM50 output.

[Fig. 2 数据分布](handoff_figures/ready_to_use/fig02_data_overview.png) · [M3 逐图分数](../results/product_background/m3/scores.csv)

### 3.2 先报告文生图辅助实验，再报告商品主结果

| 文生图确认方法 | 全图 LPIPS ↓ | 中位耗时（秒） |
| --- | ---: | ---: |
| 无缓存 DDIM50 | 0.00000 | 5.463 |
| 均匀 DeepCache | 0.19376 | 2.299 |
| 论文式非均匀 | 0.18612 | 2.302 |
| 校准 DeepCache | **0.16728** | 2.300 |
| DPM-Solver++ 20 | 0.19112 | 2.265 |

主配对差为 −0.02647，95% 区间 [−0.03477, −0.01785]。100 条中 77 条改善。论文式非均匀方案是依据原论文公式的实现适配，正文应说明这一点。额外的 B=10/17/25 消融来自同一预定 16 提示词子集，与 100 条主表分开解释。[文生图原始报告](../results/confirmation/report.md) · [文生图结果图](handoff_figures/existing/text_to_image/confirmation/quality_latency.png)

| 商品 M3 方法 | 轮廓 ROI LPIPS ↓ | 全图 LPIPS ↓ | CLIP 余弦 ↑ | 中位耗时（秒） | 峰值已分配显存（MiB） |
| --- | ---: | ---: | ---: | ---: | ---: |
| 无缓存 DDIM50 | 0.00000 | 0.00000 | 0.26175 | 5.531 | 2680 |
| 均匀 DeepCache | 0.03273 | 0.06833 | 0.25945 | 2.399 | 2796 |
| 校准 DeepCache | **0.02827** | **0.05931** | 0.25961 | 2.397 | 2796 |
| DPM-Solver++ 20 | 0.03666 | 0.07452 | 0.26014 | 2.322 | 2680 |

**英文段落素材**

> The calibrated schedule reduced full-image LPIPS to the uncached reference from 0.19376 to 0.16728 on the held-out COCO captions. On held-out products, boundary ROI LPIPS fell from 0.03273 to 0.02827, a 13.6% relative reduction. The paired mean difference was −0.00446 with a 95% product-bootstrap interval of [−0.00882, −0.00093]. The calibrated cache ran in 2.397 s per image, compared with 2.399 s for uniform caching and 5.531 s for uncached DDIM50. The 20-step DPM-Solver++ baseline ran slightly faster and had higher boundary deviation from the DDIM50 reference.

数值解释建议写明 19/24 件商品改善，CLIP 差值区间跨零。参照方法与自身的 LPIPS 为零，由参照定义决定。校准成本约 106.6 秒，单独计在商品 M2；图中耗时为逐图推理。DPM 次要对照按实测时延解读。

[Fig. 3 主结果](handoff_figures/ready_to_use/fig03_m3_main_results.png) · [M3 汇总 JSON](../results/product_background/m3/summary.json) · [M3 原始报告](../results/product_background/m3/report.md)

### 3.3 消融与失败分析

这部分用三个层次写，先说刷新表是否变化，再说参考误差，最后说应用图像：

1. **B=17 区域目标**。全图、编辑区、λ=1、λ=3 的刷新表相同，验证输出逐像素一致。M3 支持的是选定刷新表相对均匀表的收益。
2. **B=21 探索性消融**。在同一批 M3 输入上，均匀/全图/轮廓的 ROI LPIPS 分别为 0.02599/0.01934/0.02015。轮廓减全图为 +0.00081，95% 区间 [−0.00051, +0.00237]。两种校准表从第 24 步起一致。M2 的轮廓代理代价略有下降，完整生成结果没有显示出稳定额外收益。[B=21 消融图](handoff_figures/existing/product_background/ablation21/ablation_summary.png)
3. **应用效果**。高跟鞋成功换到摄影棚台座，短裤在无缓存和缓存输出都被补成长裤。黑靴和印花背心来自 M2 验证。极值图按参考误差选择，图中印花盒和蓝椅用于解释指标为何受新背景物件影响。局部修复小试在一个种子去除了杯把，同时在短裤等案例留下新结构。[Fig. 4 案例](handoff_figures/ready_to_use/fig04_application_cases.jpg) · [Fig. 5 修复小试](handoff_figures/ready_to_use/fig05_repair_cases.jpg)

**英文段落素材**

> At budget 17, the four calibration objectives produced the same schedule. At budget 21, boundary weighting changed early refresh positions, but the paired boundary LPIPS interval relative to global calibration crossed zero. This exploratory comparison reused the M3 inputs. The qualitative examples reveal a separate limitation: uncached and cached inpainting both extend the striped shorts outside the supplied mask. The local repair pilot removed an invented cup handle for one seed, while larger shape errors and replacement artifacts persisted.

### 3.4 局限与继续研究

中文写作顺序建议按“评价范围、方法范围、应用风险”组织。说明独立测试只覆盖固定底模、分辨率、采样设定和筛选后的商品分布，室外只有两件。硬合成保护像素，真实商品的整体形状、边缘衔接与背景质量需要另行评价。已有双表单 A/B 材料，尚无人工评分。可把人工评价、验证误差传播、背景结构约束列为后续工作。

## 4. Conclusion

**英文段落素材**

> Offline refresh calibration improves fidelity to uncached DDIM under the fixed 17-refresh budget in both of our held-out evaluations. Product background replacement achieved a 2.31× speedup over uncached DDIM50, with lower boundary reference deviation than uniform caching. The regional ablation and local repair examples identify two open questions: when a spatially weighted proxy improves scheduling, and how to prevent the inpainting model from drawing new product parts outside the protected region.

## 5. 可直接改写的图注

| 图 | 英文图注素材 |
| --- | --- |
| Fig. 1 | *Offline calibration measures DDIM update deviations and selects refresh indices under a fixed budget. Online inpainting applies the frozen schedule and restores the supplied foreground.* |
| Fig. 2 | *Identity splits and held-out coverage for the product dataset. The 24 test identities yield 192 primary generations across two seeds and four methods.* |
| Fig. 3 | *Median pipeline latency and mean boundary ROI LPIPS on held-out products. LPIPS uses the matching uncached DDIM50 output as the reference.* |
| Fig. 4 | *Selected qualitative examples at seed 2026. M2 validation and M3 held-out examples are marked separately. The striped shorts case shows new structures outside the protected foreground.* |
| Fig. 5 | *Diagnostic examples of a second inpainting pass. The invented cup handle is removed for this seed, while unwanted structures remain beneath the shorts.* |

## 6. 参考文献与核对

以下条目可转成组内引用格式。保留论文和数据集名称以便归因。提交材料按课程说明用正式附件承载。

1. X. Ma, G. Fang, and X. Wang. *DeepCache: Accelerating Diffusion Models for Free*. CVPR, 2024. [论文](https://arxiv.org/abs/2312.00858)
2. J. Song, C. Meng, and S. Ermon. *Denoising Diffusion Implicit Models*. ICLR, 2021. [论文](https://arxiv.org/abs/2010.02502)
3. C. Lu et al. *DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models*. arXiv:2211.01095. [论文](https://arxiv.org/abs/2211.01095)
4. R. Zhang et al. *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric*. CVPR, 2018. [论文](https://arxiv.org/abs/1801.03924)
5. Finegrain. *Finegrain Product Masks Lite*. [数据集卡](https://huggingface.co/datasets/finegrain/finegrain-product-masks-lite)
6. SD v1.5 Inpainting. [模型卡](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting)
7. T.-Y. Lin et al. *Microsoft COCO: Common Objects in Context*. ECCV, 2014. [论文](https://arxiv.org/abs/1405.0312)
8. R. Rombach et al. *High-Resolution Image Synthesis with Latent Diffusion Models*. CVPR, 2022. [论文](https://arxiv.org/abs/2112.10752)
9. A. Radford et al. *Learning Transferable Visual Models From Natural Language Supervision*. ICML, 2021. [论文](https://proceedings.mlr.press/v139/radford21a.html)

写完后核对独立样本数、两种 LPIPS 的范围、两种速度比较的分母，以及视觉案例的选取方法。复现实验入口和运行记录见[中文交接文档](project_handoff_zh.md)。
