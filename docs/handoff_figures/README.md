# 报告与展示图包

这里集中放置 68 张图，包含 63 张原有汇总图和 5 张重新绘制或拼接的图。图片可以直接插入报告或 Slides。[完整索引与来源](index.csv)

## 优先使用的五张图

| 图 | 适合放在哪里 | 图注要点 |
| --- | --- | --- |
| [Fig. 1 方法流程](ready_to_use/fig01_method_overview.png) | Methods，或讲述方法的第一张展示页 | 上半部分是离线校准，下半部分是预训练 Inpainting 的冻结策略推理 |
| [Fig. 2 商品数据](ready_to_use/fig02_data_overview.png) | 数据与实验协议 | 按商品划分，24 件独立测试商品、两个种子、四种方法。场景类别和掩码覆盖来自清单 |
| [Fig. 3 M3 主结果](ready_to_use/fig03_m3_main_results.png) | 报告主结果和展示核心页 | 两种缓存各 17 次完整刷新。左图为耗时中位数，右图为相对无缓存的轮廓 LPIPS |
| [Fig. 4 成功与失败拼图](ready_to_use/fig04_application_cases.jpg) | 应用分析、结论前一页 | 事后目视选择，左列为 M2 验证，右列为 M3 独立测试。下半部分体现掩码外误生成 |
| [Fig. 5 局部修复拼图](ready_to_use/fig05_repair_cases.jpg) | 局限、优化尝试或附录 | 杯把成功去除仅针对所示种子，短裤仍有额外结构 |

## 按研究问题找原图

| 问题 | 推荐图 |
| --- | --- |
| 文生图的质量与耗时如何变化 | [确认结果质量与耗时](existing/text_to_image/confirmation/quality_latency.png)、[配对差异区间](existing/text_to_image/confirmation/paired_differences.png)、[预算消融](existing/text_to_image/confirmation/budget_ablation.png) |
| 校准依据和刷新位置是什么 | [早期代理诊断](existing/text_to_image/round2/proxy_diagnostics.png)、[文生图刷新表](existing/text_to_image/confirmation/refresh_schedules.png) |
| 哪种提示词会使商品被补画 | [真实照片先导对照](existing/product_background/photo_pilot/comparison_seed2026.jpg)、[另一个种子](existing/product_background/photo_pilot/comparison_seed6132.jpg) |
| M3 每件商品的变化如何 | [M3 逐商品误差与时延](existing/product_background/m3/quality_latency.png)、[改善最大与退步最大](existing/product_background/m3/paired_extremes.jpg) |
| 轮廓权重在 B=21 是否有收益 | [刷新位置与配对差异](existing/product_background/ablation21/ablation_summary.png)、[台灯与袋装商品案例](existing/product_background/ablation21/paired_cases.jpg) |
| 如何展示成功和失败的完整对照 | [M3 高跟鞋与短裤](existing/product_background/m3/application_good_bad.jpg)、[M2 黑靴与印花背心](existing/product_background/m2/application_good_bad.jpg) |
| 二次重绘的完整结果 | [种子 2026](existing/product_background/repair_pilot/comparison_seed2026.jpg)、[种子 6132](existing/product_background/repair_pilot/comparison_seed6132.jpg) |

## 全部原图和来源

- [文生图](existing/text_to_image/)按 `pilot`、`round2`、`confirmation` 排列。确认阶段的 25 张比较图保留原顺序，适合补充材料。
- [商品背景替换](existing/product_background/)按真实照片先导、M2、M3、B=21 和修复小试排列。M3 的六张顺序图覆盖 24 件商品的首个种子。M1 的工程验证记录在[实验报告](../../results/product_background/m1/report.md)。
- [盲评示例图](review_materials/product_background/m3/)单独存放。它们是待评分材料，正式盲评请按[组织者说明](../../results/product_background/blind_review/README.md)分发最新离线表单。

原图按字节复制。[索引](index.csv)记录每张图的仓库相对路径、来源、像素尺寸和 SHA256。拼图的商品输入来自 Finegrain Product Masks Lite，文生图样例由 COCO 2014 caption 驱动。可在本机原始输出齐全时用[重建脚本](../../scripts/build_handoff_figures.py)重新生成整个图包，无需 GPU。

报告图注请写明数据阶段、指标参考和例图选择方式。B=21 是复用 M3 输入的探索性消融，M2 是验证集，M3 是独立测试。统计上的最好或最差按参考误差排序，视觉成功和失败是目视例图。
