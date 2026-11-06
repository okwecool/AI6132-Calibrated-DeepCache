# AI6132 项目候选调研与课程范围

调研日期：2026-11-06。硬件：本机 NVIDIA GeForce RTX 5060 Laptop GPU，8151 MiB 显存，驱动 591.86（通过 `nvidia-smi` 读取）。作业截止：2026-11-16。

本文基于本地课件、作者官方仓库和 Hugging Face Diffusers 官方文档。已核对课程主题、仓库说明及部分关键源码；尚未安装候选项目、下载权重或运行推理。因此，下文的 8GB 可行性是选型判断，不是本机实测结果；改进方案都是待验证的实验假设，不承诺提升或论文级新颖性。

## 1. 课程覆盖范围与选题边界

检索了上一级目录及 `Second Half` 中的 22 份教学 PDF，提取了文本并浏览各讲主题，重点阅读高效扩散、图像编辑、评价、Rectified Flow 和 CFG 相关内容；对包含图片公式的关键页面进行了渲染核对。没有把模拟试题、复习笔记当作课程要求来源。

| 范围 | 本地课件 | 覆盖内容及选题关联 |
|---|---|---|
| 基础与架构 | introduction to deep_learning v3.pdf、introduction to transformer.pdf | CNN、Transformer、注意力，为修改 U-Net 和注意力模块提供基础 |
| GAN | GAN.pdf、GAN advanced.pdf | 对抗训练、条件生成、训练稳定性、StyleGAN 及其扩展；支持图像翻译和 GAN 编辑方向 |
| VAE | VAE v3.pdf、VAE theory.pdf | AE、DAE、VAE、β-VAE、VQ-VAE、连续和离散潜空间 |
| 自回归 | autoregressive v4.pdf | PixelCNN、视觉 token、VAR 的 next-scale prediction；不是只涉及语言模型 |
| 扩散与流 | diffusion v7.pdf、Diffusion theory v4.pdf、rectified flow v5.pdf | DDPM、DDIM、score、SDE/ODE、Euler、Rectified Flow、Reflow |
| 条件生成 | classifier free guidance v2.pdf、stable diffusion v3.pdf | 分类器引导、CFG、潜空间扩散、SD1.5、SD3 |
| 蒸馏与加速 | model distillation.pdf、Second Half/EfficientDiffusionModels.pdf | DPM-Solver、渐进蒸馏、Consistency Models、DMD/DMD2 |
| 图像编辑与定制 | Second Half/ImageEditing.pdf | GAN inversion、DragGAN、DDIM inversion、SDEdit、DiffEdit、Imagic、InstructPix2Pix、DreamBooth、LoRA、ControlNet、DiffMorpher |
| 视频 | Second Half/VideoDiffusionModels.pdf | Video LDM、CogVideoX、Wan、Diffusion Forcing、Self Forcing、KV cache、相机控制与世界模型 |
| 3D | Second Half/3DGenerativeModels.pdf | NeRF、3DGS、SDS/VSD、Zero-1-to-3、MVDream、LRM、InstantMesh、Hunyuan3D、TRELLIS |
| 后训练 | Second Half/Diffusion_RL.pdf | Diffusion-DPO、奖励反传、DDPO、DPOK、Flow-GRPO、DanceGRPO |
| 评价 | generation evaluation.pdf | FID、IS、人类评价、VLM 评价、消融实验和指标局限 |

与本次选题最直接相关的页码（均为 PDF 物理页码）：

- `Second Half/EfficientDiffusionModels.pdf`：第 16–28 页 DPM-Solver，第 35–52 页 Consistency Models，第 54–76 页 DMD/DMD2。
- `Second Half/ImageEditing.pdf`：第 28 页 DDIM inversion，第 32–40 页 SDEdit/DiffEdit，第 44–45 页 InstructPix2Pix，第 52–65 页 LoRA/ControlNet。
- `generation evaluation.pdf`：第 2–3 页评价类别，第 10 页 FID 的实际注意事项。

课程允许做较前沿的主题，但不要求复现最大模型。结合 8GB 显存和剩余约 10 天，优先考虑 SD1.5 级别的推理阶段修改，或小规模图像生成/翻译训练。视频、3D、完整蒸馏和扩散 RL 的原始训练配置不列为首选。

## 2. 候选总览

这里列的是八个可独立立项的方向。多数方向涉及多个仓库或方法，满足作业中至少两种方法的比较要求；应保留至少两种已有方法，再加入自己的变体，避免把同一方法的两个超参数配置当作全部方法比较。

| 优先级 | 候选方向 | 推荐比较对象 | 训练需求 | 8GB 判断 | 主要风险 |
|---|---|---|---|---|---|
| A1 | 采样器与 CFG 协同优化 | DDIM、DPM-Solver++、UniPC、自定义引导调度 | 无 | 优先尝试 | 只换 scheduler 容易分析不足 |
| A2 | DeepCache 特征缓存策略 | 无缓存、少步 DPM-Solver++、固定缓存、改进缓存 | 无 | 优先尝试，需记录缓存显存 | hook 与 Diffusers 兼容性；少步时缓存收益可能降低 |
| A3 | FreeU 的时间/频率调度 | 原始 SD、固定 FreeU、动态 FreeU | 无 | 优先尝试 | 提升依赖数据，可能损伤纹理 |
| B1 | 局部图像编辑与背景保持 | SDEdit、LEDITS++、可选 DiffEdit、自定义掩码策略 | 无模型训练 | 有条件尝试 | attention map 显存；反演误差；旧接口 |
| B2 | ControlNet 控制强度与注入时机 | ControlNet、匹配条件的 T2I-Adapter、自定义控制调度 | 使用预训练权重 | 需要低显存配置 | 条件与底模不匹配会造成比较混杂 |
| B3 | LCM-LoRA 少步生成 | DDIM、DPM-Solver++、LCM-LoRA、自定义推理策略 | 使用已蒸馏权重 | SD1.5 版本优先尝试 | 不同 CFG 区间；不要直接搬普通扩散的参数 |
| C1 | CFM / OT-CFM 小规模图像生成 | Independent CFM、OT-CFM、可选 DDPM/自定义采样 | 小模型训练 | 小分辨率可尝试 | 数据难度、训练预算和模型收敛 |
| C2 | CUT / FastCUT / CycleGAN 图像翻译 | CUT、FastCUT、CycleGAN、自定义 PatchNCE | 可用预训练基线；改损失需训练 | 256 或更低分辨率可尝试 | 老环境、GAN 训练波动、完整训练时间 |

优先级是针对本次硬件和截止时间的判断，不是对方法学术价值的排名。

## 3. 候选 A1：少步扩散采样器与动态 CFG

**建议题目：** 低计算预算下扩散采样器与分类器自由引导策略的比较及优化。

**现成代码：**

- [DPM-Solver / DPM-Solver++ 官方仓库](https://github.com/LuChengTHU/dpm-solver)。核心实现 `dpm_solver_pytorch.py`。
- [UniPC 官方仓库](https://github.com/wl-zhao/UniPC)。README 提供 Diffusers 集成示例及 CIFAR-10 预训练模型示例。
- [Diffusers 的 DPM-Solver++ 实现（v0.35.1）](https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py)。查阅版本用于固定源码位置，不等于已验证的安装组合。

**为什么适合：** 直接对应 DDIM、ODE、DPM-Solver 和 CFG 课件；同一个预训练 SD1.5 模型即可比较多个算法，无需多次训练或同时加载多个模型。

**可修改内容：** 在采样循环中把固定引导强度替换成按实际噪声水平/采样阶段变化的强度；比较线性、分段或平滑调度。也可以单独研究时间步分配、solver order 与强引导下的稳定性。第一版只改一个维度。

**最低可交付实验：**

- 同一底模、分辨率、数据、提示词和随机种子。
- DDIM、DPM-Solver++、UniPC，在 5/10/20 步下比较。
- 在一个固定采样器上比较固定 CFG 与自定义 CFG，分析 3–4 个参数设定。
- 指标：文本图像相似度、人工盲评、单图耗时、显存、U-Net 调用次数；展示失败样例和相同提示词的横向图。

**数据：** [COCO 官方数据入口](https://cocodataset.org/#download)，固定一份验证集图像与对应 caption 清单。先用 20 个提示词排错，再用 100 个提示词 × 2 个种子完成主实验；更多样本视实测速率决定。验证集用于选参数，另留测试子集报告最终结果。

**风险：** 动态 CFG、分段引导本身已有相关研究，不能仅因自行实现就宣称新方法。若只展示不同 scheduler 的效果，工程修改和分析会偏浅；应围绕一个明确问题，如“强 CFG 是否在极少步采样下恶化稳定性”。

## 4. 候选 A2：DeepCache 缓存策略优化

**建议题目：** 面向低显存图像生成的扩散特征缓存策略与质量—延迟权衡。

**现成代码：** [DeepCache 官方仓库](https://github.com/horseee/DeepCache)，包含推理入口、实验代码和可接入 Diffusers 的 helper。作者说明旧实验代码可能受 Diffusers 更新影响，优先检查通用 helper。

**具体改动入口：** 已核对 [DeepCache/extension/deepcache.py](https://github.com/horseee/DeepCache/blob/master/DeepCache/extension/deepcache.py)。`set_params()` 设置缓存间隔与分支；`is_skip_step()` 决定是否复用缓存；`wrap_block_forward()` 执行缓存读取/更新。这比修改完整训练框架更容易定位。

**可修改内容：**

1. 第一版做基于校准样本的固定预算刷新时刻选择，与统一间隔和原论文的非均匀刷新比较。原论文已有非均匀 1:N 策略，简单分阶段刷新只能算复现/基线，不能直接宣称新贡献。
2. 进阶版用仍需计算的浅层特征变化作为刷新信号，加入阈值和最大复用步数；先验证该信号与缓存误差相关，再实施。
3. 不用“额外完整运行一次 U-Net”来判断是否可缓存，否则检测成本会抵消收益。

**最低可交付实验：**

- 无缓存的基准生成、无缓存的少步 DPM-Solver++、官方固定缓存、自定义缓存。
- 比较缓存间隔 2/3/4、不同分支和采样步数。
- 同时报告端到端时间、U-Net 阶段时间、峰值显存、文本图像相似度、人工偏好。
- 可用与无缓存输出之间的 LPIPS/PSNR 衡量近似偏差，但必须解释“接近无缓存输出”不等于“更高生成质量”。

**数据：** 与 A1 相同的 COCO 子集，便于复用评测框架。

**风险与价值：** 缓存增加额外张量，不能默认它节省显存；缓存与 CPU offload 的交互也需要实测。普通 U-Net 调用次数不足以描述缓存节约的计算，应额外记录完整刷新次数或实际耗时。自适应缓存已有 [AdaCache](https://github.com/AdaCache-DiT/AdaCache) 和 [DiCache](https://github.com/Bujiazi/DiCache) 等研究，本项目可以把相关思想移植到较轻的 U-Net 场景，但不能声称自适应缓存概念全新。

## 5. 候选 A3：FreeU 时间与频率调度

**建议题目：** 扩散 U-Net 中主干与跳跃特征的动态重加权分析。

**现成代码：** [FreeU 官方仓库](https://github.com/ChenyangSi/FreeU)，作者公开了 Fourier filter 和 backbone/skip 重加权代码；[Diffusers 官方使用文档](https://huggingface.co/docs/diffusers/main/en/using-diffusers/freeu) 提供 `enable_freeu()` 接口，无需重新训练。

**具体改动入口：** [Diffusers U-Net blocks](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unets/unet_2d_blocks.py) 中调用 `apply_freeu` 的上采样模块，以及官方仓库中的 `Fourier_filter`。实现时固定所用 Diffusers tag/commit。

**可修改内容：** 把固定的 `b1/b2/s1/s2` 改为随采样进度变化；或者按分辨率归一化滤波范围。先做时间调度，再决定是否加入频率调度，防止实验维度过多。

**最低可交付实验：** 原始 SD1.5、官方固定 FreeU、自定义动态 FreeU；消融只调 backbone、只调 skip、两者都调，比较低步数与常规步数。展示细纹理、复杂背景、文字或多物体场景的失败案例。

**数据与评价：** COCO 子集加上明确列出的复杂提示词子集。评价文本对齐、人工偏好、细节与伪影；锐度或高频能量仅作诊断，不能直接当作真实感指标。

**8GB 判断：** 三个高优先级方案中新增状态较少，是较稳妥的起点；仍须先验证 FP16、512×512、batch=1 的实际峰值。

**风险：** 不能把开启官方 FreeU 本身作为自己的优化；动态策略可能削弱模型原本良好的纹理，需要允许出现负结果并分析原因。最终若要强化“两种方法”的覆盖，可再加入一种采样器比较，但不要把采样器和 FreeU 同时改变后直接归因。

## 6. 候选 B1：局部编辑与背景保持

**建议题目：** 无需训练的扩散图像编辑中编辑强度与背景保持的权衡。

**现成代码：**

- [LEDITS++ 官方仓库](https://github.com/ml-research/ledits_pp)，提供反演与语义引导编辑。
- [Diffusers LEDITS++ 官方文档](https://huggingface.co/docs/diffusers/api/pipelines/ledits_pp)。
- [DiffEdit 的固定版本实现](https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/pipelines/stable_diffusion_diffedit/pipeline_stable_diffusion_diffedit.py)，对应课件中的自动掩码思想；该 pipeline 已被标记 deprecated，应固定版本并单独验证。
- SDEdit 可使用 SD 的 img2img 流程，作为简单基线。

**可修改内容：** 根据语义差异生成软掩码，改进阈值、边界平滑或编辑引导的阶段调度。在同一编辑方法中分别评估这些组件。

**数据：** [PnPInversion 官方仓库及 PIE-Bench 入口](https://github.com/cure-lab/PnPInversion)。该 benchmark 提供 700 张图像及编辑标注，适合取按编辑类型分层抽样的 50–100 个样本；官方入口需要填写表单，应先确认能否取得数据。也可查看 LEDITS++ 的 TEdBench++ 入口。若数据暂不可得，可自建标注集，但需清楚记录来源和划分。

**评价：** 目标文本对齐、编辑区域变化、非编辑区域 LPIPS/PSNR/SSIM、反演重建误差、总耗时。背景区域评价使用固定 benchmark 标注，不能让每种方法自己预测掩码再各自定义评价区域。

**重要限制：** LEDITS++ 作者和 Diffusers 文档均明确说明，Diffusers 版本因 scheduler 兼容性不保证完美反演。应单独做“无编辑重建”实验；若研究反演精度，优先核对作者实现。显存评估不能直接套用普通 SD 文生图数据，因为此类方法可能存储注意力图并同时处理多个分支。

**暂不优先的相关仓库：** [MasaCtrl](https://github.com/TencentARC/MasaCtrl) 官方示例要求至少 16GB VRAM；[PnPInversion](https://github.com/cure-lab/PnPInversion) 集成很多方法但存在多个旧依赖环境。前者降低优先级，后者更适合用作数据/评价参考，再挑一个最小实现。

## 7. 候选 B2：ControlNet 与轻量条件注入

**建议题目：** 边缘条件图像生成中控制注入时机与控制强度的优化。

**现成代码：** [ControlNet 官方仓库](https://github.com/lllyasviel/ControlNet)、[T2I-Adapter 的 SD1.4/1.5 分支](https://github.com/TencentARC/T2I-Adapter/tree/SD)。注意 T2I-Adapter 默认主分支侧重 SDXL，不能直接照主分支例子选型。

**具体入口：** [Diffusers ControlNet pipeline 固定版本](https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/pipelines/controlnet/pipeline_controlnet.py) 的控制权重和去噪循环。已有接口能设置控制开始/结束时间；若只设置现成开关，主要是参数分析，进一步优化需要实现连续调度或基于输入特征的策略。

**可修改内容：** 使用 Canny 边缘作为统一条件，比较恒定控制、前期控制、连续衰减控制；研究边缘密度与控制强度之间的关系。只用 Canny 可避免额外加载深度或姿态检测模型。

**数据与指标：** COCO 图像提取 Canny 图，加对应 caption。比较生成图的边缘一致性（使用容差匹配）、文本对齐、人工偏好、延迟和显存。边缘相似度高不一定自然，因此需要多维评价。

**8GB 判断：** 官方提供 [8GB 低显存模式](https://github.com/lllyasviel/ControlNet/blob/main/docs/low_vram.md)，并说明不保证所有显卡成功。建议只加载一个 SD1.x 底模和一个条件模块，FP16、512×512、batch=1，必要时 offload，逐方法串行运行。

**比较边界：** ControlNet 与 T2I-Adapter 的条件类型、底模版本和预处理应尽量匹配；无法统一训练数据或 checkpoint 来源时，应报告这些差异，不能把所有质量差异归结为模块结构。不要同时比较 SD1.5 ControlNet 和 SDXL Adapter 后声称是公平架构对比。

## 8. 候选 B3：LCM-LoRA 少步生成

**建议题目：** 预训练一致性适配器的少步生成及质量—效率分析。

**现成代码：** [LCM / LCM-LoRA 官方仓库](https://github.com/luosiallen/latent-consistency-model) 和 [Diffusers 官方推理教程](https://huggingface.co/docs/diffusers/main/en/using-diffusers/inference_with_lcm_lora)。选择 SD1.5 对应的预训练 LCM-LoRA，而不是从头蒸馏 SDXL。

**可修改内容：** 研究 LoRA 强度与推理步数、引导强度的交互，或实现轻量的阶段策略。若做时变 LoRA 强度，要先检查是否能不引入明显的切换开销；从头训练蒸馏模型不作为本次前提。

**实验：** 相同 SD1.5 基础模型下比较 DDIM、DPM-Solver++、加载 LCM-LoRA 的 LCM scheduler，再加入一个自定义策略。LCM-LoRA 改变了有效模型权重，应把它归类为“预训练蒸馏适配器方法”，而非完全相同模型下的纯 solver 对比。

**数据与评价：** 与 A1 相同。LCM 重点看 2/4/8 步，普通扩散重点看 5/10/20 步；同时画按步数和按实际时间的质量曲线。官方教程建议 LCM-LoRA 使用较低 CFG，不能机械使用普通 SD 常用的 CFG=7.5。

**风险：** 单纯加载 LoRA 做演示不足以体现自己的修改；全面蒸馏和大规模 LoRA 训练会扩大任务。这个方向更适合作为 A1 的扩展，或者与明确的控制任务结合。

## 9. 候选 C1：小规模 Flow Matching 训练

**建议题目：** 不同配对策略与数值求解器对小规模流匹配图像生成的影响。

**现成代码：** [TorchCFM](https://github.com/atong01/conditional-flow-matching)，提供 independent CFM、OT-CFM、VP-CFM 等损失及 MNIST 图像示例。已核对 [torchcfm/conditional_flow_matching.py](https://github.com/atong01/conditional-flow-matching/blob/main/torchcfm/conditional_flow_matching.py) 中线性路径和条件向量场实现。

**可修改内容：** 在相同网络和训练预算下比较随机配对与 minibatch OT 配对，尝试时间采样分布、损失权重或 Euler/Heun 采样。`ConditionalFlowMatcher(sigma=0)` 的独立线性路径与课件的基础直线插值训练相联系，但不能把库中所有 CFM 变体统称为同一种 Rectified Flow。

**数据：** 先用 MNIST 检查训练链路；正式实验优先 Fashion-MNIST 或小型 CIFAR-10 设置。MNIST 适合作为排错数据，但单独作为最终任务挑战性偏弱。若只跑低预算训练，结论应限定在该预算下。

**评价：** 生成多样性、采样轨迹、实际 NFE、耗时、适合该数据域的分布指标。小模型训练和多个消融会消耗时间，因此需要先做每步耗时测量再分配训练预算。

**风险：** 8GB 容量可通过缩小模型和 batch 控制，但这不等于能在 10 天内得到强 CIFAR-10 结果。加入 DDPM 基线需要额外实现或接入，并统一网络容量与训练预算。

## 10. 候选 C2：CUT / FastCUT / CycleGAN

**建议题目：** 低资源无配对图像翻译中对比学习约束与结构保持的分析。

**现成代码：** [CUT 官方仓库](https://github.com/taesungp/contrastive-unpaired-translation)、[CycleGAN / pix2pix 官方仓库](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)。CUT 作者提供 FastCUT，并说明其训练相对 CycleGAN 更节省资源；不能把作者的相对结论直接当作 RTX 5060 实测。

**具体入口：** 已核对 [models/cut_model.py](https://github.com/taesungp/contrastive-unpaired-translation/blob/master/models/cut_model.py)，可修改 PatchNCE 的层选择、温度 `nce_T`、采样 patch 数量、identity 约束以及相应损失组合。

**可修改内容：** 针对背景被错误转换的问题，尝试非均匀 patch 采样、结构保持约束或 NCE 权重调度。保留原版 CUT/FastCUT，并与 CycleGAN 比较。

**数据：** 官方示例的 grumpifycat 或 horse2zebra。先使用较小训练子集和 128/256 分辨率测吞吐；对各方法固定数据划分与更新预算。不要用配对 pix2pix 与非配对 CycleGAN 的差异直接证明架构优劣。

**风险：** CUT README 的安装指导仍指向很老的 PyTorch，列出的操作系统是 Linux/macOS；需要在支持 Blackwell 的现代环境中迁移验证。CycleGAN 官方仓库有较新的环境更新，但这不等于 CUT 原仓库自动兼容。若选此方向，训练时间和老依赖迁移比推理型项目更不确定。

## 11. 针对 RTX 5060 8GB 的共同实施原则

1. **先验证小流程。** SD1.5、FP16、512×512、batch=1、20 个样本起步；记录成功率、峰值显存和单图耗时后再扩大。各方法顺序加载，不要求同时驻留。
2. **区分 GPU 架构与 Python 包版本。** PyTorch 官方在 [2.7 发布说明](https://pytorch.org/blog/pytorch-2-7/) 中引入 Blackwell/CUDA 12.8 支持；当前 [CUDA 打包说明](https://dev-discuss.pytorch.org/t/introducing-cuda-13-2-and-deprecating-cuda-12-8-release-2-12/3337) 已继续演进。不要照抄旧仓库的 PyTorch 1.x/CUDA 11.x 命令。实施时结合当前驱动和官方安装矩阵选支持该 GPU 的构建，锁定版本；本次调研未安装软件。
3. **底模优先选可取得的 SD1.5 checkpoint。** 老 README 中的 `runwayml/...` 等路径可能已迁移，实际选定时核实下载入口、模型卡和权重许可；代码许可与权重许可分开记录。
4. **注意力与 offload 按方法验证。** [Diffusers 显存优化文档](https://huggingface.co/docs/diffusers/main/en/optimization/memory) 可作参考，但保存 attention map 的编辑方法不能假定能直接换成所有高效注意力后端。缓存 hook 与 offload hook 也可能相互影响。
5. **计时公平。** 固定精度、分辨率、batch 和 offload 配置，预热后在 CUDA 同步的边界计时；报告中位数及波动。把模型下载/初次加载与生成耗时分开。笔记本使用相同供电和功耗设置。
6. **参数选择与最终测试分离。** 在验证子集上调整阈值/调度，再冻结设置报告测试结果；不要对每张测试图人工挑最佳参数或种子。
7. **不依赖小样本 FID 单一结论。** 课件第 10 页强调样本量和参考集一致性。100–200 张图的 FID 只能是探索性结果，不能直接对照论文的 50k 样本分数。可以采用配对评价、文本对齐和人工盲评，并明确样本量与局限。LPIPS 接近原图也不能替代编辑成功率。
8. **保留源代码溯源。** 选定后记录仓库 URL、commit、许可证、预训练模型来源和自己的修改；提交包保留所需代码与许可文件，不用远程云盘替代源代码，不包含数据集和模型权重。

## 12. 推荐决策

**默认首选 A2 DeepCache，低风险备选 A1 采样器，进一步备选 A3 FreeU。** 三者都能使用同一类底模、数据和评测工具，切换方向的成本较低。

- 想让代码修改清楚、能画出有说服力的效率曲线：选 DeepCache。
- 想最贴合课件、最快建立多个已有方法的比较：选 DDIM / DPM-Solver++ / UniPC + CFG。
- 想修改点集中、减少训练和额外显存压力：选 FreeU。
- 想要更直观的应用展示，愿意承担注意力/反演适配：选局部图像编辑。
- 想自己训练并研究目标函数：选 TorchCFM；其次是 CUT，但先测训练成本。

建议第一步只对 A1/A2/A3 中最感兴趣的一个做 20 样本可运行性检查。检查通过后，冻结基础环境、模型与数据划分，再展开主实验。约 10 天的工作量应集中在一个主问题和一个改动机制上。
