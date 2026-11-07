# 商品背景替换人工盲评

使用 M3 预先固定的 8 件商品及种子 2026。主项是轮廓和背景衔接的自然度，次项是背景符合提示词的程度。[整体设计与统计协议](../../../docs/product_background_replacement_design.md)

1. 向未看过带标签结果的评价者发一份离线表单。表单在 `images/packets` 下，文件名为 `review_form_1.html` 和 `review_form_2.html`。交替发放两个版本。每人用不同的匿名编号、只评一份。目标至少三份。
2. 用浏览器打开 HTML，点击图片可以放大。评价者完成八个案例后点击“下载评分 CSV”。表单内嵌图像，可单独复制。组织者保管 `organizer_key.json`，评价者只收到自己的一份 HTML。
3. 将导出的原始 CSV 放在 `images/responses`。汇总时先核对八项完整性和匿名编号是否重复，再根据对应版本解码 A/B。

从项目根目录运行：

```powershell
& '.\.venv\Scripts\python.exe' scripts/background_blind_review.py prepare
& '.\.venv\Scripts\python.exe' scripts/background_blind_review.py score
```

第一条命令使用本机保存的 M3 输出重建表单。第二条命令在收到评分后生成 `human_review_summary.json`。摘要包含两项偏好的胜负和平局计数、以商品等权的校准偏好比例及两张均不可用的计数。少于三名评价者时标为先导结果。备注由组织者在原始 CSV 中复核。

当前状态为材料已准备，待回收人工评分。
