# 2026-10-30 论文进展公开资料 / Thesis research update

本目录汇集 9 月 30 日汇报之后的已完成研究资料，供导师和读者查看。根目录保留 10 月 12 日代码分享快照；这里追加后续源码、论文草稿章节、公开版研究报告和小型汇总结果。

## 阅读入口

| 内容 | 路径 |
| --- | --- |
| HTRU2 方法、结果、讨论及方法—证据一致性草稿 | [manuscript](manuscript/) |
| HTRU2 消融、WDBC 三折进度、A3/A4 各阶段公开报告 | [reports](reports/) |
| 传统方法与原生 PowerSHAP 五折汇总，四组合成实验结果 | [results](results/) |
| 新阶段核心源码及选定测试 | [code](code/) |
| 科学协议与数据划分契约 | [protocols](protocols/) |
| 原生选择器和下游 LR 的分别固定依赖 | [environments](environments/) |
| 来源、修改方式及 SHA-256 | [PUBLICATION_MANIFEST.json](PUBLICATION_MANIFEST.json) |

## 当前证据边界

HTRU2 传统方法五折比较、原生 PowerSHAP 正式五折比较，以及信号—噪声、相关组和 XOR（RF/LR）四层合成实验均已交付。WDBC 目前完成外层折 0–2；其余折及 Adult 正式实验尚未完成。完整 A3/A4 综合论证和整篇论文仍未结题。

这些结果属于开发阶段和预先限定的诊断场景，不支持“整体优于现有方法”“普遍鲁棒”或“显著更快”等概括。五个交叉验证折不等于五个独立数据集；合成数据中的六个嵌套重采样也不是额外独立实验单位。历史报告内的“待执行”说明以各报告日期为准。

相关组报告的汇总使用独立核对后的精确有理数重建，原冻结汇总失败保留在本地证据链。PowerSHAP 训练门控曾出现 float32 语义审计问题；正式五折汇总与该早期训练门控是不同交付，不能混用。

## 资料完整性与使用范围

`byte_preserving_copy` 项与来源字节一致。`derived` 项是公开衍生副本，移除了账户使用比例或本机路径元数据，且在 manifest 中注明；不能用其散列冒充原冻结文件。结果数值未因公开整理而改写。报告中的本地来源索引不表示完整原始实验档案已上传。

这里不含新增原始数据、逐模型输出、全部预测、虚拟环境、完整审计档案、第三方论文全文、账户日志或历史恢复 ZIP。新增代码是供研究检查的源快照，完整运行仍需要匹配的数据、环境和原始绑定；详见 [代码范围](code/README.md)。未启动任何新拟合。

原仓库 HTRU2 数据署名和 CC BY 4.0 说明继续适用。代码未新增开源许可证；公开可浏览不等于授予额外再分发许可。

## English note

This is a curated research update, not a complete experiment archive or a claim of thesis completion. Dated chapters and reports retain their original scientific limitations. The publication manifest distinguishes byte-preserving source copies from explicitly derived public copies. No new classifier fitting or prediction was performed for publication.
