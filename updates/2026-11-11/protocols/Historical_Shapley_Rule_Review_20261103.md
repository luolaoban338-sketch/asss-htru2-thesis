> Dated public research copy, prepared 11 November 2026. The source date and scientific findings are retained. Current progress is stated in the update README: Adult confirmation timing passed on 10 November; its main experiment has not run. Local source links are shown as source references because this is a curated inspection snapshot.

# 历史 Shapley 规则核对表

2026 年 11 月 3 日 · R09 / T2 · COMPAS 历史方法

| 核对项 | 签字报告所载 | 复现所需或提案补充 |
|---|---|---|
| 目标与列数 | 最大化子集效用 v(S)，入选特征数至多为 K，每个入选特征须满足 φᵢ(v) ≥ τ。报告称四种方法实际都选了 5 列。 | K = 5 是特征数。公式允许少于五列，但没有给出何时少选或停止。 |
| 效用与空集 | v(S) 是交叉验证 F1。式（4）用增列前后的效用差定义 Shapley 值。 | 未交代 F1 的阳性类及平均口径、训练分折、每个子集的预测器、空集效用 v(∅) 或最终选集为空时的预测规则。 |
| 阈值 | τ 是最低贡献门槛。表 4 的 0.0756 是该方法的平均 Shapley 值，表 3 的 0.6570 是 F1。 | 未列 τ 数值。XAI 提案拟用验证集选 τ、测试集只评价一次，未给候选网格和决胜规则。 |
| 贡献估计 | 式（4）给出对全部联盟加权的数学定义。 | 实际是否枚举、是否近似、排列数和种子未记载。排列近似出现在 XAI 提案的方法设计中。 |
| 贪心与不可行 | 表 2 称该方法在贡献约束下贪心选列，另有无约束贪心对照。 | 每步候选评分、贡献是否重估、同分顺序、合格列不足及空集处理均待证。XAI 提案仅拟报告不够五列的情形。 |
| 数据与复核 | 报告使用 COMPAS 的 is_recid 标签。表 1 列有风险评分特征。 | XAI 提案拟预测两年再犯、排除已发布的风险分数。历史规则的识别和旧表数字的复算分别需要方法配置和原划分。 |

**结论**　现有材料足以重述优化问题与 K = 5 的历史比较。复现该次选择仍需可对应签字报告的代码、配置或运行记录。学位提案将旧固定阈值法与 PowerSHAP 分列为两个基线。前瞻性设定应单独登记。

**原文**　签字报告 (source: `outputs/researchwrite/thesis-plan/sources/Report-LUO Feiyang_signed - 副本.pdf.txt`) 物理页 4–6、8（§§2.1、3.1–3.3、4.1–4.2 和后续研究） · XAI 提案 (source: `outputs/researchwrite/thesis-plan/sources/Feature_Selection_XAI_Project_Proposal.pdf.txt`) 第 1 页（Method 与 Experiments and evaluation） · 学位提案 (source: `outputs/researchwrite/thesis-plan/sources/Thesis_Proposal.pdf.txt`) 第 1 页 §3。
