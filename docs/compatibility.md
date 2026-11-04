# Polyrate 兼容矩阵

本文档是兼容性声明的唯一依据。状态含义：

- **Implemented**：具有公共 API、测试和端到端样例；
- **Partial**：基础算法可用，但尚未覆盖 Polyrate 的全部选项或固定格式行为；
- **Planned**：尚未作为可用功能发布。

| Polyrate 能力 | PyRate 0.1 状态 | 当前边界 / 验收条件 |
|---|---|---|
| `*GENERAL` 等 section 语法 | Partial | 区段、block、未知 keyword 均保留；尚未执行所有 keyword 语义 |
| 原子、几何、单位模型 | Partial | TOML 强类型模型完整；legacy 单位/同位素语义仍需逐关键字映射 |
| 理想气体平动分区函数 | Implemented | 1 bar 标准态，单/双分子单位测试 |
| 刚性转子分区函数 | Implemented | 原子、线性、非线性和对称数 |
| 谐振子分区函数与 ZPE | Implemented | 正频率；负频率作为反应坐标排除 |
| 电子分区函数 | Implemented | 多电子态、能级和简并度 |
| CTST / TST | Implemented | harmonic RRHO，单/双分子 |
| canonical VTST / CVT | Implemented | 用户提供的路径分割面；温度依赖自由能瓶颈，局部峰值保护；待官方路径回归 |
| Polyrate 3/5-point `FITMAX` 行为 | Partial | 目前采用保守三点抛物线局部精修；与官方 FITMAX 3/5-point 行为尚未对齐 |
| CUS | Planned | 需 spline-unified statistical treatment |
| Wigner | Implemented | 单一虚频 |
| ZCT / MEPSAG | Partial | 改进了分段 action、阈值和反射近似；尚非 Polyrate ZCT/MEPSAG 的等价实现 |
| SCT / CD-SCSAG | Planned | 需反应路径曲率和 small-curvature action |
| LCT / LCG3 / LCG4 | Planned | 需 large-curvature tunneling path |
| ICVT | Historical only | Polyrate 2023 已移除 ICVT；仅作为旧版本兼容范围，不计入 2023 必选功能 |
| μVT / US | Partial | Beyer–Swinehart DOS 内核完成；能量分辨 variational optimization 未接入 |
| Hindered rotor | Planned | 需 rotor potential、Pitzer–Gwinn/quantum rotor 选项 |
| Morse / QQ / anharmonic vibration | Planned | 当前仅谐振子 |
| MEP integration / Page–McIver / Euler | Planned | 当前读取预计算路径点 |
| Direct dynamics hooks | Planned | 尚无 Gaussian/GAMESS/电子结构程序适配器 |
| Dual-level correction / ISPE | Planned | 尚未实现 `fu50/fu51` 路径能量校正 |
| IVTST / IOC | Planned | 尚未实现 |
| VRC-VTST | Planned | loose transition state sampling 尚未实现 |
| SS-QRRK / pressure dependence | Partial | DOS 基础完成；master equation 与碰撞模型未实现 |
| Solvent / surface options | Planned | 尚未实现 |
| Restart files / legacy unit files | Planned | 尚未复现固定文件单元语义 |
| `fu15` summary reader | Implemented | forward/reverse 第一张速率表 |
| Polyrate fixed-format output 1:1 | Planned | 当前输出为结构化 JSON/CSV/表格 |
| 112 个 Polyrate 2023 test runs | Planned | 官方分发包和手册链接已核实，尚未完成获取/运行；manifest/import 工具已提供 |

对应的逐温度、逐列数值比较工具与严格失败语义见 [`docs/head-to-head.md`](head-to-head.md)。

## “1:1 完成”的定义

单个功能只有同时满足以下条件才可从 Partial/Planned 改为 Implemented：

1. 对应 legacy keyword 可解析并执行；
2. 与 Polyrate reference run 使用相同物理模型；
3. 关键中间量（分区函数、`ΔG(s)`、action、瓶颈位置）有回归测试；
4. 最终速率在预先声明的误差容限内；
5. 原生和 Python 回退结果一致；
6. 文档清楚说明单位、默认值和不兼容行为。

建议数值门槛：纯代数/分区函数相对误差 `≤1e-10`；路径拟合和数值积分 `≤1e-6`；完整速率对 legacy 文本舍入前数据 `≤1e-4`，对仅有三位有效数字的 `fu15` 表采用 `≤5e-3`。

## P0 数值验证状态（2026-11）

修复分支提供一个非经典反射的势垒顶抛物线延拓近似，不能将其称为已经复现 Polyrate 的全能量透射概率或其 ZCT 算法。必须补充精确 Eckart 解析对照、端点不对称阈值测试、积分收敛测试和官方中间量后才能声明等价。`constants_profile=codata2018` 当前只允许单个固定常数集，拒绝未实现的历史配置。
