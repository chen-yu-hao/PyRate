# Polyrate 兼容矩阵

本文档是兼容性声明的唯一依据。状态含义：

- **Implemented**：具有公共 API、原生/回退一致性测试和端到端验证；
- **Partial**：核心算法与常见工作流可用，但尚未覆盖 Polyrate 的全部遗留选项或外部耦合接口；
- **Planned**：尚未作为可用功能发布。

| Polyrate 能力 | PyRate 0.2 状态 | 当前边界 / 验收条件 |
|---|---|---|
| `*GENERAL` 等 section 语法 | Implemented (单能级路径工作流) | 完整解析全部 112 个 Polyrate 2023 `.dat` 输入；支持静点与 `.fu40` 路径直接执行 |
| 原子、几何、单位与同位素模型 | Implemented | 支持 IUPAC 与 Polyrate 单同位素（`interface.F` `ams`）质量约定，以及 `2H`/`13C`/`18O`/`37Cl` 等显式同位素 |
| 理想气体平动分区函数 | Implemented | 1 bar 标准态，单/双分子单位与平衡常数 $K_c/K_p$ 测试 |
| 刚性转子分区函数 | Implemented | 原子、线性、非线性和反应统计因子 `SIGMAF`/`SIGMAR` |
| 谐振子分区函数与 ZPE | Implemented | 支持频率缩放 `frequency_scale`、低频下限 `frequency_floor_cm1`；负频率排除 |
| 电子分区函数 | Implemented | 多电子态、能级和简并度 |
| CTST / TST | Implemented | 单/双分子，与 Polyrate `ch5fu40` 系列误差 `0.08%` |
| canonical VTST / CVT | Implemented | 温度依赖自由能瓶颈与 `FITMAX` 精修，与 Polyrate `ch5fu40` 系列误差 `0.08%` |
| Polyrate 3/5-point `FITMAX` 行为 | Implemented | 次端点 `index == 1` 或 `N-2` 采用 3 点抛物线，内部点采用 5 点四次多项式，及 `refine_minimum` |
| CUS (Canonical Unified Statistical) | Implemented | 多重自由能极大值与中间极小值的统一统计修正 |
| CAG (Classical Adiabatic Ground-State) | Implemented | $\kappa^{\text{CAG}}_{\text{TST}}$ 与 $\kappa^{\text{CAG}}_{\text{CVT}}$ |
| Wigner | Implemented | 单一虚频 |
| ZCT / MEPSAG | Implemented | 沿 MEP 振动绝热基态势 $V_a^G(s)$ 的 Kemble 均匀半经典 WKB 与 `BOLTZ` 非经典反射 |
| SCT / CD-SCSAG | Implemented | 反应路径曲率 $B_{mF}(s)$、`TPCDSC` 离心拐点与位置相关有效质量 $\mu_{\text{eff}}(s)$ 积分 |
| Eckart 隧穿 | Implemented | 对称与非对称 Eckart 解析透射系数与热平均 |
| COMT / μOMT | Partial | 仅在 ZCT 与 SCT 候选间逐温度或逐能量择优；由于尚无 LCT 且 $\mu_{\text{eff}}(s) \le \mu$，当前结果恒等于 SCT，**不能**与 Polyrate 的 `CVT/COMT`、`CVT/mOMT` 列等同。待 LCT 实现后升级 |
| LCT / LCG3 / LCG4 | Planned | 需非绝热区 corner-cutting 几何直线路径与激发振动态求和 |
| ICVT | Historical only | Polyrate 2023 已移除 ICVT；仅作为旧版本兼容范围，不计入 2023 必选功能 |
| μVT / US | Implemented | Beyer–Swinehart 振动态计数与经典转动态和 $N_{\text{rv}}(E, s)$，能量分辨最小通量与统一统计 |
| Hindered rotor | Implemented | Pitzer–Gwinn 与 Polyrate `BES0` 修正贝塞尔函数扭转内旋转修正 |
| Morse / anharmonic vibration | Implemented | 一阶非谐常数 $\omega_e x_e$ 精确有界 Morse 能级梯与非谐 ZPE（QQ 待定） |
| `fu40` 电子结构路径文件 | Implemented | 支持 `freqsource read` 与 `freqsource hessian`（紧缩/全矩阵）、`PROJCT` 梯度投影与 `RPHB` 曲率计算 |
| 曲线内坐标/广义频率 (`curv1/curv2/curv3`) | Implemented | Wilson GF、冗余 B 矩阵伪逆、Cartesian Hessian 投影和路径点显式 `intrinsic_frequencies_cm1` |
| MEP integration / Page–McIver / Euler | Planned | 当前读取预计算路径点或 `.fu40`；势能面上直接步进积分待实现 |
| Direct dynamics hooks (`POT`) | Planned | 尚无编译期 Fortran 势能面（77 个算例）或 Gaussian/GAMESS 实时调用适配器 |
| Dual-level correction / ISPE | Implemented | TOML `[dual_level]` 支持 high-minus-low 路径能量、坐标插值和 `fu50/fu51` 风格数据；`ivtst`/`ioc` 共享同一校正对象 |
| IVTST / IOC | Implemented | `calculate_rates(..., method="ivtst")` 在校正后的自由能路径上执行变分瓶颈搜索；`ioc` 将参考点校正传播到路径 |
| VRC-VTST | Implemented | `[vrc]` 确定性 loose-TS facet/方向采样、振动分区权重、对称数和瓶颈坐标 |
| SS-QRRK / pressure dependence | Implemented | `rrkm_microcanonical_rate` 暴露 $k(E)=N^\ddagger/(h\rho)$；`PressureDependence` 与 `pressure_dependent_rate` 提供 Lindemann/Troe 压力网格和碰撞效率 |
| Solvent / surface options | Implemented | `[environment]` 支持显式溶剂自由能、介电连续介质元数据、表面吸附能、位点密度和取向因子 |
| `fu30/fu31` legacy unit files | Partial | `.fu40` 已完整实现；旧式固定格式 `.fu30/.fu31` 待映射 |
| `fu15` summary reader | Implemented | 完整支持 Polyrate 2023 全部 101 个非空 `.fu15` 文件（含 3 类表头与多子表拼接） |
| Polyrate fixed-format output 1:1 | Planned | 当前输出为结构化 JSON/CSV/表格 |

对应的逐温度、逐列数值比较工具与严格失败语义见 [`docs/head-to-head.md`](head-to-head.md)。

## “1:1 完成”的定义

单个功能只有同时满足以下条件才可从 Partial/Planned 改为 Implemented：

1. 对应 legacy keyword 可解析并执行；
2. 与 Polyrate reference run 使用相同物理模型；
3. 关键中间量（分区函数、`ΔG(s)`、$V_a^G(s)$、$\mu_{\text{eff}}(s)$、action、瓶颈位置）有回归测试；
4. 最终速率在预先声明的误差容限内；
5. 原生和 Python 回退结果一致；
6. 文档清楚说明单位、默认值和不兼容行为。

建议数值门槛：纯代数/分区函数相对误差 `≤1e-10`；路径拟合和数值积分 `≤1e-6`；完整速率对 legacy 文本舍入前数据 `≤1e-3`，对仅有三位有效数字的 `fu15` 表采用 `≤5e-3`（对依赖五次样条导数的多维曲率隧穿采用 `≤1e-1`）。

