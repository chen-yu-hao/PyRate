# PyRate 与 Polyrate 2023 revision B 头对头审计

本报告比较当前 PyRate `0.1` 与官方 **Polyrate 2023 revision B**。基准归档为
`polyrate2023.0831.xz`，来源是 Zenodo record `8309384`，官方 MD5 为
`fa1423cc3f4a1279631b05f49228a2d2`；它对应 Mendeley Data DOI
`10.17632/vpf3y9cbhf.1`。每次 reference workflow 都重新下载并校验归档，随后生成
`report.json`、`case-manifest.json`、固定 unit 验证、legacy 执行验证和内核基准。

## 结论

**当前 PyRate 仍不是 Polyrate 的科学功能等价替代品。**

- Polyrate 在完整反应路径、ZCT/SCT/LCT/OMT、多层级校正、直接动力学、VRC、
  非谐性、状态选择和压力相关动力学方面仍具有决定性优势。
- PyRate 的优势是现代 Python API、强类型模型、JSON/CSV 输出、可读实现、
  C++17 热点内核、纯 Python 回退以及自动化的逐算例证据。
- PyRate 已跨过一个窄但真实的门槛：4 个直接频率 conventional-TST 官方输入可由
  legacy deck 和 `fu30`/`fu40` 文件直接构造并运行，打印的正向 TST 表全部通过门禁。
- 这 4 例尚未取得 Polyrate 分区函数、ZPE 和 `ΔG‡` 的独立官方 oracle，因此只记为
  **打印最终速率匹配**，不记为完整的中间量加最终速率数值复现。

## 官方 112-run 门禁

| 门禁 | 结果 | 含义 |
|---|---:|---|
| 官方主输入 | 112 | `testrun/**/*.dat` 的精确数量 |
| 语法解析 | **112/112** | section、block 和 keyword 可保留；不代表语义已执行 |
| 可由 legacy 输入直接求解 | **4/112** | 直接频率 harmonic TST 子集已贯通；其余算例保留逐例 blocker |
| 匹配打印的正向 TST 表 | **4/112** | 在声明相对容差和最后一位打印量化间隔下通过 |
| 完整中间量加最终速率复现 | **0/112** | 尚无算例同时通过分区函数、ZPE、`ΔG‡` 等 staged oracle |
| 官方 `fu15` 文件 | 102 | 另有 10 个主算例没有 `fu15` oracle |
| 可结构化读取的 `fu15` | **101/102** | 唯一例外 `oh3tr10.fu15` 是空文件 |
| 解析出的速率表 | 113 | 包括标准 summary、compact FINAL/IVTST 和 compact TST 表 |
| 显式打印速率单位的文件 | 86 | 可进入一般单位归一化流程 |
| 未打印速率单位的 compact TST 文件 | 15 | 通用读取器保留 `unspecified`；执行验证从已审计的反应级数上下文确定单位 |

已直接执行并匹配打印正向 TST 表的四例是：

- `ch5fu30tr1`
- `ch5fu30tr4`
- `ch5fu40tr1`
- `ch5fu40tr4`

“112/112 可解析”只证明语法保真，不能解释为“112/112 兼容”。同样，四个最终速率表
通过也不能代替完整 staged intermediate validation。机器可读 manifest 分别记录
`semantic_status`、`legacy_execution_status`、`printed_rate_status`、
`intermediate_validation_status` 和总体 `numerical_status`。

## 16 个核心候选的当前状态

最接近首批 TST/CVT 数值验收的 16 个候选是：

`ch5fu30tr1`–`ch5fu30tr4`、`ch5fu40tr1`–`ch5fu40tr4`、`ch5j2tr2`、
`cwmctr1`、`h3o2fu31tr1`–`h3o2fu31tr4`、`oh3fu30tr1`、`oh3fu40tr1`。

| 状态 | 数量 | 主要条件或 blocker |
|---|---:|---|
| 可执行且打印 TST 表匹配 | 4 | `fu30`/`fu40` 直接频率、conventional TST、明确几何和温度网格 |
| 尚未执行 | 12 | 主要为 Hessian-to-normal-mode、CVT、非谐/扭转语义或尚未映射的控制数据 |

当前 mapper 不会为缺失频率、未知电子激发能单位或 Hessian 模态凭空填值。被阻塞的算例
仍保留语法树、固定文件摘要和具体 blocker，供后续实现逐项消除。

## 主输入的排他分类

| 分类 | 数量 | 当前边界 |
|---|---:|---|
| 高级算法未实现 | 63 | SCT/LCT、μVT/CUS、VRC 或其他端到端高级路径 |
| 部分 ZCT | 22 | 只有一维 WKB 内核，没有完整 MEP/MEPSAG ZCT |
| 核心方法候选 | 16 | 其中 4 例已执行；其余 12 例仍被 legacy 或数值路径语义阻塞 |
| 控制/接口型输入 | 11 | 没有识别到常规速率方法，包含接口和非标准控制算例 |

官方主输入中的方法关键字出现次数为：TST 100、CVT 85、ZCT 84、SCT 61、
LCT 3、μVT 3、US 2、CUS 1、VRC 1。一个输入可同时请求多个方法，因此这些数字
不能相加作为独立算例数。

## 已实现的 legacy 执行语义

首个执行子集支持：

- `*RATE/TST` 和 `TEMP` 温度网格；
- `*ENERGETICS/POTENTIAL UNIT30|UNIT31|UNIT40`；
- `ATOMS`、各物种 `GEOM`、`INPUNIT` 的 Bohr/Å 转换；
- 固定 unit 文件中的势能、直接实频和鞍点虚频；
- 原子、线性和非线性刚性转子；
- 零激发能电子简并度；
- `SIGMAF` 作为反应路径简并因子，而不是错误地并入分子旋转对称数；
- `pyrate run`、`pyrate trace`、`pyrate inspect` 和
  `pyrate validate --require-executable` 的 legacy 路径。

当前明确拒绝：

- 只有 Hessian、尚未完成 Polyrate 投影/模态语义的数据；
- 非零 legacy 电子激发能但单位语义尚未映射的数据；
- CVT/ICVT、ZCT/SCT/LCT、μVT/US/CUS、非谐和 torsional 请求；
- 缺失势能、频率单位、几何、原子表或温度网格的输入。

## 功能矩阵

| 能力 | Polyrate 2023 | 当前 PyRate | 头对头判定 |
|---|---|---|---|
| Legacy 输入 | 原生完整语义 | 112/112 语法保留；4/112 可执行；其余逐例 blocker | Polyrate |
| 固定 unit 30/31/40 | 原生生产格式 | 保留式读取、统一 stationary-point/path 模型；直接频率 TST 子集可执行 | Polyrate 覆盖更广 |
| 平动/转动/振动/电子分区函数 | 完整生产路径，含更多模型 | 理想气体 + 刚性转子 + 谐振子/ZPE + 电子态 | PyRate 仅核心子集 |
| CTST/TST | 单/双分子及多种物理选项 | harmonic RRHO；4 个官方 legacy 最终速率表匹配 | Polyrate 覆盖更广 |
| canonical CVT | 反应路径 VTST、路径积分和多种坐标/算法 | 对用户提供的离散路径点做自由能瓶颈搜索；官方 legacy 未贯通 | Polyrate |
| microcanonical VTST / US / CUS | 完整 | 仅 Beyer–Swinehart DOS 基础 | Polyrate |
| Wigner | 支持 | 支持；legacy 映射仍需更多官方门禁 | 核心能力接近 |
| ZCT | 完整 MEP/MEPSAG | 一维 WKB building block | Polyrate |
| SCT / LCT / OMT | 完整多维半经典方法 | 未实现 | Polyrate |
| 反应路径生成 | Euler、Page–McIver、VRP 等 | 只读取预计算路径点 | Polyrate |
| 直接动力学与电子结构接口 | 多种接口与文件路径 | 未实现适配器 | Polyrate |
| Dual-level / IVTST / ISPE / IOC | 支持 | 未实现 | Polyrate |
| Loose-TS / VRC-VTST | canonical、microcanonical、E,J 分辨 | 未实现 | Polyrate |
| Hindered rotor / 非谐性 | 多种扭转与振动非谐模型 | 当前仅谐振子 | Polyrate |
| 压力相关速率 | SS-QRRK utility | 只有 DOS primitive | Polyrate |
| 溶液、固体、表面 | 官方程序覆盖相应模型 | 未实现 | Polyrate |
| 输出与自动化 | 传统固定格式和 restart/unit 文件 | JSON、CSV、表格、pytest、逐例 CI artifact | PyRate 工程体验更现代 |
| 可维护性/嵌入 Python | Fortran 90 大型成熟程序 | 模块化 Python API，易于组合和单元测试 | PyRate |

## 速率比较的容差口径

compact TST `fu15` 表只打印少量有效数字。验证器不把打印舍入误差当成算法误差，采用：

`允许绝对误差 = 5e-3 × |reference| + 最后一位打印值的半个量化间隔`

每个温度点同时保留原始打印 token、绝对误差、相对误差、量化间隔和有效相对上限。
其中 `ch5fu30tr4` 的最大原始相对误差略高于 `5e-3`，但位于三位有效数字表值的合法
舍入区间内，因此通过。该规则只适用于打印表；若取得未舍入 oracle，应使用更严格容差。

## 性能证据的边界

当前 CI 继续测量 **PyRate C++ 内核相对 PyRate Python fallback**，证明热点下沉是否有效。
四个可执行官方 TST 算例尚未形成同一硬件、同一编译配置下的 Polyrate 与 PyRate
端到端 wall-clock 基准，因此仍不宣称 PyRate 比 Polyrate 更快。当前 benchmark 数字
保存在每次 reference audit artifact 中，不将单次共享 runner 结果写成稳定性能承诺。

## 下一实现优先级

1. 实现 fixed-unit Hessian 到质量加权、平移/转动投影和 Polyrate-compatible 正规模的
   legacy 转换，争取打通其余直接 TST 核心候选。
2. 从官方详细输出中抽取分区函数、ZPE 和 `ΔG‡` oracle，让已运行的 4 例从
   “打印最终速率匹配”升级为完整 staged numerical reproduction。
3. 对齐 `SECOND`、`OPTIMIZATION`、质量/同位素、对称数和标准态默认规则。
4. 贯通官方 CVT 局部瓶颈精修和路径插值，完成 16 个核心候选的剩余部分。
5. 接入完整 MEP、振动绝热势和 MEPSAG，处理 22 个 ZCT 类算例。
6. 再实现 SCT/CD-SCSAG、LCT/LCG3/LCG4、OMT 及更高级工作流。

## 验收口径

只有同时满足以下条件，官方算例才可计为完整通过：

1. 同一 legacy 输入或经审计的无损语义转译；
2. 相同物理模型、默认值、单位和标准态；
3. 关键中间量有逐温度或逐路径点 oracle 回归；
4. 最终速率满足预声明容差；
5. C++ 与 Python 回退一致；
6. 不以可解析、可读取或单个最终数字接近代替端到端复现。
