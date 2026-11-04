# PyRate 与 Polyrate 2023 revision B 头对头审计

本报告比较当前 PyRate `0.1` 与官方 **Polyrate 2023 revision B**。基准归档为
`polyrate2023.0831.xz`，来源是 Zenodo record `8309384`，官方 MD5 为
`fa1423cc3f4a1279631b05f49228a2d2`；它对应 Mendeley Data DOI
`10.17632/vpf3y9cbhf.1`。

审计提交：`243f3cb90e293e7b114d6f8dd4c8edd2d040f753`  
GitHub Actions run：`41818116487`  
审计 artifact：`13314999383`  
artifact SHA-256：`15716f457a93a16ed2df283de53e817a722c56d425a7585cd3ff368e1c70984d`

## 结论

**当前 PyRate 不是 Polyrate 的科学功能等价替代品。**

- Polyrate 在完整反应路径、ZCT/SCT/LCT/OMT、多层级校正、直接动力学、VRC、
  非谐性、状态选择和压力相关动力学方面仍具有决定性优势。
- PyRate 的优势是现代 Python API、强类型 TOML、JSON/CSV 输出、可读实现、
  C++17 热点内核、纯 Python 回退以及自动化测试。
- PyRate 已经成为可复现的 **解析、oracle 读取和逐点比较框架**，但尚未跨过
  “执行官方输入”和“数值复现官方结果”两道门槛。

## 官方 112-run 门禁

| 门禁 | 结果 | 含义 |
|---|---:|---|
| 官方主输入 | 112 | `testrun/**/*.dat` 的精确数量 |
| 语法解析 | **112/112** | section、block 和 keyword 可保留；不代表语义已执行 |
| 可由 legacy 输入直接求解 | **0/112** | 固定 unit 文件、能量/频率语义和高级算法尚未贯通 |
| 端到端数值复现 | **0/112** | 尚无官方算例通过预声明容差门禁 |
| 官方 `fu15` 文件 | 102 | 另有 10 个主算例没有 `fu15` oracle |
| 可结构化读取的 `fu15` | **101/102** | 唯一例外 `oh3tr10.fu15` 是空文件 |
| 解析出的速率表 | 113 | 包括标准 summary、compact FINAL/IVTST 和 compact TST 表 |
| 显式打印速率单位的文件 | 86 | 可进入单位归一化流程 |
| 未打印速率单位的 compact TST 文件 | 15 | 单位记为 `unspecified`，比较器拒绝猜测反应级数 |

“112/112 可解析”只证明语法保真；不能解释为“112/112 兼容”或“112/112 通过”。
同理，读取 `fu15` 只是获得 reference oracle，不是 PyRate 的数值通过证据。

## 主输入的排他分类

| 分类 | 数量 | 当前阻塞点 |
|---|---:|---|
| 高级算法未实现 | 63 | SCT/LCT、μVT/CUS、VRC 或其他端到端高级路径 |
| 部分 ZCT | 22 | 只有一维 WKB 内核，没有完整 MEP/MEPSAG ZCT |
| 核心方法但 legacy 语义阻塞 | 16 | TST/CVT/Wigner 方法本身存在，但固定 unit/section 语义未转译 |
| 控制/接口型输入 | 11 | 没有识别到常规速率方法，包含接口和非标准控制算例 |

官方主输入中的方法关键字出现次数为：TST 100、CVT 85、ZCT 84、SCT 61、
LCT 3、μVT 3、US 2、CUS 1、VRC 1。一个输入可同时请求多个方法，因此这些数字
不能相加作为独立算例数。

## 功能矩阵

| 能力 | Polyrate 2023 | 当前 PyRate | 头对头判定 |
|---|---|---|---|
| Legacy 输入 | 原生完整语义 | 112/112 语法保留；语义不完整 | Polyrate |
| 平动/转动/振动/电子分区函数 | 完整生产路径，含更多模型 | 理想气体 + 刚性转子 + 谐振子/ZPE + 电子态 | PyRate 仅核心子集 |
| CTST/TST | 单/双分子及多种物理选项 | harmonic RRHO，TOML 模型可运行 | Polyrate 覆盖更广 |
| canonical CVT | 反应路径 VTST、路径积分和多种坐标/算法 | 对用户提供的离散路径点做自由能瓶颈搜索 | Polyrate |
| microcanonical VTST / US / CUS | 完整 | 仅 Beyer–Swinehart DOS 基础 | Polyrate |
| Wigner | 支持 | 支持 | 核心能力接近，仍需官方数值门禁 |
| ZCT | 完整 MEP/MEPSAG | 一维 WKB building block | Polyrate |
| SCT / LCT / OMT | 完整多维半经典方法 | 未实现 | Polyrate |
| 反应路径生成 | Euler、Page–McIver、VRP 等 | 只读取预计算路径点 | Polyrate |
| 直接动力学与电子结构接口 | 多种接口与文件路径 | 未实现适配器 | Polyrate |
| Dual-level / IVTST / ISPE / IOC | 支持 | 未实现 | Polyrate |
| Loose-TS / VRC-VTST | canonical、microcanonical、E,J 分辨 | 未实现 | Polyrate |
| Hindered rotor / 非谐性 | 多种扭转与振动非谐模型 | 当前仅谐振子 | Polyrate |
| 压力相关速率 | SS-QRRK utility | 只有 DOS primitive | Polyrate |
| 溶液、固体、表面 | 官方程序覆盖相应模型 | 未实现 | Polyrate |
| 并行和热点计算 | Fortran 90，部分 MPI | Python orchestration + C++17 热点内核 | 不同定位 |
| 输出与自动化 | 传统固定格式和 restart/unit 文件 | JSON、CSV、表格、pytest、CI artifact | PyRate 工程体验更现代 |
| 可维护性/嵌入 Python | Fortran 90 大型成熟程序 | 模块化 Python API，易于组合和单元测试 | PyRate |

## 性能证据的边界

本次只能公平测量 **PyRate C++ 内核相对其 Python 回退**，不能声称相对 Polyrate
更快，因为当前没有任何官方输入能在两端以同一物理模型端到端执行。

在 GitHub-hosted Ubuntu、Python 3.12.16、每项 2 次重复的诊断基准中：

| 内核 | C++17 | Python fallback | 内部加速比 |
|---|---:|---:|---:|
| Beyer–Swinehart DOS | 0.001323 s/call | 0.021503 s/call | 16.25× |
| thermal WKB | 0.000477 s/call | 0.105908 s/call | 222.23× |

这些数字只证明热点下沉到 C++ 有效；它们不是 PyRate-vs-Polyrate 性能结论。

## 最近完成的审计改进

原始读取器只识别标准 `Summary of forward/reverse rate constants`。官方全集暴露出
另外两类合法格式：

1. `FINAL RATE CONSTANTS [IVTST-n] (...)`；
2. `Forward TST rate constants:`。

读取器现已支持这两类格式，并把 compact FINAL 中的
`cm**3/molecule-sec` 归一化为二级速率单位。compact TST 表没有打印单位，仍保留
`unspecified`，以防止无证据的单位猜测。测试套件为 **36 passed**。

## 实现优先级

1. **先打通 16 个核心候选算例**：实现 `ENERGETICS`、`SECOND`、
   `OPTIMIZATION` 以及 `fu30/fu31/fu40` 固定 unit 文件的语义转译。
2. 对每个核心算例先比较分区函数、ZPE、`ΔG‡`、CVT 瓶颈位置，再比较最终速率；
   不允许只对最终数字拟合。
3. 接入完整 MEP、振动绝热势和 MEPSAG，完成 22 个 ZCT 类算例。
4. 再实现 SCT/CD-SCSAG、LCT/LCG3/LCG4 和 OMT，处理 63 个高级算法算例。
5. 最后扩展 direct dynamics、dual-level、VRC、非谐性和压力相关模块。

当前最接近首批数值通过的 16 个核心候选是：

`ch5fu30tr1`–`ch5fu30tr4`、`ch5fu40tr1`–`ch5fu40tr4`、`ch5j2tr2`、
`cwmctr1`、`h3o2fu31tr1`–`h3o2fu31tr4`、`oh3fu30tr1`、`oh3fu40tr1`。

## 验收口径

只有同时满足以下条件，官方算例才可计为通过：

1. 同一 legacy 输入或经审计的无损语义转译；
2. 相同物理模型、默认值、单位和标准态；
3. 关键中间量有逐温度/逐路径点回归；
4. 最终速率满足预声明容差；
5. C++ 与 Python 回退一致；
6. 不以可解析、可读取或单个最终数字接近代替端到端复现。
