# PyRate 审计、Polyrate 2023 对照与并行路线图

> 审计基线：`main` @ `02b7d09`（PyRate 0.1.0）。日期：2026-11-04。
> 本文档是计划文件，不是兼容性声明；兼容性以 [`compatibility.md`](compatibility.md) 为准。

## 0. 结论摘要

- **工程状态良好**：可安装，C++ 内核可编译；原生与纯 Python 回退两种后端各通过 32 个测试。
- **Polyrate 数值对照尚未开始**：目前没有任何一个 Polyrate 官方算例被数值复现。
- **有 4 个数值问题必须先修**，否则即使有了参考输出，对照也会系统性失败：隧穿 `P(E)` 形式、隧穿能量下限、CVT 极值精修防护，以及质量表和物理常数。
- 功能覆盖约为 Polyrate 2023 的**入门子集**：RRHO 下的 TST/CVT、Wigner，再加一个 1D WKB 构件。核心差距在 MEP 路径计算、ZCT/SCT/LCT、μVT/CUS、直接动力学和 VRC-VTST。

## 进展（2026-11-04 更新）

| 项目 | 状态 | 说明 |
|---|---|---|
| Wave 0：接口冻结与公共重构 | ✅ 完成 | C++/`native` 按领域拆分；跨后端 parity、金标准回归、进程内 CLI 测试；ruff 与 mypy strict 进入 CI |
| A1 / N1、N2：隧穿核 | ✅ 完成 | 一致半经典 `P(E)`、非经典反射、`E₀` 阈值；对称 Eckart 误差 ≤7e-8 |
| A2 / N3：CVT 极值精修 | ✅ 完成 | 有界三点抛物线（可选四/五点）；1.2 万例随机属性测试无违例 |
| A3 / N4（质量部分）：同位素质量 | ✅ 完成 | AME2020 表，经两个独立来源交叉核对；`mass_model` 可切换 |
| N4（常数部分）：Polyrate 常数 profile | ⏳ 待办 | 需要 Polyrate 源码中的常数 |
| A4：参考数据与事实核查 | ⏳ 待办 | 与已开 PR #6 中的官方 112 例审计工作重叠，需先合并方向 |

## 1. 仓库审计

### 1.1 已验证的事实

| 检查项 | 结果 |
|---|---|
| `pip install -e .[test,dev]` | 成功，`pyrate._kernels` 编译通过（Python 3.13） |
| `pytest -q`（C++ 后端） | 32 passed |
| `PYRATE_DISABLE_NATIVE=1 pytest -q` | 32 passed |
| 覆盖率（进程内） | 68%；`cli.py` 0%（只经子进程测试），`calculation.py` 48%，`native.py` 39% |
| `ruff check src tests tools` | **32 个错误**（CI 未运行） |
| `mypy`（`strict = true`） | **15 个错误 / 6 个文件**（CI 未运行） |
| 示例 `harmonic_cvt.toml`、`wkb_tst.toml` | 可运行，输出有限正值 |

### 1.2 数值与物理问题（按严重度排序）

**N1：隧穿透射概率 `P(E)` 不是 Polyrate 采用的形式。** 严重度：高。
`native.wkb_exponent` 和 `thermal_wkb_kappa` 使用 `P(E) = exp(-2θ)`，并假定势垒以上 `P = 1`。Polyrate 的 ZCT/SCT 使用一致半经典形式 `P(E) = 1/(1+exp(2θ))`，并在势垒以上计入非经典反射 `P(E) = 1 − P(2V^AG − E)`。
在对称 Eckart 势垒上（V₀ = 40 kJ/mol，a = 0.35 Å，μ = 1 amu，虚频约 1357 cm⁻¹）的实测比较如下：

| T / K | PyRate κ | 一致半经典 κ（Polyrate 型） | 精确量子 κ | PyRate / 半经典 |
|---|---|---|---|---|
| 200 | 235.28 | 232.81 | 272.72 | 1.011 |
| 250 | 23.12 | 21.73 | 25.26 | 1.064 |
| 300 | 8.06 | 7.06 | 8.09 | 1.142 |
| 500 | 2.37 | 1.85 | 2.02 | **1.283** |
| 1000 | 1.42 | 1.16 | 1.21 | **1.230** |

即使输入完全相同的 `V_a^G(s)`，中高温下也会相差 20–30%，远超 `fu15` 的 5e-3 门槛。

**N2：隧穿能量下限错误。** 严重度：中高。
当前积分从反应物能量 `V(−∞)` 开始。Polyrate 的下限是 `E₀ = max(V_a^G(−∞), V_a^G(+∞))`。对吸能剖面，低于产物渐近线的能量本应 `P = 0`，实测却为 `P(5 kJ/mol) = 2.6e-23`、`P(15 kJ/mol) = 3.5e-16`（产物渐近线为 20 kJ/mol）。结果还会随剖面截断长度变化。

**N3：CVT 极值精修的防护过宽。** 严重度：中高。
`refine_maximum` 对 5 点做四次插值。只有当拟合值超过采样最大值 `2·max(1 kJ/mol, span)` 时才回退，这个阈值依赖绝对单位。随机非均匀 5 点测试中，被接受的最大超调为 **6.34 kJ/mol**，相当于 300 K 下速率偏低 **12.7 倍**。Polyrate 的默认极值插值方式需按手册确认，并做同等实现。

**N4：质量表和常数与 Polyrate 不一致。** 严重度：高（对 1e-10 门槛而言）。
`elements.py` 使用**平均原子量**（H 1.00784、C 12.011、Cl 35.45），而电子结构程序和 Polyrate 通常使用**丰度最高同位素质量**（H 1.007825、C 12.000、Cl 34.96885）。CH₄ 的 `q_trans` 因此偏差约 1e-3，Cl 体系的偏差可达约 2%。常数采用 CODATA 2018；Polyrate 使用内置的原子单位换算常数。
需要一个 `constants_profile = "polyrate"`，否则 `compatibility.md` 中“分区函数 ≤1e-10”的门槛不可达。

**N5：生成正规模分析缺少梯度投影。** 严重度：中。
`normal_modes()` 只投影平动和转动。非驻点（s ≠ 0）上的广义频率必须同时投影反应坐标（梯度方向），因此目前无法由 Hessian 生成 CVT 路径点的 3N−7 个频率。

**N6：缺少 CAG 因子，ZCT 与 CVT 路径脱节。** 严重度：中。
Polyrate 的 `CVT/ZCT` 满足 `κ^{CVT/ZCT} = κ^{ZCT} · κ^{CVT/CAG}`，其中 `κ^{CVT/CAG} = exp[−β(V^AG − V_a^G(s*_CVT))]`。PyRate 的隧穿剖面由用户单独提供，不来自路径 `V_MEP + ZPE(s)`，约化质量也是人为指定的；Polyrate 用质量标度坐标。

**N7：缺少逆向速率、平衡常数和反应路径简并度。** 严重度：中。
`Reaction.products` 从未参与计算，所以 `fu15` 的 reverse 表无法对照。另外，除物种对称数之外，没有独立的反应路径对称因子。

**N8：其他小问题。**
- `CalculationOptions` 中 `method == "cvt" and output_rate_unit == ""` 是无意义条件。
- `ReactionPath.coordinate_unit` 只作标签，从不参与换算。
- `legacy._SUPPORTED_SECTIONS` 名称误导：这些区段只能被解析，不能执行。
- Wigner 被乘到 CVT 速率上（`k_final = k_CVT·κ_W(TS)`），Polyrate 没有对应列，做映射时要避免误配。

### 1.3 测试与流程缺口

- 端到端测试只断言 `k > 0`，**没有金标准数值回归**。
- CONTRIBUTING 第 3 条要求“跨后端测试”，但目前没有在同一进程内比较 C++ 与 Python 输出的 parity 测试。
- CI 不运行 ruff/mypy，`pyproject` 中配置的 strict 实际未执行。
- 文档中关于 Polyrate 2023 的若干断言**未经核实**：“已移除 ICVT”、“112 个 test runs（46/85/27）”、`fu15` 的确切格式与列名。CPC 2024 的摘要确认 2023 版包含 VRC-VTST。

## 2. 与 Polyrate 2023 的头对头对比

状态图例：✅ 可用，🟡 部分可用 / 有偏差，❌ 缺失，❓ 待核实。

| 维度 | Polyrate 2023 | PyRate 0.1 | 差距 / 主要阻塞 |
|---|---|---|---|
| 语言与架构 | Fortran，COMMON block，文件单元 fuNN | Python + C++17，强类型 API，Python 回退 | PyRate 工程化更好 |
| 许可与发布 | Apache-2.0；CPC 294, 108933 (2024) | Apache-2.0，CI 覆盖三平台、多个 Python 版本 | — |
| 数值验证 | 官方 test runs（数量 ❓） | 0 个官方算例复现 | **最大差距** |
| 常数 / 质量 | 内部原子单位常数、同位素质量 ❓ | CODATA 2018、平均原子量 | 🟡 N4 |
| 分区函数（RRHO） | ✅ | ✅ | 待用 N4 profile 对照 |
| 反应路径 MEP 计算（ES1、Page–McIver 等） | ✅ | ❌（只读预计算点） | 需 PES hook 和积分器 |
| 沿路径的广义正规模 | ✅（笛卡尔 / 冗余内坐标） | 🟡（缺梯度投影 N5） | — |
| TST | ✅ | ✅ | — |
| CVT | ✅ | 🟡（极值精修 N3，s 未做质量标度） | — |
| ICVT | ❓（仓库文档称 2023 已移除） | ❌ | 先核实 |
| μVT / US / CUS | ✅ | ❌（只有 Beyer–Swinehart 内核） | — |
| Wigner | — | ✅ | 不在 Polyrate 对照范围内 |
| ZCT | ✅ | 🟡（1D 构件；N1、N2、N6） | **首个可达目标** |
| SCT | ✅ | ❌ | 需曲率 κ(s) 与 μ_eff |
| LCT / μOMT | ✅ | ❌ | — |
| 非谐振动（Morse、受阻转子等） | ✅ | ❌ | — |
| 双层次 / IVTST / ISPE / IOC | ✅ | ❌ | — |
| VRC-VTST | ✅（CPC 摘要确认） | ❌ | — |
| 直接动力学（电子结构接口） | ✅ | ❌ | — |
| 压力依赖（SS-QRRK 工具） | ✅（随分发包提供的工具） | ❌（只有 DOS 内核） | — |
| 逆向速率 / 平衡常数 | ✅ | ❌（N7） | — |
| 旧输入 fu5 | 原生 | 🟡（语法可保真解析，不能执行） | 语义映射 |
| 输出 | fu6 / fu15 等定长文本 | JSON / CSV / 表格 | 需 fu6 中间量提取器 |
| 对照工具 | — | ✅（`fu15` 比较器，失败语义严格） | 已就绪 |

**PyRate 的优势**：可读、可测、单位显式、有结构化输出和 Python API，原生加速且带回退，失败语义保守。
**Polyrate 的优势**：方法覆盖完整，有数十年的验证积累，并自带全部参考算例。

## 3. 下一步待办（按优先级）

### P0：数值正确性（任何头对头对照的前置条件）

1. **[N1/N2] 修正隧穿核。** C++ 与 Python 两端同步完成：
   - `P(E) = 1/(1+e^{2θ})`；
   - 非经典反射，积分到 `2V^AG`；
   - 下限取 `E₀ = max(V_R, V_P)`；
   - 加入 Eckart 解析测试，门槛为 ≤1e-6 对一致半经典参考。
2. **[N3] 重写 CVT 极值精修。** 先按手册确认 Polyrate 的插值方式，然后：
   - 拟合值超过采样最大值超出容差时回退，且容差不依赖单位；
   - 检查局部凹性；
   - 加入随机属性测试，断言超调 ≤ 设定容差。
3. **[N4] 新增 `constants_profile`（`codata2018` / `polyrate`）。**
   - 提供同位素质量表，并由 profile 选择默认值；
   - 从 Polyrate 源码提取它实际使用的常数；
   - 加分区函数 1e-10 级回归。
4. **测试与 CI 硬化。**
   - 进程内跨后端 parity 测试；
   - 两个示例的金标准数值回归；
   - CI 加入 ruff 与 mypy，并修复现有 47 个问题；
   - 改为进程内测试 `cli.main`。
5. **事实核查。** 获取 Polyrate 2023 分发包和手册（DOI `10.17632/vpf3y9cbhf.1`），核实 ICVT、test run 数量，以及 `fu15` / `fu6` 的格式与列名，并更正 `head-to-head.md` 和 `compatibility.md`。

### P1：第一次真实头对头（目标：3–5 个解析势能面的单层次算例）

6. 用 gfortran 构建 Polyrate，并运行 `tools/import_polyrate_reference.py` 生成 manifest。挑选**解析 PES、无直接动力学、只有 TST/CVT/ZCT 列**的算例作为第一批。
7. **fu6 中间量提取器**：提取 `V_MEP(s)`、`V_a^G(s)`、沿路径频率、`ΔG(s,T)`、`s*_CVT`、`κ` 和 CAG。这样可以逐层对照，而不是只比较最终速率。
8. **路径管线** [N5/N6]：
   - 质量标度坐标 s；
   - 生成正规模并投影梯度；
   - 由路径自动构造 `V_a^G(s)` 作为 ZCT 剖面；
   - 计算 CAG 因子，输出 `k_cvt_zct` 列，并加入比较器白名单。
9. **[N7]** 逆向速率、平衡常数和反应路径对称因子；比较器支持 reverse 表。
10. **legacy 语义映射（子集）**：只覆盖第一批算例用到的关键字，实现 fu5 → `PathData` 的转换，加 `pyrate compare` 子命令，并生成批量对照报告。

### P2：功能扩展（每项都要有官方算例验收）

11. SCT：曲率耦合 κ(s)、μ_eff、CD-SCSAG。
12. μVT / US / CUS：微正则态和与态密度，加入转子贡献（Stein–Rabinovitch）。
13. PES hook 接口（Python callable 与 C/Fortran 插件），以及 MEP 积分器（ES1、Page–McIver）。
14. 非谐性：Morse、受阻转子。
15. 双层次、IVTST、ISPE、IOC（对应 `fu50` / `fu51` 等）。

### P3：远期

16. LCT 与 μOMT。
17. VRC-VTST。
18. SS-QRRK 与压力依赖。
19. 直接动力学适配器（Gaussian、ORCA 等）。
20. 定长输出兼容层。

## 4. 多 agent 并行计划

### 4.1 原则

- **先冻结接口，再并行**：Wave 0 由一个 agent 串行完成接口和公共重构，避免后续多个 agent 争抢同一文件。
- **文件所有权**：每个 agent 只改自己负责的文件。共享文件（`docs/compatibility.md`、`__init__.py`、`pyproject.toml`、CI 配置）只由集成 agent 修改。
- **分支**：每个任务一个分支 `agent/<id>-<topic>`，PR 合入集成分支；在 Claude Code 中可使用 `isolation: "worktree"`。
- **统一验收**：
  - 两种后端测试都通过，ruff 与 mypy 无错，parity 测试通过；
  - 新数值功能必须带解析参考或官方算例；
  - 只有通过 `compatibility.md` 的 6 条门槛才能改成 Implemented。

### 4.2 波次与依赖

```text
Wave 0 (串行, 1 agent)        Wave 1 (并行, 4 agents)          Wave 2 (并行, 4 agents)          Wave 3 (并行)
┌──────────────────────┐     ┌─ A1 隧穿核 N1/N2 ─────────┐    ┌─ B2 ZCT+CAG 集成 ◄─(A1,B1) ┐   SCT ◄─(B1 曲率)
│ W0 集成/架构         │────►├─ A2 CVT 精修 N3 ──────────┤───►├─ B1 路径管线 N5 ◄─(A3)      ├─► μVT/US/CUS
│ 接口冻结+重构+CI     │     ├─ A3 常数/质量 N4 ─────────┤    ├─ B3 逆向/K_eq/σ N7          │   PES hook + MEP
└──────────────────────┘     └─ A4 参考数据/事实核查 ────┘    └─ B4 legacy→PathData+compare ◄─(A4)  非谐性
                                                                         │
                                                集成 agent：每次合并后运行头对头看板
```

### 4.3 各 agent 职责卡

| ID | 目标 | 负责的文件 | 依赖 | 交付与验收 |
|---|---|---|---|---|
| **W0 集成 / 架构** | ① 将 `cpp/kernels.cpp` 拆成 `wkb.cpp` / `fit.cpp` / `dos.cpp` / `module.cpp`，将 `native.py` 拆成 `native/` 包；② 冻结 `PathData`、`TunnelingResult`、`ConstantsProfile` 和 manifest v2 的数据契约，只写空实现与类型；③ 修复 ruff 与 mypy 并加入 CI；④ parity 测试骨架与金标准回归 | 全仓库（独占期） | 无 | CI 全绿；接口文档 `docs/contracts.md` |
| **A1 隧穿核** | 第 1 项 | `cpp/wkb.cpp`、`native/wkb.py`、`tunneling.py`、`tests/test_tunneling*.py` | W0 | Eckart 门槛 ≤1e-6；吸能剖面 `P(E<E₀)=0`；parity ≤1e-12 |
| **A2 CVT 精修** | 第 2 项 | `cpp/fit.cpp`、`native/fit.py`、`rates.py` 中的 CVT 部分、`tests/test_rates*.py` | W0 | 随机属性测试；超调 ≤ 容差；与手册算法一致 |
| **A3 常数 / 质量** | 第 3 项 | `constants.py`、`elements.py`、`partition.py`、`tests/test_partition*.py` | W0 | 两套 profile；分区函数回归 ≤1e-10 |
| **A4 参考数据** | 第 5、6、7 项 | `reference.py`、`tools/`、`docs/head-to-head.md`、`docs/reference-validation.md` | W0；**需要用户提供 Polyrate 分发包或开放网络** | 本地 manifest；第一批 3–5 个算例清单；fu6 提取器及测试 |
| **B1 路径管线** | 第 8 项前半 | `normal_modes.py`、新增 `path.py`、`geometry.py` | A3 | 梯度投影后频率与解析模型一致；`V_a^G(s)` 构造 |
| **B2 ZCT 集成** | 第 8 项后半 | `tunneling.py` 的集成层、`calculation.py` 的隧穿分支 | A1、B1（可先用契约 mock） | 与第一批算例的 `CVT/ZCT` 列一致 |
| **B3 逆向 / 平衡** | 第 9 项 | `rates.py` 的非 CVT 部分、`calculation.py` 的输出行、`head_to_head.py` | W0 | reverse 表可比较；满足细致平衡测试 |
| **B4 legacy 翻译** | 第 10 项 | 新增 `legacy_semantics.py`、`cli.py` 中的 `compare` 子命令 | A4 | 第一批算例从 fu5 直接运行，产出对照报告 |

B2 与 B3 都会改 `calculation.py`。由 W0 预先把它拆成 `_tunneling_step()` 和 `_rate_row()` 两个函数，各归一方，从而避免冲突。

### 4.4 节奏与规模建议

- **Wave 0**：1 个 agent，最先完成，是一切并行的前提。
- **Wave 1**：4 个 agent 同时进行。A4 不需要等其他 agent，但需要用户先提供 Polyrate 分发包。
- **Wave 2**：4 个 agent。B1、B3 在 A3 合并后即可开始；B2、B4 可以先基于契约 mock 编写。
- **集成 agent**：贯穿全程，负责合并 PR、更新 `compatibility.md`，并维护“已通过算例数 / 总数”看板。
- **里程碑**：
  - **M1**：P0 全部完成；
  - **M2**：第一批 3–5 个算例的 TST/CVT/ZCT 列在 5e-3 内一致（第一次真实头对头）；
  - **M3**：SCT 与 μVT 各有 ≥2 个官方算例通过。

### 4.5 启动单个 agent 的提示模板

```text
你是 PyRate 的 <ID> agent。目标：<职责卡中的目标>。
只允许修改：<负责的文件>；其他文件只读，需要改时在 PR 描述中提出。
遵守 docs/contracts.md 中的接口契约，以及 docs/compatibility.md 中的“1:1 完成”定义。
完成标准：<验收>；pytest 在原生与 PYRATE_DISABLE_NATIVE=1 两种后端下都通过；ruff 与 mypy 无错。
在分支 agent/<id>-<topic> 上提交，PR 描述中列出数值证据（误差表）。
```
