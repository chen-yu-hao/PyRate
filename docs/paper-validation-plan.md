# 软件论文科学算例规划

本文档为 PyRate 软件论文规划一组**计算量小、可复现、以经典论文和标准数据为参照**的科学算例。每个模块至少对应一个主算例；所有算例在普通笔记本上应在数秒到数分钟内完成。

> 状态：规划 + 预研。第 7 节中的数值来自 commit `02b7d09`、Python 回退后端（`PYTHONPATH=src`，未编译 C++）的一次性探针脚本，仅用于证明算例可行、给出量级。正式数值以后续 `benchmarks/` 脚本在两个后端上的结果为准。
>
> 文中标注“需核对原文”的参数（力常数、驻点性质、表格数值）是凭经验给出的起始值，写进论文前必须对照原文逐项核实。

## 0. 原则

### 0.1 证据类型

| 记号 | 含义 | 典型阈值 |
|---|---|---|
| **V** Verification | 有闭式解或精确恒等式（解析积分、乘积规则、Laplace 变换） | 机器精度或可预测的离散误差阶 |
| **R** Reproduction | 重现经典论文或标准数据库（JANAF、CCCBDB 等）的数表/曲线 | 由参考数据的有效位数和模型差异决定 |
| **H** Head-to-head | 在**相同物理模型**下与 Polyrate 2023 官方输出比较 | 见 `docs/compatibility.md` 第 54 行门槛 |

### 0.2 约束

1. 每个算例都在 C++ 和 Python 回退两个后端上运行，并报告二者差异。
2. 只陈述算例实际证明的内容。范围与 `docs/compatibility.md` 一致：一维 WKB 不称为 ZCT/SCT，CVT 不包含 MEP 积分。
3. 所有参考数值放进 `benchmarks/*/data/*.csv`，列出文献、表号/页码和单位。不使用来源不明的数字。
4. 光谱/热化学比较必须显式给出质量约定（同位素质量或天然丰度原子量），见 §6.2。

## 1. 算例总览

| 编号 | 模块 | 算例 | 类型 | 核心参考 | 主要验收指标 | 预计耗时 | 是否需外部数据 |
|---|---|---|---|---|---|---|---|
| B1 | `geometry.py` | 五个小分子的主惯性矩与转动常数 | R + V | CCCBDB；Herzberg (1966) | A/B/C 与文献一致到几何有效位数；刚体变换不变性 ≤1e-12 | <1 s | 少量几何与转动常数 |
| B2 | `normal_modes.py` | H₂O/HDO/D₂O 谐振频率与 Teller–Redlich 乘积规则 | V + R | Hoy, Mills & Strey (1972)；Redlich (1935) | 乘积规则 ≤1e-9；与 GF 法 ≤1e-8；与实验谐振频率 ≤3 cm⁻¹ | <1 s | 4 个力常数 |
| B3 | `partition.py` | 标准熵（原子、双原子、多原子）与 H₂ + D₂ ⇌ 2HD 平衡常数 | R | NIST-JANAF (1998)；Urey (1947)；Bigeleisen & Mayer (1947) | 原子 S° ≤0.01、分子 S°(298 K) ≤0.2 J mol⁻¹ K⁻¹；K(T) 与原文表 ≤1% | <1 s | JANAF 少量数值 |
| B4 | `states.py` | Beyer–Swinehart 态计数：精确枚举、Laplace 一致性、Whitten–Rabinovitch | V + R | Beyer & Swinehart (1973)；Stein & Rabinovitch (1973)；Whitten & Rabinovitch (1963) | 枚举差为 0；Σρe^{−E/kT}/q_vib − 1 随粒度收敛 | 秒级 | 一组分子频率 |
| B5 | `rates.py`（CTST） | H + H₂ 及其同位素体的常规 TST 与 KIE | R + V（+H） | LSTH 势能面；Truhlar, Isaacson & Garrett (1985) | 与参考表 ≤5e-3；标准态不变性 ≤1e-12 | <1 s | 驻点性质 |
| B6 | `rates.py`（CVT）+ `refine_maximum` | 人造解析自由能剖面的收敛阶 + H + H₂ CVT 对照 | V（+H） | Truhlar & Garrett (1984) | s_CVT、ΔG_CVT 的收敛阶；k_CVT ≤ k_GT(s) | 秒级 | B6b 需 Polyrate 输出 |
| B7 | `tunneling.py` | Eckart 势垒：WKB 作用量、κ(T) 对精确量子解、Wigner 对 Bell | V + R | Eckart (1930)；Johnston & Heicklen (1962)；Bell (1959)；Garrett & Truhlar (1979) | θ(E) 误差 O(h²)；κ 对精确解的偏差曲线 | 秒级 | 无（解析）/ JH 表 |
| B8 | `calculation.py` / `input.py` / `cli.py` | 端到端：H + CH₄ → H₂ + CH₃ 的 TOML → 速率表 → Arrhenius 图 | R | 一篇列出驻点数据与 TST 速率的 H + CH₄ 文献（候选见 §2.8） | TST 列 ≤5e-3 | 秒级 | 驻点数据 |
| B9 | `legacy.py` / `reference.py` / `head_to_head.py` | Polyrate 2023 测试集的解析覆盖率与 fu15 头对头 | H | Polyrate 2023（DOI `10.17632/vpf3y9cbhf.1`） | 112 个输入无关键字丢失；可比列逐点 ≤5e-3 | 分钟级 | 用户本地 Polyrate 分发包 |
| B10 | `native.py` + `cpp/kernels.cpp` | C++/Python 一致性与性能标度 | V | — | 随机输入相对差 ≤1e-12；加速比与标度曲线 | 分钟级 | 无 |

`constants.py` / `elements.py` 不单独设算例。CODATA 2018 精确常数和导出量（如 hc/k = 1.438776877 cm K）的核对并入 B3 的一张小表。

## 2. 各算例设计

### B1 `geometry.py`：主惯性矩与转动常数

**目的**：验证质心、惯性张量、主惯性矩和转子分类，它们是 q_rot 唯一的几何输入。

**参考**：NIST CCCBDB (SRD 101) 实验几何与转动常数；G. Herzberg, *Molecular Spectra and Molecular Structure III* (1966)；W. Gordy, R. L. Cook, *Microwave Molecular Spectra* (1984)。

**体系**：H₂O（不对称陀螺，σ=2）、NH₃（扁对称陀螺，σ=3）、CH₄（球陀螺，σ=12）、CO₂（线性，σ=2）、C₂H₄（不对称陀螺，σ=4）。

**计算**：
1. 输入 r_e 几何，用 `mass_amu` 显式给出同位素质量。计算 I_A ≤ I_B ≤ I_C 和 A, B, C = h/(8π²cI)（cm⁻¹）。
2. 与平衡转动常数 A_e/B_e/C_e 比较。如果文献只给出 r₀/B₀，偏差主要来自振动平均，不是代码误差，要在表注中说明。
3. 性质测试：
   - 施加随机正交变换和平移，主惯性矩相对变化 ≤1e-12；
   - 交换同核原子标签，结果不变；
   - 近线性构型下，线性判别阈值（`classify_rotor` 的 1e-7）的敏感性。
4. 由主惯性矩和对称数计算 q_rot，与手写公式比较。

**验收**：转动常数与文献一致到几何参数的有效位数（典型 ≤1e-4 相对）；不变性 ≤1e-12；转子类型 5/5 正确。

**论文呈现**：表 “Rotational constants from equilibrium geometries”。

### B2 `normal_modes.py`：水同位素体谐振频率与 Teller–Redlich 乘积规则

**目的**：验证笛卡尔 Hessian 的质量加权、平动/转动投影，以及本征值到波数的换算。在 Born–Oppenheimer 近似下，Teller–Redlich 乘积规则是精确恒等式，对质量加权和投影错误非常敏感，适合作为论文中“数值正确性”的硬证据。

**参考**：
- A. R. Hoy, I. M. Mills, G. Strey, *Mol. Phys.* **24**, 1265 (1972)：H₂O 谐力场；
- W. S. Benedict, N. Gailar, E. K. Plyler, *J. Chem. Phys.* **24**, 1139 (1956)：H₂O/D₂O 谐振频率；
- O. Redlich, *Z. Phys. Chem. B* **28**, 371 (1935)；
- E. B. Wilson, J. C. Decius, P. C. Cross, *Molecular Vibrations* (1955)。

**模型**：内坐标二次力场

```text
V = ½ f_rr (Δr1² + Δr2²) + f_rr' Δr1 Δr2 + f_rθ (Δr1 + Δr2) Δθ + ½ f_θθ Δθ²
r_e = 0.9578 Å, θ_e = 104.48°
```

预研使用 f_rr = 8.454、f_rr' = −0.101 mdyn Å⁻¹，f_rθ = 0.228 mdyn rad⁻¹，f_θθ = 0.697 mdyn Å rad⁻²（需核对原文）。正式算例用 Wilson B 矩阵得到解析笛卡尔 Hessian（平衡构型无梯度项，H = BᵀFB）；预研中的有限差分只用于可行性验证。

**计算**：
1. 求 H₂O、HDO、D₂O 各自的 3N−6 个频率；
2. 用独立实现的 GF 矩阵法（det|GF − λE| = 0）对照；
3. 不分对称类的乘积规则：

   ```text
   Π_i ω'_i/ω_i = Π_a (m_a/m'_a)^{3/2} · (M'/M)^{3/2} · (I'_A I'_B I'_C / I_A I_B I_C)^{1/2}
   ```

4. 检查投影：`rigid_mode_count == 6`；坐标任意旋转后频率不变；
5. 扩展：用线性 CO₂ 的价力场检验线性分子 5 个刚体模的秩判定（`_rigid_basis` 的 SVD 分支）。

**验收**：与 GF 法 ≤1e-8 相对；乘积规则 ≤1e-9（解析 Hessian）；与实验谐振频率差 ≤3 cm⁻¹（受力场有效位数限制）。

**限制说明**：当前 `normal_modes` 只投影平动和转动，不做 Miller–Handy–Adams 梯度投影。因此只能用于驻点。论文中应写明非驻点的广义频率需由外部提供。

### B3 `partition.py`：标准熵与同位素交换平衡

#### B3a 标准熵

**参考**：M. W. Chase Jr., *NIST-JANAF Thermochemical Tables*, 4th ed., *J. Phys. Chem. Ref. Data* Monograph 9 (1998)；CCCBDB 的 RRHO 熵。

**体系**：
- 原子（只有平动和电子项，用于单独检验 `log_translational_partition` 和 `log_electronic_partition`）：O(³P₂,₁,₀；0、158.265、226.977 cm⁻¹；g = 5、3、1)，Cl(²P₃/₂,₁/₂；0、882.35 cm⁻¹；g = 4、2)；
- 分子：N₂、O₂（³Σg⁻，g=3）、H₂O、CO₂、CH₄、NH₃。

**计算**：S°(T) = −∂G°/∂T，用 h = 10⁻³ K 的中心差分（截断误差约 1e-9 J mol⁻¹ K⁻¹），温度范围 298.15–3000 K。同时给出 −[G° − H°(0)]/T。

**验收**：
- 原子：S° 与 JANAF 偏差 ≤0.01 J mol⁻¹ K⁻¹；
- 分子：298.15 K 时 ≤0.2 J mol⁻¹ K⁻¹（RRHO 极限）；
- 高温偏差曲线画成“RRHO 适用范围”图，不设通过阈值。JANAF 含非谐和离心畸变修正。

**注意**：JANAF 使用天然丰度原子量。比较时质量约定必须与之一致（见 §6.2）。

#### B3b H₂ + D₂ ⇌ 2HD

**参考**：H. C. Urey, *J. Chem. Soc.* 1947, 562；J. Bigeleisen, M. G. Mayer, *J. Chem. Phys.* **15**, 261 (1947)；Huber & Herzberg (1979) 的 ω_e 和 r_e。

**意义**：
- 同时检验平动、转动对称数、零点能和 Gibbs 能组合；
- 高温极限 K → 4 正是对称数之比，可直接检验对称数处理；
- PyRate 的经典转子 + 量子谐振子模型与 Bigeleisen–Mayer 近似一致，可以严格逐项比较；
- Δn = 0，K 与标准压力无关，可作附带的不变性检查。

**验收**：K(T) 与 Urey 表值偏差 ≤1%（表值需从原文摘录），并画出 K(T) → 4 的曲线。

**扩展（展示已知限制）**：低温下比较经典转子与量子转动能级求和（含正/仲氢核自旋权重），H₂ 的 Θ_rot ≈ 87.6 K。这张图说明为什么当前版本只适合 T ≳ 200 K 的含 H₂ 体系。

### B4 `states.py`：Beyer–Swinehart 振动态计数

**参考**：T. Beyer, D. F. Swinehart, *Commun. ACM* **16**, 379 (1973)；S. E. Stein, B. S. Rabinovitch, *J. Chem. Phys.* **58**, 2438 (1973)；G. Z. Whitten, B. S. Rabinovitch, *J. Chem. Phys.* **38**, 2466 (1963)。

**计算**：
1. **精确枚举（V）**：取 3–4 个频率为粒度整数倍的振子，与暴力枚举的整数计数逐格比较，要求完全相等。
2. **Laplace 一致性（V，跨模块）**：Σ_i N_i e^{−E_i/kT} 应等于 `partition.log_vibrational_partition` 给出的 q_vib(T)（不含 ZPE）。频率取互不可约的实际分子频率（如 CH₃NC 的 12 个模），粒度取 50、20、10、5、1 cm⁻¹，画出相对误差随粒度的收敛曲线。误差来源是频率取整到粒度。
3. **Whitten–Rabinovitch 对照（R）**：用同一分子计算态和 N(E)，与 WR 近似比较。复现 WR 原文的结论：中高能区误差为百分之几，低能区偏差较大。
4. 可选：从 Holbrook, Pilling & Robertson, *Unimolecular Reactions* (1996) 或 Baer & Hase (1996) 的表中取若干 ρ(E) 值逐点复现（需核对原文）。

**API 说明**：`vibrational_density_of_states` 返回的是每个能量格内的状态数，ρ(E) = N_i/ΔE。论文和 docstring 中应明确这一点。

### B5 `rates.py`（CTST）：H + H₂ 及同位素体

**参考**：
- D. G. Truhlar, C. J. Horowitz, *J. Chem. Phys.* **68**, 2466 (1978)；P. Siegbahn, B. Liu, *J. Chem. Phys.* **68**, 2457 (1978)（LSTH 势能面）；
- D. G. Truhlar, A. D. Isaacson, B. C. Garrett, “Generalized transition state theory”, in *Theory of Chemical Reaction Dynamics*, Vol. IV, ed. M. Baer (CRC, 1985)；
- A. Fernández-Ramos et al., *Theor. Chem. Acc.* **118**, 813 (2007)（对称数约定）；
- 可选 H 级对照：Polyrate 2023 中 H + H₂ 系列测试的 fu15 TST 列（以 manifest 中实际存在的测试为准）。

**输入**：LSTH 鞍点 R₁ = R₂ ≈ 1.757 a₀，V‡ ≈ 9.80 kcal mol⁻¹，频率约为 2058、909（二重简并）和 1510i cm⁻¹；H₂ r_e ≈ 1.401 a₀。这些都需核对原文或 Polyrate 输出。电子简并度：H 为 2，TS 为 2。

**体系**：H + H₂、D + H₂（D–H–H，C∞v，σ=1）、H + D₂、D + D₂。由此给出 KIE。

**计算与验收**：
1. 用独立的手写配分函数比值公式重算 k_TST：≤1e-12；
2. **标准态不变性**：双分子 k 与所选 p° 无关（1 bar 与 1 atm 只改变 ΔG‡）：≤1e-12；
3. 单位换算：cm³ molecule⁻¹ s⁻¹、L mol⁻¹ s⁻¹ 与 m³ mol⁻¹ s⁻¹ 互相一致；
4. 与参考表 k_TST(T)、KIE(T) 比较，三位有效数字表的阈值为 ≤5e-3；
5. 在论文中明确对称数约定（σ_H₂ = 2，σ_{H₃‡} = 2 等），并引用 Fernández-Ramos et al. (2007)。

**论文呈现**：k_TST(T) 和 KIE 表。与实验（如 NIST Kinetics Database 中的评估值）的比较只作为背景曲线，不作为验收依据。

### B6 `rates.py`（CVT）与 `refine_maximum`：变分瓶颈

#### B6a 人造解析剖面（V，核心）

**构造**：

```text
V_MEP(s) = V0 · sech²(s/L)
ω_⊥(s)   = ω_R − Δω · sech²((s − s0)/L')    # s0 ≠ 0，使零点能谷偏离势垒顶
I(s)     = I_0 + I_1 · s                    # 让转动熵也随 s 变化
```

零点能和振动熵使 V_a^G 的最大值偏离 s = 0，且随 T 移动。

**参考解**：在基准脚本中用独立的 RRHO 公式写出 ΔG(s,T) 的解析表达式，用 Brent 方法求 s*(T) 和 ΔG*(T)。不调用 PyRate。

**PyRate 输入**：在 [−1, 1] 上等距取 N = 9、17、33、65、129 个路径点，T = 200–2000 K。

**验收**：
1. |s_CVT − s*| 和 |ΔG_CVT − ΔG*| 随 Δs 的 log-log 收敛图，拟合出收敛阶（五点多项式预期为高阶）；
2. 变分不等式：k_CVT ≤ min_i k_GT(s_i)；s = 0 为路径点时 k_CVT ≤ k_TST；
3. 把最大值移到边界时触发安全回退（`fit_points == 1`）；
4. 原生与回退后端的 s_CVT 一致（≤1e-12）。

#### B6b H + H₂ CVT 对照（H，可选）

从 Polyrate H + H₂ 测试的输出中导出沿路径的 s、V_MEP、广义频率和转动惯量，转成 PyRate 的 `[[path.points]]`（需新增 `tools/export_polyrate_path.py`）。比较 s_CVT(T) 与 k_CVT/k_TST。

注意：
- Polyrate 的 s 是质量标度坐标（默认标度质量 1 amu，单位 bohr）；
- Polyrate 的插值与 `FITMAX` 细节和 PyRate 不同，需逐项记录差异；
- 不依赖 Polyrate 的替代方案：用 BEBO 解析路径复现 B. C. Garrett, D. G. Truhlar, *J. Am. Chem. Soc.* **101**, 4534 (1979) 的 k_CVT/k_TST 比值。

### B7 `tunneling.py`：Eckart 势垒

**参考**：C. Eckart, *Phys. Rev.* **35**, 1303 (1930)；H. S. Johnston, J. Heicklen, *J. Phys. Chem.* **66**, 532 (1962)；R. P. Bell, *Trans. Faraday Soc.* **55**, 1 (1959)；E. Wigner, *Z. Phys. Chem. B* **19**, 203 (1932)；B. C. Garrett, D. G. Truhlar, *J. Phys. Chem.* **83**, 2921 (1979)；R. T. Skodje, D. G. Truhlar, *J. Phys. Chem.* **85**, 624 (1981)。

**解析事实**：对对称 Eckart 势 V = V₀ sech²(x/a)：
- WKB 积分有闭式 θ(E) = π a √(2μ) (√V₀ − √E)/ħ；
- 精确量子透射概率有闭式（Landau–Lifshitz）：

  ```text
  P(E) = sinh²(πka) / [sinh²(πka) + cosh²(½π √(8μV₀a²/ħ² − 1))]
  ```

- 无限抛物势垒的精确 κ 为 Bell 公式 (u/2)/sin(u/2)；Wigner 是它的一阶展开。

**计算**：
1. **作用量（V）**：`wkb_exponent` 与 2θ(E) 比较，要求误差随网格间距呈 O(h²) 收敛；
2. **κ(T)（R）**：PyRate 的 κ_WKB 与精确 Eckart κ 比较，覆盖 u* = ħω‡/kT = 1–12；不对称 Eckart 情形与 Johnston & Heicklen 表值逐点比较；
3. **Wigner 与 Bell**：高温极限 κ_W − κ_Bell = O(u⁴)；
4. **不对称势垒**：检验“第一个能量点即正向反应物参考”的约定。

**预研发现（重要）**：见 §7.4。当前一维 WKB 使用 P(E) = e^{−2θ}，势垒顶以上取 P = 1。由此 κ − 1 ≈ u/2π，属于 O(u)，而精确结果是 O(u²)。高温下 κ 被高估：V₀ = 40 kJ mol⁻¹、ν‡ = 1500i cm⁻¹ 时，1000 K 下 1.48 对精确值 1.26，2000 K 下 1.20 对精确值 1.07。建议投稿前：
- 改用 Kemble 均匀半经典形式 P = 1/(1 + e^{2θ})，并对势垒顶以上加入非经典反射 P(E) = 1 − P(2V₀ − E)，这与 Polyrate ZCT/SCT 的处理一致；
- 原始形式可以保留为选项；
- 本算例同时展示修正前后的曲线。

### B8 端到端：H + CH₄ → H₂ + CH₃

**目的**：用一个真实多原子反应演示完整工作流（TOML → `pyrate validate` → `pyrate run --format json` → 绘图），同时检验对称数（σ_CH₄ = 12，σ_TS(C₃v) = 3，反应路径简并度 4）、非线性转子和双分子单位。

**参考候选**（选一篇明确列出驻点几何、频率和 TST 速率的文献；需核对原文）：
- J. Pu, D. G. Truhlar, *J. Chem. Phys.* **117**, 1479 (2002)；
- J. Espinosa-García, *J. Chem. Phys.* **116**, 10664 (2002)；
- 实验背景曲线：D. L. Baulch et al., *J. Phys. Chem. Ref. Data* **34**, 757 (2005) 评估值。

**计算**：250–2000 K 下的 TST、TST/Wigner 和 TST/WKB（用按 V‡、ΔE、ω‡ 拟合的 Eckart 剖面，在 B7 修正之后）。输出 JSON 和 CSV，画 Arrhenius 图。

**验收**：TST 列与原文 ≤5e-3。同时记录 CLI 总耗时，用于展示软件开销。

### B9 Polyrate 2023 兼容性：解析覆盖率与头对头

**计算**：
1. 用 `tools/import_polyrate_reference.py` 生成 manifest。统计 112 个输入中被解析的区段和关键字、未知关键字和 `compatibility_warnings`，以及请求的方法分布。
2. 把测试分成三类：可执行、部分可执行、仅可检查。
3. 对可执行子集（TST、CVT（需 B6b 导出器）、Wigner）运行 `tools/compare_polyrate_rates.py --strict`，逐点报告。

**验收**：解析不丢失任何关键字；论文只计入数值比较已通过的测试数（与 `docs/head-to-head.md` 的规则一致）。

**论文呈现**：兼容性表（按方法分组列出测试数、可执行数和通过数）。这比单纯声称“兼容 Polyrate”有说服力得多。

### B10 `native.py` / `cpp/kernels.cpp`：后端一致性与性能

**一致性**：对每个内核（`harmonic_log_partition`、`wigner_kappa`、`refine_maximum`、`wkb_exponent`、`thermal_wkb_kappa`、`beyer_swinehart`）各生成 10⁴ 组随机输入，比较原生与回退结果，报告最大相对差。阈值 ≤1e-12；BS 在频率可公约时要求整数完全相等。

**性能**：扩展 `tools/benchmark_kernels.py`：
- BS 的格数取 10³–10⁶、模数取 10–100，验证 O(s·N) 标度；
- thermal WKB 的剖面点数取 50–2000、能量网格取 100–4000；
- 报告加速比，并记录 CPU、编译器、编译选项（默认不开 `PYRATE_NATIVE_ARCH`）和 NumPy 版本。

## 3. 工程化建议

```text
benchmarks/
  README.md                 # 运行方式、预计耗时、依赖
  common/provenance.py      # git commit、后端、平台、编译器、时间戳
  common/plotting.py
  b01_rotational_constants/
    run.py
    data/reference.csv      # 每行注明文献、表号/页码、单位
  b02_water_normal_modes/
  ...
  b10_native_parity/
  run_all.py                # 产出 results/<case>.json 和 figures/<case>.pdf
tests/test_paper_cases.py   # @pytest.mark.paper，降维版本，CI 中总时长 <30 s
```

- 正式结果用 `PYRATE_DISABLE_NATIVE=0/1` 各跑一次，结果 JSON 中记录两个后端的差异。
- 参考数据只摘录论文中需要比较的少量数值，并在 CSV 头部写明来源。
- 绘图脚本与计算分离，方便审稿人只复算数值。
- `pyproject.toml` 中注册 `paper` marker。CI 跑降维版本；完整版本作为发布前检查。

## 4. 优先级

| 优先级 | 算例 | 理由 |
|---|---|---|
| P0 | B2、B4、B6a、B7（含 WKB 修正）、B10 | 无需外部数据，可立即实施；B7 会触发代码修正 |
| P1 | B1、B3、B5 | 只需摘录少量文献数值 |
| P2 | B6b、B8、B9 | 需要 Polyrate 分发包、新导出工具或较多文献摘录 |

## 5. 论文中的组织建议

- **表 1**：算例矩阵（即 §1 的精简版），列出模块、参考和达到的精度。
- **图 1**：Eckart κ(T)，曲线包括精确量子解、Wigner、Bell、PyRate WKB（修正前/后）。
- **图 2**：CVT 收敛阶（B6a）。
- **图 3**：相对 JANAF 的 S°(T) 偏差（B3a），附 K(H₂ + D₂ ⇌ 2HD) 小图（B3b）。
- **图 4**：H + CH₄ 的 Arrhenius 图（B8）。
- **表 2**：Teller–Redlich 乘积规则与 GF 对照（B2）。
- **表 3**：Polyrate 2023 头对头统计（B9）。
- **表 4**：后端一致性与性能（B10）。

## 6. 规划中发现的、建议投稿前处理的问题

1. **一维 WKB 的透射概率形式**（B7，影响最大）：见 §2 B7 和 §7.4。
2. **默认原子质量约定**：`elements.py` 中 H 为标准原子量 1.00784，D 却是同位素质量 2.014102。混用会让 H/D 同位素效应带入约 1.5×10⁻⁵ 的质量不一致。建议增加“最丰同位素”选项，或在文档中明确约定；所有基准算例显式写 `mass_amu`。
3. **热力学函数 API**：目前只能拿到 G°；S°、H°、C_p° 需外部差分。建议增加解析的 `standard_entropy` / `enthalpy` / `heat_capacity`，B3 可直接调用，并用差分结果互验。
4. **态计数返回值语义**：明确 `vibrational_density_of_states` 返回的是每格计数，不是 ρ(E)。
5. **非驻点频率**：`normal_modes` 不含梯度投影，论文需说明 CVT 的广义频率由外部提供。

## 7. 附录：预研数值

运行环境：commit `02b7d09`，Python 3.13，`PYTHONPATH=src`（Python 回退后端）。探针脚本没有提交，正式脚本按 §3 实现。

### 7.1 B2：水的谐振频率（有限差分 Hessian，¹H/²H/¹⁶O 同位素质量）

| 体系 | ω₁ / ω₂ / ω₃（cm⁻¹） | 去除刚体模数 |
|---|---|---|
| H₂O | 1646.6 / 3831.9 / 3942.4 | 6 |
| HDO | 1443.1 / 2823.7 / 3889.6 | 6 |
| D₂O | 1204.6 / 2763.5 / 2888.6 | 6 |

Teller–Redlich（H₂O/D₂O，全部 3N−6 个模）：左边 0.386578425358，右边 0.386578425290，相对差 1.8×10⁻¹⁰，受有限差分步长限制。

### 7.2 B3：标准熵（1 bar，298.15 K，J mol⁻¹ K⁻¹）

| 物种 | PyRate RRHO | JANAF | 说明 |
|---|---|---|---|
| O(³P) | 161.056 | 161.059 | 用 ¹⁶O 同位素质量；换成 JANAF 原子量 15.9994 后约 +0.0035 |
| Cl(²P) | 165.190 | 165.190 | 原子量 35.45 |
| N₂ | 191.561 | 191.609 | r_e、ω_e |
| O₂ | 205.079 | 205.147 | g_e = 3 |
| H₂O | 188.666 | 188.834 | 用基频 3657/1595/3756 cm⁻¹ |

H₂ + D₂ ⇌ 2HD（ω_e = 4401.21/3813.15/3115.50 cm⁻¹，r_e = 0.74144 Å）：

| T (K) | 200 | 298.15 | 500 | 1000 | 3000 |
|---|---|---|---|---|---|
| K | 2.860 | 3.256 | 3.623 | 3.902 | 3.997 |

### 7.3 B4 与 B5

- Beyer–Swinehart（500/800/1200 cm⁻¹，粒度 1 cm⁻¹，E ≤ 6000 cm⁻¹）与暴力枚举的最大差为 0。
- 9 模体系（频率为 10 cm⁻¹ 的整数倍）的 Laplace 一致性 ≤7×10⁻¹⁶。正式算例需用不可公约频率才能体现粒度收敛。
- `examples/wkb_tst.toml` 在 1 bar 与 1 atm 下 ΔG‡ 不同，但 k_TST 相对差 ≤5×10⁻¹⁵，标准态不变性成立。

### 7.4 B7：对称 Eckart 势垒（V₀ = 40 kJ mol⁻¹，μ = 1 amu，ν‡ = 1500i cm⁻¹，4001 点剖面）

WKB 指数与闭式 2θ(E) 的相对差：E/V₀ = 0.1、0.5、0.9 时分别为 1.6×10⁻⁶、1.2×10⁻⁴、1.5×10⁻⁴。

| T (K) | u* | Wigner | Bell（抛物） | 精确 Eckart | PyRate WKB（当前） | 均匀半经典（探针实现） |
|---|---|---|---|---|---|---|
| 200 | 10.79 | 5.85 | — | 797 | 672 | 669 |
| 300 | 7.19 | 3.16 | — | 13.0 | 12.2 | 11.1 |
| 400 | 5.40 | 2.21 | 6.28 | 3.87 | 4.16 | 3.40 |
| 600 | 3.60 | 1.54 | 1.85 | 1.81 | 2.14 | 1.66 |
| 1000 | 2.16 | 1.19 | 1.22 | 1.26 | 1.48 | 1.19 |
| 2000 | 1.08 | 1.05 | 1.05 | 1.07 | 1.20 | 1.04 |

结论：低温深隧穿区，当前实现与精确值同量级，这是半经典的典型精度。u* ≲ 4 的高温区，当前实现有系统性高估，这与 κ − 1 ≈ u/2π 的分析一致。均匀半经典形式消除了这一 O(u) 偏差。

## 8. 已在 GitHub Actions 中自动运行的算例

`.github/workflows/paper-benchmarks.yml` 会在 `src/`、`cpp/`、`benchmarks/`、`examples/` 变更或手动触发时运行 `python -m benchmarks.run_all`。每个算例分别用 C++ 后端和 Python 回退后端各跑一次，然后把 `benchmarks/results/*.json` 和 `SUMMARY.md` 提交回触发分支。PR 上的运行只出报告，不提交。普通 CI 中的 `tests/test_paper_benchmarks.py` 也会检查这些算例没有失败项。

| 算例 | 文件 | CI | 说明 |
|---|---|---|---|
| B2 | `benchmarks/b02_water_normal_modes.py` | ✅ | 解析 B 矩阵 Hessian，GF 对照，Teller–Redlich |
| B3 | `benchmarks/b03_thermochemistry.py` | ✅ | JANAF S°(298.15)；H₂ + D₂ ⇌ 2HD；Urey 表值对比待摘录 |
| B4 | `benchmarks/b04_state_counting.py` | ✅ | 枚举、Laplace 恒等式、粒度收敛、WR |
| B5 | `benchmarks/b05_tst_consistency.py` | ✅（部分） | 独立 Eyring 公式、标准态不变性、单位；H + H₂ 文献复现待驻点数据核对 |
| B6a | `benchmarks/b06_cvt_convergence.py` | ✅ | 人造剖面收敛阶、变分不等式 |
| B7 | `benchmarks/b07_eckart_tunneling.py` | ✅ | 作用量误差、κ 与精确 Eckart（作为信息报告当前 WKB 的高温偏差） |
| B10 | `benchmarks/b10_native_parity.py` | ✅ | 随机输入一致性与加速比 |
| B1、B8 | — | 可以但未做 | 需要先摘录并核对文献数据，之后即可加入 CI |
| B6b、B9 | — | ❌ | 依赖 Polyrate 2023 分发包（不随仓库分发），只能本地运行 |

## 9. 更新：WKB 高温偏差已修正

`thermal_wkb_kappa` 默认改为 Kemble 均匀半经典透射概率加非经典反射（C++ 与 Python 回退同步），旧形式保留为 `transmission = "primitive"`。B7 现在要求：
- 与独立实现的均匀半经典参考 ≤1e-4；
- 抛物势垒与 Bell 公式 ≤1e-4（实测约 1e-5，来自分段线性网格）。

对 Eckart 势垒（V₀ = 40 kJ/mol，ν‡ = 1500i cm⁻¹），κ/κ_exact 在 1000 K 从 1.18 变为 0.95，在 2000 K 从 1.12 变为 0.98。低温深隧穿区（300 K 约 0.85）的低估是一维半经典近似本身的误差，不是实现问题。
