# 架构与数值约定

## 1. 分层原则

PyRate 不机械翻译 Fortran 全局数组，而是把 Polyrate 的职责拆成四层：

1. **模型层**：`Atom`、`HinderedRotor`、`Species`、`PathPoint`、`Reaction`、`ReactionPath`、`TunnelingProfile`、`LctOptions` 和 `CalculationOptions`；
2. **物理层**：谐振/Morse 非谐/Pitzer–Gwinn 受阻转子分区函数、标准 Gibbs 能、CTST/CVT/CUS、微正则 μVT/US、以及 CAG/Wigner/ZCT/SCT/LCT(LCG3 & LCG4)/COMT/μOMT/Eckart 隧穿修正；
3. **数值层**：C++17 热点与 1:1 等价的 Python 回退；
4. **适配层**：TOML、Polyrate `.dat + .fu40` 转换与执行器、`fu15` reference parser 和 CLI。

这种分层使每个公式可以独立测试，也避免将 Polyrate 的固定大小 COMMON block 和文件单元号传播到现代 API。

## 2. Python / C++ 边界

`cpp/kernels.cpp` 当前提供：

- 谐振子 `log(q_vib)` 批处理；
- Wigner 系数；
- 与 Polyrate `FITMAX` 对齐的 3 点抛物线 / 5 点四次多项式局部极值精修（`refine_maximum` 与 `refine_minimum`）；
- 常数约化质量与位置相关有效质量 $\mu_{\text{eff}}(s)$ 的分段线性势垒 WKB 作用量指数 $\theta(E)$；
- 包含 Kemble 均匀半经典透射与 `BOLTZ` 非经典反射的 canonical thermal WKB / SCT 积分；
- Gauss–Legendre 积分节点与权重（`gauss_legendre_quadrature`）、Neville 有理多项式插值（`neville_interpolate` / `POLINT`）、大曲率路径端点振幅积分（`lct_integrate_amplitudes` / `LCSINT`）、多振动态有理均匀化与单调包络排序（`lct_uniformize_probabilities` / `LCPROB` + `LCGSRT`）、以及对数空间防溢出的任意能量网格透射概率热平均（`thermal_boltz_probabilities_kappa` / `BOLTZ`）；
- 对称与非对称 Eckart 势垒解析透射系数与热积分；
- Beyer–Swinehart 振动态密度与状态和计数。

C++ 函数在读取 Python 序列后释放 GIL 执行长循环。`src/pyrate/native.py` 是唯一公共桥接层；导入扩展失败或设置 `PYRATE_DISABLE_NATIVE=1` 时调用等价 Python 算法。

正规模的对称本征问题交给 NumPy/LAPACK；质量加权、静点直接对角化（`FDIAG`）、广义过渡态上的梯度与刚体空间投影（`PROJCT`）、以及反应路径曲率 $B_{mF}(s)$ 计算（`RPHB`）保留在 `normal_modes.py`、`input.py` 与 `lct.py` 中，便于审计。

## 3. 标准态、CTST、CVT 与 CUS

每个分子的平动分区函数在标准压力对应的每分子体积上求值：

```text
V°_molecule = k_B T / p°
q_trans = (2π m k_B T / h²)^(3/2) V°_molecule
```

分子的标准 Gibbs 能为（含可选的 Morse 非谐与 Pitzer–Gwinn / `BES0` 受阻内旋转修正）：

```text
G° = E_electronic + ZPE - R T ln(q_trans q_rot q_vib q_elec q_hin)
```

m 分子反应的 Eyring 速率：

```text
k = (k_B T / h) exp(-ΔG°‡ / RT) (c°)^(1-m),   c° = p° / RT
```

对每个温度，PyRate 在所有反应路径分割面上计算 $\Delta G^{\circ\ddagger}(s,T)$，使用 `FITMAX`（次端点 `index == 1` 或 `N-2` 用 3 点抛物线，内部点用 5 点四次多项式）寻找最大值 $s_{\text{CVT}}(T)$。当自由能曲线存在双瓶颈和中间阱时，CUS 速率定义为：

```text
R(T) = [ 1 - k_CVT(T)/k_max2(T) + k_CVT(T)/k_min(T) ]^(-1)
k_CUS(T) = k_CVT(T) * R(T)
```

## 4. 微正则变分理论（μVT 与 US）

使用 Beyer–Swinehart 算法在 $1\text{ cm}^{-1}$ 网格上计算每个分割面的振动态和 $N_{\text{vib}}(E - V_a^G(s))$，并与经典刚性转子态密度卷积得到总振转态和 $N_{\text{rv}}(E, s)$：

```text
N_μVT(E) = min_s N_rv(E, s)
```

积分 Boltzmann 权重后得到微正则变分速率 $k_{\mu\text{VT}}(T)$ 与微正则统一统计速率 $k_{\text{US}}(T)$。

## 5. 振动绝热势、Kemble WKB、ZCT、SCT 与大曲率隧穿（LCG3 / LCG4 / COMT / μOMT）

沿 MEP 各分割面的振动绝热基态势定义为：

```text
V_a^G(s) = V_MEP(s) + ZPE(s) - ZPE_reactants
```

相位积分采用位置相关有效质量 $\mu_{\text{eff}}(s)$（ZCT 下为常数 $\mu$）：

```text
θ(E) = (1/ħ) ∫ sqrt(2 μ_eff(s) [V_a^G(s) - E]) ds
P(E) = 1 / [1 + exp(2 θ(E))]
```

与 Polyrate 的 `KAPVA` / `BOLTZ` 一致，热透射系数同时包含低于势垒顶 $V_{\max}$ 的量子隧穿与高于 $V_{\max}$ 的非经典反射对称双曲正弦积分：

```text
κ(T) = exp((V_max - E_0)/k_B T) + 2 β ∫_{E_0}^{V_max} P(E) sinh(β (V_max - E)) dE
```

在离心主导小曲率隧穿（SCT / CD-SCSAG，对应 Polyrate `polysz.F` 中的 `CURV2` 与 `TPCDSC`）中，模式曲率 $B_{mF}(s)$ 与模式拐点合成有效离心拐点：

```text
t(s) = sqrt(Σ_m B_{mF}(s)^2) / [ Σ_m B_{mF}(s)^2 (μ ω_m(s) / ħ)^2 ]^(1/4)
a(s) = |κ(s)| t(s) - (1/2) t(s)^2 (d²t / ds²)
μ_eff(s) = μ * min(1, exp(-2 a(s) - a(s)^2 + (dt/ds)^2))
```

在大曲率隧穿（LCT：LCG3 与默认 LCG4，对应 Polyrate `polyhl.F` / `polymq.F`）中，PyRate 在质量标度笛卡尔空间 $\tilde{\mathbf{x}}_a = \sqrt{m_a / \mu}\,(\mathbf{r}_a - \mathbf{r}_{\text{COM}})$ 构造连接转折点内侧起点 $s_0$ 与终点 $s_1$ 的直线 corner-cutting 弦 $\tilde{\mathbf{x}}(\xi) = (1 - \xi)\tilde{\mathbf{x}}(s_0) + \xi\tilde{\mathbf{x}}(s_1)$：

- 通过牛顿法（`LCNWTN`）投影局部正交横坐标 $\mathbf{q}(s)$ 并由多值雅可比判据 $1 - \sum_m B_m(s) q_m(s) > 0$ 与拐点振幅条件 $|q_m(s)| \le q_{m,\text{turn}}(s)$ 识别绝热区与非绝热区（`LCZCRT`）；
- 在绝热区使用振动绝热有效势 $V_{\text{eff}}$（含激发终态量子的谐振修正 `LCVCOR`），在非绝热区使用离路径经典势加残余模式修正，LCG4 通过 `IZMX` / `LCTV4` 保证绝热-非绝热交界处有效势连续并防止非物理负值；
- 对所有可及振动终态 $n = 0, \dots, N_{\text{prod}}$ 积分起点振幅（`LCSINT`），经有理 Kemble 均匀化与单调排序（`LCPROB` + `LCGSRT`）得到 $P_{\text{LCT}}(E)$；当未提供显式离路径势函数 `LctOptions.potential_fn` 时，自动由路径几何、能量、切向与法向频率/本征向量构造光滑欧氏 Shepard 二阶反应路径哈密顿面；
- 由此在同一 Gauss–Legendre 能量网格上给出正则最优模式隧穿 $\kappa_{\text{COMT}}(T) = \max(\kappa_{\text{ZCT}}, \kappa_{\text{SCT}}, \kappa_{\text{LCT}})$ 与真正微正则最优模式隧穿：

```text
P_μOMT(E) = max(P_ZCT(E), P_SCT(E), P_LCT(E))
```

## 6. 线程与可复现性

- 所有输入在进入内核前复制成连续的 `double` 向量；
- 默认不启用 `-march=native`，从而保证可移植 wheel；
- 设置 `PYRATE_NATIVE_ARCH=1` 才启用本机指令优化；
- 原生和回退后端共享测试容差；
- 参考输出比较以速率相对误差和关键中间量误差分别验收。

