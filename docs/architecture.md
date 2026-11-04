# 架构与数值约定

## 1. 分层原则

PyRate 不机械翻译 Fortran 全局数组，而是把 Polyrate 的职责拆成四层：

1. **模型层**：`Atom`、`Species`、`Reaction`、`ReactionPath` 和 `CalculationOptions`；
2. **物理层**：分区函数、标准 Gibbs 能、CTST/CVT 和隧穿修正；
3. **数值层**：C++ 热点与对应 Python 回退；
4. **适配层**：TOML、Polyrate legacy parser、`fu15` reference parser 和 CLI。

这种分层使每个公式可以独立测试，也避免将 Polyrate 的固定大小 COMMON block 和文件单元号传播到现代 API。

## 2. Python / C++ 边界

原生扩展 `pyrate._kernels` 按数值领域拆成多个翻译单元，`src/pyrate/native/` 下有一一对应的 Python 模块：

| 领域 | C++ | Python | 内容 |
|---|---|---|---|
| thermo | `cpp/thermo.cpp` | `native/thermo.py` | 谐振子 `log(q_vib)`、Wigner 系数 |
| fit | `cpp/fit.cpp` | `native/fit.py` | CVT 局部极值精修 |
| wkb | `cpp/wkb.cpp` | `native/wkb.py` | 分段线性势垒 WKB 作用量、canonical 透射积分 |
| dos | `cpp/dos.cpp` | `native/dos.py` | Beyer–Swinehart 状态计数 |

`cpp/common.hpp` 存放常数和序列转换。每个翻译单元导出一张以空项结尾的 `PyMethodDef` 表，由 `cpp/module.cpp` 在初始化时逐一注册，所以新增内核不需要改动其他领域的文件。

每个 Python 模块提供两类函数：

- 公共函数 `<name>`：只做一次输入校验，然后分派到 C++ 或 Python；
- 纯 Python 参考实现 `py_<name>`：作为回退，也用于 `tests/test_parity_*.py` 的同进程跨后端比较。

C++ 函数在读取 Python 序列后释放 GIL 执行长循环。扩展无法导入，或设置了 `PYRATE_DISABLE_NATIVE=1` 时，使用 Python 实现。

正规模的对称本征问题交给 NumPy/LAPACK；质量加权、刚体空间构造和投影逻辑保留在 `normal_modes.py`，便于审计。

## 3. 标准态和 CTST

每个分子的平动分区函数在标准压力对应的每分子体积上求值：

```text
V°_molecule = k_B T / p°
q_trans = (2π m k_B T / h²)^(3/2) V°_molecule
```

分子的 RRHO 标准 Gibbs 能为：

```text
G° = E_electronic + ZPE - R T ln(q_trans q_rot q_vib q_elec)
```

活化自由能：

```text
ΔG°‡ = G°_dividing_surface - Σ G°_reactants
```

m 分子反应的 Eyring 速率：

```text
k = (k_B T / h) exp(-ΔG°‡ / RT) (c°)^(1-m)
c° = p° / RT
```

因此双分子内部结果是 `m³ mol⁻¹ s⁻¹`。这与 Polyrate 在自由能中加入标准气体浓度并在最终速率中除以 `C0` 的处理等价。

## 4. CVT

对每个温度，PyRate 在所有反应路径分割面上计算 `ΔG°‡(s,T)`，然后寻找最大值：

```text
s_CVT(T) = arg max_s ΔG°‡(s,T)
k_CVT(T) = min_s k_GT(s,T)
```

采样最大值由 `native.refine_maximum(x, y, *, max_points=3)` 精修，C++ 与 Python 两端逐运算一致。设 `i` 为**第一个**采样最大值的下标，相邻步长 `h_l = x[i] − x[i−1]`、`h_r = x[i+1] − x[i]`，相邻落差 `d_l = y[i] − y[i−1] > 0`、`d_r = y[i] − y[i+1] ≥ 0`。

**默认：三点抛物线。** 过 `(i−1, i, i+1)` 三点的抛物线写成 `p(x[i]+t) = y[i] + b t + a t²`：

```text
H  = h_l h_r (h_l + h_r)
a  = −(d_l h_r + d_r h_l) / H                 (< 0，必然下凹)
b  = (d_l h_r² − d_r h_l²) / H
t* = −b / (2a) = (d_l h_r² − d_r h_l²) / [2 (d_l h_r + d_r h_l)]
s_CVT = x[i] + t*,   ΔG_CVT = y[i] + b t* / 2
```

可以证明 `t* ∈ [−h_l/2, h_r/2]`，且 `b t* ≥ 0`。

**保证。** 记 `B = y[i] − min(y[i−1], y[i+1]) = max(d_l, d_r)`，即到较低邻点的落差。`B` 与 `y` 的单位和零点无关。所有精修结果满足：

1. 最大值在端点或少于 3 个点时，返回采样点本身，`fit_points = 1`；
2. `x[i−1] ≤ s_CVT ≤ x[i+1]`（抛物线顶点在解析上已落在半步长内，实现中的截断只吸收舍入）；
3. `y[i] ≤ ΔG_CVT ≤ y[i] + B`；
4. 数据本身是二次函数时结果精确（误差 ≤1e-12），网格可以不均匀，只要 B 判据接受；
5. C++ 与 Python 的坐标误差 ≤1e-9，函数值误差 ≤1e-12。

**违反保证时回退，而不是截断。** 拟合越界，或结果非有限值时，返回采样点（`fit_points = 1`），不把数值截到 `y[i] + B`。采样点给出的是一个真实分割面的 `k_GT`，因此仍是变分上界。

均匀网格上抛物线的超调至多为 `B/8`。只有局部步长比 `max(h_l,h_r)/min(h_l,h_r)` 超过 `2 + 2√2 ≈ 4.83` 时，`B` 才可能起作用。此时抛物线是在用一侧的陡坡外推另一侧的宽区间，数据同样可以由平台解释。因此二次数据在这种网格上也可能被保守地拒绝，例如顶点远离采样点时。步长比不超过 4.83 时，二次数据总被接受，结果精确。

**可选：四 / 五点插值（`max_points=4` 或 `5`）。**

- 在包含 `i−1, i, i+1` 的窗口内取 `max_points` 个点，用 Newton 差商构造插值多项式；
- 只在 `[x[i−1], x[i+1]]` 内寻找局部极大：以 `p''` 的根把区间分成 `p'` 单调的几段，用二分法定位 `p'` 由正变负的点；
- 结果必须严格高于 `y[i]` 且满足保证 2、3，否则依次退回三点抛物线和采样点。

`fit_points` 记录实际采用的点数。审计中的旧四次拟合在一个随机非均匀 5 点例子上给出 `+6.37` 的超调，而 `B = 3.32`。现在五点模式拒绝它并退回抛物线。`rates.cvt_rate_si(..., max_fit_points=3)` 透传该选项。

**待与手册核对。** Polyrate 手册目前不可得，`FITMAX` 等选项实际采用的插值方式（三点或五点、是否在 `s` 的质量标度坐标上插值、回退规则）尚未核实。当前默认取保守的三点抛物线，与 Polyrate 的逐例一致性要等拿到手册和官方算例后再确认。

## 5. WKB / ZCT 核心

分段线性势垒的透射指数（`wkb_exponent`）为：

```text
γ(E) = 2θ(E) = 2/ħ ∫ sqrt(2 μ [V(x)-E]) dx
```

积分对每个线性片段的 `sqrt(V-E)` 使用解析积分。

透射概率（`semiclassical_transmission`）采用 Polyrate ZCT/SCT 使用的一致半经典形式，并计入势垒以上的非经典反射。记 `V^AG = max(V)`，阈值 `E₀ = max(V[0], V[-1])`：

```text
P(E) = 0                       E < E₀
P(E) = 1 / (1 + exp[γ(E)])     E₀ ≤ E ≤ V^AG
P(E) = 1 − P(2V^AG − E)        V^AG < E < 2V^AG − E₀
P(E) = 1                       E ≥ 2V^AG − E₀
```

下限取两侧渐近线的较高者，因此吸能剖面在产物渐近线以下 `P = 0`，κ 与产物平台的截断长度无关。所有能量与剖面能量处于同一绝对标度。

canonical 修正（`thermal_wkb_kappa`）相对于 `V^AG` 处的经典阈值：

```text
κ(T) = β exp(βV^AG) ∫_{E₀}^{∞} P(E) exp(−βE) dE
```

令 `δ = V^AG − E`、`Δ = V^AG − E₀`，并把反射区映射回 `[0, Δ]`，得到等价形式：

```text
κ = β ∫_0^Δ [P(V^AG−δ) e^{βδ} + (1 − P(V^AG−δ)) e^{−βδ}] dδ + e^{−βΔ}
```

积分对 δ 用复合 Simpson 公式，区间数为 `energy_grid_points`（奇数时加一）。第一项按 `exp(βδ − log(1+e^γ))` 在对数空间求值，低温深隧穿时既不上溢也不下溢；真正超出 double 范围时返回 `+inf`。`Δ ≤ 0`（没有高于两侧渐近线的势垒）时 κ = 1。

数值验收见 `tests/test_tunneling.py`：对称 Eckart 势垒（`θ` 有解析式）在 8001 点分段线性网格、16000 个能量区间下，200–1000 K 的 κ 与独立 Gauss–Legendre 参考的相对误差 ≤1e-7。分段线性剖面的 `γ(E)` 逐点带有随 E 振荡的误差（约 `h^1.5`），能量积分会将其平均掉，所以 κ 的收敛明显好于逐点 `P(E)`。

当前实现是 **一维 zero-curvature building block**。Polyrate 的完整 ZCT 还依赖 MEP 上的振动绝热基态势 `V_a^G(s)`（以及质量标度坐标和 CAG 因子）；本核只接收用户给出的一维剖面。SCT/LCT 还需要曲率与 corner-cutting path，不能把本实现误标为这些完整算法。

## 6. 线程与可复现性

- 所有输入在进入内核前复制成连续的 `double` 向量；
- 默认不启用 `-march=native`，从而保证可移植 wheel；
- 设置 `PYRATE_NATIVE_ARCH=1` 才启用本机指令优化；
- 原生和回退后端共享测试容差；
- 参考输出比较以速率相对误差和关键中间量误差分别验收。
