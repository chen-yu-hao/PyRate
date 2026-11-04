# 架构与数值约定

## 1. 分层原则

PyRate 不机械翻译 Fortran 全局数组，而是把 Polyrate 的职责拆成四层：

1. **模型层**：`Atom`、`Species`、`Reaction`、`ReactionPath` 和 `CalculationOptions`；
2. **物理层**：分区函数、标准 Gibbs 能、CTST/CVT 和隧穿修正；
3. **数值层**：C++ 热点与对应 Python 回退；
4. **适配层**：TOML、Polyrate legacy parser、`fu15` reference parser 和 CLI。

这种分层使每个公式可以独立测试，也避免将 Polyrate 的固定大小 COMMON block 和文件单元号传播到现代 API。

## 2. Python / C++ 边界

`cpp/kernels.cpp` 当前提供：

- 谐振子 `log(q_vib)` 批处理；
- Wigner 系数；
- 最多五点的局部极值精修；
- 分段线性势垒的 WKB action exponent；
- canonical thermal WKB 积分；
- Beyer–Swinehart 状态计数。

C++ 函数在读取 Python 序列后释放 GIL 执行长循环。`src/pyrate/native.py` 是唯一公共桥接层；导入扩展失败时调用等价 Python 算法。

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

采样最大值附近最多使用五点多项式拟合。若最大值在边界、矩阵病态或拟合发生不合理过冲，则安全回退到采样点。

## 5. WKB / ZCT 核心

分段线性势垒的透射指数为：

```text
γ(E) = 2/ħ ∫ sqrt(2 μ [V(x)-E]) dx
P(E) = exp[-γ(E)]
```

积分对每个线性片段的 `sqrt(V-E)` 使用解析积分。canonical 修正用 Simpson 能量积分，并采用移位指数形式避免低温下的上溢/下溢。

当前实现是 **一维 zero-curvature building block**。Polyrate 的完整 ZCT 还依赖 MEP 上的振动绝热基态势；SCT/LCT 还需要曲率与 corner-cutting path，不能把本实现误标为这些完整算法。

## 6. 线程与可复现性

- 所有输入在进入内核前复制成连续的 `double` 向量；
- 默认不启用 `-march=native`，从而保证可移植 wheel；
- 设置 `PYRATE_NATIVE_ARCH=1` 才启用本机指令优化；
- 原生和回退后端共享测试容差；
- 参考输出比较以速率相对误差和关键中间量误差分别验收。
