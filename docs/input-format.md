# TOML 输入格式

## `calculation`

```toml
[calculation]
name = "reaction name"
temperatures_k = [200.0, 300.0, 600.0]
method = "tst"             # tst | cvt
tunneling = "wigner"       # none | wigner | wkb | zct
standard_pressure_bar = 1.0
rate_unit = "cm^3 molecule^-1 s^-1"
mass_model = "isotopic"    # isotopic | average
```

也可以用 `standard_pressure_pa`。当两者同时出现时，Pa 值优先。

### 原子质量模型 `mass_model`

`mass_model` 决定**未显式给出 `mass_amu` 的原子**取什么质量，作用于全部物种的原子：反应物、过渡态、产物和 `[[path.points]]`。

| 取值 | 含义 | 例：H / C / Cl |
|---|---|---|
| `isotopic`（默认） | 丰度最高同位素的精确质量（AME2020），与电子结构程序和 Polyrate 的惯例一致 | 1.0078250319 / 12 / 34.968852694 |
| `average` | 常规平均原子量，即 PyRate 早期版本的默认值 | 1.00784 / 12.011 / 35.45 |

- 默认值已由 `average` 改为 `isotopic`。旧输入若依赖平均原子量，请写明 `mass_model = "average"`。
- `D`、`T` 在两种模型下都表示氘和氚。
- 取值不区分大小写；其他值会报 `ValueError`。
- 原子表中显式给出的 `mass_amu` 永远优先，不受 `mass_model` 影响。
- 物种级的 `mass_amu`（见下文）只替换平动用的总质量；若同时给出原子坐标而未给 `moments_of_inertia_amu_a2`，转动惯量仍由各原子质量计算，因此仍受 `mass_model` 影响。
- 数值来源与交叉核对记录在 `src/pyrate/elements.py` 的模块文档中。
- `pyrate inspect` 的摘要会显示所用的 `mass_model`。

## `reactants` / `products`

每个物种使用 array table：

```toml
[[reactants]]
name = "H2"
electronic_energy_kj_mol = 0.0
symmetry_number = 2
frequencies_cm1 = [4401.0]

atoms = [
  {symbol = "H", xyz = [-0.37, 0.0, 0.0]},
  {symbol = "H", xyz = [ 0.37, 0.0, 0.0]},
]
```

没有几何时可直接给质量、转子类型和主惯性矩：

```toml
mass_amu = 28.0
rotor_type = "linear"
moments_of_inertia_amu_a2 = [8.0]
```

`rotor_type` 为 `atomic`、`linear`、`nonlinear` 或 `auto`。

单个原子可以显式指定质量，它优先于 `mass_model`：

```toml
atoms = [
  {symbol = "H",  xyz = [0.0, 0.0, -0.74]},
  {symbol = "Cl", xyz = [0.0, 0.0,  0.53], mass_amu = 36.965902573},  # 37Cl (AME2020)
]
```

`symbol = "D"` 或 `"T"` 直接给出氘、氚的质量。

电子激发态：

```toml
electronic_states = [
  {degeneracy = 2, energy_kj_mol = 0.0},
  {degeneracy = 4, energy_kj_mol = 12.5},
]
```

## `transition_state`

字段与物种相同。虚频可单独声明：

```toml
[transition_state]
name = "TS"
# ... mass/geometry/moments ...
electronic_energy_kj_mol = 35.0
frequencies_cm1 = [500.0, 900.0, 1400.0]
imaginary_frequency_cm1 = 1100.0
```

输入值的正负均可，内部统一存为负频率。过渡态最多允许一个虚频。

## CVT 路径

```toml
[path]
coordinate_unit = "angstrom"

[[path.points]]
coordinate = -0.5
name = "s=-0.5"
# 完整 Species 字段

[[path.points]]
coordinate = 0.0
# ...

[[path.points]]
coordinate = 0.5
# ...
```

至少需要三个唯一、可排序坐标。每个点是一个广义分割面，应提供该点的能量、实频率、质量和转动信息。

## WKB / ZCT profile

```toml
[tunneling]
method = "wkb"

[tunneling.profile]
coordinates_angstrom = [-1.0, -0.5, 0.0, 0.5, 1.0]
energies_kj_mol = [0.0, 12.0, 25.0, 12.0, 0.0]
reduced_mass_amu = 1.0
energy_grid_points = 400
```

第一个能量是正向反应物参考。坐标必须严格递增。
