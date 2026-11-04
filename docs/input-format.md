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
```

也可以用 `standard_pressure_pa`。当两者同时出现时，Pa 值优先。

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
