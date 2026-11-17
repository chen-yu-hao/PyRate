# 输入格式说明（TOML 与 Polyrate Legacy）

## 1. `calculation`

```toml
[calculation]
name = "reaction name"
temperatures_k = [200.0, 300.0, 600.0]
method = "cvt"             # tst | cvt | cus | muvt | us | all
tunneling = "sct"          # none | wigner | wkb | zct | mepsag | sct | cd-scsag | eckart | omt | comt | muomt（后三者在 LCT 实现前等同 sct）
standard_pressure_bar = 1.0
rate_unit = "cm^3 molecule^-1 s^-1"
energy_grid_points = 400
state_bin_cm1 = 1.0
state_max_energy_kj_mol = 250.0
```

也可以用 `standard_pressure_pa`。当两者同时出现时，Pa 值优先。

## 2. `reactants` / `products`

每个物种使用 array table：

```toml
[[reactants]]
name = "H2"
electronic_energy_kj_mol = 0.0
symmetry_number = 2
frequencies_cm1 = [4401.0]
frequency_scale = 1.0
frequency_floor_cm1 = 0.0
anharmonicities_cm1 = [121.3]  # 可选：一阶 Morse 非谐常数 ω_e x_e

atoms = [
  {symbol = "H", xyz = [-0.37, 0.0, 0.0], mass_convention = "polyrate"},
  {symbol = "2H", xyz = [ 0.37, 0.0, 0.0]},
]
```

- `symbol` 支持标准元素符号及同位素符号（`D`/`2H`、`T`/`3H`、`13C`、`15N`、`18O`、`37Cl` 等）；
- `mass_convention` 可选 `"iupac"`（默认，IUPAC 标准原子量）或 `"polyrate"`（Polyrate 2023 `interface.F` 单同位素质量）；
- `hindered_rotors` 可声明一维扭转受阻转子修正（Pitzer–Gwinn 或 Polyrate `BES0`）：

```toml
hindered_rotors = [
  {frequency_cm1 = 180.0, barrier_kj_mol = 12.0, symmetry_number = 3, scheme = "bes0"},
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

## 3. `transition_state`

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

## 4. 反应路径（CVT / CUS / μVT / US / ZCT / SCT）

```toml
[path]
coordinate_unit = "bohr"   # angstrom | bohr
reduced_mass_amu = 1.007825

[[path.points]]
coordinate = -0.5
name = "s=-0.5"
effective_mass_amu = 0.62  # 可选：直接提供 SCT 有效质量
curvatures_bohr_inv = [0.15, 0.04, 0.01]  # 可选：各横模反应路径曲率 B_{mF}(s)
# 以及完整 Species 字段

[[path.points]]
coordinate = 0.0
# ...

[[path.points]]
coordinate = 0.5
# ...
```

至少需要三个唯一、可排序坐标。当启用 `zct`/`mepsag` 或 `sct`/`cd-scsag` 且未显式指定 `[tunneling.profile]` 时，PyRate 会自动从 `[path]` 构造振动绝热基态势 $V_a^G(s)$ 与 `TPCDSC` 小曲率有效质量分布 $\mu_{\text{eff}}(s)$。

`coordinate_system` 可选 `cartesian`、`curv1`、`curv2`、`curv3` 或 `intrinsic`。在非静点路径上，推荐使用曲线坐标并在每个点给出 `intrinsic_frequencies_cm1`。如果电子结构程序同时输出 Wilson $B$ 矩阵和力常数，可将它们放入物种的 `metadata`（键名 `wilson_b` 与 `force_constants`），然后通过 `pyrate.curvilinear_frequencies` 计算广义频率。

## 5. 双层、VRC、压力和环境模型

双层路径校正的数组单位为 kJ mol$^{-1}$，`high - low` 的符号约定由程序统一处理：

```toml
[calculation]
method = "ivtst"       # ivtst | vrc-vtst | pressure | master-equation | all
temperatures_k = [300.0, 500.0]

[dual_level]
method = "ispe"        # ispe | ivtst | ioc
coordinates = [-1.0, 0.0, 1.0]
low_level_energies_kj_mol = [42.0, 50.0, 43.0]
high_level_energies_kj_mol = [43.2, 52.0, 44.1]
reference_coordinate = 0.0 # ioc 可选
```

VRC-VTST 使用可重复的 loose-TS 面元求积。每个面元的 `energy_kj_mol` 是相对于反应物的自由能，`weight` 是角度/距离求积权重：

```toml
[vrc]
symmetry_factor = 2.0
[[vrc.samples]]
coordinate = 3.0
energy_kj_mol = 36.0
weight = 0.25
frequencies_cm1 = [120.0, 180.0]
[[vrc.samples]]
coordinate = 4.0
energy_kj_mol = 34.5
weight = 0.75
```

压力依赖计算使用 SS-QRRK 的能量转移参数和 Lindemann/Troe falloff：

```toml
[calculation]
method = "pressure"
pressures_bar = [0.001, 1.0, 100.0]
[pressure_dependence]
collider = "N2"
energy_transfer_cm1 = 200.0
falloff_model = "troe"
f_cent = 0.6
```

溶剂或表面修正通过过渡态与反应物的显式自由能差加入活化势垒；表面还可指定吸附能和取向因子：

```toml
[environment]
phase = "solvent"       # solvent | surface
solvent = "water"
dielectric = 78.4
species_solvation_kj_mol = { TS = -18.0, A = -5.0 }

# surface 示例：phase = "surface"
# surface_site_density_m2 = 1.0e19
# adsorption_energy_kj_mol = -12.0
# orientation_factor = 0.5
```

## 6. 显式 `[tunneling.profile]`

```toml
[tunneling]
method = "wkb"

[tunneling.profile]
coordinates_angstrom = [-1.0, -0.5, 0.0, 0.5, 1.0]
energies_kj_mol = [0.0, 12.0, 25.0, 12.0, 0.0]
reduced_mass_amu = 1.0
effective_masses_amu = [1.0, 0.85, 0.72, 0.85, 1.0]  # 可选
energy_grid_points = 400
```

## 7. Polyrate `.dat` + `.fu40` 输入

`load_input("case.dat", fu40_path="case.fu40")` 或 `pyrate run case.dat --fu40 case.fu40` 可直接读取 Polyrate 2023 格式的 `.dat` 控制文件与 `.fu40` 电子结构路径文件（支持 `*point40` 下的 `freqsource read` 显式频率/曲率，以及 `freqsource hessian` 紧缩下三角或全矩阵笛卡尔力常数自动 `PROJCT` 投影与 `RPHB` 曲率计算）。

