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

## 5. 显式 `[tunneling.profile]`

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

## 6. 在线势能面与直接动力学（`[potential]`）

可通过 `[potential]` 指定内置解析势能面或外部量子化学子进程；开启后只需给出各物种（及过渡态）的 `atoms`，无需预先手工填写 `electronic_energy_kj_mol` 与 `frequencies_cm1`：

```toml
[potential]
type = "cl_hbr_leps"         # 内置：cl_hbr_leps | h3_leps | extended_leps | muller_brown | eckart_valley
zero_at_reactants = true
```

自定义三体扩展 LEPS 参数使用 `type = "extended_leps"`（需提供长度为 3 的 `de_kcal_mol`、`re_angstrom`、`beta_angstrom_inv`、`sato` 及可选 `atom_symbols`）。

外部电子结构接口与持久化重启缓存（支持 `command`、`gaussian`、`gamess`）：

```toml
[potential]
type = "command"             # command | gaussian | gamess
command = "python3 my_qm_driver.py {input} {output}"
restart_file = "cache/restart.json"   # 相对当前 .toml 文件解析
rst_tol_bohr = 1e-8
supermolecule = true                  # 多体专用势能面需要原子碎片按 200 Å 隔离求值时设为 true
```

- `type = "gaussian"` 可配置 `command`（默认 `["g16"]`）、`method`、`basis`、`mem`、`nprocshared`、`extra_keywords`、`extra_sections`、`fchk_command`；
- `type = "gamess"` 可配置 `command`（默认 `["rungms"]`）、`scftyp`、`contrl`、`basis`、`extra_groups`；
- 物种表与过渡态支持 `optimize = true`（自动极小值 BFGS / 鞍点 PRFO 优化，默认收敛阈值 `gtol_au = 1e-6`）；
- **注意**：与 `[potential]` 配合声明自旋多重度或电子态简并度时，请使用物种表中的 `electronic_states = [{degeneracy = 2, energy_kj_mol = 0.0}]`（TOML 解析器不读取 `electronic_degeneracy` 键）。

## 7. 反应路径在线积分（`[mep]` 与 `[path.integration]`）

与 `[potential]`（或 CLI `--pes`）结合时，可用 `[mep]` 或 `[path.integration]`（二选一互斥，不可同时提供）自动从过渡态或初始非驻点积分散布反应路径；完整端到端配置见 [`examples/cl_hbr_mep.toml`](../examples/cl_hbr_mep.toml)：

```toml
[mep]                        # 或写为 [path.integration]
method = "pm"                # pm (同 rpm/lqa) | es1 | euler (同 esd)
coordinate_system = "curvilinear"  # curvilinear | rectilinear
curvature_method = "rphb"    # rphb | none
step_size_bohr = 0.03
hessian_stride = 1           # 每隔多少积分步求值一次 Hessian（INH）
save_stride = 1
s_min_bohr = -1.2
s_max_bohr = 1.2
scale_mass_amu = 1.0
start_from_saddle = true
direction_sign = 1           # 1 | -1（或提供 direction_vector_angstrom）
mode_tracking = true
optimize_stationary_points = false
```

## 8. Polyrate `.dat` + `.fu40` 输入与命令行参数

`load_input("case.dat")`、`load_input("case.dat", fu40_path="case.fu40")` 或命令行可直接读取与执行 Polyrate 2023 格式 `.dat`：

- `.fu40` 模式：配合 `--fu40 case.fu40` 支持 `*point40` 下 `freqsource read` 与 `freqsource hessian`（紧缩/全矩阵 `PROJCT` + `CURV`/`RECT` 投影 + `RPHB` 曲率）；当 `.fu40` 的 `*R140`/`*R240`/`*P140`/`*P240`/`*SADDLE40` 包含 `GEOM` 块时优先使用其平衡构型；
- 在线势能面模式：支持 `*ENERGETICS POTENTIAL BUILTIN:<name>`（如 `POTENTIAL BUILTIN:cl_hbr_leps`、`BUILTIN:h3_leps`）或通过 CLI `--pes <name|command:<cmd>>` 注入势能面；支持 `*OPTIMIZATION`（`OPTMIN`、`OPTTS`、`BFGS`、`PRFO`、`GTOL`）、`*PATH`（`PM`/`RPM`/`LQA`/`ES1`/`ESD`/`EULER`、`SSTEP`、`SRANGE`/`SMIN`/`SMAX`、`INH`、`SCALEMASS`、`NOSADDLE`、`SIGN`、`CURV`/`RECT`）、`*GENERAL MASS` 逐原子质量覆盖、`FREQSCALE` 及 `.fu40` 导出；
- 常用新增 CLI 参数：
  - `pyrate run input.dat --pes cl_hbr_leps --export-fu40 out.fu40`：在线求解并导出 `.fu40` 供后续 `--fu40` 复算；
  - `pyrate run input.toml --pes "command:python3 driver.py" --restart cache.json`：注入外部驱动及持久化重启缓存；
  - `pyrate validate input.dat --strict`：启用严格语义校验（对应 `validate_legacy_semantics`），遇到不支持的区段/关键字（如 `LCG4`、`ISPE`、`ICVT`、`VRCTST`、`SOLV`、`RODS`）或非法单位/参数时以退出码 `2` 报告精确的 `*SECTION KEYWORD line N` 错误。

