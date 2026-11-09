# PyRPMD 双分子 RPMD 速率：输入语法核对、带注释输入与参数对比

核对对象：[chen-yu-hao/PyRPMD](https://github.com/chen-yu-hao/PyRPMD) `main` @ `306a482`（2026-11）。
本目录的输入文件需要复制到 PyRPMD 源码的 `examples/rate/` 下使用（相对路径按输入文件所在目录解析）。

| 文件 | 内容 |
| --- | --- |
| `h_h2_bkmp2.inp` | 物理体系 H + H₂（BKMP2 PES），schema 2 显式写法，逐行中文注释；按文献设置：300 K、`force_constant = 0.1`（即 RPMDrate 的 kforce = 0.1·T = 30 hartree）、111 个窗口、R∞ = 30 bohr |
| `toy_exchange_bimolecular.inp` | 同一流程换成仓库自带的解析教学势，不需要 Fortran，几分钟即可跑完，用于参数对比 |
| `h_h2_convergence_study.toml` | `pyrpmd study` 扫描：珠子数 / 步长 / R∞ / 温度，独立种子重复 |
| `collect_results.py` | 汇总多个运行目录的 `results/summary.json`，输出对比表和 CSV |
| `results_0p1T_300K.csv` | 本次实际运行的 6 组结果（第 3 节） |

## 1. 核对结果：rate 输入的实际语法

PyRPMD 的速率任务是 `task = "rate"`，流程为 Bennett–Chandler：
**伞形采样 → PMF → k_QTST**，再 **约束 parent 链 → 成对 ±p 子轨迹 → κ(t)**，最终 `k = k_QTST · κ`。

当前有两种写法，都被解析器接受：

| | schema 1（旧，`pilot.inp`、`bkmp2_pilot.inp`） | schema 2（新，推荐，`research.inp`） |
| --- | --- | --- |
| 反应定义 | `[reaction] fragments=…` | `definition="explicit"`，`reactant_fragments`/`product_fragments`/`forming_bonds`/`breaking_bonds` |
| TS 检查 | 隐式 | `[preparation] ts_action = "validate" \| "refine" \| "none"`，`check_channel` |
| 窗口 | `[umbrella] start/stop/count` + `kappa`（物理单位） | `[[umbrella.regions]]` 分区间距；`force_constant` 裸数字 = **k/T**（hartree/K） |
| 伞形采样 | 全局 `[sampling]` | `[umbrella.sampling]` + `[umbrella.sampling.thermostat]` |
| 透射系数 | `[recrossing] parent_replicas/parent_samples/child_pairs/child_time` | `[recrossing.parent]`、`[recrossing.children]`、`[recrossing.analysis]` 分开 |
| PMF 估计器 | `[rate] pmf_method = "ui" \| "mbar"` | `[rate.pmf] method="umbrella", estimator="ui" \| "mbar"` |

从源码（`input_stages.py`、`input_reaction.py`、`workflows/umbrella_plan.py`、`input_schema.py`）确认的几个容易出错的点：

- **`force_constant` 裸数字是 k/T，不是 k。** `0.00016` 在 1000 K 下是 0.16 hartree；照抄 RPMDrate 示例的 `0.1` 会变成 100 hartree。要固定物理弹簧就写 `"1.9 hartree"`。旧的 `kappa`/`stiffness` 仍按物理单位解释。不要乘以珠子数。
- `regions` 的 `(stop-start)/spacing` 必须是整数（容差 1e-10），相邻区域用 `include_endpoints="left"` 避免重复中心。
- `centers`/`grid`/`regions`/`windows`/`start-stop-count` 只能选一种。
- schema 2 的 parent 链**不继承**伞形采样恒温器；不写时默认 Andersen 每 10 步刷新、`configurations_per_chain=4`。
- `definition="explicit"` 必须同时给出反应物碎片和成对的 forming/breaking bonds；`"infer"` 必须 `check_channel=true`；`ts_action="none"` 只允许显式反应且 `check_channel=false`。
- 两条等价产物通道用 `equivalent_paths = [[0,2,1]]` 计入，此时 `symmetry_factor` 保持 1.0，**不要再乘 2**。
- 时间可以写 `"5 fs"` 或 `{steps = N}`，必须是步长的整数倍。
- 命令：`validate [--check-pes]` → `prepare --pilot` → `run` → `analyze` → `resume`；`run` 拒绝写入已存在的非空目录。

## 2. 运行方法

```bash
git clone https://github.com/chen-yu-hao/PyRPMD.git && cd PyRPMD
python -m pip install -e '.[test]'
cp /path/to/PyRate/examples/pyrpmd_bimolecular/* examples/rate/
```

快速版（教学势，不需要 Fortran）：

```bash
pyrpmd validate examples/rate/toy_exchange_bimolecular.inp --check-pes
pyrpmd run examples/rate/toy_exchange_bimolecular.inp --output runs/toy-P4
```

物理版（BKMP2，需要 gfortran；构建脚本会下载并校验固定版本的源码）：

```bash
python benchmarks/h3/build_pes.py
python benchmarks/h3/pes.py --check --output runs/bkmp2-pes-check.json
pyrpmd validate examples/rate/h_h2_bkmp2.inp --check-pes
pyrpmd run examples/rate/h_h2_bkmp2.inp --output runs/h_h2_bkmp2
pyrpmd analyze runs/h_h2_bkmp2
```

需要看的输出字段在 `results/summary.json` 里（另有 `rate_pmf.json`、`rate_pmf.csv`、`recrossing.csv`）：

| 字段 | 含义 |
| --- | --- |
| `pmf.free_energy_barrier_hartree`、`pmf.xi_star` | PMF 势垒和分割面 |
| `pmf.k_qtst_cm3_s` | 静态因子 k_QTST |
| `kappa`、`kappa_standard_error`、`plateau_detected` | 透射系数及平台 |
| `rate_estimate_cm3_s` | 有限时间速率估计（一定有） |
| `rate_cm3_s` | 仅当 PMF 支撑、平台、能量漂移检查都通过时给出，否则为 `null` |
| `status`、`converged`、`warnings` | pilot 输入一般为 `pilot_requires_convergence`、`converged=false` |

## 3. 不同参数设置的输出对比

参考值为 Suleimanov, Allen & Green, *CPC* 184, 833 (2013) 表 2（300 K）：
经典 TST 1.56×10⁻¹⁸，RPMD(P=128) QTST 2.51×10⁻¹⁶，κ = 0.776，k = 1.95×10⁻¹⁶ cm³ molecule⁻¹ s⁻¹。

### 3.0 本次实际运行：文献弹簧 0.1·T

环境：PyRPMD 0.2.0 @ `306a482`，native C++/OpenMP 内核，BKMP2 用 GNU Fortran 13.3 编译，并通过 `pes.py --check`（势垒 9.613 kcal/mol，一个虚频）。4 核，`--workers 4`。
公共设置：`h_h2_bkmp2.inp`，300 K，`force_constant = 0.1` → k = 30.0 hartree（`validate` 输出的 `stiffness_au` 已核对），111 个窗口 × 2 条链，R∞ = 30 bohr，0.1 fs，MBAR（`compare_ui` 默认同时给出 UI），16 个 parent 构型，64 条子轨迹，30 fs。每组只改一个参数。

| 运行 | P | 种子 | 平衡+采样 / 每链 | ξ* | 势垒 MBAR / UI (hartree) | k_QTST | κ ± SE | k (cm³ s⁻¹) | k / 文献 | 中位 ESS/链 | 用时 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P1_200fs（经典） | 1 | 721062 | 50+200 fs | 0.999 | 0.0360 / 0.0361 | 8.8×10⁻²⁵ | 0.996 ± 0.003 | 8.8×10⁻²⁵ | 4.5×10⁻⁹ | 9.8 | 238 s |
| P16_200fs（基准） | 16 | 721062 | 50+200 fs | 1.008 | 0.0306 / 0.0302 | 2.4×10⁻²² | 0.48 ± 0.08 | 1.2×10⁻²² | 6.0×10⁻⁷ | 10.2 | 238 s |
| P16_seed2 | 16 | 991073 | 50+200 fs | 0.995 | 0.0302 / 0.0303 | 3.7×10⁻²² | 0.79 ± 0.09 | 2.9×10⁻²² | 1.5×10⁻⁶ | 10.0 | 237 s |
| P32_200fs | 32 | 721062 | 50+200 fs | 0.999 | 0.0322 / 0.0318 | 4.7×10⁻²³ | 0.81 ± 0.06 | 3.8×10⁻²³ | 1.9×10⁻⁷ | 10.7 | 240 s |
| P128_200fs | 128 | 721062 | 50+200 fs | 0.987 | 0.0365 / 0.0368 | 4.9×10⁻²⁵ | 0.41 ± 0.16 | 2.0×10⁻²⁵ | 1.0×10⁻⁹ | 10.3 | 249 s |
| **P16_1ps** | 16 | 721062 | **200+1000 fs** | 1.000 | **0.0224 / 0.0224** | 1.4×10⁻¹⁸ | 0.56 ± 0.12 | **7.5×10⁻¹⁹** | 3.9×10⁻³ | 32.9 | 1241 s |

所有运行的 `status` 都是 `plateau_estimate`，`converged = false`，子轨迹端点能量误差 ≤ 0.065 kBT。完整字段见 `results_0p1T_300K.csv`。

解读：

- **主导因素是采样长度。** 只把每窗口采样从 50+200 fs 延长到 200+1000 fs，PMF 势垒就从 0.0306 降到 0.0224 hartree，速率升高约 6500 倍。BKMP2 的电子势垒是 0.0153 hartree，而量子化的 PMF 势垒应低于经典值，所以 0.0224 仍然偏高，说明 1 ps 仍未平衡（官方 P=128、1.5 ps 的审计结果是 0.0166–0.0181）。
- **在 200 fs 采样下，珠子数的趋势被采样噪声掩盖。** P=128 的势垒反而比 P=16 高，与物理预期（P 越大隧穿越强、势垒越低）相反。只换种子，速率就相差 2.5 倍，κ 从 0.48 变到 0.79。因此短采样时比较 P 没有意义。
- **MBAR 与 UI 用同一批样本时，势垒相差 ≤ 0.0004 hartree**（300 K 下约 0.4 kBT），是次要因素。
- **κ 的趋势符合物理：** 经典 P=1 时 κ ≈ 0.996（文献 0.994），RPMD 时 0.4–0.8（文献 0.776）。但只有 16 个 parent 构型，标准误差较大。
- **用时与珠子数基本无关**（P=1 和 P=128 都约 240 s），而与模拟时间成正比。BKMP2 本身很便宜，开销在每步的调度上，所以算力应该优先用于延长采样。
- 想接近文献值，需要每窗口 ps 级采样（文献为 20 ps 平衡 + 100 ps 采样）、更多独立链，并检查前后两半数据的 PMF 是否一致。按本机速度估算，P=16、每窗口 1 ps 约需 20 分钟，耗时随采样长度线性增长。

下面 3.1–3.3 是 PyRPMD 仓库中**已提交的官方运行记录**（不是本次运行），可与 3.0 对照。

### 3.1 采样长度与随机种子（`docs/validation/literature-audit-2026-11-01/h3/compact-results.csv`）

P=128，111 个窗口 × 2 条链，R∞ = 30 bohr，k = 30 hartree，0.1 fs。

| 算例 | 平衡 + 采样 / 每链 | UI 势垒 / hartree | k_QTST | κ | k 候选值 | 与文献之比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 5 + 10 fs | 0.0953 | 6.43×10⁻⁵² | 0.973 | 6.25×10⁻⁵² | 3×10⁻³⁶ |
| pilot | 50 + 100 fs | 0.0260 | 3.12×10⁻²⁰ | 0.617 | 1.93×10⁻²⁰ | 9.9×10⁻⁵ |
| long-a（种子 831072） | 500 + 1000 fs | 0.01658 | 6.43×10⁻¹⁶ | 0.775 | 4.98×10⁻¹⁶ | 2.56 |
| long-b（种子 991073） | 500 + 1000 fs | 0.01807 | 1.33×10⁻¹⁶ | 0.715 | 9.54×10⁻¹⁷ | 0.49 |
| 经典 P=1 | 500 + 1000 fs | 0.02323 | 5.87×10⁻¹⁹ | 0.997 | 5.86×10⁻¹⁹ | 0.38 |
| 经典 TST 积分（无采样） | — | — | 1.553×10⁻¹⁸ | — | — | 0.996（TST/TST） |

结论：

- **采样不足会让速率低几十个数量级。** 短轨迹下各窗口远未平衡，PMF 势垒被高估（0.095 → 0.017 hartree），而速率对势垒是指数敏感的。
- 采样 1 ps 时，两个种子仍相差 5.2 倍；long-a 的前后两半 QTST 分别为 1.68×10⁻¹⁶ 和 2.42×10⁻¹⁵，**PMF 还在漂移**。
- κ 收敛得快得多（0.62 → 0.72–0.77，文献 0.776）；误差主要来自 PMF，不是 recrossing。
- 不经采样的经典 TST 积分与文献相差 0.44%，说明 PES、测度和双通道计数是对的，问题在于采样的收敛。

### 3.2 弹簧强度（`docs/validation/tier1-2026-10-31/{broad,stiff}-pilot/summary.json`）

P=128，每窗口 50 + 100 fs，2 条链，UI 估计器，ξ* = 0.992。

| 量 | 弱弹簧 0.095 hartree（100 kBT） | 文献匹配 30 hartree |
| --- | ---: | ---: |
| PMF 势垒 / hartree | 0.01073 | 0.02601 |
| k_QTST / cm³ s⁻¹ | 3.04×10⁻¹³ | 3.12×10⁻²⁰ |
| κ | 0.576 ± 0.094 | 0.617 ± 0.103 |
| 有限时间速率 | 1.75×10⁻¹³（文献的 899 倍） | 1.93×10⁻²⁰（文献的 10⁻⁴ 倍） |
| 每链有效样本中位数 | 4.2 | 9.7 |
| 分割面处有直方图支撑 | 否 | 是 |
| `rate_cm3_s` | `null` | `null` |

弱弹簧时窗口分布太宽，势垒顶部没有采到样，Gaussian UI 是从没有数据的区域外推出来的，因此严重低估了势垒（MBAR 重放会拒绝这个分割面）。强弹簧在势垒顶部有支撑，但只有 150 fs 时同样未平衡。两组 `rate_cm3_s` 都被程序扣留为 `null`，这是程序的正确行为。

### 3.3 估计器 MBAR 与 UI（同一批样本重放，`extended-pmf-replay.json`）

| 算例 | UI 势垒 | MBAR 势垒 | QTST(MBAR)/QTST(UI) |
| --- | ---: | ---: | ---: |
| long-a | 0.016577 | 0.016439 | 1.156 |
| long-b | 0.018072 | 0.017944 | 1.144 |
| 经典 | 0.023226 | 0.023102 | 1.140 |

在采样充分、窗口重叠良好的前提下，估计器的选择只带来约 15% 的差别，远小于种子或采样长度造成的数倍差异。

### 3.4 对本目录输入的建议

- `h_h2_bkmp2.inp` 的采样预算（50 + 200 fs/窗口）**只能用来检验流程是否跑通**，3.0 节的结果说明速率会差 6 个数量级。生产计算时，每窗口需要 ps 级采样，并分别比较各条链和前后两半数据的 PMF。
- 优先加长 `umbrella.sampling.production` 并增加 `chains_per_window`；κ 部分的预算可以留到最后再加。
- 用 `h_h2_convergence_study.toml` 做 P、dt、R∞ 的单因素扫描；`force_constant = 0.1` 即 k = 0.1·T，窗口宽度 σ = sqrt(k_B/0.1) ≈ 0.0056 与温度无关，因此 0.01 的窗口间距在各温度下都适用。
- 在 `toy_exchange_bimolecular.inp` 上可以几分钟内重复上面的对比：用 `--beads 4/8/16`，把 `estimator` 在 `ui`/`mbar` 之间切换，`production` 设为 `20 fs` 和 `100 fs`，`force_constant` 取 `0.00016` 或 `0.00004`。它的数值没有物理意义，只用来观察参数的影响趋势。
