# PyRate

**PyRate** 是一个面向 Polyrate 2023 工作流兼容性的现代反应速率计算程序：流程、输入模型与结果组织使用强类型 Python，分区函数批处理、变分极值（`FITMAX`）、Kemble 均匀半经典 WKB / SCT / Eckart 隧穿积分与非经典反射（`BOLTZ`）、以及 Beyer–Swinehart 状态计数等热点使用 C++17。Python 后端提供 1:1 逐函数回退，因此源码检出后即使尚未编译扩展也可以完整运行。

> 当前版本为 **0.2.0**。已实现单能级热力学、主流变分过渡态理论（CTST、CVT、CUS、μVT、US）、振动绝热与多维小曲率隧穿（Wigner、CAG、ZCT/MEPSAG、SCT/CD-SCSAG、Eckart、COMT、μOMT）、Morse 非谐与 Pitzer–Gwinn（`BES0`）受阻转子修正，以及 Polyrate `.dat + .fu40` 电子结构路径文件的直接端到端求解。精确兼容边界见 [`docs/compatibility.md`](docs/compatibility.md)。

## 已实现

- 理想气体标准态下的平动、刚性转子、谐振子与多能级电子分区函数；
- Polyrate 单同位素原子质量约定（`mass_convention = "polyrate"`）与显式同位素符号（`D`/`2H`、`T`/`3H`、`13C`、`18O`、`37Cl` 等）；
- 频率缩放因子（`frequency_scale`）、低频截断下限（`frequency_floor_cm1`）、一阶非谐振子（Morse 能级梯与非谐 ZPE）和 Pitzer–Gwinn / Polyrate `BES0` 受阻内旋转修正；
- 单分子和双分子常规过渡态理论（CTST/Eyring）及正逆向平衡常数 $K_c(T)$、$K_p(T)$；
- 沿反应路径最大化广义活化自由能的 canonical VTST（CVT）与 Polyrate `FITMAX` 3 点抛物线 / 5 点四次多项式瓶颈精修；
- 考虑多重自由能极大值与中间极小值的 Canonical Unified Statistical（CUS）速率；
- 基于 Beyer–Swinehart 振动态密度与经典转动态卷积的微正则变分过渡态理论（μVT）与微正则统一统计理论（US）；
- 经典绝热基态修正因子（CAG：$\kappa^{\text{CAG}}_{\text{TST}}$、$\kappa^{\text{CAG}}_{\text{CVT}}$）；
- Kemble 均匀半经典透射 $P(E) = 1 / [1 + \exp(2\theta(E))]$ 与高于势垒的非经典反射积分（Polyrate `BOLTZ`）；
- Wigner、一维 WKB、沿 MEP 振动绝热基态势 $V_a^G(s)$ 的零曲率隧穿（ZCT/MEPSAG）、耦合反应路径曲率 $B_{mF}(s)$ 与离心主导拐点（`TPCDSC`）的小曲率隧穿（SCT/CD-SCSAG）、非对称 Eckart 解析透射、以及 canonical / microcanonical 优化多维隧穿（COMT / μOMT）；
- 笛卡尔 Hessian 质量加权、反応路径梯度与平移/转动投影（Polyrate `FDIAG` / `PROJCT`）及反应路径曲率向量计算（`RPHB`）；
- Polyrate `.dat` 输入与伴随 `.fu40`（`freqsource read` 与 `freqsource hessian`）的直接转换与端到端速率计算；
- Polyrate 2023 全部 3 种 `fu15` 速率汇总表头与多子表（`CVT/COMT`、`CVT/mOMT` 等）无损解析与逐列数值比较；
- TOML 输入、命令行、JSON/CSV/表格输出、Python API 与 CI。

## 安装

需要 Python 3.10+、C++17 编译器和 NumPy。

```bash
git clone https://github.com/chen-yu-hao/PyRate.git
cd PyRate
python -m pip install -e ".[test]"
python -m pyrate native-info
pytest -q
```

安装时会构建 `pyrate._kernels`。调试纯 Python 回退时：

```bash
PYRATE_DISABLE_NATIVE=1 pytest -q
```

## 快速运行

```bash
pyrate validate examples/harmonic_cvt.toml
pyrate run examples/harmonic_cvt.toml
pyrate run examples/wkb_tst.toml --format json -o rates.json
```

示例输出包含每个温度的 `ΔG‡_TST`、`k_TST`、CVT 瓶颈坐标、`ΔG‡_CVT`、`k_CVT`（以及启用 `"all"` 时的 `k_CUS`、`k_muVT`、`k_US`、平衡常数与 CAG 因子）、隧穿系数 `κ` 和最终速率。

### 直接运行或检查旧 Polyrate 输入

对于包含完整静点或预计算路径（`.fu40`）的 Polyrate `.dat` 输入，可直接验证并求解：

```bash
pyrate inspect path/to/ch5fu40tr1.dat --json
pyrate validate path/to/ch5fu40tr1.dat
pyrate run path/to/ch5fu40tr1.dat --fu40 path/to/ch5fu40tr1.fu40
```

解析器会保留全部区段与未知关键字。依赖编译期解析势能面 Fortran 子程序（`POT` hook）或 `fu30/fu31/fu50` 双层修正文件的算例可通过 `inspect` 检查结构。

## Python API

```python
from pyrate import calculate_rates, load_input, load_toml

job = load_toml("examples/harmonic_cvt.toml")
result = calculate_rates(job.reaction, job.options, name=job.name)
for row in result.rows:
    print(row.temperature_k, row.k_final, row.rate_unit)

# 也可直接加载 Polyrate .dat + .fu40
legacy_job = load_input("ch5fu40tr5.dat", fu40_path="ch5fu40tr5.fu40")
legacy_result = calculate_rates(legacy_job.reaction, legacy_job.options, name=legacy_job.name)
```

## 数值与单位约定

- 公开输入：能量 `kJ mol⁻¹`、频率 `cm⁻¹`、质量 `amu`、长度 `Å`（或 `bohr`）；
- 内核：SI，每分子能量用于隧穿与状态计数积分；
- 默认气相标准压力：`1 bar`；
- 单分子速率：`s⁻¹`；
- 双分子内部单位：`m³ mol⁻¹ s⁻¹`，可输出 `L mol⁻¹ s⁻¹` 或 `cm³ molecule⁻¹ s⁻¹`；
- `electronic_energy_kj_mol` 不含零点能；默认由正频率加入谐振（或非谐 Morse）零点能；
- 负频率不进入振动分区函数，过渡态最多允许一个负频率。

公式与 Polyrate 标准态、Kemble WKB、`TPCDSC` 离心拐点及 `PROJCT` 投影的对应关系见 [`docs/architecture.md`](docs/architecture.md)。

## 项目结构

```text
src/pyrate/          Python 数据模型、解析、热力学、变分/微正则速率、隧穿和 CLI
cpp/kernels.cpp      C++17 数值热点（分区函数、FITMAX、Kemble WKB/SCT、Eckart、DOS）
examples/            可直接运行的 TOML 样例
tests/               单元、原生/回退一致性、物理模型和 Polyrate 对照测试
tools/               参考分发包导入与基准对照工具
docs/                架构、输入格式、兼容矩阵和验证方法
```

## 兼容性路线

当前 0.2.0 版本已完整覆盖单能级谐振/非谐/受阻转子热力学、CTST/CVT/CUS/μVT/US 变分方法、Wigner/CAG/ZCT/SCT/Eckart/COMT/μOMT 隧穿以及 `.dat + .fu40` 反应路径求解。在 Polyrate 2023 官方 `ch5fu40` 基准系列（`tr1`–`tr6`）上，TST 与 CVT 速率相对误差为 `0.08%`，低温强隧穿（298 K 下 $\kappa \approx 1.85 \times 10^5$）的 CVT/ZCT 与 CVT/SCT 全温区吻合度达 `91.6%–97.9%`。

下一阶段待移植的 Polyrate 2023 高级模块包括：大曲率隧穿（LCT/LCG3/LCG4）、反应路径内蕴积分器（Page–McIver / Euler）、双层插值修正（ISPE/IVTST/IOC、`fu50/fu51`）、可变反应坐标（VRC-VTST）、SS-QRRK 压力依赖与外部电子结构/解析势能面直接动力学接口。

### Polyrate 官方输出逐列对照

```bash
pyrate run path/to/ch5fu40tr5.dat --fu40 path/to/ch5fu40tr5.fu40 --format json -o /tmp/pyrate.json
python tools/compare_polyrate_rates.py /path/to/ch5fu40tr5.fu15 /tmp/pyrate.json --rtol 0.01
```

工具支持自动映射 `TST`、`CVT`、`CUS`、`muVT`、`US` 以及 `/W`、`/ZCT`、`/MEPSAG`、`/SCT`、`/CD-SCSAG`、`/COMT`、`/mOMT` 等复合速率列，详情见 [头对头测试规范](docs/head-to-head.md) 与 [`docs/compatibility.md`](docs/compatibility.md)。

## 许可与来源

PyRate 使用 Apache License 2.0。Polyrate 2023 参考分发包的 DOI 为 `10.17632/vpf3y9cbhf.1`；本仓库不内置该分发包，测试导入工具接受用户本地的合法副本。具体归属见 [`NOTICE`](NOTICE)。

