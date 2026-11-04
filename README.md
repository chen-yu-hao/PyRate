# PyRate

**PyRate** 是一个面向 Polyrate 工作流兼容性的反应速率程序：流程、输入模型与结果组织使用可读的 Python，分区函数批处理、变分极值、WKB 隧穿积分和状态计数等热点使用 C++。Python 后端提供逐函数回退，因此源码检出后即使尚未编译扩展也可以运行。

> 当前版本为 **0.1.0 / 核心阶段**。它已经是可安装、可运行、可测试的软件，但尚未宣称与 Polyrate 2023 全功能 1:1。精确边界见 [`docs/compatibility.md`](docs/compatibility.md)。

## 已实现

- 理想气体标准态下的平动、刚性转子、谐振子与电子分区函数；
- 单分子和双分子常规过渡态理论（CTST/Eyring）；
- 沿反应路径最大化广义活化自由能的 canonical VTST/CVT；
- 最多五点的局部多项式瓶颈精修；
- Wigner 隧穿修正；
- 一维热 WKB / zero-curvature 风格透射修正；
- Beyer–Swinehart 振动态密度计数 C++ 内核；
- 笛卡尔 Hessian 质量加权、平移/转动投影和正规模分析；
- Polyrate `*GENERAL`、`*REACT1`、`*START`、`*PATH`、`*TUNNEL`、`*RATE` 等区段的保留式解析；
- Polyrate `fu15` 速率汇总表解析与数值误差比较；
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

示例输出包含每个温度的 `ΔG‡_TST`、`k_TST`、CVT 瓶颈坐标、`ΔG‡_CVT`、`k_CVT`、隧穿系数 `κ` 和最终速率。

### 检查旧 Polyrate 输入

```bash
pyrate inspect path/to/h3tr1.dat --json
pyrate validate path/to/h3tr1.dat
```

解析器会保留未知关键字，而不是静默删除。依赖外部势能面 hook、`fu30/fu31/fu40` 或尚未实现算法的旧输入目前只能检查，不能直接求解。

## Python API

```python
from pyrate import calculate_rates, load_toml

job = load_toml("examples/harmonic_cvt.toml")
result = calculate_rates(job.reaction, job.options, name=job.name)
for row in result.rows:
    print(row.temperature_k, row.k_final, row.rate_unit)
```

## 数值与单位约定

- 公开输入：能量 `kJ mol⁻¹`、频率 `cm⁻¹`、质量 `amu`、长度 `Å`；
- 内核：SI，每分子能量用于隧穿积分；
- 默认气相标准压力：`1 bar`；
- 单分子速率：`s⁻¹`；
- 双分子内部单位：`m³ mol⁻¹ s⁻¹`，可输出 `L mol⁻¹ s⁻¹` 或 `cm³ molecule⁻¹ s⁻¹`；
- `electronic_energy_kj_mol` 不含零点能；默认由正频率加入谐振零点能；
- 负频率不进入振动分区函数，过渡态最多允许一个负频率。

这些定义避免旧程序中原子单位、cgs 与 SI 混用造成的隐式换算。公式与 Polyrate 标准态处理的对应关系见 [`docs/architecture.md`](docs/architecture.md)。

## 项目结构

```text
src/pyrate/          Python 数据模型、解析、热力学、编排和 CLI
cpp/kernels.cpp      C++17 数值热点与 CPython 扩展
examples/            可直接运行的 TOML 样例
tests/               单元、原生后端和端到端测试
tools/               参考分发包导入与基准工具
docs/                架构、输入格式、兼容矩阵和验证方法
```

## 兼容性路线

当前核心覆盖 harmonic CTST/CVT、Wigner 与一维 WKB。以下 Polyrate 2023 能力仍需逐项移植并以官方测试输出验收：SCT/LCT、μVT/US、VRC-VTST、反应路径积分器、hindered rotor/非谐振动、双层校正、IVTST/ISPE/IOC、SS-QRRK 与压力依赖动力学。ICVT 已在 Polyrate 2023 中移除，仅作为旧版兼容性话题。

### Polyrate 官方输出逐列对照

在使用相同物理模型和温度网格生成 PyRate JSON 后，比较 Polyrate `fu15` 中实际可验证的速率列：

```bash
pyrate run examples/harmonic_cvt.toml --format json -o /tmp/pyrate.json
python tools/compare_polyrate_rates.py /path/to/case.fu15 /tmp/pyrate.json --rtol 0.005
```

工具将匹配温度、检查速率量纲、执行单位换算并输出每项误差；未映射的高级方法不会被误报为通过。该工具的合成测试通过**不代表**已经完成 112 个官方算例的数值验收，说明见 [头对头测试规范](docs/head-to-head.md)。

完整模块矩阵和“完成”的数值验收条件见 [`docs/compatibility.md`](docs/compatibility.md)。参考测试工作流见 [`docs/reference-validation.md`](docs/reference-validation.md)。

## 许可与来源

PyRate 使用 Apache License 2.0。Polyrate 2023 参考分发包的 DOI 为 `10.17632/vpf3y9cbhf.1`；本仓库不内置该分发包，测试导入工具接受用户本地的合法副本。具体归属见 [`NOTICE`](NOTICE)。
