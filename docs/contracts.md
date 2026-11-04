# 并行开发契约（Wave 1）

本文件冻结 Wave 1 各 agent 的**接口**、**文件所有权**和**验收门槛**。背景与优先级见 [`roadmap.md`](roadmap.md) 的 §3 与 §4。

## 通用规则

1. **只修改自己负责的文件。** 需要改动别人的文件时，在最终报告中写明，由集成方处理。
2. **同时维护两种后端。** 每个原生内核都同时存在于 `cpp/<area>.cpp` 和 `src/pyrate/native/<area>.py`：
   - `py_<name>` 是纯 Python 参考实现；
   - 公共 `<name>` 只做一次输入校验，然后分派到 C++ 或 Python。
   - 新增 C++ 函数时，只往本领域的 `pyrate_<area>_methods[]` 表里加条目。`module.cpp` 会自动注册，不需要修改。
3. **门禁。** 以下检查全部通过才能交付：
   - `ruff check src tests tools`；
   - `mypy`（strict）；
   - `pytest -q`；
   - `PYRATE_DISABLE_NATIVE=1 pytest -q`；
   - 本领域的 `tests/test_parity_<area>.py`。
4. **金标准值。** 只有在算法确有变更时才更新 `tests/test_golden_*.py` 的数值，并在提交信息中写明原因和变化幅度。
5. **不要宣称兼容。** 在没有 Polyrate 官方算例的情况下，不得把 `docs/compatibility.md` 中任何条目改为 Implemented。该文件由集成方维护。
6. **环境隔离。** 每个工作树使用自己的虚拟环境，避免导入到别的工作树的代码：

   ```bash
   python -m venv --system-site-packages .venv
   .venv/bin/pip install -e ".[test,dev]"
   ```

   然后用 `.venv/bin/python -c "import pyrate; print(pyrate.__file__)"` 确认路径在本工作树内。

## A1：隧穿核（N1 / N2）

**负责的文件**
- `cpp/wkb.cpp`
- `src/pyrate/native/wkb.py`
- `src/pyrate/tunneling.py`
- `tests/test_tunneling.py`
- `tests/test_parity_wkb.py`
- `tests/test_golden_wkb.py`
- `docs/architecture.md` §5

**接口（签名保持不变）**
- `wkb_exponent(coordinates_m, potentials_j, energy_j, reduced_mass_kg) -> float`：返回 `2θ(E)`，语义不变。
- `thermal_wkb_kappa(coordinates_m, potentials_j, temperature_k, reduced_mass_kg, energy_grid_points=400) -> float`：改为以下语义。
  - `V^AG = max(V)`，阈值 `E₀ = max(V[0], V[-1])`。
  - 透射概率：
    - `P(E) = 0`，当 `E < E₀`；
    - `P(E) = 1/(1+exp(2θ(E)))`，当 `E₀ ≤ E ≤ V^AG`；
    - `P(E) = 1 − P(2V^AG − E)`，当 `V^AG < E < 2V^AG − E₀`（非经典反射）；
    - `P(E) = 1`，当 `E ≥ 2V^AG − E₀`。
  - `κ(T) = β·exp(βV^AG) ∫_{E₀}^{∞} P(E) exp(−βE) dE`，相对于在 `V^AG` 处的经典阈值。
- 新增 `semiclassical_transmission(coordinates_m, potentials_j, energy_j, reduced_mass_kg) -> float`，同时提供 C++ 与 `py_` 两个版本。
- `tunneling.profile_transmission_probability(profile, energy_kj_mol)` 返回上述 `P(E)`。
- `tunneling.profile_wkb_kappa(profile, T)` 签名不变。

**验收**
- 对称 Eckart 势垒（θ 有解析式）：与高精度独立数值积分的一致半经典 κ 比较，相对误差 ≤1e-6（可用加密网格）。
- 非对称（吸能）剖面：`E < E₀` 时 `P = 0`；把产物平台延长一倍后，κ 的相对变化 ≤1e-10。
- 极限行为：
  - 高温时 κ → 1；
  - 平坦剖面时 κ = 1；
  - 低温时 κ 单调增大。
- C++ 与 Python 一致：`P` ≤1e-12，`κ` ≤1e-10。
- 更新 `test_golden_wkb.py`，并报告新旧 κ 的对比。

## A2：CVT 极值精修（N3）

**负责的文件**
- `cpp/fit.cpp`
- `src/pyrate/native/fit.py`
- `src/pyrate/rates.py`
- `tests/test_rates.py`
- `tests/test_native.py`
- `tests/test_parity_fit.py`
- `tests/test_golden_cvt.py`
- `docs/architecture.md` §4

**接口**
- `refine_maximum(x, y) -> RefinedMaximum(coordinate, value, sample_index, fit_points)`，签名不变。可以增加带默认值的仅关键字参数。

**必须满足的保证**（设 `i` 为采样最大值的下标）
1. 最大值在端点或少于 3 个点时，返回采样点本身，且 `fit_points = 1`。
2. 精修后的坐标落在 `[x[i-1], x[i+1]]` 内。
3. `y[i] ≤ value ≤ y[i] + B`，其中上界 `B` 不依赖单位，必须在文档中写明。示例：`B = y[i] − min(y[i-1], y[i+1])`。
4. 对二次型数据精确（误差 ≤1e-12）。
5. C++ 与 Python 一致：坐标误差 ≤1e-9，函数值误差 ≤1e-12。

**验收**
- 随机属性测试 ≥1e4 例，包括非均匀网格、噪声和平台，断言保证 2 与 3。
- 审计中发现的超调反例（`docs/roadmap.md` N3）不再出现。
- 更新 `test_golden_cvt.py` 并报告 `s_CVT` 与 `k_CVT` 的变化。
- Polyrate 手册目前不可得，因此实现取保守方案，并在 §4 中注明“待与手册核对”。

## A3：同位素质量（N4，质量部分）

**负责的文件**
- `src/pyrate/elements.py`
- `src/pyrate/models.py`（仅限 `Atom` 和 `Species.from_mapping` 的质量相关逻辑）
- `src/pyrate/input.py`
- `docs/input-format.md`
- `tests/test_geometry.py`
- `tests/test_input.py`
- 新增 `tests/test_elements.py`

**接口**
- `MassModel = Literal["isotopic", "average"]`。
- `atomic_mass_amu(symbol, model: MassModel = "isotopic") -> float`。
  - `isotopic`：丰度最高同位素的精确质量（AME2020 / NIST）；
  - `average`：现有的平均原子量表。
- `Atom` 的默认质量改为 `isotopic`。显式给出的 `mass_amu` 永远优先。
- TOML 中 `[calculation] mass_model = "isotopic" | "average"`，默认 `isotopic`，并传递给所有物种的原子。`LoadedCalculation.summary()` 中要包含 `mass_model`。
- `D`、`T` 仍按现有方式识别。

**验收**
- 同位素表覆盖现有平均原子量表的全部元素，数值来源在文档中写明。
- 用独立数据源（例如临时安装、不加入依赖的 `periodictable` 包）交叉核对，误差 ≤1e-6 amu。
- H、C、N、O、Cl 的关键值有显式测试。
- 用 `mass_amu` 覆写原子数据的示例结果不变（金标准测试不受影响）。

## 暂缓：A4（参考数据）

A4 需要 Polyrate 2023 分发包和手册，本会话无法获取（网络受限）。获得之后再启动；见 roadmap §3 的第 5–7 项。
