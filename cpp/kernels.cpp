#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr double kBoltzmann = 1.380649e-23;
constexpr double kPlanck = 6.62607015e-34;
constexpr double kHbar = kPlanck / (2.0 * 3.141592653589793238462643383279502884);
constexpr double kLightSpeed = 299792458.0;

bool sequence_to_vector(PyObject* object, std::vector<double>& values, const char* name) {
    PyObject* sequence = PySequence_Fast(object, "expected a sequence of real numbers");
    if (sequence == nullptr) {
        return false;
    }
    const Py_ssize_t size = PySequence_Fast_GET_SIZE(sequence);
    values.resize(static_cast<std::size_t>(size));
    PyObject** items = PySequence_Fast_ITEMS(sequence);
    for (Py_ssize_t index = 0; index < size; ++index) {
        const double value = PyFloat_AsDouble(items[index]);
        if (PyErr_Occurred() != nullptr) {
            Py_DECREF(sequence);
            PyErr_Format(PyExc_TypeError, "%s must contain only real numbers", name);
            return false;
        }
        values[static_cast<std::size_t>(index)] = value;
    }
    Py_DECREF(sequence);
    return true;
}

bool strictly_increasing(const std::vector<double>& values) {
    for (std::size_t index = 1; index < values.size(); ++index) {
        if (!(values[index] > values[index - 1])) {
            return false;
        }
    }
    return true;
}

double harmonic_log_partition_impl(const std::vector<double>& frequencies_cm1,
                                   const double temperature) {
    double result = 0.0;
    for (const double frequency : frequencies_cm1) {
        if (frequency <= 0.0) {
            continue;
        }
        const double theta_over_t =
            kPlanck * kLightSpeed * 100.0 * frequency / (kBoltzmann * temperature);
        result -= std::log1p(-std::exp(-theta_over_t));
    }
    return result;
}

double wigner_impl(const double imaginary_frequency_cm1, const double temperature) {
    const double ratio = kPlanck * kLightSpeed * 100.0 * std::abs(imaginary_frequency_cm1) /
                         (kBoltzmann * temperature);
    return 1.0 + ratio * ratio / 24.0;
}

// Integral of sqrt(max(u(t), 0)) on a linearly interpolated segment t in [0,1].
double linear_sqrt_integral(const double u0, const double u1) {
    if (u0 <= 0.0 && u1 <= 0.0) {
        return 0.0;
    }
    const double delta = u1 - u0;
    const double scale = std::max({1.0, std::abs(u0), std::abs(u1)});
    if (std::abs(delta) <= std::numeric_limits<double>::epsilon() * scale) {
        return std::sqrt(std::max(0.0, u0));
    }
    if (u0 > 0.0 && u1 > 0.0) {
        return (2.0 / 3.0) * (std::pow(u1, 1.5) - std::pow(u0, 1.5)) / delta;
    }
    if (u0 > 0.0) {
        return (2.0 / 3.0) * std::pow(u0, 1.5) / (u0 - u1);
    }
    return (2.0 / 3.0) * std::pow(u1, 1.5) / (u1 - u0);
}

double wkb_exponent_impl(const std::vector<double>& coordinates_m,
                         const std::vector<double>& potentials_j,
                         const double energy_j,
                         const double reduced_mass_kg) {
    double integral = 0.0;
    for (std::size_t index = 1; index < coordinates_m.size(); ++index) {
        const double dx = coordinates_m[index] - coordinates_m[index - 1];
        const double u0 = potentials_j[index - 1] - energy_j;
        const double u1 = potentials_j[index] - energy_j;
        integral += dx * linear_sqrt_integral(u0, u1);
    }
    return (2.0 / kHbar) * std::sqrt(2.0 * reduced_mass_kg) * integral;
}

double safe_exp(const double exponent) {
    if (exponent < -745.0) {
        return 0.0;
    }
    if (exponent > 709.0) {
        return std::numeric_limits<double>::infinity();
    }
    return std::exp(exponent);
}

double thermal_wkb_impl(const std::vector<double>& coordinates_m,
                        std::vector<double> potentials_j,
                        const double temperature,
                        const double reduced_mass_kg,
                        int energy_grid_points,
                        const bool uniform) {
    const double reference = potentials_j.front();
    for (double& value : potentials_j) {
        value -= reference;
    }
    const double barrier = *std::max_element(potentials_j.begin(), potentials_j.end());
    if (barrier <= 0.0) {
        return 1.0;
    }

    if (energy_grid_points % 2 != 0) {
        ++energy_grid_points;
    }
    const double beta = 1.0 / (kBoltzmann * temperature);
    const double step = barrier / static_cast<double>(energy_grid_points);

    auto integrand = [&](const double energy) {
        const double exponent = wkb_exponent_impl(
            coordinates_m, potentials_j, energy, reduced_mass_kg);
        if (uniform) {
            // Kemble uniform semiclassical P(E) = 1 / (1 + exp(2 theta)) below the
            // barrier top and nonclassical reflection P(E) = 1 - P(2 V - E) above it.
            // Folding the above-barrier part onto E < V gives
            //   kappa = 1 + int_0^V beta P(E) 2 sinh(beta (V - E)) dE.
            const double a = beta * (barrier - energy);
            const double log_one_plus = exponent > 0.0
                ? exponent + std::log1p(std::exp(-exponent))
                : std::log1p(std::exp(exponent));
            return beta * (safe_exp(a - log_one_plus) - safe_exp(-a - log_one_plus));
        }
        // This is the integrand after multiplying the canonical transmission
        // integral by beta * exp(beta * V_barrier).  It avoids subtracting two
        // exponentially small numbers at low temperature.
        return beta * safe_exp(beta * (barrier - energy) - exponent);
    };

    double sum = integrand(0.0) + integrand(barrier);
    for (int index = 1; index < energy_grid_points; ++index) {
        const double energy = step * static_cast<double>(index);
        sum += (index % 2 == 0 ? 2.0 : 4.0) * integrand(energy);
    }
    const double below_barrier = step * sum / 3.0;
    return 1.0 + below_barrier;
}

bool solve_linear_system(std::vector<std::vector<double>>& matrix,
                         std::vector<double>& rhs,
                         std::vector<double>& solution) {
    const std::size_t n = matrix.size();
    for (std::size_t pivot = 0; pivot < n; ++pivot) {
        std::size_t best = pivot;
        for (std::size_t row = pivot + 1; row < n; ++row) {
            if (std::abs(matrix[row][pivot]) > std::abs(matrix[best][pivot])) {
                best = row;
            }
        }
        if (std::abs(matrix[best][pivot]) < 1.0e-14) {
            return false;
        }
        if (best != pivot) {
            std::swap(matrix[best], matrix[pivot]);
            std::swap(rhs[best], rhs[pivot]);
        }
        const double diagonal = matrix[pivot][pivot];
        for (std::size_t column = pivot; column < n; ++column) {
            matrix[pivot][column] /= diagonal;
        }
        rhs[pivot] /= diagonal;
        for (std::size_t row = 0; row < n; ++row) {
            if (row == pivot) {
                continue;
            }
            const double factor = matrix[row][pivot];
            for (std::size_t column = pivot; column < n; ++column) {
                matrix[row][column] -= factor * matrix[pivot][column];
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }
    solution = rhs;
    return true;
}

double polynomial_value(const std::vector<double>& coefficients, const double x) {
    double result = 0.0;
    for (auto iterator = coefficients.rbegin(); iterator != coefficients.rend(); ++iterator) {
        result = result * x + *iterator;
    }
    return result;
}

double polynomial_derivative(const std::vector<double>& coefficients, const double x) {
    if (coefficients.size() <= 1) {
        return 0.0;
    }
    double result = 0.0;
    for (std::size_t degree = coefficients.size() - 1; degree > 0; --degree) {
        result = result * x + static_cast<double>(degree) * coefficients[degree];
    }
    return result;
}

struct RefinedMaximum {
    double x;
    double y;
    std::size_t sample_index;
    std::size_t fit_points;
};

RefinedMaximum refine_maximum_impl(const std::vector<double>& x,
                                   const std::vector<double>& y) {
    const auto maximum_iterator = std::max_element(y.begin(), y.end());
    const std::size_t maximum_index =
        static_cast<std::size_t>(std::distance(y.begin(), maximum_iterator));
    if (x.size() < 3 || maximum_index == 0 || maximum_index + 1 == x.size()) {
        return {x[maximum_index], y[maximum_index], maximum_index, 1};
    }

    const std::size_t count = std::min<std::size_t>(5, x.size());
    std::size_t start = maximum_index > count / 2 ? maximum_index - count / 2 : 0;
    if (start + count > x.size()) {
        start = x.size() - count;
    }

    const double center = x[maximum_index];
    const double scale = std::max(
        std::abs(x[start] - center), std::abs(x[start + count - 1] - center));
    if (scale <= 0.0) {
        return {x[maximum_index], y[maximum_index], maximum_index, 1};
    }

    std::vector<std::vector<double>> matrix(count, std::vector<double>(count, 0.0));
    std::vector<double> rhs(count, 0.0);
    for (std::size_t row = 0; row < count; ++row) {
        const double t = (x[start + row] - center) / scale;
        double power = 1.0;
        for (std::size_t column = 0; column < count; ++column) {
            matrix[row][column] = power;
            power *= t;
        }
        rhs[row] = y[start + row];
    }
    std::vector<double> coefficients;
    if (!solve_linear_system(matrix, rhs, coefficients)) {
        return {x[maximum_index], y[maximum_index], maximum_index, 1};
    }

    const double lower = (x[start] - center) / scale;
    const double upper = (x[start + count - 1] - center) / scale;
    double best_t = (x[maximum_index] - center) / scale;
    double best_y = polynomial_value(coefficients, best_t);

    auto consider = [&](const double candidate) {
        if (candidate < lower || candidate > upper) {
            return;
        }
        const double value = polynomial_value(coefficients, candidate);
        if (std::isfinite(value) && value > best_y) {
            best_y = value;
            best_t = candidate;
        }
    };

    consider(lower);
    consider(upper);
    constexpr int intervals = 512;
    double previous_x = lower;
    double previous_value = polynomial_derivative(coefficients, previous_x);
    for (int index = 1; index <= intervals; ++index) {
        const double current_x = lower + (upper - lower) * static_cast<double>(index) /
                                             static_cast<double>(intervals);
        const double current_value = polynomial_derivative(coefficients, current_x);
        if (current_value == 0.0) {
            consider(current_x);
        } else if (previous_value == 0.0 || current_value * previous_value < 0.0) {
            double left = previous_x;
            double right = current_x;
            double left_value = previous_value;
            for (int iteration = 0; iteration < 80; ++iteration) {
                const double midpoint = 0.5 * (left + right);
                const double midpoint_value = polynomial_derivative(coefficients, midpoint);
                if (std::abs(midpoint_value) < 1.0e-14) {
                    left = right = midpoint;
                    break;
                }
                if (left_value * midpoint_value <= 0.0) {
                    right = midpoint;
                } else {
                    left = midpoint;
                    left_value = midpoint_value;
                }
            }
            consider(0.5 * (left + right));
        }
        previous_x = current_x;
        previous_value = current_value;
    }

    // Guard against an oscillatory polynomial overshooting by an unreasonable
    // amount.  Falling back to the sampled maximum is safer than reporting a
    // spurious variational bottleneck.
    const double local_span = *std::max_element(y.begin() + static_cast<std::ptrdiff_t>(start),
                                                y.begin() + static_cast<std::ptrdiff_t>(start + count)) -
                              *std::min_element(y.begin() + static_cast<std::ptrdiff_t>(start),
                                                y.begin() + static_cast<std::ptrdiff_t>(start + count));
    if (!std::isfinite(best_y) || best_y > y[maximum_index] + 2.0 * std::max(1.0, local_span)) {
        return {x[maximum_index], y[maximum_index], maximum_index, 1};
    }
    return {center + scale * best_t, best_y, maximum_index, count};
}

PyObject* py_harmonic_log_partition(PyObject*, PyObject* args) {
    PyObject* sequence = nullptr;
    double temperature = 0.0;
    if (!PyArg_ParseTuple(args, "Od:harmonic_log_partition", &sequence, &temperature)) {
        return nullptr;
    }
    if (temperature <= 0.0) {
        PyErr_SetString(PyExc_ValueError, "temperature must be positive");
        return nullptr;
    }
    std::vector<double> frequencies;
    if (!sequence_to_vector(sequence, frequencies, "frequencies_cm1")) {
        return nullptr;
    }
    return PyFloat_FromDouble(harmonic_log_partition_impl(frequencies, temperature));
}

PyObject* py_wigner_kappa(PyObject*, PyObject* args) {
    double frequency = 0.0;
    double temperature = 0.0;
    if (!PyArg_ParseTuple(args, "dd:wigner_kappa", &frequency, &temperature)) {
        return nullptr;
    }
    if (temperature <= 0.0) {
        PyErr_SetString(PyExc_ValueError, "temperature must be positive");
        return nullptr;
    }
    return PyFloat_FromDouble(wigner_impl(frequency, temperature));
}

PyObject* py_refine_maximum(PyObject*, PyObject* args) {
    PyObject* x_object = nullptr;
    PyObject* y_object = nullptr;
    if (!PyArg_ParseTuple(args, "OO:refine_maximum", &x_object, &y_object)) {
        return nullptr;
    }
    std::vector<double> x;
    std::vector<double> y;
    if (!sequence_to_vector(x_object, x, "x") || !sequence_to_vector(y_object, y, "y")) {
        return nullptr;
    }
    if (x.size() != y.size() || x.empty()) {
        PyErr_SetString(PyExc_ValueError, "x and y must have equal, non-zero length");
        return nullptr;
    }
    if (!strictly_increasing(x)) {
        PyErr_SetString(PyExc_ValueError, "x must be strictly increasing");
        return nullptr;
    }
    const RefinedMaximum result = refine_maximum_impl(x, y);
    return Py_BuildValue("ddnn", result.x, result.y,
                         static_cast<Py_ssize_t>(result.sample_index),
                         static_cast<Py_ssize_t>(result.fit_points));
}

PyObject* py_wkb_exponent(PyObject*, PyObject* args) {
    PyObject* coordinate_object = nullptr;
    PyObject* potential_object = nullptr;
    double energy = 0.0;
    double reduced_mass = 0.0;
    if (!PyArg_ParseTuple(args, "OOdd:wkb_exponent", &coordinate_object, &potential_object,
                          &energy, &reduced_mass)) {
        return nullptr;
    }
    std::vector<double> coordinates;
    std::vector<double> potentials;
    if (!sequence_to_vector(coordinate_object, coordinates, "coordinates_m") ||
        !sequence_to_vector(potential_object, potentials, "potentials_j")) {
        return nullptr;
    }
    if (coordinates.size() != potentials.size() || coordinates.size() < 2) {
        PyErr_SetString(PyExc_ValueError,
                        "coordinates and potentials require matching lengths of at least 2");
        return nullptr;
    }
    if (!strictly_increasing(coordinates)) {
        PyErr_SetString(PyExc_ValueError, "coordinates must be strictly increasing");
        return nullptr;
    }
    if (reduced_mass <= 0.0) {
        PyErr_SetString(PyExc_ValueError, "reduced mass must be positive");
        return nullptr;
    }
    double result = 0.0;
    Py_BEGIN_ALLOW_THREADS
    result = wkb_exponent_impl(coordinates, potentials, energy, reduced_mass);
    Py_END_ALLOW_THREADS
    return PyFloat_FromDouble(result);
}

PyObject* py_thermal_wkb_kappa(PyObject*, PyObject* args) {
    PyObject* coordinate_object = nullptr;
    PyObject* potential_object = nullptr;
    double temperature = 0.0;
    double reduced_mass = 0.0;
    int energy_grid_points = 400;
    int uniform = 1;
    if (!PyArg_ParseTuple(args, "OOdd|ip:thermal_wkb_kappa", &coordinate_object,
                          &potential_object, &temperature, &reduced_mass,
                          &energy_grid_points, &uniform)) {
        return nullptr;
    }
    std::vector<double> coordinates;
    std::vector<double> potentials;
    if (!sequence_to_vector(coordinate_object, coordinates, "coordinates_m") ||
        !sequence_to_vector(potential_object, potentials, "potentials_j")) {
        return nullptr;
    }
    if (coordinates.size() != potentials.size() || coordinates.size() < 3) {
        PyErr_SetString(PyExc_ValueError,
                        "coordinates and potentials require matching lengths of at least 3");
        return nullptr;
    }
    if (!strictly_increasing(coordinates)) {
        PyErr_SetString(PyExc_ValueError, "coordinates must be strictly increasing");
        return nullptr;
    }
    if (temperature <= 0.0 || reduced_mass <= 0.0) {
        PyErr_SetString(PyExc_ValueError, "temperature and reduced mass must be positive");
        return nullptr;
    }
    if (energy_grid_points < 20) {
        PyErr_SetString(PyExc_ValueError, "energy_grid_points must be at least 20");
        return nullptr;
    }
    double result = 0.0;
    Py_BEGIN_ALLOW_THREADS
    result = thermal_wkb_impl(
        coordinates, potentials, temperature, reduced_mass, energy_grid_points, uniform != 0);
    Py_END_ALLOW_THREADS
    return PyFloat_FromDouble(result);
}

PyObject* py_beyer_swinehart(PyObject*, PyObject* args) {
    PyObject* frequency_object = nullptr;
    double grain_cm1 = 0.0;
    Py_ssize_t grain_count = 0;
    if (!PyArg_ParseTuple(args, "Odn:beyer_swinehart", &frequency_object, &grain_cm1,
                          &grain_count)) {
        return nullptr;
    }
    std::vector<double> frequencies;
    if (!sequence_to_vector(frequency_object, frequencies, "frequencies_cm1")) {
        return nullptr;
    }
    if (grain_cm1 <= 0.0 || grain_count <= 0) {
        PyErr_SetString(PyExc_ValueError, "grain_cm1 and grain_count must be positive");
        return nullptr;
    }
    std::vector<double> density(static_cast<std::size_t>(grain_count), 0.0);
    density[0] = 1.0;
    Py_BEGIN_ALLOW_THREADS
    for (const double frequency : frequencies) {
        if (frequency <= 0.0) {
            continue;
        }
        const std::size_t step = static_cast<std::size_t>(
            std::max(1.0, std::round(frequency / grain_cm1)));
        for (std::size_t index = step; index < density.size(); ++index) {
            density[index] += density[index - step];
        }
    }
    Py_END_ALLOW_THREADS

    PyObject* result = PyList_New(grain_count);
    if (result == nullptr) {
        return nullptr;
    }
    for (Py_ssize_t index = 0; index < grain_count; ++index) {
        PyObject* value = PyFloat_FromDouble(density[static_cast<std::size_t>(index)]);
        if (value == nullptr) {
            Py_DECREF(result);
            return nullptr;
        }
        PyList_SET_ITEM(result, index, value);
    }
    return result;
}

PyObject* py_native_version(PyObject*, PyObject*) {
    return PyUnicode_FromString("pyrate-cpp-0.1");
}

PyMethodDef module_methods[] = {
    {"harmonic_log_partition", py_harmonic_log_partition, METH_VARARGS,
     "Return the harmonic-oscillator log partition function."},
    {"wigner_kappa", py_wigner_kappa, METH_VARARGS,
     "Return the Wigner tunneling correction."},
    {"refine_maximum", py_refine_maximum, METH_VARARGS,
     "Refine a sampled maximum using up to five local points."},
    {"wkb_exponent", py_wkb_exponent, METH_VARARGS,
     "Return the dimensionless WKB transmission exponent."},
    {"thermal_wkb_kappa", py_thermal_wkb_kappa, METH_VARARGS,
     "Return a canonical one-dimensional WKB transmission correction."},
    {"beyer_swinehart", py_beyer_swinehart, METH_VARARGS,
     "Return a vibrational density-of-states grain array."},
    {"native_version", py_native_version, METH_NOARGS,
     "Return the native-kernel ABI version."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module_definition = {
    PyModuleDef_HEAD_INIT,
    "_kernels",
    "Native numerical kernels for PyRate.",
    -1,
    module_methods,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

}  // namespace

PyMODINIT_FUNC PyInit__kernels() {
    return PyModule_Create(&module_definition);
}
