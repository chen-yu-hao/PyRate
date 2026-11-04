#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace {

constexpr double kBoltzmann = 1.380649e-23;
constexpr double kPlanck = 6.62607015e-34;
constexpr double kHbar = kPlanck / (2.0 * 3.141592653589793238462643383279502884);

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

bool all_finite(const std::vector<double>& values) {
    return std::all_of(values.begin(), values.end(), [](const double value) {
        return std::isfinite(value);
    });
}

bool strictly_increasing(const std::vector<double>& values) {
    for (std::size_t index = 1; index < values.size(); ++index) {
        if (!(values[index] > values[index - 1])) {
            return false;
        }
    }
    return true;
}

// Integral of sqrt(max(u(t), 0)) on a linearly interpolated segment t in [0,1].
// Every branch is scaled by the local energy magnitude. This avoids both the
// cancellation in (u1^1.5-u0^1.5)/(u1-u0) and the old absolute-1-joule test,
// which is invalid for molecular barriers of order 1e-20 joule.
double linear_sqrt_integral(const double u0, const double u1) {
    if (u0 <= 0.0 && u1 <= 0.0) {
        return 0.0;
    }
    if (u0 > 0.0 && u1 > 0.0) {
        const double scale = std::max(u0, u1);
        const double root0 = std::sqrt(u0 / scale);
        const double root1 = std::sqrt(u1 / scale);
        return (2.0 / 3.0) * std::sqrt(scale) *
               (root0 * root0 + root0 * root1 + root1 * root1) /
               (root0 + root1);
    }
    if (u0 > 0.0) {
        const double scale = std::max(u0, -u1);
        const double positive = u0 / scale;
        const double negative = -u1 / scale;
        return (2.0 / 3.0) * std::sqrt(scale) * std::pow(positive, 1.5) /
               (positive + negative);
    }
    const double scale = std::max(-u0, u1);
    const double negative = -u0 / scale;
    const double positive = u1 / scale;
    return (2.0 / 3.0) * std::sqrt(scale) * std::pow(positive, 1.5) /
           (positive + negative);
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

double thermal_wkb_impl(const std::vector<double>& coordinates_m,
                        std::vector<double> potentials_j,
                        const double temperature,
                        const double reduced_mass_kg,
                        int energy_grid_points) {
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

    const double beta_barrier = barrier / (kBoltzmann * temperature);
    if (!std::isfinite(beta_barrier)) {
        return std::numeric_limits<double>::infinity();
    }

    std::vector<double> log_values(
        static_cast<std::size_t>(energy_grid_points + 1), 0.0);
    double maximum_log = -std::numeric_limits<double>::infinity();
    for (int index = 0; index <= energy_grid_points; ++index) {
        const double fraction = static_cast<double>(index) /
                                static_cast<double>(energy_grid_points);
        const double energy = barrier * fraction;
        const double exponent = wkb_exponent_impl(
            coordinates_m, potentials_j, energy, reduced_mass_kg);
        const double log_value = beta_barrier * (1.0 - fraction) - exponent;
        log_values[static_cast<std::size_t>(index)] = log_value;
        maximum_log = std::max(maximum_log, log_value);
    }

    double scaled_sum = 0.0;
    for (int index = 0; index <= energy_grid_points; ++index) {
        const double weight =
            (index == 0 || index == energy_grid_points)
                ? 1.0
                : (index % 2 == 0 ? 2.0 : 4.0);
        scaled_sum += weight * std::exp(
            log_values[static_cast<std::size_t>(index)] - maximum_log);
    }

    const double log_below = std::log(beta_barrier) -
                             std::log(3.0 * static_cast<double>(energy_grid_points)) +
                             maximum_log + std::log(scaled_sum);
    const double log_max = std::log(std::numeric_limits<double>::max());
    if (log_below > log_max) {
        return std::numeric_limits<double>::infinity();
    }
    if (log_below < -745.0) {
        return 1.0;
    }
    return 1.0 + std::exp(log_below);
}

bool validate_wkb_input(const std::vector<double>& coordinates,
                        const std::vector<double>& potentials,
                        const std::size_t minimum_size,
                        const double reduced_mass) {
    if (coordinates.size() != potentials.size() || coordinates.size() < minimum_size) {
        PyErr_Format(
            PyExc_ValueError,
            "coordinates and potentials require matching lengths of at least %zu",
            minimum_size);
        return false;
    }
    if (!all_finite(coordinates) || !all_finite(potentials)) {
        PyErr_SetString(PyExc_ValueError,
                        "coordinates and potentials must contain only finite values");
        return false;
    }
    if (!strictly_increasing(coordinates)) {
        PyErr_SetString(PyExc_ValueError, "coordinates must be strictly increasing");
        return false;
    }
    if (!std::isfinite(reduced_mass) || reduced_mass <= 0.0) {
        PyErr_SetString(PyExc_ValueError, "reduced mass must be positive and finite");
        return false;
    }
    return true;
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
    if (!std::isfinite(energy)) {
        PyErr_SetString(PyExc_ValueError, "energy must be finite");
        return nullptr;
    }
    if (!validate_wkb_input(coordinates, potentials, 2, reduced_mass)) {
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
    if (!PyArg_ParseTuple(args, "OOdd|i:thermal_wkb_kappa", &coordinate_object,
                          &potential_object, &temperature, &reduced_mass,
                          &energy_grid_points)) {
        return nullptr;
    }
    std::vector<double> coordinates;
    std::vector<double> potentials;
    if (!sequence_to_vector(coordinate_object, coordinates, "coordinates_m") ||
        !sequence_to_vector(potential_object, potentials, "potentials_j")) {
        return nullptr;
    }
    if (!validate_wkb_input(coordinates, potentials, 3, reduced_mass)) {
        return nullptr;
    }
    if (!std::isfinite(temperature) || temperature <= 0.0) {
        PyErr_SetString(PyExc_ValueError, "temperature must be positive and finite");
        return nullptr;
    }
    if (energy_grid_points < 20) {
        PyErr_SetString(PyExc_ValueError, "energy_grid_points must be at least 20");
        return nullptr;
    }
    double result = 0.0;
    Py_BEGIN_ALLOW_THREADS
    result = thermal_wkb_impl(
        coordinates, potentials, temperature, reduced_mass, energy_grid_points);
    Py_END_ALLOW_THREADS
    return PyFloat_FromDouble(result);
}

PyObject* py_native_version(PyObject*, PyObject*) {
    return PyUnicode_FromString("pyrate-wkb-cpp-0.2");
}

PyMethodDef module_methods[] = {
    {"wkb_exponent", py_wkb_exponent, METH_VARARGS,
     "Return the dimensionless WKB transmission exponent."},
    {"thermal_wkb_kappa", py_thermal_wkb_kappa, METH_VARARGS,
     "Return a log-stable canonical one-dimensional WKB correction."},
    {"native_version", py_native_version, METH_NOARGS,
     "Return the native WKB-kernel ABI version."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module_definition = {
    PyModuleDef_HEAD_INIT,
    "_wkb_kernels",
    "Numerically stable native WKB kernels for PyRate.",
    -1,
    module_methods,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

}  // namespace

PyMODINIT_FUNC PyInit__wkb_kernels() {
    return PyModule_Create(&module_definition);
}
