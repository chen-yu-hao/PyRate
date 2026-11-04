// One-dimensional WKB barrier-transmission kernels.
#include "common.hpp"

#include <algorithm>

namespace {

using namespace pyrate;

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
    const double beta = 1.0 / (kBoltzmann * temperature);
    const double step = barrier / static_cast<double>(energy_grid_points);

    auto integrand = [&](const double energy) {
        const double exponent = wkb_exponent_impl(
            coordinates_m, potentials_j, energy, reduced_mass_kg);
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
        coordinates, potentials, temperature, reduced_mass, energy_grid_points);
    Py_END_ALLOW_THREADS
    return PyFloat_FromDouble(result);
}

}  // namespace

PyMethodDef pyrate_wkb_methods[] = {
    {"wkb_exponent", py_wkb_exponent, METH_VARARGS,
     "Return the dimensionless WKB transmission exponent."},
    {"thermal_wkb_kappa", py_thermal_wkb_kappa, METH_VARARGS,
     "Return a canonical one-dimensional WKB transmission correction."},
    {nullptr, nullptr, 0, nullptr},
};
