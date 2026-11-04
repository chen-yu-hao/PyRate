// One-dimensional semiclassical barrier-transmission kernels.
//
// wkb_exponent returns the barrier-penetration integral gamma(E) = 2 theta(E)
// of a piecewise-linear profile.  semiclassical_transmission and
// thermal_wkb_kappa use the uniform semiclassical probability of Polyrate's
// ZCT/SCT methods, including nonclassical reflection above the barrier:
//
//   V_AG = max(V),  E0 = max(V[0], V[-1])
//   P(E) = 0                          E < E0
//   P(E) = 1 / (1 + exp(gamma(E)))    E0 <= E <= V_AG
//   P(E) = 1 - P(2 V_AG - E)          V_AG < E < 2 V_AG - E0
//   P(E) = 1                          E >= 2 V_AG - E0
//
// kappa(T) = beta exp(beta V_AG) int_{E0}^{inf} P(E) exp(-beta E) dE.
// The pure-Python mirror is src/pyrate/native/wkb.py.
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

double semiclassical_transmission_impl(const std::vector<double>& coordinates_m,
                                       const std::vector<double>& potentials_j,
                                       const double energy_j,
                                       const double reduced_mass_kg) {
    const double barrier = *std::max_element(potentials_j.begin(), potentials_j.end());
    const double threshold = std::max(potentials_j.front(), potentials_j.back());
    if (energy_j < threshold) {
        return 0.0;
    }
    if (energy_j <= barrier) {
        // P = 1/(1+e^gamma) written as e^-gamma/(1+e^-gamma): gamma >= 0, so
        // nothing overflows.
        const double exponent =
            wkb_exponent_impl(coordinates_m, potentials_j, energy_j, reduced_mass_kg);
        const double damping = std::exp(-exponent);
        return damping / (1.0 + damping);
    }
    if (energy_j < 2.0 * barrier - threshold) {
        // Nonclassical reflection: P(E) = 1 - P(2 V_AG - E) = 1/(1+e^-gamma(2 V_AG - E)).
        const double exponent = wkb_exponent_impl(
            coordinates_m, potentials_j, 2.0 * barrier - energy_j, reduced_mass_kg);
        return 1.0 / (1.0 + std::exp(-exponent));
    }
    return 1.0;
}

double thermal_wkb_impl(const std::vector<double>& coordinates_m,
                        std::vector<double> potentials_j,
                        const double temperature,
                        const double reduced_mass_kg,
                        int energy_grid_points) {
    // Measure energies from the barrier top, so E = V_AG - delta becomes E = -delta.
    const double barrier = *std::max_element(potentials_j.begin(), potentials_j.end());
    for (double& value : potentials_j) {
        value -= barrier;
    }
    const double depth = -std::max(potentials_j.front(), potentials_j.back());  // V_AG - E0
    if (depth <= 0.0) {
        return 1.0;
    }

    if (energy_grid_points % 2 != 0) {
        ++energy_grid_points;
    }
    const double beta = 1.0 / (kBoltzmann * temperature);
    const double step = depth / static_cast<double>(energy_grid_points);

    // kappa = beta int_0^depth [P(V_AG - delta) e^{beta delta}
    //                           + (1 - P(V_AG - delta)) e^{-beta delta}] d delta
    //         + e^{-beta depth}.
    // The first term is evaluated as exp(beta delta - log(1 + e^gamma)) so that
    // deep tunneling at low temperature neither overflows nor underflows.
    auto integrand = [&](const int index) {
        const double delta =
            index == energy_grid_points ? depth : static_cast<double>(index) * step;
        const double exponent =
            wkb_exponent_impl(coordinates_m, potentials_j, -delta, reduced_mass_kg);
        const double damping = std::exp(-exponent);
        const double scaled = beta * delta;
        const double log_tunneling = scaled - (exponent + std::log1p(damping));
        const double reflected = std::exp(-scaled) / (1.0 + damping);
        return safe_exp(log_tunneling) + reflected;
    };

    double sum = integrand(0) + integrand(energy_grid_points);
    for (int index = 1; index < energy_grid_points; ++index) {
        sum += (index % 2 == 0 ? 2.0 : 4.0) * integrand(index);
    }
    return (beta * step) * sum / 3.0 + std::exp(-beta * depth);
}

// Parse (coordinates, potentials, energy, reduced_mass) for the per-energy
// kernels.  Returns false with a Python exception set on invalid input.
bool parse_profile_energy(PyObject* args, const char* format,
                          std::vector<double>& coordinates, std::vector<double>& potentials,
                          double& energy, double& reduced_mass) {
    PyObject* coordinate_object = nullptr;
    PyObject* potential_object = nullptr;
    if (!PyArg_ParseTuple(args, format, &coordinate_object, &potential_object, &energy,
                          &reduced_mass)) {
        return false;
    }
    if (!sequence_to_vector(coordinate_object, coordinates, "coordinates_m") ||
        !sequence_to_vector(potential_object, potentials, "potentials_j")) {
        return false;
    }
    if (coordinates.size() != potentials.size() || coordinates.size() < 2) {
        PyErr_SetString(PyExc_ValueError,
                        "coordinates and potentials require matching lengths of at least 2");
        return false;
    }
    if (!strictly_increasing(coordinates)) {
        PyErr_SetString(PyExc_ValueError, "coordinates must be strictly increasing");
        return false;
    }
    if (reduced_mass <= 0.0) {
        PyErr_SetString(PyExc_ValueError, "reduced mass must be positive");
        return false;
    }
    return true;
}

PyObject* py_wkb_exponent(PyObject*, PyObject* args) {
    std::vector<double> coordinates;
    std::vector<double> potentials;
    double energy = 0.0;
    double reduced_mass = 0.0;
    if (!parse_profile_energy(args, "OOdd:wkb_exponent", coordinates, potentials, energy,
                              reduced_mass)) {
        return nullptr;
    }
    double result = 0.0;
    Py_BEGIN_ALLOW_THREADS
    result = wkb_exponent_impl(coordinates, potentials, energy, reduced_mass);
    Py_END_ALLOW_THREADS
    return PyFloat_FromDouble(result);
}

PyObject* py_semiclassical_transmission(PyObject*, PyObject* args) {
    std::vector<double> coordinates;
    std::vector<double> potentials;
    double energy = 0.0;
    double reduced_mass = 0.0;
    if (!parse_profile_energy(args, "OOdd:semiclassical_transmission", coordinates,
                              potentials, energy, reduced_mass)) {
        return nullptr;
    }
    double result = 0.0;
    Py_BEGIN_ALLOW_THREADS
    result = semiclassical_transmission_impl(coordinates, potentials, energy, reduced_mass);
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
     "Return the dimensionless WKB barrier-penetration exponent 2*theta(E)."},
    {"semiclassical_transmission", py_semiclassical_transmission, METH_VARARGS,
     "Return the uniform semiclassical transmission probability P(E)."},
    {"thermal_wkb_kappa", py_thermal_wkb_kappa, METH_VARARGS,
     "Return the canonical one-dimensional uniform semiclassical transmission coefficient."},
    {nullptr, nullptr, 0, nullptr},
};
