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
constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kHbar = kPlanck / (2.0 * kPi);
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

double wkb_exponent_profile_impl(const std::vector<double>& coordinates_m,
                                 const std::vector<double>& potentials_j,
                                 const double energy_j,
                                 const std::vector<double>& effective_masses_kg) {
    double integral = 0.0;
    for (std::size_t index = 1; index < coordinates_m.size(); ++index) {
        const double dx = coordinates_m[index] - coordinates_m[index - 1];
        const double u0 = 2.0 * effective_masses_kg[index - 1] * (potentials_j[index - 1] - energy_j);
        const double u1 = 2.0 * effective_masses_kg[index] * (potentials_j[index] - energy_j);
        integral += dx * linear_sqrt_integral(u0, u1);
    }
    return (2.0 / kHbar) * integral;
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

// Polyrate-consistent SAG/WKB Boltzmann average with Kemble uniform P(E) = 1 / (1 + exp(2*theta))
// and parabolic nonclassical reflection above the barrier (see polyhl.F:kapva and polyag.F:boltz).
template <typename ExponentFn>
double thermal_sag_integral_impl(const std::vector<double>& potentials_j,
                                 const double temperature,
                                 int energy_grid_points,
                                 ExponentFn&& exponent_fn) {
    const double threshold = std::max(potentials_j.front(), potentials_j.back());
    const double vmax = *std::max_element(potentials_j.begin(), potentials_j.end());
    const double vm = vmax - threshold;
    if (vm <= 0.0) {
        return 1.0;
    }

    if (energy_grid_points % 2 != 0) {
        ++energy_grid_points;
    }
    const double beta = 1.0 / (kBoltzmann * temperature);
    const double step = vm / static_cast<double>(energy_grid_points);

    auto integrand = [&](const double energy) {
        const double gamma = exponent_fn(energy);
        const double delta = std::max(0.0, beta * (vmax - energy));
        if (delta == 0.0) {
            return 0.0;
        }
        const double log_num = delta - gamma;
        const double exp_num = safe_exp(log_num);
        if (exp_num == 0.0) {
            return 0.0;
        }
        const double reflection_factor = -std::expm1(-2.0 * delta);
        const double kemble_denom = 1.0 + safe_exp(-gamma);
        return beta * exp_num * (reflection_factor / kemble_denom);
    };

    double sum = integrand(threshold) + integrand(vmax);
    for (int index = 1; index < energy_grid_points; ++index) {
        const double energy = threshold + step * static_cast<double>(index);
        sum += (index % 2 == 0 ? 2.0 : 4.0) * integrand(energy);
    }
    return 1.0 + step * sum / 3.0;
}

double thermal_wkb_impl(const std::vector<double>& coordinates_m,
                        const std::vector<double>& potentials_j,
                        const double temperature,
                        const double reduced_mass_kg,
                        int energy_grid_points) {
    return thermal_sag_integral_impl(
        potentials_j, temperature, energy_grid_points, [&](const double energy) {
            return wkb_exponent_impl(coordinates_m, potentials_j, energy, reduced_mass_kg);
        });
}

double thermal_wkb_profile_impl(const std::vector<double>& coordinates_m,
                                const std::vector<double>& potentials_j,
                                const double temperature,
                                const std::vector<double>& effective_masses_kg,
                                int energy_grid_points) {
    return thermal_sag_integral_impl(
        potentials_j, temperature, energy_grid_points, [&](const double energy) {
            return wkb_exponent_profile_impl(
                coordinates_m, potentials_j, energy, effective_masses_kg);
        });
}

double eckart_transmission_impl(const double energy_j,
                                const double barrier_forward_j,
                                const double barrier_reverse_j,
                                const double imaginary_frequency_cm1) {
    if (barrier_forward_j <= 0.0 || barrier_reverse_j <= 0.0) {
        return energy_j >= std::max(0.0, barrier_forward_j - barrier_reverse_j) ? 1.0 : 0.0;
    }
    const double threshold = std::max(0.0, barrier_forward_j - barrier_reverse_j);
    if (energy_j <= threshold) {
        return 0.0;
    }
    const double u_star = kPlanck * kLightSpeed * 100.0 * std::abs(imaginary_frequency_cm1);
    if (u_star <= 0.0) {
        return energy_j >= barrier_forward_j ? 1.0 : 0.0;
    }
    const double inv_sqrt_sum =
        (1.0 / std::sqrt(barrier_forward_j)) + (1.0 / std::sqrt(barrier_reverse_j));
    // Johnston & Heicklen (1962): 2*pi*a = 2*sqrt(alpha_1*xi) / (alpha_1^-1/2 + alpha_2^-1/2)
    // with alpha_i = 2*pi*V_i / (h*nu*), i.e. a prefactor of 4*pi / (h*nu*).
    const double two_pi_alpha = (4.0 * kPi / u_star) * std::sqrt(energy_j) / inv_sqrt_sum;
    const double two_pi_beta =
        (4.0 * kPi / u_star) *
        std::sqrt(std::max(0.0, energy_j - barrier_forward_j + barrier_reverse_j)) / inv_sqrt_sum;
    const double a_plus = two_pi_alpha + two_pi_beta;
    const double a_minus = std::abs(two_pi_alpha - two_pi_beta);
    const double d_arg = 16.0 * barrier_forward_j * barrier_reverse_j / (u_star * u_star) - 1.0;

    if (d_arg >= 0.0) {
        const double two_pi_delta = kPi * std::sqrt(d_arg);
        const double scale = std::max(a_plus, two_pi_delta);
        const double num =
            (safe_exp(a_plus - scale) + safe_exp(-a_plus - scale)) -
            (safe_exp(a_minus - scale) + safe_exp(-a_minus - scale));
        const double den =
            (safe_exp(a_plus - scale) + safe_exp(-a_plus - scale)) +
            (safe_exp(two_pi_delta - scale) + safe_exp(-two_pi_delta - scale));
        return std::clamp(num / den, 0.0, 1.0);
    }
    const double cos_term = 2.0 * std::cos(kPi * std::sqrt(-d_arg));
    const double scale = a_plus;
    const double num =
        (1.0 + safe_exp(-2.0 * a_plus)) -
        (safe_exp(a_minus - scale) + safe_exp(-a_minus - scale));
    const double den = (1.0 + safe_exp(-2.0 * a_plus)) + cos_term * safe_exp(-scale);
    return std::clamp(num / den, 0.0, 1.0);
}

double thermal_eckart_impl(const double barrier_forward_j,
                           const double barrier_reverse_j,
                           const double imaginary_frequency_cm1,
                           const double temperature,
                           int energy_grid_points) {
    const double threshold = std::max(0.0, barrier_forward_j - barrier_reverse_j);
    const double vm = barrier_forward_j - threshold;
    if (vm <= 0.0 || std::abs(imaginary_frequency_cm1) <= 0.0) {
        return 1.0;
    }
    if (energy_grid_points % 2 != 0) {
        ++energy_grid_points;
    }
    const double beta = 1.0 / (kBoltzmann * temperature);
    // Integrate from threshold to barrier_forward + 12 * k_B * T (covering both tunneling
    // and above-barrier Eckart reflection).
    const double e_upper = barrier_forward_j + 16.0 * kBoltzmann * temperature;
    const double step = (e_upper - threshold) / static_cast<double>(energy_grid_points);

    auto integrand = [&](const double energy) {
        const double prob = eckart_transmission_impl(
            energy, barrier_forward_j, barrier_reverse_j, imaginary_frequency_cm1);
        if (prob <= 0.0) {
            return 0.0;
        }
        return beta * prob * safe_exp(beta * (barrier_forward_j - energy));
    };

    double sum = integrand(threshold) + integrand(e_upper);
    for (int index = 1; index < energy_grid_points; ++index) {
        const double energy = threshold + step * static_cast<double>(index);
        sum += (index % 2 == 0 ? 2.0 : 4.0) * integrand(energy);
    }
    return step * sum / 3.0 + safe_exp(beta * (barrier_forward_j - e_upper));
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

struct RefinedExtremum {
    double x;
    double y;
    std::size_t sample_index;
    std::size_t fit_points;
};

RefinedExtremum refine_maximum_impl(const std::vector<double>& x,
                                    const std::vector<double>& y) {
    const auto maximum_iterator = std::max_element(y.begin(), y.end());
    const std::size_t maximum_index =
        static_cast<std::size_t>(std::distance(y.begin(), maximum_iterator));
    if (x.size() < 3 || maximum_index == 0 || maximum_index + 1 == x.size()) {
        return {x[maximum_index], y[maximum_index], maximum_index, 1};
    }

    // Match Polyrate FITMAX (polyag.F:1864): use 5-point quartic fit when at least two
    // points exist on both sides of the maximum, otherwise use 3-point parabolic fit.
    const bool can_use_five = (x.size() >= 5 && maximum_index > 1 && maximum_index + 2 < x.size());
    const std::size_t count = can_use_five ? 5 : 3;
    const std::size_t start = maximum_index - count / 2;

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
    const RefinedExtremum result = refine_maximum_impl(x, y);
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

PyObject* py_wkb_exponent_profile(PyObject*, PyObject* args) {
    PyObject* coordinate_object = nullptr;
    PyObject* potential_object = nullptr;
    double energy = 0.0;
    PyObject* mass_object = nullptr;
    if (!PyArg_ParseTuple(args, "OOdO:wkb_exponent_profile", &coordinate_object,
                          &potential_object, &energy, &mass_object)) {
        return nullptr;
    }
    std::vector<double> coordinates;
    std::vector<double> potentials;
    std::vector<double> effective_masses;
    if (!sequence_to_vector(coordinate_object, coordinates, "coordinates_m") ||
        !sequence_to_vector(potential_object, potentials, "potentials_j") ||
        !sequence_to_vector(mass_object, effective_masses, "effective_masses_kg")) {
        return nullptr;
    }
    if (coordinates.size() != potentials.size() ||
        coordinates.size() != effective_masses.size() || coordinates.size() < 2) {
        PyErr_SetString(
            PyExc_ValueError,
            "coordinates, potentials, and effective_masses require matching lengths >= 2");
        return nullptr;
    }
    if (!strictly_increasing(coordinates)) {
        PyErr_SetString(PyExc_ValueError, "coordinates must be strictly increasing");
        return nullptr;
    }
    for (const double m : effective_masses) {
        if (m <= 0.0) {
            PyErr_SetString(PyExc_ValueError, "effective masses must be positive");
            return nullptr;
        }
    }
    double result = 0.0;
    Py_BEGIN_ALLOW_THREADS
    result = wkb_exponent_profile_impl(coordinates, potentials, energy, effective_masses);
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

PyObject* py_thermal_wkb_profile_kappa(PyObject*, PyObject* args) {
    PyObject* coordinate_object = nullptr;
    PyObject* potential_object = nullptr;
    double temperature = 0.0;
    PyObject* mass_object = nullptr;
    int energy_grid_points = 400;
    if (!PyArg_ParseTuple(args, "OOdO|i:thermal_wkb_profile_kappa", &coordinate_object,
                          &potential_object, &temperature, &mass_object,
                          &energy_grid_points)) {
        return nullptr;
    }
    std::vector<double> coordinates;
    std::vector<double> potentials;
    std::vector<double> effective_masses;
    if (!sequence_to_vector(coordinate_object, coordinates, "coordinates_m") ||
        !sequence_to_vector(potential_object, potentials, "potentials_j") ||
        !sequence_to_vector(mass_object, effective_masses, "effective_masses_kg")) {
        return nullptr;
    }
    if (coordinates.size() != potentials.size() ||
        coordinates.size() != effective_masses.size() || coordinates.size() < 3) {
        PyErr_SetString(
            PyExc_ValueError,
            "coordinates, potentials, and effective_masses require matching lengths >= 3");
        return nullptr;
    }
    if (!strictly_increasing(coordinates)) {
        PyErr_SetString(PyExc_ValueError, "coordinates must be strictly increasing");
        return nullptr;
    }
    if (temperature <= 0.0) {
        PyErr_SetString(PyExc_ValueError, "temperature must be positive");
        return nullptr;
    }
    for (const double m : effective_masses) {
        if (m <= 0.0) {
            PyErr_SetString(PyExc_ValueError, "effective masses must be positive");
            return nullptr;
        }
    }
    if (energy_grid_points < 20) {
        PyErr_SetString(PyExc_ValueError, "energy_grid_points must be at least 20");
        return nullptr;
    }
    double result = 0.0;
    Py_BEGIN_ALLOW_THREADS
    result = thermal_wkb_profile_impl(
        coordinates, potentials, temperature, effective_masses, energy_grid_points);
    Py_END_ALLOW_THREADS
    return PyFloat_FromDouble(result);
}

PyObject* py_eckart_transmission(PyObject*, PyObject* args) {
    double energy = 0.0;
    double barrier_forward = 0.0;
    double barrier_reverse = 0.0;
    double imag_freq = 0.0;
    if (!PyArg_ParseTuple(args, "dddd:eckart_transmission", &energy, &barrier_forward,
                          &barrier_reverse, &imag_freq)) {
        return nullptr;
    }
    return PyFloat_FromDouble(
        eckart_transmission_impl(energy, barrier_forward, barrier_reverse, imag_freq));
}

PyObject* py_thermal_eckart_kappa(PyObject*, PyObject* args) {
    double barrier_forward = 0.0;
    double barrier_reverse = 0.0;
    double imag_freq = 0.0;
    double temperature = 0.0;
    int energy_grid_points = 400;
    if (!PyArg_ParseTuple(args, "dddd|i:thermal_eckart_kappa", &barrier_forward,
                          &barrier_reverse, &imag_freq, &temperature, &energy_grid_points)) {
        return nullptr;
    }
    if (temperature <= 0.0) {
        PyErr_SetString(PyExc_ValueError, "temperature must be positive");
        return nullptr;
    }
    if (energy_grid_points < 20) {
        PyErr_SetString(PyExc_ValueError, "energy_grid_points must be at least 20");
        return nullptr;
    }
    double result = 0.0;
    Py_BEGIN_ALLOW_THREADS
    result = thermal_eckart_impl(
        barrier_forward, barrier_reverse, imag_freq, temperature, energy_grid_points);
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

std::vector<double> project_symmetric_matrix_impl(const std::vector<double>& matrix,
                                                  const std::vector<double>& basis,
                                                  const std::size_t dim,
                                                  const std::size_t n_basis) {
    std::vector<double> out(dim * dim, 0.0);
    for (std::size_t row = 0; row < dim; ++row) {
        out[row * dim + row] = matrix[row * dim + row];
        for (std::size_t col = row + 1; col < dim; ++col) {
            const double avg = 0.5 * (matrix[row * dim + col] + matrix[col * dim + row]);
            out[row * dim + col] = avg;
            out[col * dim + row] = avg;
        }
    }
    if (n_basis == 0) {
        return out;
    }

    // Compute T = M * B (dim x n_basis)
    std::vector<double> t_mat(dim * n_basis, 0.0);
    for (std::size_t row = 0; row < dim; ++row) {
        for (std::size_t col = 0; col < dim; ++col) {
            const double m_rc = out[row * dim + col];
            if (m_rc == 0.0) {
                continue;
            }
            for (std::size_t k = 0; k < n_basis; ++k) {
                t_mat[row * n_basis + k] += m_rc * basis[col * n_basis + k];
            }
        }
    }

    // Compute C = B^T * T (n_basis x n_basis)
    std::vector<double> c_mat(n_basis * n_basis, 0.0);
    for (std::size_t row = 0; row < dim; ++row) {
        for (std::size_t k1 = 0; k1 < n_basis; ++k1) {
            const double b_rk1 = basis[row * n_basis + k1];
            for (std::size_t k2 = 0; k2 < n_basis; ++k2) {
                c_mat[k1 * n_basis + k2] += b_rk1 * t_mat[row * n_basis + k2];
            }
        }
    }

    // Compute W = T - 0.5 * B * C (dim x n_basis)
    std::vector<double> w_mat = t_mat;
    for (std::size_t row = 0; row < dim; ++row) {
        for (std::size_t k1 = 0; k1 < n_basis; ++k1) {
            double bc = 0.0;
            for (std::size_t k2 = 0; k2 < n_basis; ++k2) {
                bc += basis[row * n_basis + k2] * c_mat[k2 * n_basis + k1];
            }
            w_mat[row * n_basis + k1] -= 0.5 * bc;
        }
    }

    // Out = Out - W * B^T - B * W^T, guaranteed symmetric
    for (std::size_t row = 0; row < dim; ++row) {
        for (std::size_t col = row; col < dim; ++col) {
            double correction = 0.0;
            for (std::size_t k = 0; k < n_basis; ++k) {
                correction += w_mat[row * n_basis + k] * basis[col * n_basis + k] +
                              basis[row * n_basis + k] * w_mat[col * n_basis + k];
            }
            const double val = out[row * dim + col] - correction;
            out[row * dim + col] = val;
            out[col * dim + row] = val;
        }
    }
    return out;
}

struct LqaStepResult {
    std::vector<double> delta_x;
    std::vector<double> end_tangent;
    double u_star;
};

LqaStepResult page_mciver_lqa_step_impl(const std::vector<double>& gradient,
                                        const std::vector<double>& eigenvalues,
                                        const std::vector<double>& eigenvectors,
                                        const std::size_t dim,
                                        const double step_size) {
    // Expand gradient in Hessian eigenbasis: g_i = v_i^T g
    std::vector<double> g_mode(dim, 0.0);
    double g_norm2 = 0.0;
    for (std::size_t mode = 0; mode < dim; ++mode) {
        double dot = 0.0;
        for (std::size_t row = 0; row < dim; ++row) {
            dot += eigenvectors[row * dim + mode] * gradient[row];
        }
        g_mode[mode] = dot;
        g_norm2 += dot * dot;
    }

    if (g_norm2 <= 0.0) {
        return {std::vector<double>(dim, 0.0), std::vector<double>(dim, 0.0), 0.0};
    }
    const double g_norm = std::sqrt(g_norm2);

    // 10-point Gauss-Legendre quadrature on [0, 1]
    constexpr double gl_nodes[10] = {
        0.013046735741414128, 0.06746831665550773, 0.16029521585048778, 0.2833023029353764,
        0.4255628305091844,  0.5744371694908156,  0.7166976970646236,  0.8397047841495122,
        0.9325316833444923,  0.9869532642585859,
    };
    constexpr double gl_weights[10] = {
        0.03333567215434407, 0.0747256745752902,  0.109543181257991,   0.13463335965499826,
        0.1477621123573764,  0.1477621123573764,  0.13463335965499826, 0.109543181257991,
        0.0747256745752902,  0.03333567215434407,
    };

    auto speed = [&](const double tau) -> double {
        double sum_sq = 0.0;
        for (std::size_t mode = 0; mode < dim; ++mode) {
            const double gm = g_mode[mode];
            if (gm == 0.0) {
                continue;
            }
            const double arg = std::clamp(-eigenvalues[mode] * tau, -300.0, 50.0);
            const double scaled = gm * std::exp(arg);
            sum_sq += scaled * scaled;
        }
        return std::sqrt(sum_sq);
    };

    auto arc_length = [&](const double u) -> double {
        if (u <= 0.0) {
            return 0.0;
        }
        double integral = 0.0;
        for (int q = 0; q < 10; ++q) {
            integral += gl_weights[q] * speed(u * gl_nodes[q]);
        }
        return u * integral;
    };

    double u_lo = 0.0;
    double u_hi = step_size / g_norm;
    double phi_hi = arc_length(u_hi);
    bool finite_limit = false;
    for (int expand = 0; expand < 60 && phi_hi < step_size; ++expand) {
        const double sp_hi = speed(u_hi);
        if (sp_hi <= 1.0e-12 * g_norm) {
            finite_limit = true;
            break;
        }
        u_lo = u_hi;
        u_hi *= 2.0;
        phi_hi = arc_length(u_hi);
        if (u_hi > 1.0e6 * (step_size / g_norm)) {
            finite_limit = true;
            break;
        }
    }

    double u_star = u_hi;
    if (!finite_limit) {
        double u = 0.5 * (u_lo + u_hi);
        const double tol = 1.0e-13 * std::max(1.0, step_size);
        for (int iter = 0; iter < 50; ++iter) {
            const double phi = arc_length(u);
            const double diff = phi - step_size;
            if (std::abs(diff) <= tol || (u_hi - u_lo) <= 1.0e-14 * std::max(1.0, u)) {
                u_star = u;
                break;
            }
            if (diff < 0.0) {
                u_lo = u;
            } else {
                u_hi = u;
            }
            const double deriv = speed(u);
            double candidate = 0.5 * (u_lo + u_hi);
            if (deriv > 1.0e-15 * g_norm) {
                const double newton = u - diff / deriv;
                if (newton > u_lo && newton < u_hi) {
                    candidate = newton;
                }
            }
            u = candidate;
            u_star = u;
        }
    }

    std::vector<double> coeffs(dim, 0.0);
    std::vector<double> d_end_mode(dim, 0.0);
    for (std::size_t mode = 0; mode < dim; ++mode) {
        const double gm = g_mode[mode];
        if (gm == 0.0) {
            continue;
        }
        const double lam = eigenvalues[mode];
        const double x = lam * u_star;
        if (std::abs(x) <= 1.0e-5) {
            const double series =
                1.0 - 0.5 * x + (x * x) / 6.0 - (x * x * x) / 24.0 + (x * x * x * x) / 120.0;
            coeffs[mode] = -gm * u_star * series;
        } else {
            const double arg = std::clamp(-x, -300.0, 50.0);
            coeffs[mode] = gm * std::expm1(arg) / lam;
        }
        const double end_arg = std::clamp(-x, -300.0, 50.0);
        d_end_mode[mode] = -gm * std::exp(end_arg);
    }

    std::vector<double> delta_x(dim, 0.0);
    std::vector<double> end_tangent(dim, 0.0);
    for (std::size_t row = 0; row < dim; ++row) {
        double dx_val = 0.0;
        double t_val = 0.0;
        for (std::size_t mode = 0; mode < dim; ++mode) {
            const double v_rm = eigenvectors[row * dim + mode];
            dx_val += v_rm * coeffs[mode];
            t_val += v_rm * d_end_mode[mode];
        }
        delta_x[row] = dx_val;
        end_tangent[row] = t_val;
    }

    double t_norm2 = 0.0;
    for (const double v : end_tangent) {
        t_norm2 += v * v;
    }
    if (t_norm2 > 0.0) {
        const double inv_t = 1.0 / std::sqrt(t_norm2);
        for (double& v : end_tangent) {
            v *= inv_t;
        }
    }
    return {std::move(delta_x), std::move(end_tangent), u_star};
}

std::pair<double, std::vector<double>> leps_energy_and_gradient_impl(
    const std::vector<double>& coords_angstrom,
    const std::vector<double>& de_hartree,
    const std::vector<double>& re_bohr,
    const std::vector<double>& beta_bohr_inv,
    const std::vector<double>& sato,
    const double bohr_angstrom) {
    const double inv_bohr = 1.0 / bohr_angstrom;
    double r_bohr[9];
    for (int idx = 0; idx < 9; ++idx) {
        r_bohr[idx] = coords_angstrom[static_cast<std::size_t>(idx)] * inv_bohr;
    }

    // Atom pairs: 0 -> (0, 1) AB, 1 -> (1, 2) BC, 2 -> (0, 2) AC
    constexpr int pair_a[3] = {0, 1, 0};
    constexpr int pair_b[3] = {1, 2, 2};
    double dx[3][3];
    double r[3];
    for (int p = 0; p < 3; ++p) {
        const int ia = pair_a[p] * 3;
        const int ib = pair_b[p] * 3;
        double dist2 = 0.0;
        for (int c = 0; c < 3; ++c) {
            const double diff = r_bohr[ia + c] - r_bohr[ib + c];
            dx[p][c] = diff;
            dist2 += diff * diff;
        }
        r[p] = std::sqrt(dist2);
    }

    constexpr double kRoot2 = 1.4142135623730951;
    double x[3], op3z[3], zp3[3], coul[3], ex[3];
    for (int p = 0; p < 3; ++p) {
        const double zp = sato[static_cast<std::size_t>(p)];
        op3z[p] = 1.0 + 3.0 * zp;
        zp3[p] = zp + 3.0;
        const double do4z = de_hartree[static_cast<std::size_t>(p)] / (4.0 * (1.0 + zp));
        const double xp = std::exp(-beta_bohr_inv[static_cast<std::size_t>(p)] *
                                   (r[p] - re_bohr[static_cast<std::size_t>(p)]));
        x[p] = xp;
        coul[p] = do4z * (zp3[p] * xp - 2.0 * op3z[p]) * xp;
        ex[p] = do4z * (op3z[p] * xp - 2.0 * zp3[p]) * xp;
    }

    const double d01 = ex[0] - ex[1];
    const double d12 = ex[1] - ex[2];
    const double d20 = ex[2] - ex[0];
    const double rad = std::sqrt(d01 * d01 + d12 * d12 + d20 * d20);
    const double energy = coul[0] + coul[1] + coul[2] - rad / kRoot2;
    const double sum_ex = ex[0] + ex[1] + ex[2];

    double dedr[3];
    for (int p = 0; p < 3; ++p) {
        const double zp = sato[static_cast<std::size_t>(p)];
        const double b_coef = beta_bohr_inv[static_cast<std::size_t>(p)] *
                              (de_hartree[static_cast<std::size_t>(p)] / (2.0 * (1.0 + zp)));
        double ex_deriv_term = 0.0;
        if (rad > 1.0e-14) {
            ex_deriv_term = ((3.0 * ex[p] - sum_ex) * (op3z[p] * x[p] - zp3[p])) / (kRoot2 * rad);
        }
        dedr[p] = b_coef * x[p] * (ex_deriv_term - zp3[p] * x[p] + op3z[p]);
    }

    std::vector<double> grad(9, 0.0);
    for (int p = 0; p < 3; ++p) {
        if (r[p] <= 0.0) {
            continue;
        }
        const double factor = dedr[p] / r[p];
        const int ia = pair_a[p] * 3;
        const int ib = pair_b[p] * 3;
        for (int c = 0; c < 3; ++c) {
            const double g_comp = factor * dx[p][c];
            grad[static_cast<std::size_t>(ia + c)] += g_comp;
            grad[static_cast<std::size_t>(ib + c)] -= g_comp;
        }
    }
    return {energy, std::move(grad)};
}

PyObject* py_project_symmetric_matrix(PyObject*, PyObject* args) {
    PyObject* matrix_obj = nullptr;
    PyObject* basis_obj = nullptr;
    Py_ssize_t dim_s = 0;
    Py_ssize_t n_basis_s = 0;
    if (!PyArg_ParseTuple(args, "OOnn:project_symmetric_matrix", &matrix_obj, &basis_obj,
                          &dim_s, &n_basis_s)) {
        return nullptr;
    }
    if (dim_s <= 0 || n_basis_s < 0) {
        PyErr_SetString(PyExc_ValueError, "dim must be positive and n_basis non-negative");
        return nullptr;
    }
    std::vector<double> matrix;
    std::vector<double> basis;
    if (!sequence_to_vector(matrix_obj, matrix, "matrix") ||
        !sequence_to_vector(basis_obj, basis, "basis")) {
        return nullptr;
    }
    const std::size_t dim = static_cast<std::size_t>(dim_s);
    const std::size_t n_basis = static_cast<std::size_t>(n_basis_s);
    if (matrix.size() != dim * dim || basis.size() != dim * n_basis) {
        PyErr_SetString(PyExc_ValueError, "invalid matrix or basis flattened dimensions");
        return nullptr;
    }
    std::vector<double> out;
    Py_BEGIN_ALLOW_THREADS
    out = project_symmetric_matrix_impl(matrix, basis, dim, n_basis);
    Py_END_ALLOW_THREADS

    PyObject* result = PyList_New(static_cast<Py_ssize_t>(out.size()));
    if (result == nullptr) {
        return nullptr;
    }
    for (std::size_t idx = 0; idx < out.size(); ++idx) {
        PyObject* value = PyFloat_FromDouble(out[idx]);
        if (value == nullptr) {
            Py_DECREF(result);
            return nullptr;
        }
        PyList_SET_ITEM(result, static_cast<Py_ssize_t>(idx), value);
    }
    return result;
}

PyObject* py_page_mciver_lqa_step(PyObject*, PyObject* args) {
    PyObject* grad_obj = nullptr;
    PyObject* evals_obj = nullptr;
    PyObject* evecs_obj = nullptr;
    Py_ssize_t dim_s = 0;
    double step_size = 0.0;
    if (!PyArg_ParseTuple(args, "OOOnd:page_mciver_lqa_step", &grad_obj, &evals_obj,
                          &evecs_obj, &dim_s, &step_size)) {
        return nullptr;
    }
    if (dim_s <= 0 || step_size <= 0.0) {
        PyErr_SetString(PyExc_ValueError, "dim and step_size must be positive");
        return nullptr;
    }
    std::vector<double> gradient;
    std::vector<double> eigenvalues;
    std::vector<double> eigenvectors;
    if (!sequence_to_vector(grad_obj, gradient, "gradient") ||
        !sequence_to_vector(evals_obj, eigenvalues, "eigenvalues") ||
        !sequence_to_vector(evecs_obj, eigenvectors, "eigenvectors")) {
        return nullptr;
    }
    const std::size_t dim = static_cast<std::size_t>(dim_s);
    if (gradient.size() != dim || eigenvalues.size() != dim || eigenvectors.size() != dim * dim) {
        PyErr_SetString(PyExc_ValueError, "inconsistent gradient/eigenpair dimensions");
        return nullptr;
    }
    LqaStepResult step;
    Py_BEGIN_ALLOW_THREADS
    step = page_mciver_lqa_step_impl(gradient, eigenvalues, eigenvectors, dim, step_size);
    Py_END_ALLOW_THREADS

    PyObject* dx_list = PyList_New(dim_s);
    PyObject* tan_list = PyList_New(dim_s);
    if (dx_list == nullptr || tan_list == nullptr) {
        Py_XDECREF(dx_list);
        Py_XDECREF(tan_list);
        return nullptr;
    }
    for (std::size_t idx = 0; idx < dim; ++idx) {
        PyObject* vx = PyFloat_FromDouble(step.delta_x[idx]);
        PyObject* vt = PyFloat_FromDouble(step.end_tangent[idx]);
        if (vx == nullptr || vt == nullptr) {
            Py_XDECREF(vx);
            Py_XDECREF(vt);
            Py_DECREF(dx_list);
            Py_DECREF(tan_list);
            return nullptr;
        }
        PyList_SET_ITEM(dx_list, static_cast<Py_ssize_t>(idx), vx);
        PyList_SET_ITEM(tan_list, static_cast<Py_ssize_t>(idx), vt);
    }
    PyObject* u_obj = PyFloat_FromDouble(step.u_star);
    if (u_obj == nullptr) {
        Py_DECREF(dx_list);
        Py_DECREF(tan_list);
        return nullptr;
    }
    PyObject* tuple = PyTuple_New(3);
    if (tuple == nullptr) {
        Py_DECREF(dx_list);
        Py_DECREF(tan_list);
        Py_DECREF(u_obj);
        return nullptr;
    }
    PyTuple_SET_ITEM(tuple, 0, dx_list);
    PyTuple_SET_ITEM(tuple, 1, tan_list);
    PyTuple_SET_ITEM(tuple, 2, u_obj);
    return tuple;
}

PyObject* py_leps_energy_and_gradient(PyObject*, PyObject* args) {
    PyObject* coords_obj = nullptr;
    PyObject* de_obj = nullptr;
    PyObject* re_obj = nullptr;
    PyObject* beta_obj = nullptr;
    PyObject* sato_obj = nullptr;
    double bohr_angstrom = 0.0;
    if (!PyArg_ParseTuple(args, "OOOOOd:leps_energy_and_gradient", &coords_obj, &de_obj,
                          &re_obj, &beta_obj, &sato_obj, &bohr_angstrom)) {
        return nullptr;
    }
    if (bohr_angstrom <= 0.0) {
        PyErr_SetString(PyExc_ValueError, "bohr_angstrom must be positive");
        return nullptr;
    }
    std::vector<double> coords, de, re, beta, sato;
    if (!sequence_to_vector(coords_obj, coords, "coords_angstrom") ||
        !sequence_to_vector(de_obj, de, "de_hartree") ||
        !sequence_to_vector(re_obj, re, "re_bohr") ||
        !sequence_to_vector(beta_obj, beta, "beta_bohr_inv") ||
        !sequence_to_vector(sato_obj, sato, "sato")) {
        return nullptr;
    }
    if (coords.size() != 9 || de.size() != 3 || re.size() != 3 || beta.size() != 3 ||
        sato.size() != 3) {
        PyErr_SetString(PyExc_ValueError, "LEPS requires 3-atom coordinates (9) and 3 bond parameters");
        return nullptr;
    }
    std::pair<double, std::vector<double>> eval_out;
    Py_BEGIN_ALLOW_THREADS
    eval_out = leps_energy_and_gradient_impl(coords, de, re, beta, sato, bohr_angstrom);
    Py_END_ALLOW_THREADS

    PyObject* grad_list = PyList_New(9);
    PyObject* e_obj = PyFloat_FromDouble(eval_out.first);
    if (grad_list == nullptr || e_obj == nullptr) {
        Py_XDECREF(grad_list);
        Py_XDECREF(e_obj);
        return nullptr;
    }
    for (std::size_t idx = 0; idx < 9; ++idx) {
        PyObject* g_val = PyFloat_FromDouble(eval_out.second[idx]);
        if (g_val == nullptr) {
            Py_DECREF(grad_list);
            Py_DECREF(e_obj);
            return nullptr;
        }
        PyList_SET_ITEM(grad_list, static_cast<Py_ssize_t>(idx), g_val);
    }
    PyObject* tuple = PyTuple_New(2);
    if (tuple == nullptr) {
        Py_DECREF(grad_list);
        Py_DECREF(e_obj);
        return nullptr;
    }
    PyTuple_SET_ITEM(tuple, 0, e_obj);
    PyTuple_SET_ITEM(tuple, 1, grad_list);
    return tuple;
}

PyObject* py_native_version(PyObject*, PyObject*) {
    return PyUnicode_FromString("pyrate-cpp-0.3");
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
    {"wkb_exponent_profile", py_wkb_exponent_profile, METH_VARARGS,
     "Return the dimensionless WKB/SCT transmission exponent for variable effective mass."},
    {"thermal_wkb_kappa", py_thermal_wkb_kappa, METH_VARARGS,
     "Return a canonical one-dimensional WKB/ZCT transmission correction."},
    {"thermal_wkb_profile_kappa", py_thermal_wkb_profile_kappa, METH_VARARGS,
     "Return a canonical SCT/variable-mass WKB transmission correction."},
    {"eckart_transmission", py_eckart_transmission, METH_VARARGS,
     "Return the analytical Eckart barrier transmission probability."},
    {"thermal_eckart_kappa", py_thermal_eckart_kappa, METH_VARARGS,
     "Return the canonical thermal Eckart transmission coefficient."},
    {"beyer_swinehart", py_beyer_swinehart, METH_VARARGS,
     "Return a vibrational density-of-states grain array."},
    {"project_symmetric_matrix", py_project_symmetric_matrix, METH_VARARGS,
     "Orthogonally project a symmetric matrix P M P with P = I - B B^T."},
    {"page_mciver_lqa_step", py_page_mciver_lqa_step, METH_VARARGS,
     "Compute an exact arc-length Page-McIver LQA steepest-descent step."},
    {"leps_energy_and_gradient", py_leps_energy_and_gradient, METH_VARARGS,
     "Evaluate 3-atom Extended LEPS energy (Hartree) and gradient (Hartree/bohr)."},
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
