// Variational-maximum refinement kernels used by CVT.
#include "common.hpp"

#include <algorithm>
#include <iterator>
#include <utility>

namespace {

using namespace pyrate;

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

}  // namespace

PyMethodDef pyrate_fit_methods[] = {
    {"refine_maximum", py_refine_maximum, METH_VARARGS,
     "Refine a sampled maximum using up to five local points."},
    {nullptr, nullptr, 0, nullptr},
};
