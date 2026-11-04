// Variational-maximum refinement kernels used by CVT.
//
// refine_maximum(x, y, max_points=3) refines the first sampled maximum i of a
// curve y(x) (for CVT, the generalized free energy of activation along the
// path).  Every refined result satisfies, with B = y[i] - min(y[i-1], y[i+1]),
//
//   x[i-1] <= coordinate <= x[i+1]   and   y[i] <= value <= y[i] + B;
//
// a fit that would violate either bound is not reported.  The default is the
// parabola through (i-1, i, i+1).  max_points = 4 or 5 opts in to the
// interpolating polynomial through that many points; when it has no admissible
// maximum the parabola is tried next, and the sample itself is the last resort
// (fit_points = 1).  The arithmetic mirrors src/pyrate/native/fit.py operation
// by operation so that both backends agree to rounding.
#include "common.hpp"

#include <algorithm>
#include <iterator>
#include <utility>

namespace {

using namespace pyrate;

constexpr Py_ssize_t kMinFitPoints = 3;
constexpr Py_ssize_t kMaxFitPoints = 5;
constexpr int kBisectionSteps = 100;

struct RefinedMaximum {
    double x;
    double y;
    std::size_t sample_index;
    std::size_t fit_points;
};

struct Candidate {
    bool accepted;
    double x;
    double y;
};

double horner(const std::vector<double>& coefficients, const double t) {
    double result = 0.0;
    for (auto iterator = coefficients.rbegin(); iterator != coefficients.rend(); ++iterator) {
        result = result * t + *iterator;
    }
    return result;
}

std::vector<double> derivative(const std::vector<double>& coefficients) {
    std::vector<double> result;
    for (std::size_t degree = 1; degree < coefficients.size(); ++degree) {
        result.push_back(static_cast<double>(degree) * coefficients[degree]);
    }
    return result;
}

// Accept a refined point only inside the bracket and below the drop bound.
Candidate admit(const double coordinate, const double value, const std::vector<double>& x,
                const std::vector<double>& y, const std::size_t i) {
    const double ceiling = y[i] + (y[i] - std::min(y[i - 1], y[i + 1]));
    if (!std::isfinite(coordinate) || !std::isfinite(value) || !(value >= y[i]) ||
        !(value <= ceiling)) {
        return {false, 0.0, 0.0};
    }
    // The fits below keep the coordinate inside the bracket analytically; the
    // clamp only absorbs rounding in x[i] + offset.
    return {true, std::min(std::max(coordinate, x[i - 1]), x[i + 1]), value};
}

// Vertex of the parabola through (i-1, i, i+1), written as
// p(x[i] + t) = y[i] + b t + a t^2.  With spacings h_l, h_r and drops
// d_l = y[i] - y[i-1] > 0 (i is the first maximum) and d_r = y[i] - y[i+1] >= 0,
//   a = -(d_l h_r + d_r h_l) / H,   b = (d_l h_r^2 - d_r h_l^2) / H,
//   H = h_l h_r (h_l + h_r),
// so a < 0, the vertex t* = -b / (2a) lies in [-h_l / 2, h_r / 2] and
// p(t*) = y[i] + b t* / 2 >= y[i].
Candidate quadratic_vertex(const std::vector<double>& x, const std::vector<double>& y,
                           const std::size_t i) {
    const double h_left = x[i] - x[i - 1];
    const double h_right = x[i + 1] - x[i];
    const double drop_left = y[i] - y[i - 1];
    const double drop_right = y[i] - y[i + 1];
    const double numerator = drop_left * h_right * h_right - drop_right * h_left * h_left;
    const double weight = drop_left * h_right + drop_right * h_left;
    const double spacing = h_left * h_right * (h_left + h_right);
    if (!(weight > 0.0) || !(spacing > 0.0)) {
        return {false, 0.0, 0.0};
    }
    const double offset = numerator / (2.0 * weight);
    const double slope = numerator / spacing;
    return admit(x[i] + offset, y[i] + 0.5 * slope * offset, x, y, i);
}

// Monomial coefficients (lowest degree first) of the polynomial interpolating
// (t[k], values[k]); built from Newton divided differences.
bool interpolating_polynomial(const std::vector<double>& t, const std::vector<double>& values,
                              std::vector<double>& coefficients) {
    const std::size_t count = t.size();
    std::vector<double> divided = values;
    for (std::size_t level = 1; level < count; ++level) {
        for (std::size_t k = count - 1; k >= level; --k) {
            const double denominator = t[k] - t[k - level];
            if (!(denominator > 0.0)) {
                return false;
            }
            divided[k] = (divided[k] - divided[k - 1]) / denominator;
        }
    }
    coefficients.assign(1, divided[count - 1]);
    for (std::size_t k = count - 1; k-- > 0;) {
        // coefficients <- coefficients * (t - t[k]) + divided[k]
        std::vector<double> expanded(coefficients.size() + 1, 0.0);
        for (std::size_t degree = 0; degree < expanded.size(); ++degree) {
            const double shifted = degree >= 1 ? coefficients[degree - 1] : 0.0;
            const double scaled = degree < coefficients.size() ? t[k] * coefficients[degree] : 0.0;
            expanded[degree] = shifted - scaled;
        }
        expanded[0] = expanded[0] + divided[k];
        coefficients = std::move(expanded);
    }
    return true;
}

// Real roots of a polynomial of degree <= 2 strictly inside (lower, upper).
void append_interior_roots(const std::vector<double>& coefficients, const double lower,
                           const double upper, std::vector<double>& roots) {
    std::vector<double> found;
    const double constant = coefficients.size() > 0 ? coefficients[0] : 0.0;
    const double linear = coefficients.size() > 1 ? coefficients[1] : 0.0;
    const double quadratic = coefficients.size() > 2 ? coefficients[2] : 0.0;
    if (quadratic != 0.0) {
        const double discriminant = linear * linear - 4.0 * quadratic * constant;
        if (discriminant >= 0.0) {
            const double q = -0.5 * (linear + std::copysign(std::sqrt(discriminant), linear));
            found.push_back(q / quadratic);
            if (q != 0.0) {
                found.push_back(constant / q);
            }
        }
    } else if (linear != 0.0) {
        found.push_back(-constant / linear);
    }
    for (const double root : found) {
        if (root > lower && root < upper) {
            roots.push_back(root);
        }
    }
}

// Highest local maximum of the interpolating polynomial through
// x[start .. start+count) that lies inside [x[i-1], x[i+1]].  The search uses
// the scaled variable t = (x - x[i]) / scale; the roots of p'' split the
// bracket into pieces on which p' is monotonic, and every + to - sign change of
// p' is located by bisection.
Candidate polynomial_maximum(const std::vector<double>& x, const std::vector<double>& y,
                             const std::size_t i, const std::size_t count) {
    std::size_t start = i >= count / 2 ? i - count / 2 : 0;
    if (start + count > x.size()) {
        start = x.size() - count;
    }
    const double scale = std::max(x[i] - x[start], x[start + count - 1] - x[i]);
    if (!std::isfinite(scale) || !(scale > 0.0)) {
        return {false, 0.0, 0.0};
    }
    std::vector<double> t(count);
    std::vector<double> values(count);
    for (std::size_t k = 0; k < count; ++k) {
        t[k] = (x[start + k] - x[i]) / scale;
        values[k] = y[start + k];
    }
    std::vector<double> coefficients;
    if (!interpolating_polynomial(t, values, coefficients)) {
        return {false, 0.0, 0.0};
    }
    const std::vector<double> slope = derivative(coefficients);
    const std::vector<double> curvature = derivative(slope);

    const double lower = t[i - 1 - start];
    const double upper = t[i + 1 - start];
    std::vector<double> breaks;
    append_interior_roots(curvature, lower, upper, breaks);
    std::sort(breaks.begin(), breaks.end());
    breaks.insert(breaks.begin(), lower);
    breaks.push_back(upper);

    bool found = false;
    double best_t = 0.0;
    double best_value = y[i];
    for (std::size_t piece = 0; piece + 1 < breaks.size(); ++piece) {
        double left = breaks[piece];
        double right = breaks[piece + 1];
        if (!(horner(slope, left) > 0.0) || !(horner(slope, right) <= 0.0)) {
            continue;
        }
        for (int step = 0; step < kBisectionSteps; ++step) {
            const double middle = 0.5 * (left + right);
            if (middle <= left || middle >= right) {
                break;
            }
            if (horner(slope, middle) > 0.0) {
                left = middle;
            } else {
                right = middle;
            }
        }
        const double root = 0.5 * (left + right);
        const double value = horner(coefficients, root);
        if (value > best_value) {
            found = true;
            best_t = root;
            best_value = value;
        }
    }
    if (!found) {
        return {false, 0.0, 0.0};
    }
    return admit(x[i] + scale * best_t, best_value, x, y, i);
}

RefinedMaximum refine_maximum_impl(const std::vector<double>& x, const std::vector<double>& y,
                                   const std::size_t max_points) {
    const auto maximum_iterator = std::max_element(y.begin(), y.end());
    const std::size_t i = static_cast<std::size_t>(std::distance(y.begin(), maximum_iterator));
    const RefinedMaximum sample = {x[i], y[i], i, 1};
    if (x.size() < 3 || i == 0 || i + 1 == x.size()) {
        return sample;
    }
    const std::size_t count = std::min(max_points, x.size());
    if (count > 3) {
        const Candidate polynomial = polynomial_maximum(x, y, i, count);
        if (polynomial.accepted) {
            return {polynomial.x, polynomial.y, i, count};
        }
    }
    const Candidate parabola = quadratic_vertex(x, y, i);
    if (parabola.accepted) {
        return {parabola.x, parabola.y, i, 3};
    }
    return sample;
}

bool all_finite(const std::vector<double>& values) {
    return std::all_of(values.begin(), values.end(),
                       [](const double value) { return std::isfinite(value); });
}

PyObject* py_refine_maximum(PyObject*, PyObject* args) {
    PyObject* x_object = nullptr;
    PyObject* y_object = nullptr;
    Py_ssize_t max_points = kMinFitPoints;
    if (!PyArg_ParseTuple(args, "OO|n:refine_maximum", &x_object, &y_object, &max_points)) {
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
    if (!all_finite(x) || !all_finite(y)) {
        PyErr_SetString(PyExc_ValueError, "x and y must be finite");
        return nullptr;
    }
    if (!strictly_increasing(x)) {
        PyErr_SetString(PyExc_ValueError, "x must be strictly increasing");
        return nullptr;
    }
    if (max_points < kMinFitPoints || max_points > kMaxFitPoints) {
        PyErr_SetString(PyExc_ValueError, "max_points must be 3, 4 or 5");
        return nullptr;
    }
    const RefinedMaximum result =
        refine_maximum_impl(x, y, static_cast<std::size_t>(max_points));
    return Py_BuildValue("ddnn", result.x, result.y,
                         static_cast<Py_ssize_t>(result.sample_index),
                         static_cast<Py_ssize_t>(result.fit_points));
}

}  // namespace

PyMethodDef pyrate_fit_methods[] = {
    {"refine_maximum", py_refine_maximum, METH_VARARGS,
     "refine_maximum(x, y, max_points=3) -> (coordinate, value, sample_index, fit_points)\n\n"
     "Refine the first sampled maximum inside [x[i-1], x[i+1]] with the parabola through\n"
     "three points (or an interpolant through up to max_points points), never exceeding\n"
     "y[i] + (y[i] - min(y[i-1], y[i+1])).  Falls back to the sample (fit_points = 1)."},
    {nullptr, nullptr, 0, nullptr},
};
