// Shared helpers for the PyRate native extension.
//
// Each numerical area (thermo, fit, wkb, dos) lives in its own translation
// unit and exports a null-terminated PyMethodDef table.  module.cpp registers
// every table, so adding a kernel never requires editing another area's file.
#pragma once

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace pyrate {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kBoltzmann = 1.380649e-23;
constexpr double kPlanck = 6.62607015e-34;
constexpr double kHbar = kPlanck / (2.0 * kPi);
constexpr double kLightSpeed = 299792458.0;

// Copy a Python sequence of real numbers into a contiguous vector.
inline bool sequence_to_vector(PyObject* object, std::vector<double>& values, const char* name) {
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

inline bool strictly_increasing(const std::vector<double>& values) {
    for (std::size_t index = 1; index < values.size(); ++index) {
        if (!(values[index] > values[index - 1])) {
            return false;
        }
    }
    return true;
}

// exp() clamped to the finite double range used by the Python fallbacks.
inline double safe_exp(const double exponent) {
    if (exponent < -745.0) {
        return 0.0;
    }
    if (exponent > 709.0) {
        return std::numeric_limits<double>::infinity();
    }
    return std::exp(exponent);
}

}  // namespace pyrate

// Method tables exported by each translation unit (null-terminated).
extern PyMethodDef pyrate_thermo_methods[];
extern PyMethodDef pyrate_fit_methods[];
extern PyMethodDef pyrate_wkb_methods[];
extern PyMethodDef pyrate_dos_methods[];
