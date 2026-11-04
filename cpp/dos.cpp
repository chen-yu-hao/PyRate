// Microcanonical state-counting kernels.
#include "common.hpp"

#include <algorithm>

namespace {

using namespace pyrate;

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
        const std::size_t step =
            static_cast<std::size_t>(std::max(1.0, std::round(frequency / grain_cm1)));
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

}  // namespace

PyMethodDef pyrate_dos_methods[] = {
    {"beyer_swinehart", py_beyer_swinehart, METH_VARARGS,
     "Return a vibrational density-of-states grain array."},
    {nullptr, nullptr, 0, nullptr},
};
