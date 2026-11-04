// Harmonic-oscillator partition function and Wigner tunneling kernels.
#include "common.hpp"

namespace {

using namespace pyrate;

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

}  // namespace

PyMethodDef pyrate_thermo_methods[] = {
    {"harmonic_log_partition", py_harmonic_log_partition, METH_VARARGS,
     "Return the harmonic-oscillator log partition function."},
    {"wigner_kappa", py_wigner_kappa, METH_VARARGS, "Return the Wigner tunneling correction."},
    {nullptr, nullptr, 0, nullptr},
};
