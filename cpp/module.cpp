// Module definition: registers every per-area method table.
#include "common.hpp"

namespace {

PyObject* py_native_version(PyObject*, PyObject*) {
    return PyUnicode_FromString("pyrate-cpp-0.2");
}

PyMethodDef module_methods[] = {
    {"native_version", py_native_version, METH_NOARGS, "Return the native-kernel ABI version."},
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
    PyObject* module = PyModule_Create(&module_definition);
    if (module == nullptr) {
        return nullptr;
    }
    PyMethodDef* tables[] = {
        pyrate_thermo_methods,
        pyrate_fit_methods,
        pyrate_wkb_methods,
        pyrate_dos_methods,
    };
    for (PyMethodDef* table : tables) {
        if (PyModule_AddFunctions(module, table) != 0) {
            Py_DECREF(module);
            return nullptr;
        }
    }
    return module;
}
