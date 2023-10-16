#include <pybind11/pybind11.h>


int mul(int x, int y) {
    return x * y;
}

int add(int x, int y) {
    return x + y;
}

// FLOAT_COMPATIBLE is defined in setup.py
#ifdef FLOAT_COMPATIBLE
float add(float x, float y) {
    return x + y;
}
#endif

namespace py = pybind11;

PYBIND11_MODULE(my_cpp_module, m) {
    m.def("mul", &mul);                                     // plain C++ function
    m.def("add", py::overload_cast<int, int>(&add));        // overloaded function
#ifdef FLOAT_COMPATIBLE
    m.def("add", py::overload_cast<float, float>(&add));    // overloaded function
#endif
    m.def("subtract", [](int x, int y) { return x - y; });  // lambda function
}
