## pybind11
pybind11 is a library for calling C++ libary from python and vice versa.

## Example: Call add function written in C++ from Python
### my_module.cpp
We define four simple math functions, where addition has two overloaded versions for int and float types, 
and subtraction is a lambda function. The overloaded version for float type is compiled if and only if FLOAT_COMPATIBLE is defined.
We pass the function pointer for a plain C++ function. 
We use a special `pybind11::overload_cast<>()` function for overloaded functions.
```bash
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
```

### setup.py
We can build the C++ source code using `setuptools`.
The C++ source code is built while installing the package by `pip install .`.
Note that we define FLOAT_COMPATIBLE macro here.

```bash
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension("my_cpp_module",  # import this-name
                      ["my_module.cpp"],  # source files
                      define_macros = [('FLOAT_COMPATIBLE', True)]),  # define macros when compile
]

setup(
    name="my_cpp_module",  # package name install in the os
    version="0.0.0", 
    author="author name", 
    author_email="author email", 
    url="home page",  # 
    description="summary",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},  # search for the highest supported C++ standard
    zip_safe=False,  # whether the project can be safely installed and run from a zip file.
    python_requires=">=3.7",
    install_requires=[],
)
```

### main.py
After `pip install .`, we can import the library and call the function in Python script.

```bash
import my_cpp_module

result = my_cpp_module.mul(3, 3)
print(result)

result = my_cpp_module.add(5, 4)
print(result)

result = my_cpp_module.add(4.0, 5.0)
print(result)

result = my_cpp_module.subtract(11, 2)
print(result)
```

### Usage
```bash
pip install .
python main.py
```

### Result
```bash
9
9
9.0
9
```


## PYBIND11_MODULE(name, variable)
This macro crates the entry point that will be invoked when the Python interpreter imports an extension module. The module name is given as the first argument (name) and it should not be in quotes. The second macro argument defines a variable of type py::module\_ which can be used to initialize the module.

### class module\_ : public object
Wrapper for Python extension modules.

**class hierarchy**
```bash
class object: public handle
// holds a reference to a Python object (with reference counting)
    inline object(const object &o)  // copy constructor. always increases the reference count
    inline object(object &&other) noexcept  // move constructor. steals the object from other and preserves its reference count
    infline ~object()  // destructor. automatically calls handle::dec_ref()
    infline handle release()  // resets the internal pointer to nullptr without decreasing the reference count of the object. the function returns a raw handle to the original Pytho object.

class handle: public detail::object_api<handle>
// holds a reference to a Python object (no reference counting)
    handle() = default  // the default constructor creates a handle with a nullptr-valued pointer
    inline const handle &inc_ref() const &  // manually increase the reference count of the Python object. 
    inline const handle *dec_ref() const &  // manually decrease the reference count of the Python object.
    obj_attr_accessor attr(handle key) const  // return an internal functor to access the object's attributes.


class object_api: public pyobject_tag
// a mixin class (an uninstantiable class that provides functionality to be inferited by a subclass) which adds common functions to handle, object and various accessors.
    str_attr_accessor doc() const  // get or set the object's docstring, i.e., obj.__doc__
```

**Public Functions**
```bash
class module_: public object
    template<typename Func, typename ...Extra>
    inline module_ &def(const char *name_, Func && f, const Extra&...  extra)
    // create python binding for a new function within the module scope. Func can be a plain C++ function, a function pointer, or a lmabda function.
```
