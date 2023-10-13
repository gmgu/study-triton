## pybind11
pybind11 is a library for calling C++ libary from python and vice versa.

## Example: Call add function written in C++ from Python
```bash
// my_module.cpp
#include <pybind11/pybind11.h>

int mul(int x, int y) {
    return x * y;
}

int add(int x, int y) {
    return x + y;
}

float add(float x, float y) {
    return x + y;
}

namespace py = pybind11;

PYBIND11_MODULE(my_cpp_module, m) {
    m.def("mul", &mul);                                     // plain C++ function
    m.def("add", py::overload_cast<int, int>(&add));        // overloaded function
    m.def("add", py::overload_cast<float, float>(&add));    // overloaded function
    m.def("subtract", [](int x, int y) { return x - y; });  // lambda function
}
```

```bash
setup.py
```

```bash
# main.py
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

```bash
9
9
9.0
9
```

## Usage
```bash
pip install .
python main.py
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
    infline handle release()  // resets the internal pointer to nullptr without decreasing the object\'s reference count. the function returns a raw handle to the original Pytho object.

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
