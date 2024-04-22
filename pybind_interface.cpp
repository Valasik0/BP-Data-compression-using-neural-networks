#include <pybind11/stl.h>
#include "KthEntropyCalculator.h"

namespace py = pybind11;

PYBIND11_MODULE(entropy, m){
    py::class_<KthEntropyCalculator>(m, "KthEntropyCalculator")
        .def(py::init([](const std::string &text, int k){
            std::vector<unsigned char> text_vector(text.begin(), text.end());
            return new KthEntropyCalculator(text_vector, k); 
            }))
        .def("calculate_kth_entropy", &KthEntropyCalculator::calculate_kth_entropy);
}