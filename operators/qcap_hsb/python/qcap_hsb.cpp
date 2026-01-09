/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 YUAN High-Tech Development Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "../qcap_hsb.hpp"
#include "./qcap_hsb_pydoc.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // for unordered_map -> dict, etc.

#include <cstdint>
#include <memory>
#include <string>

#include "../../operator_util.hpp"
#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator_spec.hpp>
#include "holoscan/core/gxf/gxf_operator.hpp"

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace holoscan::ops {

/* Trampoline classes for handling Python kwargs
 *
 * These add a constructor that takes a Fragment for which to initialize the operator.
 * The explicit parameter list and default arguments take care of providing a Pythonic
 * kwarg-based interface with appropriate default values matching the operator's
 * default parameters in the C++ API `setup` method.
 *
 * The sequence of events in this constructor is based on Fragment::make_operator<OperatorT>
 */

class PyQcapHSBOp : public QcapHSBOp {
 public:
  /* Inherit the constructors */
  using QcapHSBOp::QcapHSBOp;

  // Define a constructor that fully initializes the object.
  PyQcapHSBOp(Fragment* fragment, const py::args& args,
                 const std::string& hololink_ip = "192.168.0.2"s,
                 const std::string& hololink_mac = ""s,
                 const std::string& ibv_name = "mlx5_0"s,
                 uint32_t ibv_port = 1,
                 ULONG width = 3840,
                 ULONG height = 2160,
                 const std::string& name = "qcap_hsb" )
      : QcapHSBOp(ArgList{Arg{"hololink_ip", hololink_ip},
                          Arg{"hololink_mac", hololink_mac},
                          Arg{"ibv_name", ibv_name},
                          Arg{"ibv_port", ibv_port},
                          Arg{"width", width},
                          Arg{"height", height}}) {
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

PYBIND11_MODULE(_qcap_hsb, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK Python Bindings
        ---------------------------------------
        .. currentmodule:: _qcap_hsb
        .. autosummary::
           :toctree: _generate
           add
           subtract
    )pbdoc";

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif

  py::class_<QcapHSBOp, PyQcapHSBOp, Operator, std::shared_ptr<QcapHSBOp>>(
      m, "QcapHSBOp", doc::QcapHSBOp::doc_QcapHSBOp)
      .def(py::init<Fragment*,
                    const py::args&,
                    const std::string&,
                    const std::string&,
                    const std::string&,
                    uint32_t,
                    ULONG,
                    ULONG,
                    const std::string&>(),
           "fragment"_a,
           "hololink_ip"_a = "192.168.0.2"s,
           "hololink_mac"_a = ""s,
           "ibv_name"_a = "roceP5p3s0f0"s,
           "ibv_port"_a = 1,
           "width"_a = 3840,
           "height"_a = 2160,
           "name"_a = "qcap_hsb"s,
           doc::QcapHSBOp::doc_QcapHSBOp_python)
      .def("initialize", &QcapHSBOp::initialize, doc::QcapHSBOp::doc_initialize)
      .def("setup", &QcapHSBOp::setup, "spec"_a, doc::QcapHSBOp::doc_setup);
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::ops
