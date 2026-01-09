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

#include "../hdmi_converter.hpp"
#include "./hdmi_converter_pydoc.hpp"

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

class PyHDMIConverterOp : public HDMIConverterOp {
 public:
  /* Inherit the constructors */
  using HDMIConverterOp::HDMIConverterOp;

  // Define a constructor that fully initializes the object.
  PyHDMIConverterOp(Fragment* fragment, const py::args& args,
                 const std::shared_ptr<holoscan::Allocator>& allocator,
                 int cuda_device_ordinal,
                 const std::string& out_tensor_name = "",
                 const std::string& left_tensor_name = "",
                 const std::string& right_tensor_name = "",
                 int input_3d_format = 0,
                 int output_3d_format = 0,
                 const std::string& name = "hdmi_converter")
      : HDMIConverterOp(ArgList{Arg{"allocator", allocator},
                             Arg{"cuda_device_ordinal", cuda_device_ordinal},
                             Arg{"out_tensor_name", out_tensor_name},
                             Arg{"left_tensor_name", left_tensor_name},
                             Arg{"right_tensor_name", right_tensor_name},
                             Arg{"input_3d_format", input_3d_format},
                             Arg{"output_3d_format", output_3d_format}
                             }) {
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

PYBIND11_MODULE(_hdmi_converter, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK Python Bindings
        ---------------------------------------
        .. currentmodule:: _hdmi_converter
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

  py::class_<HDMIConverterOp, PyHDMIConverterOp, Operator, std::shared_ptr<HDMIConverterOp>>(
      m, "HDMIConverterOp", doc::HDMIConverterOp::doc_HDMIConverterOp)
      .def(py::init<Fragment*,
                    const py::args&,
                    const std::shared_ptr<holoscan::Allocator>&,
                    int,
                    const std::string&,
                    const std::string&,
                    const std::string&,
                    int,
                    int,
                    const std::string&
                    >(),
           "fragment"_a,
           "allocator"_a,
           "cuda_device_ordinal"_a = 0,
           "out_tensor_name"_a = ""s,
           "left_tensor_name"_a = ""s,
           "right_tensor_name"_a = ""s,
           "input_3d_format"_a = 0,
           "output_3d_format"_a = 0,
           "name"_a = "hdmi_converter"s,
           doc::HDMIConverterOp::doc_HDMIConverterOp_python)
      .def("initialize", &HDMIConverterOp::initialize, doc::HDMIConverterOp::doc_initialize)
      .def("setup", &HDMIConverterOp::setup, "spec"_a, doc::HDMIConverterOp::doc_setup)
      .def("configure", &HDMIConverterOp::configure,
           "start_byte"_a, "bytes_per_line"_a, "pixel_width"_a, "pixel_height"_a,
           "pixel_format"_a, "trailing_bytes"_a = 0);

  // 綁定 PixelFormat enum
  py::enum_<HDMIConverterOp::PixelFormat>(m.attr("HDMIConverterOp"), "PixelFormat")
      .value("RAW_8", HDMIConverterOp::PixelFormat::RAW_8)
      .value("RAW_10", HDMIConverterOp::PixelFormat::RAW_10)
      .value("RAW_12", HDMIConverterOp::PixelFormat::RAW_12)
      .value("YUYV_8", HDMIConverterOp::PixelFormat::YUYV_8)
      .value("RGB_8", HDMIConverterOp::PixelFormat::RGB_8)
      .value("RGBA_8", HDMIConverterOp::PixelFormat::RGBA_8)
      .export_values();  // 讓 enum 成員可以直接當作全域使用

  // 綁定 Video3DFormat enum
  py::enum_<HDMIConverterOp::Video3DFormat>(m.attr("HDMIConverterOp"), "Video3DFormat")
        .value("INVALID", HDMIConverterOp::Video3DFormat::INVALID)
        .value("FRAME_PACKING", HDMIConverterOp::Video3DFormat::FRAME_PACKING)
        .value("SIDE_BY_SIDE_HALF", HDMIConverterOp::Video3DFormat::SIDE_BY_SIDE_HALF)
        .value("TOP_AND_BOTTOM", HDMIConverterOp::Video3DFormat::TOP_AND_BOTTOM)
        .value("LINE_BY_LINE", HDMIConverterOp::Video3DFormat::LINE_BY_LINE)
        .value("FIELD_ALTERNATIVE", HDMIConverterOp::Video3DFormat::FIELD_ALTERNATIVE)
        .value("VIDEO_PLUS_DEPTH", HDMIConverterOp::Video3DFormat::VIDEO_PLUS_DEPTH)
        .value("SIDE_BY_SIDE_FULL", HDMIConverterOp::Video3DFormat::SIDE_BY_SIDE_FULL)
        .export_values();

}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::ops
