# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.20)
project(qcap_hsb)

find_package(holoscan 0.6 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

add_library(qcap_hsb_lib STATIC IMPORTED)
set_target_properties(qcap_hsb_lib PROPERTIES
  IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/libqcap_hsb.a"
)

add_library(qcap_hololink_core_lib STATIC IMPORTED)
set_target_properties(qcap_hololink_core_lib PROPERTIES
  IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/libqcap_hololink_core.a"
)

add_library(qcap_hsb INTERFACE)
target_link_libraries(qcap_hsb INTERFACE qcap_hsb_lib)
target_link_libraries(qcap_hsb INTERFACE qcap_hololink_core_lib)

add_library(holoscan::ops::qcap_hsb ALIAS qcap_hsb)

add_library(qcap SHARED IMPORTED)
set_target_properties(qcap PROPERTIES
    IMPORTED_LOCATION "/opt/yuan/qcap/lib/libqcap.so"
    INTERFACE_INCLUDE_DIRECTORIES "/opt/yuan/qcap/include"
    INTERFACE_LINK_DIRECTORIES "/usr/lib/aarch64-linux-gnu/nvidia"
)

target_compile_definitions(qcap_hsb INTERFACE QCAP_HSB)

target_include_directories(qcap_hsb INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(qcap_hsb
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../../>
    $<INSTALL_INTERFACE:src>
  )

target_link_libraries(qcap_hsb
  INTERFACE
    holoscan::core
    qcap
  )

if(HOLOHUB_BUILD_PYTHON)
    add_subdirectory(python)
endif()
