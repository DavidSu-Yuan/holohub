# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(preview CXX)

find_package(holoscan 1.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

add_executable(preview
  main.cpp
)

target_link_libraries(preview
  PRIVATE
  holoscan::core
  holoscan::ops::video_stream_replayer
  holoscan::ops::format_converter
  holoscan::ops::holoviz
)

target_link_libraries(preview PRIVATE $<TARGET_NAME_IF_EXISTS:holoscan::qcap_source>)

# Copy the config to the binary directory
add_custom_target(preview_deps
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/preview.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "preview.yaml"
  BYPRODUCTS "preview.yaml"
)
add_dependencies(preview preview_deps)
