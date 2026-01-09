/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 YUAN High-Tech Development Co., Ltd. All rights reserved.
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

#pragma once

#include <string>

#include "macros.hpp"

namespace holoscan::doc {

namespace HDMIConverterOp {

// Constructor
PYDOC(HDMIConverterOp, R"doc(
Operator to get a video stream from an YUAN High-Tech capture card.
)doc")

// PyHDMIConverterOp Constructor
PYDOC(HDMIConverterOp_python, R"doc(
Operator to get a video stream from an YUAN High-Tech capture card.

Parameters
----------
fragment : Fragment
    The fragment that the operator belongs to.
allocator : Allocator, optional
    The allocator of cuda memory.
cuda_device_ordinal : int, optional
    Cuda device ordinal.
out_tensor_name : str, optional
    The tensor name of ouptut.
left_tensor_name : str, optional
    The tensor name of left eye.
right_tensor_name : str, optional
    The tensor name of right eye.
)doc")

PYDOC(setup, R"doc(
Define the operator specification.

Parameters
----------
spec : ``holoscan.core.OperatorSpec``
    The operator specification.
)doc")

PYDOC(initialize, R"doc(
Initialize the operator.

This method is called only once when the operator is created for the first time,
and uses a light-weight initialization.
)doc")

}  // namespace HDMIConverterOp
}  // namespace holoscan::doc
