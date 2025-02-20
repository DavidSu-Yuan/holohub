# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Make sure to order by alphabetical list

add_holohub_operator(basic_network)
add_holohub_operator(emergent_source DEPENDS EXTENSIONS emergent_source)
add_holohub_operator(lstm_tensor_rt_inference DEPENDS EXTENSIONS lstm_tensor_rt_inference)
add_holohub_operator(qcap_source DEPENDS EXTENSIONS qcap_source)
add_holohub_operator(tensor_to_video_buffer)
add_holohub_operator(tool_tracking_postprocessor)
add_holohub_operator(v4l2_video_capture DEPENDS EXTENSIONS v4l2_video_capture)
add_holohub_operator(video_decoder)
add_holohub_operator(video_encoder)
add_holohub_operator(video_read_bitstream)
add_holohub_operator(video_write_bitstream)
add_holohub_operator(videomaster DEPENDS EXTENSIONS videomaster)
add_holohub_operator(visualizer_icardio)
