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

import os
from argparse import ArgumentParser

import holoscan

from holoscan.core import Application
from holoscan.operators import (
    FormatConverterOp,
    HolovizOp,
    VideoStreamReplayerOp,
)
from holoscan.resources import (
    BlockMemoryPool,
    CudaStreamPool,
    MemoryStorageType,
    UnboundedAllocator,
)

# Enable this line for Yuan capture card
from holohub.qcap_source import QCAPSourceOp

class PreviewApp(Application):
    def __init__(self, data, source="yuan"):
        """Initialize the endoscopy tool tracking application

        Parameters
        ----------
        source : {"replayer", "yuan"}
            When set to "replayer" (the default), pre-recorded sample video data is
            used as the application input. Otherwise, the video stream from Yuan
            capture card is used.
        """
        super().__init__()

        # set name
        self.name = "Preview App"

        self.source = source

        if data == "none":
            data = os.environ.get("HOLOHUB_DATA_PATH", "../data")

        self.sample_data_path = data

    def compose(self):
        is_overlay_enabled = False
        renderer = self.kwargs("visualizer")["visualizer"]
        input_video_signal = "receivers" if renderer == "holoviz" else "videostream"

        if self.source.lower() == "yuan":
            script_path = os.path.abspath(__file__)
            parent_dir = os.path.dirname(script_path)
            left_top_yuan_kwargs = self.kwargs("yuan_left_top")
            left_top_source = QCAPSourceOp(self,
                name="yuan_left_top",
                image_directory=parent_dir,
                **left_top_yuan_kwargs)
            right_top_yuan_kwargs = self.kwargs("yuan_right_top")
            right_top_source = QCAPSourceOp(self,
                name="yuan_right_top",
                image_directory=parent_dir,
                **right_top_yuan_kwargs)
            left_bottom_yuan_kwargs = self.kwargs("yuan_left_bottom")
            left_bottom_source = QCAPSourceOp(self,
                name="yuan_left_bottom",
                image_directory=parent_dir,
                **left_bottom_yuan_kwargs)
            right_bottom_yuan_kwargs = self.kwargs("yuan_right_bottom")
            right_bottom_source = QCAPSourceOp(self,
                name="yuan_right_bottom",
                image_directory=parent_dir,
                **right_bottom_yuan_kwargs)

            # 4 bytes/channel, 4 channels
            width = 3840
            height = 2160
            source_block_size = width * height * 4 * 4
            source_num_blocks = 4
        else:
            width = 854
            height = 480
            video_dir = self.sample_data_path
            if not os.path.exists(video_dir):
                raise ValueError(f"Could not find video data: {video_dir=}")
            source = VideoStreamReplayerOp(
                self,
                name="replayer",
                directory=video_dir,
                **self.kwargs("replayer"),
            )
            # 4 bytes/channel, 3 channels
            source_block_size = width * height * 3 * 4
            source_num_blocks = 2

        cuda_stream_pool = CudaStreamPool(
            self,
            name="cuda_stream",
            dev_id=0,
            stream_flags=0,
            stream_priority=0,
            reserved_size=1,
            max_size=5,
        )

        if self.source.lower() == "yuan":
            specs = []

            left_top_view = holoscan.operators.HolovizOp.InputSpec.View()
            left_top_view.offset_x = 0
            left_top_view.offset_y = 0
            left_top_view.width = 0.5
            left_top_view.height = 0.5

            left_top_spec = holoscan.operators.HolovizOp.InputSpec(
                "Left_Top", holoscan.operators.HolovizOp.InputType.COLOR
            )
            left_top_spec.views = [left_top_view]
            specs = specs + [left_top_spec]

            right_top_view = holoscan.operators.HolovizOp.InputSpec.View()
            right_top_view.offset_x = 0.5
            right_top_view.offset_y = 0
            right_top_view.width = 0.5
            right_top_view.height = 0.5

            right_top_spec = holoscan.operators.HolovizOp.InputSpec(
                "Right_Top", holoscan.operators.HolovizOp.InputType.COLOR
            )
            right_top_spec.views = [right_top_view]
            specs = specs + [right_top_spec]

            left_bottom_view = holoscan.operators.HolovizOp.InputSpec.View()
            left_bottom_view.offset_x = 0
            left_bottom_view.offset_y = 0.5
            left_bottom_view.width = 0.5
            left_bottom_view.height = 0.5

            left_bottom_spec = holoscan.operators.HolovizOp.InputSpec(
                "Left_Bottom", holoscan.operators.HolovizOp.InputType.COLOR
            )
            left_bottom_spec.views = [left_bottom_view]
            specs = specs + [left_bottom_spec]

            right_bottom_view = holoscan.operators.HolovizOp.InputSpec.View()
            right_bottom_view.offset_x = 0.5
            right_bottom_view.offset_y = 0.5
            right_bottom_view.width = 0.5
            right_bottom_view.height = 0.5

            right_bottom_spec = holoscan.operators.HolovizOp.InputSpec(
                "Right_Bottom", holoscan.operators.HolovizOp.InputType.COLOR
            )
            right_bottom_spec.views = [right_bottom_view]
            specs = specs + [right_bottom_spec]

            if renderer == "holoviz":
                visualizer = HolovizOp(
                    self,
                    name="holoviz",
                    width=width,
                    height=height,
                    tensors=specs,
                    enable_render_buffer_input=False,
                    enable_render_buffer_output=False,
                    allocator=None,
                    cuda_stream_pool=cuda_stream_pool
                )

            # Flow definition
            self.add_flow(left_top_source, visualizer,
                {
                    (
                        "video_buffer_output" if self.source != "replayer" else "output",
                        input_video_signal,
                    )
                },
            )
            self.add_flow(right_top_source, visualizer,
                {
                    (
                        "video_buffer_output" if self.source != "replayer" else "output",
                        input_video_signal,
                    )
                },
            )
            self.add_flow(left_bottom_source, visualizer,
                {
                    (
                        "video_buffer_output" if self.source != "replayer" else "output",
                        input_video_signal,
                    )
                },
            )
            self.add_flow(right_bottom_source, visualizer,
                {
                    (
                        "video_buffer_output" if self.source != "replayer" else "output",
                        input_video_signal,
                    )
                },
            )
        else:
            if renderer == "holoviz":
                visualizer = HolovizOp(
                    self,
                    name="holoviz",
                    width=width,
                    height=height,
                    enable_render_buffer_input=False,
                    enable_render_buffer_output=False,
                    allocator=None,
                    cuda_stream_pool=cuda_stream_pool,
                    **self.kwargs("holoviz"),
                )

            # Flow definition
            self.add_flow(source, visualizer,
                {
                    (
                        "video_buffer_output" if self.source != "replayer" else "output",
                        input_video_signal,
                    )
                },
            )


if __name__ == "__main__":
    # Parse args
    parser = ArgumentParser(description="Preview application.")
    parser.add_argument(
        "-s",
        "--source",
        choices=["replayer", "yuan"],
        default="replayer",
        help=(
            "If 'replayer', replay a prerecorded video. Otherwise use a "
            "capture card as the source (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "-c",
        "--config",
        default="none",
        help=("Set config path to override the default config file location"),
    )
    parser.add_argument(
        "-d",
        "--data",
        default="none",
        help=("Set the data path"),
    )
    args = parser.parse_args()

    if args.config == "none":
        config_file = os.path.join(os.path.dirname(__file__), "quad_preview.yaml")
    else:
        config_file = args.config

    app = PreviewApp(source=args.source, data=args.data)
    app.config(config_file)
    app.run()
