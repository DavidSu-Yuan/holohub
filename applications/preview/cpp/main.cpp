/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <getopt.h>

#include "holoscan/holoscan.hpp"
#include <holoscan/operators/format_converter/format_converter.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <holoscan/operators/video_stream_replayer/video_stream_replayer.hpp>

#ifdef YUAN_QCAP
#include <qcap_source.hpp>
#endif

class App : public holoscan::Application {
 public:
  void set_source(const std::string& source) { source_ = source; }
  void set_visualizer_name(const std::string& visualizer_name) {
    this->visualizer_name = visualizer_name;
  }

  enum class Record { NONE, INPUT, VISUALIZER };

  void set_datapath(const std::string& path) { datapath = path; }

  void compose() override {
    using namespace holoscan;

    std::shared_ptr<Operator> source;
    std::shared_ptr<Operator> visualizer_operator;

    const bool use_rdma = from_config("external_source.rdma").as<bool>();
    const bool overlay_enabled = (source_ != "replayer") && (this->visualizer_name == "holoviz") &&
                                 from_config("external_source.enable_overlay").as<bool>();

    const std::string input_video_signal =
        this->visualizer_name == "holoviz" ? "receivers" : "videostream";
    const std::string input_annotations_signal =
        this->visualizer_name == "holoviz" ? "receivers" : "annotations";

    uint32_t width = 0;
    uint32_t height = 0;
    uint64_t source_block_size = 0;
    uint64_t source_num_blocks = 0;

    if (source_ == "yuan") {
      width = from_config("yuan.width").as<uint32_t>();
      height = from_config("yuan.height").as<uint32_t>();
#ifdef YUAN_QCAP
      source = make_operator<ops::QCAPSourceOp>("yuan", from_config("yuan"));
#endif
      source_block_size = width * height * 4 * 4;
      source_num_blocks = use_rdma ? 3 : 4;
    } else {  // Replayer
      width = 854;
      height = 480;
      source = make_operator<ops::VideoStreamReplayerOp>(
          "replayer", from_config("replayer"), Arg("directory", datapath));
      source_block_size = width * height * 3 * 4;
      source_num_blocks = 2;
    }

    std::shared_ptr<BlockMemoryPool> visualizer_allocator;
    if ((record_type_ == Record::VISUALIZER) && source_ == "replayer") {
        visualizer_allocator =
	    make_resource<BlockMemoryPool>("allocator", 1, source_block_size, source_num_blocks);
    }

    const std::shared_ptr<CudaStreamPool> cuda_stream_pool =
        make_resource<CudaStreamPool>("cuda_stream", 0, 0, 0, 1, 5);

    visualizer_operator = make_operator<ops::HolovizOp>(
        "holoviz",
	from_config(overlay_enabled ? "holoviz_overlay" : "holoviz"),
	Arg("width") = width,
	Arg("height") = height,
	Arg("enable_render_buffer_input") = false,
	Arg("enable_render_buffer_output") = false,
	Arg("allocator") = visualizer_allocator,
	Arg("cuda_stream_pool") = cuda_stream_pool);

    std::string output_signal = "output";  // replayer output signal name
    if (source_ == "yuan") {
      output_signal = "video_buffer_output";
    }

    add_flow(source, visualizer_operator, {{output_signal, input_video_signal}});
  }

 private:
  std::string source_ = "replayer";
  std::string visualizer_name = "holoviz";
  Record record_type_ = Record::NONE;
  std::string datapath = "data/endoscopy";
};

/** Helper function to parse the command line arguments */
bool parse_arguments(int argc, char** argv, std::string& config_name, std::string& data_path) {
  static struct option long_options[] = {{"data", required_argument, 0, 'd'}, {0, 0, 0, 0}};

  while (int c = getopt_long(argc, argv, "d", long_options, NULL)) {
    if (c == -1 || c == '?') break;

    switch (c) {
      case 'd':
        data_path = optarg;
        break;
      default:
        std::cout << "Unknown arguments returned: " << c << std::endl;
        return false;
    }
  }

  if (optind < argc) { config_name = argv[optind++]; }
  return true;
}

/** Main function */
int main(int argc, char** argv) {
  auto app = holoscan::make_application<App>();

  // Parse the arguments
  std::string data_path = "";
  std::string config_name = "";
  if (!parse_arguments(argc, argv, config_name, data_path)) { return 1; }

  if (config_name != "") {
    app->config(config_name);
  } else {
    auto config_path = std::filesystem::canonical(argv[0]).parent_path();
    config_path += "/preview.yaml";
    app->config(config_path);
  }

  auto source = app->from_config("source").as<std::string>();
  app->set_source(source);

  auto visualizer_name = app->from_config("visualizer").as<std::string>();
  app->set_visualizer_name(visualizer_name);

  if (data_path != "") app->set_datapath(data_path);

  app->run();

  return 0;
}
