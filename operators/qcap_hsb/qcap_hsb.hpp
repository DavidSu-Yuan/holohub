
#pragma once

#include "qcap_worker.hpp"
#include "qcap_buffer.hpp"
#include "qcap_event.hpp"
#include "qcap_event_handler.hpp"
#include "qcap_video_source.hpp"
#include "qcap_thread_safe_map.hpp"
#include "qcap_thread_safe_queue.hpp"

#include <holoscan/holoscan.hpp>

#include <cassert>
#include <memory>
#include <stdexcept>

// forward declarations
struct qcap2_video_source_t;

namespace holoscan::ops {

class QcapHSBOp : public holoscan::Operator {
public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(QcapHSBOp)
    constexpr static uint32_t VERSION = 1;

    int get_version();

    void setup(holoscan::OperatorSpec& spec) override;
    void start() override;
    void stop() override;
    void compute(holoscan::InputContext& op_input,
        holoscan::OutputContext& op_output,
        holoscan::ExecutionContext& context) override;

    // those functions design to call on event handler
    QRETURN start_video();
    QRETURN stop_video();
    QRETURN receive_video(std::shared_ptr<QcapVideoSource> pVsrc);

    nvidia::gxf::Expected<void> buffer_release_callback(void* pointer);
private:
    holoscan::Parameter<std::string> hololink_ip_;
    holoscan::Parameter<std::string> hololink_mac_;
    holoscan::Parameter<std::string> ibv_name_;
    holoscan::Parameter<uint32_t> ibv_port_;
    holoscan::Parameter<ULONG> nVideoWidth;
    holoscan::Parameter<ULONG> nVideoHeight;

    std::shared_ptr<QcapEventHandler> handler_ = nullptr;
    std::shared_ptr<QcapEvent> video_event_ = nullptr;;
    std::shared_ptr<QcapVideoSource> video_source_ = nullptr;

    ThreadSafeQueue<std::shared_ptr<QcapRCBuffer>> queue_;
    ThreadSafeMap<void*, std::shared_ptr<QcapRCBuffer>> pending_buffers_;
    std::mutex pending_buffers_mutex_;

    ULONG nColorSpaceType = 0;
    BOOL bVideoIsInterleaved = false;
    double dVideoFrameRate = 60;
};

} // namespace hololink::ops
