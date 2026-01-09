#pragma once
#include "qcap2.h"
#include "qcap.linux.h"
#include "qcap2.user.h"
#include "qcap2.hsb.h"

#include "qcap_event.hpp"
#include "qcap_video_format.hpp"

#include <cassert>
#include <stdexcept>

namespace holoscan::ops {

class QcapVideoSource {
public:
    QcapVideoSource()
        : pSource(nullptr),
        bIsStart(false) {
        pSource = qcap2_video_source_new();
    }

    ~QcapVideoSource() {
        stop();

        if (pSource) {
            qcap2_video_source_delete(pSource);
        }
    }

    qcap2_video_source_t* get() {
        assert(pSource != nullptr);
        return pSource;
    }   

    void start() {
        assert(pSource != nullptr);
        QRESULT qres = qcap2_video_source_start(pSource);
        if(qres != QCAP_RS_SUCCESSFUL) {
            throw std::runtime_error("QcapVideoSource: Failed to start qcap");
        }
        bIsStart = true;
    }

    void stop() {
        assert(pSource != nullptr);
        if (bIsStart) {
            QRESULT qres = qcap2_video_source_stop(pSource);
            if(qres != QCAP_RS_SUCCESSFUL) {
                throw std::runtime_error("QcapVideoSource: Failed to stop qcap");
            }
        }
        bIsStart = false;
    }

    void set_backend_type(int nBackendType) {
        assert(pSource != nullptr);
        qcap2_video_source_set_backend_type(pSource, nBackendType);
    }

    void set_event(std::shared_ptr<QcapEvent> pEvent) {
        assert(pSource != nullptr);
        qcap2_video_source_set_event(pSource, pEvent->get());
    }

    void set_video_format(std::shared_ptr<QcapVideoFormat> pFormat) {
        assert(pSource != nullptr);
        qcap2_video_source_set_video_format(pSource, pFormat->get());
    }

    void set_frame_count(int nFrameCount) {
        assert(pSource != nullptr);
        qcap2_video_source_set_frame_count(pSource, nFrameCount);
    }

    void set_device_ordinal(int nDeviceOrdinal) {
        assert(pSource != nullptr);
        qcap2_video_source_set_device_ordinal(pSource, nDeviceOrdinal);
    }

    void set_hololink_ip(const char* strHololinkIP) {
        assert(pSource != nullptr);
        qcap2_video_source_set_hololink_ip(pSource, strHololinkIP);
    }

    void set_ibv_name(const char* strIBVName) {
        assert(pSource != nullptr);
        qcap2_video_source_set_ibv_name(pSource, strIBVName);
    }

    void set_ibv_port(uint32_t nIBVPort) {
        assert(pSource != nullptr);
        qcap2_video_source_set_ibv_port(pSource, nIBVPort);
    }

    std::shared_ptr<QcapRCBuffer> pop_video() {
        qcap2_rcbuffer_t* pRCBuffer;
        QRESULT qres = qcap2_video_source_pop(pSource, &pRCBuffer);
        if(qres != QCAP_RS_SUCCESSFUL) {
            throw std::runtime_error("QcapVideoSource: Failed to pop video");
        }
        return std::make_shared<QcapRCBuffer>(pRCBuffer);
    }

private:
    qcap2_video_source_t* pSource;
    bool bIsStart;
};

} // namespace hololink::ops
