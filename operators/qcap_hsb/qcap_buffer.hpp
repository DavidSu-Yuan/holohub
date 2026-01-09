#pragma once
#include "qcap2.h"
#include "qcap.linux.h"
#include "qcap2.user.h"
//#include "qcap2.cuda.h"

#include <cassert>
#include <stdexcept>
#include <iostream>

namespace holoscan::ops {

class QcapAVFrame {
public:

    QcapAVFrame(qcap2_av_frame_t* frame)
        : pAVFrame(frame),
        bhasVideoProperty(false),
        bhasCudaBuffer(false),
        bhasSystemBuffer(false) {
    }

    ~QcapAVFrame() {
#if 0
        if (bhasCudaBuffer) {
            qcap2_av_frame_free_cuda_buffer(pAVFrame);
            bhasCudaBuffer = false;
        }
#endif

        if (bhasSystemBuffer) {
            qcap2_av_frame_free_buffer(pAVFrame);
            bhasSystemBuffer = false;
        }

        if (pAVFrame) {
            pAVFrame = nullptr;
        }
    }

    qcap2_av_frame_t* get() {
        assert(pAVFrame != nullptr);
        return pAVFrame;
    }   

    void set_video_property(ULONG nColorSpaceType, ULONG nWidth, ULONG nHeight) {
        assert(pAVFrame != nullptr);
        qcap2_av_frame_set_video_property(pAVFrame, nColorSpaceType, nWidth, nHeight);
        bhasVideoProperty = true;
    }

    void alloc_buffer(int align = 32, int valign = 1) {
        assert(pAVFrame != nullptr);
        if (!bhasVideoProperty) {
            throw std::runtime_error("QCapAVFrame: Failed to allocate system buffer. Must call alloc_buffer after set_video_property");
        }

		qcap2_av_frame_alloc_buffer(pAVFrame, 32, 1);
        bhasSystemBuffer = true;
    }

#if 0
    void alloc_cuda_buffer(int align = 32, int valign = 1) {
        assert(pAVFrame != nullptr);
        if (!bhasVideoProperty) {
            throw std::runtime_error("QCapAVFrame: Failed to allocate cuda buffer. Must call alloc_cuda_buffer after set_video_property");
        }

		qcap2_av_frame_alloc_cuda_buffer(pAVFrame, 32, 1);
        bhasCudaBuffer = true;
    }
#endif

    void get_raw_buffer(uint8_t** ppBuffer, int* pStride) {
        assert(pAVFrame != nullptr);
        qcap2_av_frame_get_buffer(pAVFrame, ppBuffer, pStride);
    }   

private:
    qcap2_av_frame_t* pAVFrame;
    bool bhasVideoProperty;
    bool bhasCudaBuffer;
    bool bhasSystemBuffer;
};

class QcapRCBuffer {
public:
    QcapRCBuffer()
        : pRCBuffer(nullptr),
          nLockCount(0),
          bIsOwner(true) {
        pRCBuffer = qcap2_rcbuffer_new_av_frame();
    }

    QcapRCBuffer(qcap2_rcbuffer_t* buffer)
        : pRCBuffer(buffer),
          nLockCount(0),
          bIsOwner(false) {
    }

    ~QcapRCBuffer() {
        if (nLockCount != 0) {
            std::cout << "[WARN] QcapRCBuffer: please unlock "
                << pRCBuffer <<" before delete. count " << nLockCount << std::endl;
        }

        if (pRCBuffer) {
            if (bIsOwner) {
                qcap2_rcbuffer_delete(pRCBuffer);
            } else {
                qcap2_rcbuffer_release(pRCBuffer);
            }
        }
    }

    qcap2_rcbuffer_t* get() {
        assert(pRCBuffer != nullptr);
        return pRCBuffer;
    }   

    std::shared_ptr<QcapAVFrame> lock() {
        assert(pRCBuffer != nullptr);
        auto frame = static_cast<qcap2_av_frame_t*>(qcap2_rcbuffer_lock_data(pRCBuffer));
        nLockCount++;
        return std::make_shared<QcapAVFrame>(frame);
    }

    void unlock() {
        assert(pRCBuffer != nullptr);
        qcap2_rcbuffer_unlock_data(pRCBuffer);
        nLockCount--;
    }

    std::shared_ptr<QcapAVFrame> get_data() {
        assert(pRCBuffer != nullptr);
        auto buffer = static_cast<qcap2_av_frame_t*>(qcap2_rcbuffer_get_data(pRCBuffer));
        // ToDo: do we need to keep this pointer???
        return std::make_shared<QcapAVFrame>(buffer);
    }

private:
    qcap2_rcbuffer_t* pRCBuffer;
    unsigned int nLockCount;
    bool bIsOwner;
};

} // namespace hololink::ops
