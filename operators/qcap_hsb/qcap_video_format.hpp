#pragma once
#include "qcap2.h"
#include "qcap.linux.h"
#include "qcap2.user.h"

#include <cassert>

namespace holoscan::ops {

class QcapVideoFormat {
public:
    QcapVideoFormat()
        : pFormat(nullptr) {
        pFormat = qcap2_video_format_new();
    }

    ~QcapVideoFormat() {
        if (pFormat) {
            qcap2_video_format_delete(pFormat);
        }
    }

    qcap2_video_format_t* get() {
        assert(pFormat != nullptr);
        return pFormat;
    }   

    void set_property(ULONG nColorSpaceType,
            ULONG nVideoWidth, ULONG nVideoHeight,
            BOOL bVideoIsInterleaved = false, double dVideoFrameRate = 60.0) {
        assert(pFormat != nullptr);

        qcap2_video_format_set_property(pFormat,
                nColorSpaceType, nVideoWidth,
                bVideoIsInterleaved ? nVideoHeight / 2 : nVideoHeight,
                bVideoIsInterleaved, dVideoFrameRate);
    }

private:
    qcap2_video_format_t* pFormat;
};

} // namespace hololink::ops
