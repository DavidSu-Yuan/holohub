#pragma once
#include "qcap2.h"
#include "qcap.linux.h"
#include "qcap2.user.h"

#include <cassert>
#include <stdexcept>

namespace holoscan::ops {

class QcapEvent {
public:
    QcapEvent()
        : pEvent(nullptr),
        bIsStart(false) {
        pEvent = qcap2_event_new();
    }

    ~QcapEvent() {
        stop();

        if (pEvent) {
            qcap2_event_delete(pEvent);
        }
    }

    void start() {
        assert(pEvent != nullptr);
        QRESULT qres = qcap2_event_start(pEvent);
        if(qres != QCAP_RS_SUCCESSFUL) {
            throw std::runtime_error("QcapEvent: Failed to start qcap");
        }
        bIsStart = true;
    }

    void stop() {
        assert(pEvent != nullptr);
        if (bIsStart) {
            QRESULT qres = qcap2_event_stop(pEvent);
            if(qres != QCAP_RS_SUCCESSFUL) {
                throw std::runtime_error("QcapEvent: Failed to stop qcap");
            }
        }
        bIsStart = false;
    }

    qcap2_event_t* get() {
        assert(pEvent != nullptr);
        return pEvent;
    }   

private:
    qcap2_event_t* pEvent;
    bool bIsStart;
};

} // namespace hololink::ops
