#pragma once
#include "qcap2.h"
#include "qcap.linux.h"
#include "qcap2.user.h"

#include "qcap_event.hpp"

#include <cassert>
#include <stdexcept>

namespace holoscan::ops {

class QcapEventHandler {
public:
	struct callback_t {
		typedef callback_t self_t;
		typedef std::function<QRETURN ()> cb_func_t;

		cb_func_t func;

		template<class FUNC>
		callback_t(FUNC func) : func(func) {
		}

		static QRETURN _func(PVOID pUserData) {
			self_t* pThis = (self_t*)pUserData;

			return pThis->func();
		}
	};

    QcapEventHandler() {
        pEventHandlers = qcap2_event_handlers_new();
        if (pEventHandlers == nullptr) {
            throw std::runtime_error("QcapEventHandler: Failed to allocate qcap event handler");
        }
    }

    ~QcapEventHandler() {
        stop();

        if (nHandle) {
            QRESULT qres = qcap2_event_handlers_remove_handler(pEventHandlers, nHandle);
            if(qres != QCAP_RS_SUCCESSFUL) {
                //throw std::runtime_error("QcapEventHandler: Failed to remove event to event handler");
                nHandle = 0;
            }
        }

        if (pEventCallback) {
            delete pEventCallback;
            pEventCallback = nullptr;
        }

        if (pEventHandlers == nullptr) {
            qcap2_event_handlers_delete(pEventHandlers);
            pEventHandlers = nullptr;
        }
    }


    void start() {
        assert(pEventHandlers != nullptr);
        QRESULT qres = qcap2_event_handlers_start(pEventHandlers);
        if(qres != QCAP_RS_SUCCESSFUL) {
            throw std::runtime_error("QcapEventHandler: Failed to start qcap event handler");
        }
        bIsStart = true;
    }

    void stop() {
        assert(pEventHandlers != nullptr);
        if (bIsStart) {
            QRESULT qres = qcap2_event_handlers_stop(pEventHandlers);
            if(qres != QCAP_RS_SUCCESSFUL) {
                throw std::runtime_error("QcapEventHandler: Failed to start qcap event handler");
            }
        }
        bIsStart = false;
    }

    template<class FUNC>
    void execute_function(FUNC func) {
        assert(pEventHandlers != nullptr);
        std::shared_ptr<callback_t> pCallback(new callback_t(func));

        QRESULT qres = qcap2_event_handlers_invoke(pEventHandlers,
                callback_t::_func, pCallback.get());
        if(qres != QCAP_RS_SUCCESSFUL) {
            throw std::runtime_error("QcapEventHandler: Failed to exec function on qcap event handler");
        }
    }

    template<class FUNC>
    void add_event(std::shared_ptr<QcapEvent> pEvent, FUNC func) {
        assert(pEventHandlers != nullptr);
        if (nHandle) {
            throw std::runtime_error("QcapEventHandler: AddEventHandler() can't call twice");
        }

        QRESULT qres = qcap2_event_get_native_handle(pEvent->get(), &nHandle);
        if(qres != QCAP_RS_SUCCESSFUL) {
            throw std::runtime_error("QcapEventHandler: Failed to get handle of qcap event");
        }

        pEventCallback = new callback_t(func);
        qres = qcap2_event_handlers_add_handler(pEventHandlers, nHandle,
                callback_t::_func, pEventCallback);
        if(qres != QCAP_RS_SUCCESSFUL) {
            throw std::runtime_error("QcapEventHandler: Failed to add event to event handler");
        }
    }

private:
    qcap2_event_handlers_t* pEventHandlers = nullptr;
    bool bIsStart = false;
    callback_t* pEventCallback = nullptr;
    std::uintptr_t nHandle = 0;
};

} // namespace hololink::ops
