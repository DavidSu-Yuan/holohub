/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 YUAN High-Tech Development Co., Ltd. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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
#include "qcap_source.hpp"

#include <qcap.common.h>
#include <qcap.h>
#include <qcap.linux.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <npp.h>

#include <sstream>
#include <string>
#include <utility>

#include "gxf/multimedia/video.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#define EXPAND_FILE(filename)                                                 \
  extern const char _binary_##filename##_start[], _binary_##filename##_end[]; \
  const char* filename##_ptr = _binary_##filename##_start;                    \
  const char* filename##_end = _binary_##filename##_end;                      \
  size_t filename##_size = filename##_end - filename##_ptr;

EXPAND_FILE(no_device_png)
EXPAND_FILE(no_signal_png)
EXPAND_FILE(no_sdk_png)

extern cudaError_t convert_YUYV_10c_RGB_8s_C2C1R(
       const void* pSrc, int srcStep,
       void* pDst, int dstStep, int nWidth, int nHeight);

extern cudaError_t convert_YUYV_10c_RGB_8s_C2C1R_sqd(
        const int header_size, const void* pSrc, const int srcStep,
        void* pDst, const int dstStep, const int nWidth, const int nHeight);

namespace nvidia {
namespace holoscan {

QRETURN on_process_signal_removed(PVOID pDevice, ULONG nVideoInput, ULONG nAudioInput,
                                  PVOID pUserData) {
  struct QCAPSource* qcap = (struct QCAPSource*)pUserData;

  GXF_LOG_INFO("QCAP Source: signal removed \n");

  if (qcap->input_type_ == INPUTTYPE_AUTO
    && qcap->m_autoDetectState == STATE_DETECTED) {
    qcap->m_autoDetectState = STATE_AUTO;
    qcap->m_needToChangeInputType = true;
  }

  qcap->m_status = STATUS_SIGNAL_REMOVED;
  qcap->m_queue.signal(false);

  return QCAP_RT_OK;
}

QRETURN on_process_no_signal_detected(PVOID pDevice, ULONG nVideoInput, ULONG nAudioInput,
                                      PVOID pUserData) {
  struct QCAPSource* qcap = (struct QCAPSource*)pUserData;

  GXF_LOG_INFO("QCAP Source: no signal Detected \n");

  if (qcap->input_type_ == INPUTTYPE_AUTO
    && qcap->m_autoDetectState == STATE_DETECTED) {
    qcap->m_autoDetectState = STATE_AUTO;
    qcap->m_needToChangeInputType = true;
  }

  qcap->m_status = STATUS_NO_SIGNAL;
  qcap->m_queue.signal(false);

  return QCAP_RT_OK;
}

inline unsigned long getVideoSizeByPixelFormat(uint32_t pixel_format_, unsigned long video_width, unsigned long video_height) {
      unsigned long video_size = 0;
      switch (pixel_format_) {
          case PIXELFORMAT_YUY2:
              video_size = video_width * video_height * 2;
              break;
          case PIXELFORMAT_BGR24:
              video_size = video_width * video_height * 3;
              break;
          case PIXELFORMAT_Y210:
              video_size = video_width * video_height * 10 / 8 * 2; // bits
              break;
          case PIXELFORMAT_NV12:
              video_size = video_width * video_height * 3 / 2;
              break;
      }
      //GXF_LOG_INFO("QCAP Source: getVideoSizeByPixelFormat %08x %ldx%ld to %ld\n", pixel_format_, video_width, video_height, video_size);
      return video_size;
}

const char* VideoInputTypeStr(ULONG nVideoInput) {
    const char* _str = " ";
    switch (nVideoInput) {
        case 0:
            _str = "COMPOSITE";
            break;
        case 1:
            _str = "SVIDEO";
            break;
        case 2:
            _str = "HDMI";
            break;
        case 3:
            _str = "DVI_D";
            break;
        case 4:
            _str = "COMPONENTS (YCBCR)";
            break;
        case 5:
            _str = "DVI_A (RGB / VGA)";
            break;
        case 6:
            _str = "SDI";
            break;
        case 7:
            _str = "AUTO";
            break;
        default:
            _str = "UNKNOW VIDEO";
            break;
    }
    return _str;
}

const char* AudioInputTypeStr(ULONG nAudioInput) {
    const char* _str = " ";
    switch (nAudioInput) {
        case 0:
            _str = "EMBEDDED_AUDIO";
            break;
        case 1:
            _str = "LIINE_IN";
            break;
        case 2:
            _str = "SOUNDCARD_MICROPHONE";
            break;
        case 3:
            _str = "SOUNDCARD_LINE_IN";
            break;
        default:
            _str = "UNKNOW AUDIO";
            break;
    }
    return _str;
}

QRETURN on_process_format_changed(PVOID pDevice, ULONG nVideoInput, ULONG nAudioInput,
                                  ULONG nVideoWidth, ULONG nVideoHeight, BOOL bVideoIsInterleaved,
                                  double dVideoFrameRate, ULONG nAudioChannels,
                                  ULONG nAudioBitsPerSample, ULONG nAudioSampleFrequency,
                                  PVOID pUserData) {
  struct QCAPSource* qcap = (struct QCAPSource*)pUserData;

  // GXF_LOG_INFO("QCAP Source: format changed Detected");

  qcap->m_nVideoWidth = nVideoWidth;
  qcap->m_nVideoHeight = nVideoHeight;
  qcap->m_bVideoIsInterleaved = bVideoIsInterleaved;
  qcap->m_dVideoFrameRate = dVideoFrameRate;
  qcap->m_nAudioChannels = nAudioChannels;
  qcap->m_nAudioBitsPerSample = nAudioBitsPerSample;
  qcap->m_nAudioSampleFrequency = nAudioSampleFrequency;
  qcap->m_nVideoInput = nVideoInput;
  qcap->m_nAudioInput = nAudioInput;

  GXF_LOG_INFO(
      "QCAP Source: INFO %ld x %ld%s @%2.3f FPS,"
      " %ld CH x %ld BITS x %ld HZ, VIDEO INPUT: %s, AUDIO INPUT: %s",
      nVideoWidth,
      bVideoIsInterleaved == TRUE ? nVideoHeight / 2 : nVideoHeight,
      bVideoIsInterleaved == TRUE ? " I " : " P ",
      dVideoFrameRate,
      nAudioChannels,
      nAudioBitsPerSample,
      nAudioSampleFrequency,
      VideoInputTypeStr(nVideoInput),
      AudioInputTypeStr(nAudioInput));

  unsigned long video_size = getVideoSizeByPixelFormat(qcap->pixel_format_,
          qcap->m_nVideoWidth, qcap->m_nVideoHeight);
  for (int i = 0; i < kDefaultColorConvertBufferSize; i++) {
      if (qcap->m_cuConvertBuffer[i] != 0) {
          if (cuMemFree(qcap->m_cuConvertBuffer[i]) != CUDA_SUCCESS) {
              throw std::runtime_error("cuMemFree failed.");
          }
          qcap->m_cuConvertBuffer[i] = 0;
      }
  }

  if (qcap->input_type_ == INPUTTYPE_AUTO
    && qcap->m_autoDetectState == STATE_AUTO) {
    qcap->m_autoDetectState = STATE_DETECTED;
    qcap->m_needToChangeInputType = true;
  }

  qcap->m_status = STATUS_SIGNAL_LOCKED;
  qcap->m_queue.signal(true);

  return QCAP_RT_OK;
}

QRETURN on_process_video_preview(PVOID pDevice, double dSampleTime, BYTE* pFrameBuffer,
                                 ULONG nFrameBufferLen, PVOID pUserData) {
  struct QCAPSource* qcap = (struct QCAPSource*)pUserData;

  PVOID pRCBuffer = QCAP_BUFFER_GET_RCBUFFER(pFrameBuffer, nFrameBufferLen);
  QCAP_RCBUFFER_ADD_REF(pRCBuffer);

  PreviewFrame preview;
  preview.pFrameBuffer = pFrameBuffer;
  preview.nFrameBufferLen = nFrameBufferLen;

  qcap->m_queue.push_and_drop(preview, [](PreviewFrame drop) {
    PVOID pRCBuffer = QCAP_BUFFER_GET_RCBUFFER(drop.pFrameBuffer, drop.nFrameBufferLen);
    QCAP_RCBUFFER_RELEASE(pRCBuffer);
  });

  return QCAP_RT_OK;
}

QRETURN on_process_audio_preview(PVOID pDevice, double dSampleTime, BYTE* pFrameBuffer,
                                 ULONG nFrameBufferLen, PVOID pUserData) {
  return QCAP_RT_OK;
}

QCAPSource::QCAPSource()
    : pixel_format_(kDefaultPixelFormat),
      output_pixel_format_(kDefaultOutputPixelFormat),
      input_type_(kDefaultInputType) {}

gxf_result_t QCAPSource::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;
  result &= registrar->parameter(video_buffer_output_,
                                 "video_buffer_output",
                                 "VideoBufferOutput",
                                 "Output for the video buffer.");
  result &= registrar->parameter(
      device_specifier_, "device", "Device", "Device specifier.", std::string(kDefaultDevice));
  result &=
      registrar->parameter(channel_, "channel", "Channel", "Channel to use.", kDefaultChannel);
  result &= registrar->parameter(width_, "width", "Width", "Width of the stream.", kDefaultWidth);
  result &=
      registrar->parameter(height_, "height", "Height", "Height of the stream.", kDefaultHeight);
  result &= registrar->parameter(
      framerate_, "framerate", "Framerate", "Framerate of the stream.", kDefaultFramerate);
  result &= registrar->parameter(use_rdma_, "rdma", "RDMA", "Enable RDMA.", kDefaultRDMA);

  result &= registrar->parameter(pixel_format_str_,
                                 "pixel_format",
                                 "PixelFormat",
                                 "Pixel Format.",
                                 std::string(kDefaultPixelFormatStr));

  result &= registrar->parameter(
      input_type_str_, "input_type", "InputType", "Input Type.", std::string(kDefaultInputTypeStr));

  result &= registrar->parameter(mst_mode_,
                                 "mst_mode",
                                 "DisplayPortMSTMode",
                                 "Display port MST mode.",
                                 kDefaultDisplayPortMstMode);

  result &= registrar->parameter(
      sdi12g_mode_, "sdi12g_mode", "SDI12GMode", "SDI 12G Mode.", kDefaultSDI12GMode);

  m_status = STATUS_NO_DEVICE;

  return gxf::ToResultCode(result);
}

void QCAPSource::loadImage(const char* filename, const unsigned char* buffer, const size_t size,
                           struct Image* image) {
  if (image == nullptr) {
    GXF_LOG_INFO("QCAP Source: invalid parameter, image is null\n");
    return;
  }

  // Init
  image->width = 0;
  image->height = 0;
  image->components = 0;
  image->data = nullptr;
  image->cu_src = 0;
  image->cu_dst = 0;

  // Loading
  image->data = stbi_load_from_memory(buffer,
                                      size,
                                      reinterpret_cast<int*>(&image->width),
                                      reinterpret_cast<int*>(&image->height),
                                      &image->components,
                                      0);

  if (image->data == nullptr) {
    GXF_LOG_INFO("QCAP Source: load image %s fail", filename);
    return;
  }

  GXF_LOG_INFO("QCAP Source: load image %s %dx%d %d",
               filename,
               image->width,
               image->height,
               image->components);

  // memset(image->data, 128, image->width * image->height * image->components);

  if (image->components == 4) {
    int width = image->width;
    int height = image->height;

    if (cuMemAlloc(&image->cu_src, width * height * 4) != CUDA_SUCCESS) {
      throw std::runtime_error("cuMemAlloc failed.");
    }
    if (cuMemAlloc(&image->cu_dst, width * height * 3) != CUDA_SUCCESS) {
      throw std::runtime_error("cuMemAlloc failed.");
    }
    if (cuMemcpyHtoD(image->cu_src, image->data, width * height * 4) != CUDA_SUCCESS) {
      throw std::runtime_error("cuMemcpyHtoD failed.");
    }

    if (output_pixel_format_ == PIXELFORMAT_RGB24) {  // RGBA to RGB
      NppStatus status;
      NppiSize oSizeROI;
      int video_width = width;
      int video_height = height;
      oSizeROI.width = video_width;
      oSizeROI.height = video_height;
      const int aDstOrder[3] = {0, 1, 2};
      status = nppiSwapChannels_8u_C4C3R((Npp8u*)image->cu_src,
                                         video_width * 4,
                                         (Npp8u*)image->cu_dst,
                                         video_width * 3,
                                         oSizeROI,
                                         aDstOrder);
      if (status != 0) {
        GXF_LOG_INFO(
            "QCAP Source: image convert error %d %dx%d", status, video_width, video_height);
      }
    }
  }
}

void QCAPSource::destroyImage(struct Image* image) {
  if (image->cu_src && cuMemFree(image->cu_src) != CUDA_SUCCESS) {
    throw std::runtime_error("cuMemFree failed.");
  }
  if (image->cu_dst && cuMemFree(image->cu_dst) != CUDA_SUCCESS) {
    throw std::runtime_error("cuMemFree failed.");
  }
}

void QCAPSource::initCuda() {
  if (cuInit(0) != CUDA_SUCCESS) { throw std::runtime_error("cuInit failed."); }

  if (cuDevicePrimaryCtxRetain(&m_CudaContext, 0) != CUDA_SUCCESS) {
    throw std::runtime_error("cuDevicePrimaryCtxRetain failed.");
  }

  if (cuCtxPushCurrent(m_CudaContext) != CUDA_SUCCESS) {
    throw std::runtime_error("cuDevicePrimaryCtxRetain failed.");
  }
}

void QCAPSource::cleanupCuda() {
  if (m_CudaContext) {
    if (cuCtxPopCurrent(&m_CudaContext) != CUDA_SUCCESS) {
      throw std::runtime_error("cuCtxPopCurrent failed.");
    }
    m_CudaContext = nullptr;

    if (cuDevicePrimaryCtxRelease(0) != CUDA_SUCCESS) {
      throw std::runtime_error("cuDevicePrimaryCtxRelease failed.");
    }
  }
}

void QCAPSource::configureInput() {
  ULONG nSelectedInputType = QCAP_INPUT_TYPE_AUTO;
  // QCAP_SET_AUDIO_SOUND_RENDERER(m_hDevice, 0);
  if (input_type_ == INPUTTYPE_AUTO) {
      if (m_autoDetectState == STATE_AUTO) {
          nSelectedInputType = QCAP_INPUT_TYPE_AUTO;
      } else if (m_autoDetectState == STATE_DETECTED) {
          nSelectedInputType = m_nVideoInput;
      }
  } else {
    m_autoDetectState = STATE_FORCED;
    if (input_type_ == INPUTTYPE_DVI_D) {
        nSelectedInputType = QCAP_INPUT_TYPE_DVI_D;
    } else if (input_type_ == INPUTTYPE_DISPLAY_PORT) {
      GXF_LOG_INFO("QCAP Source: DP MST mode is %d", mst_mode_.get());
      if (mst_mode_ == DISPLAYPORT_MST_MODE) {
        nSelectedInputType = QCAP_INPUT_TYPE_DISPLAY_PORT_MST;
      } else {
        nSelectedInputType = QCAP_INPUT_TYPE_DISPLAY_PORT_SST;
      }
    } else if (input_type_ == INPUTTYPE_SDI) {
      nSelectedInputType = QCAP_INPUT_TYPE_SDI;
    } else if (input_type_ == INPUTTYPE_HDMI) {
      nSelectedInputType = QCAP_INPUT_TYPE_HDMI;
    } else {  // INPUTTYPE_AUTO or Default
      /* do nothing, we don't change input type. Let driver select it. */
    }
  }
  GXF_LOG_INFO("QCAP Source: Current Input type is %s(%ld) state %d",
          VideoInputTypeStr(nSelectedInputType), nSelectedInputType,
          m_autoDetectState);
  QCAP_SET_VIDEO_INPUT(m_hDevice, nSelectedInputType);

  ULONG nVideoInput = 0;
  QCAP_GET_VIDEO_INPUT(m_hDevice, &nVideoInput);
  GXF_LOG_INFO("QCAP Source: Use input %lu", nVideoInput);
  if (input_type_ == INPUTTYPE_AUTO) {
      pixel_format_ = PIXELFORMAT_YUY2;
  } else if (nVideoInput == QCAP_INPUT_TYPE_SDI) {
    GXF_LOG_INFO("QCAP Source: SDI 12G mode is %d", sdi12g_mode_.get());

    if (sdi12g_mode_.get() != SDI12G_DEFAULT_MODE) {
      int qcap_sdi_mode = (sdi12g_mode_.get() == SDI12G_QUADLINK_MODE ? 0 : 1);
      QCAP_SET_DEVICE_CUSTOM_PROPERTY(m_hDevice, QCAP_DEVPROP_SDI12G_MODE, qcap_sdi_mode);
      QCAP_SET_VIDEO_INPUT(m_hDevice, QCAP_INPUT_TYPE_SDI);
    }

    // Workaround
    if (pixel_format_ == PIXELFORMAT_BGR24) {
      GXF_LOG_INFO("QCAP Source: SDI only support YUY2 or NV12, switch to yuy2");
      pixel_format_ = PIXELFORMAT_YUY2;
    }
  }
  QCAP_SET_VIDEO_DEFAULT_OUTPUT_FORMAT(m_hDevice, pixel_format_, 0, 0, 0, 0);

  if (m_autoDetectState == STATE_DETECTED || m_autoDetectState == STATE_FORCED) {
    QCAP_REGISTER_VIDEO_PREVIEW_CALLBACK(m_hDevice, on_process_video_preview, this);
    QCAP_REGISTER_AUDIO_PREVIEW_CALLBACK(m_hDevice, on_process_audio_preview, this);
  } else {
    QCAP_REGISTER_VIDEO_PREVIEW_CALLBACK(m_hDevice, nullptr, this);
    QCAP_REGISTER_AUDIO_PREVIEW_CALLBACK(m_hDevice, nullptr, this);
  }
}

gxf_result_t QCAPSource::start() {

  if (pixel_format_str_.get().compare("yuy2") == 0) {
    pixel_format_ = PIXELFORMAT_YUY2;
  } else if (pixel_format_str_.get().compare("nv12") == 0) {
    pixel_format_ = PIXELFORMAT_NV12;
  } else if (pixel_format_str_.get().compare("y210") == 0) {
    pixel_format_ = PIXELFORMAT_Y210;
  } else {
    pixel_format_ = PIXELFORMAT_BGR24;
  }

  if (input_type_str_.get().compare("dvi_d") == 0) {
    input_type_ = INPUTTYPE_DVI_D;
  } else if (input_type_str_.get().compare("dp") == 0) {
    input_type_ = INPUTTYPE_DISPLAY_PORT;
  } else if (input_type_str_.get().compare("sdi") == 0) {
    input_type_ = INPUTTYPE_SDI;
  } else if (input_type_str_.get().compare("hdmi") == 0) {
    input_type_ = INPUTTYPE_HDMI;
  } else {
    input_type_ = INPUTTYPE_AUTO;
  }

  GXF_LOG_INFO("QCAP Source: Using channel %d", (channel_.get() + 1));
  GXF_LOG_INFO("QCAP Source: RDMA is %s", use_rdma_ ? "enabled" : "disabled");
  GXF_LOG_INFO("QCAP Source: Resolution %dx%d", width_.get(), height_.get());
  GXF_LOG_INFO(
      "QCAP Source: Pixel format is %s (%d)", pixel_format_str_.get().c_str(), pixel_format_);
  GXF_LOG_INFO("QCAP Source: Input type is %s (%d)", input_type_str_.get().c_str(), input_type_);

  initCuda();

  loadImage(
      "no_device.png", (unsigned char*)no_device_png_ptr, no_device_png_size, &m_iNoDeviceImage);
  loadImage(
      "no_signal.png", (unsigned char*)no_signal_png_ptr, no_signal_png_size, &m_iNoSignalImage);
  loadImage("no_sdk.png", (unsigned char*)no_sdk_png_ptr, no_sdk_png_size, &m_iNoSdkImage);
  // loadImage("signal_remove.png",
  //          (unsigned char*)signal_remove_png_ptr,
  //          signal_remove_png_size,
  //          &m_iSignalRemovedImage);

  for (int i = 0; i < kDefaultColorConvertBufferSize; i++) {
    cudaMalloc((void**)&m_pRGBBUffer[i], kDefaultPreviewSize);
  }

  m_status = STATUS_NO_SDK;
  m_autoDetectState = STATE_AUTO;
  m_hDevice = nullptr;
  m_bHasSignal = false;
  m_nVideoWidth = 0;
  m_nVideoHeight = 0;
  m_bVideoIsInterleaved = false;
  m_dVideoFrameRate = 0.0f;
  m_nAudioChannels = 0;
  m_nAudioBitsPerSample = 0;
  m_nAudioSampleFrequency = 0;
  m_nVideoInput = 0;
  m_nAudioInput = 0;
  m_needToChangeInputType = false;

  QCAP_CREATE((char*)device_specifier_.get().c_str(), 0, nullptr, &m_hDevice, TRUE);

  QCAP_SET_DEVICE_CUSTOM_PROPERTY(m_hDevice, QCAP_DEVPROP_IO_METHOD, 1);
  QCAP_SET_DEVICE_CUSTOM_PROPERTY(m_hDevice, QCAP_DEVPROP_VO_BACKEND, 2);

  QCAP_REGISTER_NO_SIGNAL_DETECTED_CALLBACK(m_hDevice, on_process_no_signal_detected, this);
  QCAP_REGISTER_SIGNAL_REMOVED_CALLBACK(m_hDevice, on_process_signal_removed, this);
  QCAP_REGISTER_FORMAT_CHANGED_CALLBACK(m_hDevice, on_process_format_changed, this);

  configureInput();

  if (use_rdma_) {
    for (int i = 0; i < kDefaultGPUDirectRingQueueSize; i++) {
      QCAP_ALLOC_VIDEO_GPUDIRECT_PREVIEW_BUFFER(m_hDevice, &m_pGPUDirectBuffer[i], kDefaultPreviewSize);
      QCAP_BIND_VIDEO_GPUDIRECT_PREVIEW_BUFFER(
              m_hDevice, i, m_pGPUDirectBuffer[i], kDefaultPreviewSize);
      GXF_LOG_INFO("QCAP Source: Allocate gpu buffer id:%d, pointer:%p size:%d",
              i,
              m_pGPUDirectBuffer[i],
              kDefaultPreviewSize);
    }
  }

  QCAP_RUN(m_hDevice);

  return GXF_SUCCESS;
}

gxf_result_t QCAPSource::stop() {
  if (m_hDevice) {
    QCAP_STOP(m_hDevice);

    m_queue.quit();

    if (use_rdma_) {
      for (int i = 0; i < kDefaultGPUDirectRingQueueSize; i++) {
        QCAP_UNBIND_VIDEO_GPUDIRECT_PREVIEW_BUFFER(
            m_hDevice, i, m_pGPUDirectBuffer[i], kDefaultPreviewSize);
        QCAP_FREE_VIDEO_GPUDIRECT_PREVIEW_BUFFER(m_hDevice, m_pGPUDirectBuffer[i],
            kDefaultPreviewSize);
        cudaFree((void**)&m_pGPUDirectBuffer[i]);
      }
    }

    for (int i = 0; i < kDefaultColorConvertBufferSize; i++) { cudaFree((void**)&m_pRGBBUffer[i]); }

    destroyImage(&m_iNoDeviceImage);
    destroyImage(&m_iNoSignalImage);
    destroyImage(&m_iSignalRemovedImage);

    cleanupCuda();

    QCAP_DESTROY(m_hDevice);
    m_hDevice = nullptr;
  }
  return GXF_SUCCESS;
}

gxf_result_t QCAPSource::tick() {
  PreviewFrame preview;

  // Pass the frame downstream.
  auto message = gxf::Entity::New(context());
  if (!message) {
    GXF_LOG_ERROR("QCAP Source: Failed to allocate message; terminating.");
    return GXF_FAILURE;
  }

  auto buffer = message.value().add<gxf::VideoBuffer>();
  if (!buffer) {
    GXF_LOG_ERROR("QCAP Source: Failed to allocate video buffer; terminating.");
    return GXF_FAILURE;
  }

  if (m_needToChangeInputType) {
    // 1. stop qcap first.
    QCAP_STOP(m_hDevice);

    // 2. configure to new input type.
    configureInput();

    // 3. run qcap again.
    QCAP_RUN(m_hDevice);

    // 4. back to normal procedure
    m_needToChangeInputType = false;
    return GXF_SUCCESS;
  }

  // GXF_LOG_ERROR("QCAP Source: status %d in tick", m_status);
  // Show error image
  if (m_status != STATUS_SIGNAL_LOCKED) {
    struct Image* image = nullptr;
    switch (m_status) {
      case STATUS_NO_SDK:
        image = &m_iNoSdkImage;
        break;
      case STATUS_NO_DEVICE:
        image = &m_iNoDeviceImage;
        break;
      case STATUS_NO_SIGNAL:
        image = &m_iNoSignalImage;
        break;
      case STATUS_SIGNAL_REMOVED:
        image = &m_iNoSignalImage;
        break;
    }
    int out_width = image->width;
    int out_height = image->height;
    int out_size = out_width * out_height * (output_pixel_format_ == PIXELFORMAT_RGB24 ? 3 : 4);

    // GXF_LOG_ERROR("QCAP Source: show %d image %dx%d", m_status, out_width, out_height);
    gxf::VideoTypeTraits<gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB> video_type;
    gxf::VideoFormatSize<gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB> color_format;
    auto color_planes = color_format.getDefaultColorPlanes(out_width, out_height);
    color_planes[0].stride = out_width * 3;
    gxf::VideoBufferInfo info{(uint32_t)out_width,
                              (uint32_t)out_height,
                              video_type.value,
                              color_planes,
                              gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR};
    auto storage_type = gxf::MemoryStorageType::kDevice;
    // auto storage_type = gxf::MemoryStorageType::kHost;
    buffer.value()->wrapMemory(info, out_size, storage_type, (void*)image->cu_dst, nullptr);
    // buffer.value()->wrapMemory(info, out_size, storage_type, (void*) image->data, nullptr);
    const auto result = video_buffer_output_->publish(std::move(message.value()));
    return gxf::ToResultCode(message);
  }

  // GXF_LOG_ERROR("QCAP Source: status %d block >>", m_status);
  if (m_queue.pop_block(preview) == false) {
    // GXF_LOG_ERROR("QCAP Source: status %d block <<<", m_status);
    return GXF_SUCCESS;
  }
  // GXF_LOG_ERROR("QCAP Source: status %d block <<", m_status);

  PVOID pRCBuffer = QCAP_BUFFER_GET_RCBUFFER(preview.pFrameBuffer, preview.nFrameBufferLen);
  qcap_av_frame_t* pAVFrame = (qcap_av_frame_t*)QCAP_RCBUFFER_LOCK_DATA(pRCBuffer);

  PVOID frame = pAVFrame->pData[0];
  NppStatus status = NPP_NO_ERROR;
  cudaError_t cuda_status = cudaSuccess;
  NppiSize oSizeROI;
  int video_width = m_nVideoWidth;
  int video_height = m_nVideoHeight;
  oSizeROI.width = video_width;
  oSizeROI.height = video_height;
  m_nRGBBufferIndex = (m_nRGBBufferIndex + 1) % kDefaultColorConvertBufferSize;
  frame = m_pRGBBUffer[m_nRGBBufferIndex];
  auto storage_type = use_rdma_ ? gxf::MemoryStorageType::kDevice : gxf::MemoryStorageType::kHost;

#if 0  // for debug
  struct cudaPointerAttributes attributes;
  if (cudaPointerGetAttributes(&attributes, pAVFrame->pData[0]) != cudaSuccess) {
      throw std::runtime_error("cudaPointerGetAttributes failed.");
  }
  GXF_LOG_INFO("video preview cb frame: %p type: %d\n", pAVFrame->pData[0], attributes.type);
#endif

  unsigned long video_size = getVideoSizeByPixelFormat(pixel_format_,
          m_nVideoWidth, m_nVideoHeight);
  for (int i = 0; i < kDefaultColorConvertBufferSize; i++) {
      if (m_cuConvertBuffer[i] == 0 && video_size != 0) {
          cuMemAlloc(&(m_cuConvertBuffer[i]), video_size);
          //GXF_LOG_INFO("QCAP Source: convert buffer %lld %ld\n", m_cuConvertBuffer[i], video_size);
      }
  }

  m_nConvertBufferIndex = (m_nConvertBufferIndex + 1) % kDefaultColorConvertBufferSize;
  CUdeviceptr convert_buf = m_cuConvertBuffer[m_nConvertBufferIndex];
  unsigned char *video_src = nullptr;
  storage_type = gxf::MemoryStorageType::kDevice;
  if (pixel_format_ == PIXELFORMAT_YUY2 &&
      output_pixel_format_ == PIXELFORMAT_RGB24) {  // YUY2 to RGB
    if (use_rdma_ == false) {
        cuMemcpyHtoD(convert_buf, pAVFrame->pData[0], video_size);
        video_src = (unsigned char*) convert_buf;
    } else {
        video_src = pAVFrame->pData[0];
    }
    status = nppiYCbCr422ToRGB_8u_C2C3R(
        video_src, video_width * 2, (Npp8u*)frame, video_width * 3, oSizeROI);
    storage_type = gxf::MemoryStorageType::kDevice;
  } else if (pixel_format_ == PIXELFORMAT_BGR24 &&
             output_pixel_format_ == PIXELFORMAT_RGB24) {  // Default is BGR. BGR to RGB
    const int aDstOrder[3] = {2, 1, 0};
    if (use_rdma_ == false) {
        cuMemcpyHtoD(convert_buf, pAVFrame->pData[0], video_size);
        video_src = (unsigned char*) convert_buf;
    } else {
        video_src = pAVFrame->pData[0];
    }
    status = nppiSwapChannels_8u_C3R(
        video_src, video_width * 3, (Npp8u*)frame, video_width * 3, oSizeROI, aDstOrder);
    storage_type = gxf::MemoryStorageType::kDevice;
  } else if (pixel_format_ == PIXELFORMAT_Y210 &&
             output_pixel_format_ == PIXELFORMAT_RGB24) {  // Default is BGR. BGR to RGB
    if (use_rdma_ == false) {
        cuMemcpyHtoD(convert_buf, pAVFrame->pData[0], video_size);
        video_src = (unsigned char*) convert_buf;
        cuda_status = convert_YUYV_10c_RGB_8s_C2C1R(
                video_src, video_width / 2 * 5, (Npp8u*)frame, video_width * 3, video_width, video_height);
        storage_type = gxf::MemoryStorageType::kDevice;
    } else {
        video_src = pAVFrame->pData[0];
        if (sdi12g_mode_.get() == SDI12G_QUADLINK_MODE) { // SQD with header
            cuda_status = convert_YUYV_10c_RGB_8s_C2C1R_sqd(
                    10, video_src, (video_width / 4 * 5) + 16, (Npp8u*)frame, video_width * 3, video_width, video_height);
        } else {
            cuda_status = convert_YUYV_10c_RGB_8s_C2C1R(
                    video_src, video_width / 2 * 5, (Npp8u*)frame, video_width * 3, video_width, video_height);
        }
        storage_type = gxf::MemoryStorageType::kDevice;
    }
  } else if (pixel_format_ == PIXELFORMAT_NV12 &&
             output_pixel_format_ == PIXELFORMAT_RGB24) {  // NV12 to RGB
    const int aDstOrder[3] = {2, 1, 0};
    Npp8u* input[2];
    if (use_rdma_ == false) {
        cuMemcpyHtoD(convert_buf, pAVFrame->pData[0], video_size);
        video_src = (unsigned char*) convert_buf;
        input[0] = (Npp8u*)video_src;
        input[1] = (Npp8u*)(video_src + video_width * video_height);
    } else {
        input[0] = (Npp8u*)pAVFrame->pData[0];
        input[1] = (Npp8u*)pAVFrame->pData[1];
    }
    status = nppiNV12ToRGB_8u_P2C3R(input, video_width, (Npp8u*)frame, video_width * 3, oSizeROI);
    storage_type = gxf::MemoryStorageType::kDevice;
  } else {
    status = NPP_ERROR;
  }

  QCAP_RCBUFFER_UNLOCK_DATA(pRCBuffer);
  QCAP_RCBUFFER_RELEASE(pRCBuffer);

  if (status != NPP_NO_ERROR || cuda_status != cudaSuccess) {
    GXF_LOG_INFO("QCAP Source: convert error %d %d buffer %p(%08x) convert %llx %ld to %p(%08x) %dx%d\n",
                 status,
                 cuda_status,
                 pAVFrame->pData[0],
                 pixel_format_,
                 convert_buf,
                 video_size,
                 frame,
                 output_pixel_format_,
                 video_width,
                 video_height);
    return GXF_FAILURE;
  }

  int out_width = m_nVideoWidth;
  int out_height = m_nVideoHeight;
  int out_size = out_width * out_height * 3;

  gxf::VideoTypeTraits<gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB> video_type;
  gxf::VideoFormatSize<gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB> color_format;
  auto color_planes = color_format.getDefaultColorPlanes(out_width, out_height);
  color_planes[0].stride = out_width * 3;
  gxf::VideoBufferInfo info{(uint32_t)out_width,
                            (uint32_t)out_height,
                            video_type.value,
                            color_planes,
                            gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR};
  buffer.value()->wrapMemory(info, out_size, storage_type, frame, nullptr);
  const auto result = video_buffer_output_->publish(std::move(message.value()));

  return gxf::ToResultCode(message);
}

}  // namespace holoscan
}  // namespace nvidia
