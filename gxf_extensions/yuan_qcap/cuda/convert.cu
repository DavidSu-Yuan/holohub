#include <stdio.h>

typedef unsigned short uint16_t;
__device__ float clamp(float x, float a, float b)
{
    return max(a, min(b, x));
}

__device__ void cvt_yuyv10c_2_rgb8(uchar1* src, uchar1* dst) {
       uchar1 s0 = src[0];
       uchar1 s1 = src[1];
       uchar1 s2 = src[2];
       uchar1 s3 = src[3];
       uchar1 s4 = src[4];

       int y1 = (int(s1.x & 0x03) << 8) |
               int((s0.x & 0xFF));

       int v = (int(s2.x & 0x0F) << 6) |
               int((s1.x & 0xFC) >> 2);

       int y2 = (int(s3.x & 0x3F) << 4) |
               int((s2.x & 0xF0) >> 4);

       int u =(int(s4.x & 0xFF) << 2) |
               int((s3.x & 0xC0) >> 6);

    y1 -= 64;
    y2 -= 64;
    u -= 512;
    v -= 512;

    dst[2].x = clamp(((y1 * 1.164384) + (v * 1.792741)) / 4, 16, 235);
    dst[1].x = clamp(((y1 * 1.164384) - (u * 0.213249) - (v * 0.532909)) / 4, 16, 240);
    dst[0].x = clamp(((y1 * 1.164384) + (u * 2.112402)) / 4, 16, 240);

    dst[5].x = clamp(((y2 * 1.164384) + (v * 1.792741)) / 4, 16, 235);
    dst[4].x = clamp(((y2 * 1.164384) - (u * 0.213249) - (v * 0.532909)) / 4, 16, 240);
    dst[3].x = clamp(((y2 * 1.164384) + (u * 2.112402)) / 4, 16, 240);
}

__global__ void kernel_YUYV_10c_RGB_8s_C2C1R(uchar1* pSrc, int srcStep,
       uchar1* pDst, int dstStep, int nWidth, int nHeight) {
       int x = blockIdx.x * blockDim.x + threadIdx.x;
       int y = blockIdx.y * blockDim.y + threadIdx.y;

       if(x < nWidth && y < nHeight) {
               int nSrcIdx = y * srcStep + x * 5;
               int nDstIdx = y * dstStep + x * 6;

               cvt_yuyv10c_2_rgb8(&pSrc[nSrcIdx + 0], (uchar1*)&pDst[nDstIdx + 0]);
       }
}

cudaError_t convert_YUYV_10c_RGB_8s_C2C1R(
       const void* pSrc, int srcStep,
       void* pDst, int dstStep, int nWidth, int nHeight) {
       const int BLOCK_W = 16;
       const int BLOCK_H = 16;

       nWidth /= 2;

       dim3 grid((nWidth + BLOCK_W-1) / BLOCK_W, (nHeight + BLOCK_H-1) / BLOCK_H, 1);
       dim3 block(BLOCK_W, BLOCK_H, 1);

       kernel_YUYV_10c_RGB_8s_C2C1R<<<grid, block>>>(
               (uchar1*)pSrc, srcStep, (uchar1*)pDst, dstStep, nWidth, nHeight);

       return cudaDeviceSynchronize();
}

__global__ void kernel_YUYV_10c_RGB_8s_C2C1R_sqd(const int header_size, uchar1* pSrc, const int srcStep,
        uchar1* pDst, const int dstStep, const int nWidth, const int nHeight) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x < nWidth && y < nHeight) {
        int nHeaderIdx = y * srcStep;
        uchar1 h5 = pSrc[nHeaderIdx + 5];
        uchar1 h6 = pSrc[nHeaderIdx + 6];
        uchar1 h7 = pSrc[nHeaderIdx + 7];
        int nChannel = (h5.x >> 2 & 0x3);
        int nLine = ((h5.x >> 5) & 0x7) + ((h6.x & 0x3) << 3) +
            ((h6.x >> 4) << 5) + ((h7.x & 0x7) << 9) - 1; // -1 for rebase to 0
        int nSrcIdx = nHeaderIdx + header_size + x * 5;
        int nXoffset = (nChannel == 1 || nChannel == 3) ? dstStep/2 : 0;
        int nYoffset = (nChannel == 2 || nChannel == 3) ? nHeight/4 : 0;
        int nDstIdx = ((nLine + nYoffset) * dstStep) + (x * 6 + nXoffset);

        cvt_yuyv10c_2_rgb8(&pSrc[nSrcIdx + 0], &pDst[nDstIdx + 0]);
    }
}

cudaError_t convert_YUYV_10c_RGB_8s_C2C1R_sqd(
        const int header_size, const void* pSrc, const int srcStep,
        void* pDst, const int dstStep, const int nWidth, const int nHeight) {
    const int BLOCK_W = 16;
    const int BLOCK_H = 16;
    int _nWidth = nWidth;
    int _nHeight = nHeight;

    _nWidth /= 2;

    _nWidth /= 2;
    _nHeight *= 2;

    dim3 grid((_nWidth + BLOCK_W-1) / BLOCK_W, (_nHeight + BLOCK_H-1) / BLOCK_H, 1);
    dim3 block(BLOCK_W, BLOCK_H, 1);

    kernel_YUYV_10c_RGB_8s_C2C1R_sqd<<<grid, block>>>(
            header_size, (uchar1*)pSrc, srcStep, (uchar1*)pDst, dstStep, _nWidth, _nHeight);

    return cudaDeviceSynchronize();
}
