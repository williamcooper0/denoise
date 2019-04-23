import timer


import numpy as np
import math

from skimage.restoration import inpaint
import skimage.io as io

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule


_image_size = None
_work_group_size = None
_grid_size = None

_tmp0_buffer = None
_tmp1_buffer = None
_value0_buffer = None
_value1_buffer = None

_image_max = None
_image_min = None
_percentile = None


def find_value(function, image_buffer):
    gs = _image_size
    i = 0

    while True:
        if i == 0:
            inb = image_buffer
            outb = _tmp0_buffer
        elif i % 2:
            inb = _tmp0_buffer
            outb = _tmp1_buffer
        else:
            inb = _tmp1_buffer
            outb = _tmp0_buffer

        gs_new = gs // _work_group_size // 2
        if gs_new:
            ws = _work_group_size
            gs = gs_new
        else:
            ws = gs // 2
            gs = 1
        if gs == 1:
            outb = _value0_buffer

        function(inb, outb, block = (ws, 1, 1), grid = (gs, 1), shared = _work_group_size * 4)

        if outb == _value0_buffer:
            return

        i += 1


_middle = None

def percentile(image_buffer, image_tmp0_buffer, image_tmp1_buffer, q):
    find_value(_image_max, image_buffer)
    max = np.zeros(1, np.float32)
    cuda.memcpy_dtoh(max, _value0_buffer)

    find_value(_image_min, image_buffer)
    min = np.zeros(1, np.float32)
    cuda.memcpy_dtoh(min, _value0_buffer)


    less = np.zeros(1, np.int32)
    global _middle
    _middle = np.zeros(1, np.float32)

    def set_middle():
        global _middle
        _middle = (min + max) / 2

    set_middle()

    start = np.int32(0)
    border = -1
    end = np.int32(_image_size)
    final_border = q * _image_size // 100

    inb = image_buffer
    outb = image_tmp0_buffer

    i = 0
    while True:
        cuda.memcpy_htod(_value0_buffer, np.int32(0))
        cuda.memcpy_htod(_value1_buffer, np.int32(0))

        global_work_size = end - start
        grid_size = global_work_size // _work_group_size
        if global_work_size % _work_group_size:
            grid_size += 1

        if i == 0:
            i += 1
        elif i == 1:
            i += 1
            inb = image_tmp0_buffer
            outb = image_tmp1_buffer
        else:
            tmp = inb
            inb = outb
            outb = tmp

        _percentile(inb, outb, start, end, _middle, _value0_buffer, _value1_buffer, block = (_work_group_size, 1, 1), grid = (int(grid_size), 1))
        cuda.memcpy_dtoh(less, _value0_buffer)
        border = start + less

        if abs(border - final_border) < 2:
            break
        elif border < final_border:
            min = _middle
            start = border
        else:
            max = _middle
            end = border

        set_middle()

    result = np.zeros(1, np.float32)
    cuda.memcpy_dtoh(result, int(outb) + int(border - 1) * 4)

    return result


def do(img, avg_window_size, select_window_size, bright_average):
    image_side = len(img)

    global _image_size
    _image_size = image_side * image_side

    global _work_group_size
    _work_group_size = cuda.Device.get_attribute(pycuda.autoinit.device, pycuda._driver.device_attribute.MAX_THREADS_PER_BLOCK)

    block_side = int(math.sqrt(_work_group_size))

    global _grid_size
    _grid_size = _image_size // _work_group_size

    half_window_blur = avg_window_size // 2
    block_side_blur = block_side - half_window_blur * 2
    block_side_3x3 = block_side - 1 * 2
    block_side_median = block_side - select_window_size * 2
    grid_side = image_side // block_side

    def get_grid_side(bs):
        result = image_side // bs
        if image_side % bs:
            result += 1
        return result

    grid_side_blur = get_grid_side(block_side_blur)
    grid_side_3x3 = get_grid_side(block_side_3x3)
    grid_side_median = get_grid_side(block_side_median)

    image_bool_buffer = cuda.mem_alloc(_image_size * 1)
    image_ushort0_buffer = cuda.mem_alloc(_image_size * 2)
    image_ushort1_buffer = cuda.mem_alloc(_image_size * 2)
    image_float0_buffer = cuda.mem_alloc(_image_size * 4)
    image_float1_buffer = cuda.mem_alloc(_image_size * 4)
    image_float2_buffer = cuda.mem_alloc(_image_size * 4)

    global _tmp0_buffer
    _tmp0_buffer = cuda.mem_alloc(_grid_size * 4)

    global _tmp1_buffer
    _tmp1_buffer = cuda.mem_alloc(_grid_size * 4)

    global _value0_buffer
    _value0_buffer = cuda.mem_alloc(1 * 4)

    global _value1_buffer
    _value1_buffer = cuda.mem_alloc(1 * 4)


    module = SourceModule("""
    const int imageSide = """ + str(image_side) + """;

    const int windowBlur = """ + str(avg_window_size) + """;
    const int halfwindowBlur = """ + str(half_window_blur) + """;
    const int windowMedian = """ + str(select_window_size * 2 + 1) + """;
    const int halfWindowMedian = """ + str(select_window_size) + """;

    const int blockSide = """ + str(block_side) + """;
    const int blockSideBlur = """ + str(block_side_blur) + """;
    const int blockSide3x3 = """ + str(block_side_3x3) + """;
    const int blockSideMedian = """ + str(block_side_median) + """;


    __device__ int reflect(int z)
    {
        return z < 0 ? -z : (z >= imageSide ? 2 * imageSide - z - 2 : z);
    }

    __device__ int clamp(int z)
    {
        return z < 0 ? 0 : (z >= imageSide ? imageSide - 1 : z);
    }

    __device__ int indexAtR(int x, int y)
    {
        return reflect(y) * imageSide + reflect(x);
    }

    __device__ int indexAtC(int x, int y)
    {
        return clamp(y) * imageSide + clamp(x);
    }

    __device__ int indexAt(int x, int y)
    {
        return y * imageSide + x;
    }

    __global__ void blur(const float *imageIn, float *imageOut)
    {
        const int wgx = threadIdx.x - halfwindowBlur;
        const int wgy = threadIdx.y - halfwindowBlur;

        const int x = blockIdx.x * blockSideBlur + wgx;
        const int y = blockIdx.y * blockSideBlur + wgy;

        __shared__ float tmp[blockSide][blockSide];
        const float pixel = imageIn[indexAtR(x, y)];
        tmp[threadIdx.y][threadIdx.x] = pixel;
        __syncthreads();

        if(wgx < 0 || wgx >= blockSideBlur || wgy < 0 || wgy >= blockSideBlur)
            return;

        if(x >= imageSide || y >= imageSide)
            return;

        const int index = indexAt(x, y);


        float sum = 0;

        for(int i = 0; i < windowBlur; i++) {
            for(int j = 0; j < windowBlur; j++)
                sum += tmp[threadIdx.y - halfwindowBlur + i][threadIdx.x - halfwindowBlur + j];
        }

        imageOut[index] = pixel * windowBlur * windowBlur / sum;
    }


    __device__ void writeMax(int i, float *tmp)
    {
        if(threadIdx.x < i)
            tmp[threadIdx.x] = max(tmp[threadIdx.x], tmp[threadIdx.x + i]);
    }

    __global__ void imageMax(const float *imageIn, float *maxValue)
    {
        const int gid = blockIdx.x * blockDim.x + threadIdx.x;

        extern __shared__ float tmp[];
        tmp[threadIdx.x] = max(imageIn[2 * gid], imageIn[2 * gid + 1]);
        __syncthreads();


        const int halfWorkGroupSize = blockDim.x / 2;

        for(int i = halfWorkGroupSize; i > 32; i >>= 1) {
            writeMax(i, tmp);
            __syncthreads();
        }

        for(int i = min(halfWorkGroupSize, 32); i > 1; i >>= 1)
            writeMax(i, tmp);

        if(threadIdx.x == 0)
            maxValue[blockIdx.x] = max(tmp[0], tmp[1]);
    }


    __device__ void writeMin(int i, float *tmp)
    {
        if(threadIdx.x < i)
            tmp[threadIdx.x] = min(tmp[threadIdx.x], tmp[threadIdx.x + i]);
    }

    __global__ void imageMin(const float *imageIn, float *minValue)
    {
        const int gid = blockIdx.x * blockDim.x + threadIdx.x;

        extern __shared__ float tmp[];
        tmp[threadIdx.x] = min(imageIn[2 * gid], imageIn[2 * gid + 1]);
        __syncthreads();


        const int halfWorkGroupSize = blockDim.x / 2;

        for(int i = halfWorkGroupSize; i > 32; i >>= 1) {
            writeMin(i, tmp);
            __syncthreads();
        }

        for(int i = min(halfWorkGroupSize, 32); i > 1; i >>= 1)
            writeMin(i, tmp);

        if(threadIdx.x == 0)
            minValue[blockIdx.x] = min(tmp[0], tmp[1]);
    }


    __device__ float gradient(const float tmp[blockSide][blockSide], bool dx, bool dy)
    {
        return 0
            - tmp[threadIdx.y - 1][threadIdx.x - 1]
            - tmp[threadIdx.y - 1][threadIdx.x    ] * 2 * dy
            + tmp[threadIdx.y - 1][threadIdx.x + 1] * (dx - dy)
            - tmp[threadIdx.y    ][threadIdx.x - 1] * 2 * dx
            + tmp[threadIdx.y    ][threadIdx.x + 1] * 2 * dx
            + tmp[threadIdx.y + 1][threadIdx.x - 1] * (dy - dx)
            + tmp[threadIdx.y + 1][threadIdx.x    ] * 2 * dy
            + tmp[threadIdx.y + 1][threadIdx.x + 1]
        ;
    }

    __global__ void sobel(const float *imageIn, float *imageOut, float *maxValue)
    {
        const int wgx = threadIdx.x - 1;
        const int wgy = threadIdx.y - 1;

        const int x = blockIdx.x * blockSide3x3 + wgx;
        const int y = blockIdx.y * blockSide3x3 + wgy;

        __shared__ float tmp[blockSide][blockSide];
        tmp[threadIdx.y][threadIdx.x] = min(imageIn[indexAtR(x, y)] / (*maxValue), 1.0);
        __syncthreads();

        if(wgx < 0 || wgx >= blockSide3x3 || wgy < 0 || wgy >= blockSide3x3)
            return;

        if(x >= imageSide || y >= imageSide)
            return;

        const int index = indexAt(x, y);

        if(x == 0 || x == imageSide - 1 || y == 0 || y == imageSide - 1) {
            imageOut[index] = 0;
            return;
        }

        const float gradientX = gradient(tmp, 1, 0);
        const float gradientY = gradient(tmp, 0, 1);

        imageOut[index] = sqrt((gradientX * gradientX + gradientY * gradientY) / 32);
    }


    __global__ void percentile(const float *imageIn, float *imageOut, int start, int end, float value, int *gLess, int *gBigger)
    {
        const int indexIn = blockIdx.x * blockDim.x + threadIdx.x;

        if(indexIn >= end - start)
            return;

        __shared__ int lLess;
        __shared__ int lBigger;

        if(threadIdx.x == 0) {
            lLess = 0;
            lBigger = 0;
        }

        const int workGroupSize = blockSide * blockSide;
        __shared__ float lImage[workGroupSize];

        int indexLocal;
        const float pixel = imageIn[start + indexIn];
        __syncthreads();

        if(pixel < value) {
            const int old = atomicAdd(&lLess, 1);
            indexLocal = old;
        }
        else {
            const int old = atomicAdd(&lBigger, 1);
            indexLocal = workGroupSize - 1 - old;
        }

        lImage[indexLocal] = pixel;

        __shared__ int oldLess;
        __shared__ int oldBigger;

        __syncthreads();

        if(threadIdx.x == 0) {
            oldLess = atomicAdd(gLess, lLess);
            oldBigger = atomicAdd(gBigger, lBigger);
        }

        __syncthreads();

        int indexOut;

        if(threadIdx.x < lLess) {
            indexOut = start + oldLess + threadIdx.x;
            indexLocal = threadIdx.x;
        }
        else {
            indexOut = end - 1 - oldBigger - threadIdx.x + lLess;
            indexLocal = workGroupSize - 1 - threadIdx.x + lLess;
        }

        imageOut[indexOut] = lImage[indexLocal];
    }


    __global__ void binaryDilation(const float *imageIn, bool *imageOut, float edgeTop)
    {
        const int wgx = threadIdx.x - 1;
        const int wgy = threadIdx.y - 1;

        const int x = blockIdx.x * blockSide3x3 + wgx;
        const int y = blockIdx.y * blockSide3x3 + wgy;

        __shared__ bool tmp[blockSide][blockSide];
        tmp[threadIdx.y][threadIdx.x] = (imageIn[indexAtR(x, y)] > edgeTop);
        __syncthreads();

        if(wgx < 0 || wgx >= blockSide3x3 || wgy < 0 || wgy >= blockSide3x3)
            return;

        if(x >= imageSide || y >= imageSide)
            return;

        imageOut[indexAt(x, y)] = false
            | tmp[threadIdx.y - 1][threadIdx.x    ]
            | tmp[threadIdx.y    ][threadIdx.x - 1]
            | tmp[threadIdx.y    ][threadIdx.x    ]
            | tmp[threadIdx.y    ][threadIdx.x + 1]
            | tmp[threadIdx.y + 1][threadIdx.x    ]
        ;
    }


    __device__ void write(float *imageOut, int index, unsigned short value, const unsigned short *brightAverage)
    {
        imageOut[index] = min(float(value) / brightAverage[index], 1.0);
    }

    __global__ void median(const unsigned short *imageIn, const bool *noiseMap, const unsigned short *brightAverage, float *imageOut)
    {
        const int wgx = threadIdx.x - halfWindowMedian;
        const int wgy = threadIdx.y - halfWindowMedian;

        const int x = blockIdx.x * blockSideMedian + wgx;
        const int y = blockIdx.y * blockSideMedian + wgy;

        __shared__ unsigned short lTmp[blockSide][blockSide];
        const unsigned short pixel = imageIn[indexAtC(x, y)];
        lTmp[threadIdx.y][threadIdx.x] = pixel;
        __syncthreads();

        if(wgx < 0 || wgx >= blockSideMedian || wgy < 0 || wgy >= blockSideMedian)
            return;

        if(x >= imageSide || y >= imageSide)
            return;

        const int index = indexAt(x, y);

        if(!noiseMap[index]) {
            write(imageOut, index, pixel, brightAverage);
            return;
        }

        const int windowSize = windowMedian * windowMedian;
        unsigned short pTmp[windowSize];

        for(int i = 0; i < windowMedian; i++) {
            for(int j = 0; j < windowMedian; j++)
                pTmp[i * windowMedian + j] = lTmp[wgy + i][wgx + j];
        }


        for(int j = 0; j < windowSize - 1; j++) {
            unsigned short m = pTmp[j];
            int mi = j;

            for(int i = j + 1; i < windowSize; i++) {
                if(pTmp[i] > m) {
                    m = pTmp[i];
                    mi = i;
                }
            }

            pTmp[mi] = pTmp[j];
            pTmp[j] = m;
        }

        write(imageOut, index, pTmp[windowSize / 2], brightAverage);
    }
    """)


    blur = module.get_function("blur")

    global _image_max
    _image_max = module.get_function("imageMax")

    global _image_min
    _image_min = module.get_function("imageMin")

    sobel = module.get_function("sobel")

    global _percentile
    _percentile = module.get_function("percentile")

    binary_dilation = module.get_function("binaryDilation")
    median = module.get_function("median")


    timer.start()

    cuda.memcpy_htod(image_float0_buffer, img.astype(np.float32))
    blur(image_float0_buffer, image_float1_buffer, block = (block_side, block_side, 1), grid = (grid_side_blur, grid_side_blur))
    find_value(_image_max, image_float1_buffer)
    sobel(image_float1_buffer, image_float0_buffer, _value0_buffer, block = (block_side, block_side, 1), grid = (grid_side_3x3, grid_side_3x3))
    edge_top = percentile(image_float0_buffer, image_float1_buffer, image_float2_buffer, 95)
    binary_dilation(image_float0_buffer, image_bool_buffer, edge_top, block = (block_side, block_side, 1), grid = (grid_side_3x3, grid_side_3x3))
    cuda.memcpy_htod(image_ushort0_buffer, img)
    cuda.memcpy_htod(image_ushort1_buffer, bright_average)
    median(image_ushort0_buffer, image_bool_buffer, image_ushort1_buffer, image_float0_buffer, block = (block_side, block_side, 1), grid = (grid_side_median, grid_side_median))
    img_corrected = np.empty_like(img).astype(np.float32)
    cuda.memcpy_dtoh(img_corrected, image_float0_buffer)

    timer.end()

    return img_corrected
