#include "cuascend.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

extern "C" {
CUresult cuInit(unsigned Flags);
CUresult cuDriverGetVersion(int *v);
CUresult cuDeviceGetCount(int *n);
CUresult cuDeviceGet(CUdevice *dev, int ordinal);
CUresult cuDeviceGetName(char *name, int len, CUdevice dev);
CUresult cuDevicePrimaryCtxRetain(CUcontext *ctx, CUdevice dev);
CUresult cuCtxSetCurrent(CUcontext ctx);
CUresult cuModuleLoadData(CUmodule *mod, const void *image);
CUresult cuModuleGetFunction(CUfunction *fn, CUmodule mod, const char *name);
CUresult cuMemAlloc_v2(CUdeviceptr *p, size_t bytes);
CUresult cuMemFree_v2(CUdeviceptr p);
CUresult cuMemcpyHtoD(CUdeviceptr dst, const void *src, size_t bytes);
CUresult cuMemcpyDtoH(void *dst, CUdeviceptr src, size_t bytes);
CUresult cuLaunchKernel(CUfunction f, unsigned gx, unsigned gy, unsigned gz,
                       unsigned bx, unsigned by, unsigned bz,
                       unsigned shmem, CUstream s, void **args, void **extra);
CUresult cuCtxSynchronize(void);
CUresult cuGetErrorString(CUresult e, const char **s);
const char *cuascendLastError(void);
}

namespace {

constexpr int kAxpySize = 64;     
constexpr int kGridBlock = 256;   
constexpr int kGridBlocks = 4;    
constexpr int kGridSize = kGridBlock * kGridBlocks;
constexpr int kGridActive = 1000; 

int g_fail = 0;

void step(const char *what) { std::printf("  ...   %s\n", what); }
void ok(const char *what) { std::printf("  ok    %s\n", what); }

bool check(CUresult r, const char *what) {
    if (r == CUDA_SUCCESS) return true;
    const char *msg = nullptr;
    cuGetErrorString(r, &msg);
    std::printf("  FAIL  %s -> %d (%s)\n", what, static_cast<int>(r),
                msg ? msg : "?");
    ++g_fail;
    return false;
}

std::string readFile(const char *path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

bool blockedByImage() {
    const char *why = cuascendLastError();
    return why && (std::strstr(why, "operator-development SDK") ||
                   std::strstr(why, "tilingdata_base.h"));
}

bool launchOrSkip(CUfunction fn, unsigned gx, unsigned bx, void **args,
                  const char *name) {
    const CUresult r =
        cuLaunchKernel(fn, gx, 1, 1, bx, 1, 1, 0, nullptr, args, nullptr);
    if (r == CUDA_SUCCESS) return true;
    if (blockedByImage()) {
        std::printf("  SKIP  %s: this CANN image has no operator-development SDK\n",
                    name);
        return false;
    }
    check(r, name);
    return false;
}

void checkAxpy(CUmodule mod) {
    CUfunction fn = nullptr;
    if (!check(cuModuleGetFunction(&fn, mod, "axpy"), "cuModuleGetFunction"))
        return;
    ok("function axpy obtained from the intercepted cuModuleGetFunction");

    CUdeviceptr dx = 0, dy = 0, dout = 0;
    const size_t bytes = static_cast<size_t>(kAxpySize) * sizeof(float);
    if (!check(cuMemAlloc_v2(&dx, bytes), "cuMemAlloc_v2 x") ||
        !check(cuMemAlloc_v2(&dy, bytes), "cuMemAlloc_v2 y") ||
        !check(cuMemAlloc_v2(&dout, bytes), "cuMemAlloc_v2 out"))
        return;
    step("device memory allocated through the intercepted allocator");

    std::vector<float> hostX(kAxpySize), hostY(kAxpySize);
    std::vector<float> hostOut(kAxpySize, 0.0f);
    for (int i = 0; i < kAxpySize; ++i) {
        hostX[i] = static_cast<float>(i) * 0.5f;
        hostY[i] = static_cast<float>(i);
    }
    if (!check(cuMemcpyHtoD(dx, hostX.data(), bytes), "cuMemcpyHtoD x") ||
        !check(cuMemcpyHtoD(dy, hostY.data(), bytes), "cuMemcpyHtoD y"))
        return;
    step("host to device copy works");

    unsigned n = static_cast<unsigned>(kAxpySize);
    float a = 2.0f, b = 1.0f;
    void *args[] = {&dx, &dy, &n, &a, &b, &dout};
    if (!launchOrSkip(fn, 1, kAxpySize, args, "axpy")) return;
    if (!check(cuCtxSynchronize(), "cuCtxSynchronize") ||
        !check(cuMemcpyDtoH(hostOut.data(), dy, bytes), "cuMemcpyDtoH y"))
        return;

    std::vector<float> hostSum(kAxpySize, 0.0f);
    if (!check(cuMemcpyDtoH(hostSum.data(), dout, bytes), "cuMemcpyDtoH out"))
        return;

    double maxAbs = 0.0;
    int wrong = 0;
    for (int i = 0; i < kAxpySize; ++i) {
        const double want = static_cast<double>(hostX[i]) * a + b;
        const double diff = std::fabs(want - static_cast<double>(hostOut[i]));
        if (diff > maxAbs) maxAbs = diff;
        if (diff > 1e-4) ++wrong;
    }
    const double sumWant = static_cast<double>(hostX[0]) * a + b;
    const double sumDiff = std::fabs(sumWant - static_cast<double>(hostSum[0]));
    std::printf("  ...   max abs error %g over %d elements (%d wrong)\n", maxAbs,
                kAxpySize, wrong);
    std::printf("  ...   checksum out[0] = %g, want %g\n",
                static_cast<double>(hostSum[0]), sumWant);
    if (wrong != 0 || sumDiff > 1e-4) {
        std::printf("  FAIL  numbers differ from the CPU reference\n");
        ++g_fail;
    } else {
        ok("OWN KERNEL ON THE REAL NPU PRODUCED CORRECT NUMBERS");
    }

    cuMemFree_v2(dx);
    cuMemFree_v2(dy);
    cuMemFree_v2(dout);
}

void checkScaleGrid(CUmodule mod) {
    CUfunction fn = nullptr;
    if (!check(cuModuleGetFunction(&fn, mod, "scale_grid"),
               "cuModuleGetFunction scale_grid"))
        return;

    CUdeviceptr din = 0, dout = 0;
    const size_t bytes = static_cast<size_t>(kGridSize) * sizeof(float);
    if (!check(cuMemAlloc_v2(&din, bytes), "cuMemAlloc_v2 in") ||
        !check(cuMemAlloc_v2(&dout, bytes), "cuMemAlloc_v2 out"))
        return;

    std::vector<float> hostIn(kGridSize);
    std::vector<float> hostOut(kGridSize, -1.0f); 
    for (int i = 0; i < kGridSize; ++i)
        hostIn[i] = static_cast<float>(i) * 0.25f;
    if (!check(cuMemcpyHtoD(din, hostIn.data(), bytes), "cuMemcpyHtoD in") ||
        !check(cuMemcpyHtoD(dout, hostOut.data(), bytes), "cuMemcpyHtoD out"))
        return;

    unsigned n = static_cast<unsigned>(kGridActive);
    float c = 3.0f;
    void *args[] = {&din, &dout, &n, &c};
    if (!launchOrSkip(fn, kGridBlocks, kGridBlock, args, "scale_grid")) return;
    if (!check(cuCtxSynchronize(), "cuCtxSynchronize") ||
        !check(cuMemcpyDtoH(hostOut.data(), dout, bytes), "cuMemcpyDtoH out"))
        return;

    int wrong = 0;
    int tailWrong = 0;
    for (int i = 0; i < kGridSize; ++i) {
        if (i < kGridActive) {
            const double want = static_cast<double>(hostIn[i]) * c;
            if (std::fabs(want - static_cast<double>(hostOut[i])) > 1e-4) ++wrong;
        } else if (hostOut[i] != -1.0f) {
            ++tailWrong; 
        }
    }
    std::printf("  ...   scale_grid: %d wrong of %d, tail touched %d\n", wrong,
                kGridActive, tailWrong);
    if (wrong) {
        for (int i = 0; i < 5; ++i)
            std::printf("    [%3d] got %g want %g\n", i,
                        static_cast<double>(hostOut[i]),
                        static_cast<double>(hostIn[i]) * c);
    }
    if (wrong != 0 || tailWrong != 0) {
        std::printf("  FAIL  grid launch produced wrong numbers\n");
        ++g_fail;
    } else {
        ok("GRID LAUNCH: 4 BLOCKS, GUARDED TAIL, CORRECT NUMBERS");
    }

    cuMemFree_v2(din);
    cuMemFree_v2(dout);
}

} 

int main(int argc, char **argv) {
    const char *ptxPath = argc > 1 ? argv[1] : "tests/sample.cu.ptx";

    std::printf("[1] normal CUDA application, no knowledge of Ascend\n");

    if (!check(cuInit(0), "cuInit")) return 1;
    step("interpreter ready");
    int major = 0;
    if (!check(cuDriverGetVersion(&major), "cuDriverGetVersion")) return 1;
    std::printf("  ...   driver version %d\n", major);

    int deviceCount = 0;
    if (!check(cuDeviceGetCount(&deviceCount), "cuDeviceGetCount")) return 1;
    if (deviceCount < 1) {
        
        
        
        std::printf("  SKIP  no device visible: run this test on a host with CANN\n");
        return 77;
    }

    CUdevice dev = 0;
    if (!check(cuDeviceGet(&dev, 0), "cuDeviceGet")) return 1;
    char name[128] = {0};
    cuDeviceGetName(name, sizeof name, dev);
    std::printf("  ...   device 0: %s\n", name);

    CUcontext ctx = nullptr;
    if (!check(cuDevicePrimaryCtxRetain(&ctx, dev), "cuDevicePrimaryCtxRetain"))
        return 1;
    if (!check(cuCtxSetCurrent(ctx), "cuCtxSetCurrent")) return 1;
    ok("the application sees the device through the intercepted CUDA API");

    const std::string ptx = readFile(ptxPath);
    if (ptx.empty()) {
        std::printf("  FAIL  cannot read PTX from %s\n", ptxPath);
        return 1;
    }

    CUmodule mod = nullptr;
    if (!check(cuModuleLoadData(&mod, ptx.data()), "cuModuleLoadData")) return 1;
    ok("PTX loaded by the intercepted cuModuleLoadData");

    checkAxpy(mod);
    checkScaleGrid(mod);

    std::printf("\nRESULT: %s (%d)\n", g_fail ? "FAILED" : "PASSED", g_fail);
    return g_fail ? 1 : 0;
}
