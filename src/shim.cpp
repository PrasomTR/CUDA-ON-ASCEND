// MAIN: the interception of the CUDA Driver API.
#include "cuascend.h"

// MAIN: the interception of the CUDA Driver API

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <mutex>
#include <unordered_map>
#include <vector>

constexpr uintptr_t kImplicitPool = 0x1;

namespace ca {
namespace {

std::once_flag g_once;
Backend *g_backend = nullptr;

struct ShimEnv {
    bool verbose = false;
    bool strict = false;
    FILE *log = nullptr;
};

ShimEnv &env() {
    static ShimEnv e = [] {
        ShimEnv x;
        const char *v = std::getenv("CUASCEND_VERBOSE");
        x.verbose = v && *v && std::strcmp(v, "0") != 0;
        const char *s = std::getenv("CUASCEND_STRICT");
        x.strict = s && *s && std::strcmp(s, "0") != 0;
        const char *l = std::getenv("CUASCEND_LOG");
        if (l && *l) x.log = std::fopen(l, "a");
        return x;
    }();
    return e;
}

std::mutex g_ctxMu;
std::unordered_map<void *, CUcontext> g_contexts;
std::unordered_map<CUcontext, int> g_ctxDevice;

} 

bool shimVerbose() { return env().verbose; }

void shimLog(const char *fmt, ...) {
    if (!env().verbose && !env().log) return;
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);

    if (env().verbose) {
        std::fprintf(stderr, "[cuascend] %s\n", buf);
    }
    if (env().log) {
        std::fprintf(env().log, "[cuascend] %s\n", buf);
        std::fflush(env().log);
    }
}

Backend *backend() {
    std::call_once(g_once, [] { g_backend = makeBackend(); });
    return g_backend;
}

CUresult ensureInit() {
    Backend *b = backend();
    if (b->initialized()) return CUDA_SUCCESS;
    return toCuda(b->init(0));
}

const char *shimErrorString(CUresult r) {
    
    switch (r) {
    case CUDA_SUCCESS: return "no error";
    case CUDA_ERROR_INVALID_VALUE: return "cuascend: invalid argument";
    case CUDA_ERROR_OUT_OF_MEMORY: return "cuascend: out of memory";
    case CUDA_ERROR_NOT_INITIALIZED:
        return "cuascend: CUDA driver not initialized";
    case CUDA_ERROR_DEINITIALIZED:
        return "cuascend: CUDA driver is deinitialized";
    case CUDA_ERROR_NO_DEVICE: return "cuascend: no Ascend device available";
    case CUDA_ERROR_INVALID_DEVICE: return "cuascend: invalid device ordinal";
    case CUDA_ERROR_INVALID_IMAGE: return "cuascend: invalid module image";
    case CUDA_ERROR_INVALID_CONTEXT: return "cuascend: invalid context";
    case CUDA_ERROR_MAP_FAILED: return "cuascend: memory mapping failed";
    
    case CUDA_ERROR_OUT_OF_RANGE:
        return "cuascend: index out of range or named symbol not found";
    case CUDA_ERROR_NOT_SUPPORTED: return "cuascend: operation not supported";
    case CUDA_ERROR_LAUNCH_FAILED: return "cuascend: kernel launch failed";
    default: break;
    }
    return "cuascend: unknown error";
}

CUcontext contextFrom(void *p) { return reinterpret_cast<CUcontext>(p); }

void registerContext(CUcontext ctx) {
    std::lock_guard<std::mutex> lk(g_ctxMu);
    g_contexts[reinterpret_cast<void *>(ctx)] = ctx;
}

void unregisterContext(CUcontext ctx) {
    std::lock_guard<std::mutex> lk(g_ctxMu);
    g_contexts.erase(reinterpret_cast<void *>(ctx));
    g_ctxDevice.erase(ctx);
}

CUstream streamFrom(CUstream s) { return s; }
CUevent eventFrom(CUevent e) { return e; }

} 

extern "C" {

CUresult cuInit(unsigned int Flags) {
    if (CUresult r = ca::ensureInit()) return r;
    int dev = 0;
    if (CUresult r = toCuda(ca::backend()->deviceGet(0, dev))) return r;
    ca::shimLog("cuInit -> ok (backend=%s)", ca::backend()->name());
    return CUDA_SUCCESS;
}

CUresult cuDriverGetVersion(int *v) {
    if (!v) return CUDA_ERROR_INVALID_VALUE;
    
    *v = CUDA_VERSION;
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetCount(int *count) {
    if (CUresult r = ca::ensureInit()) return r;
    if (!count) return CUDA_ERROR_INVALID_VALUE;
    *count = ca::backend()->deviceCount();
    return CUDA_SUCCESS;
}

CUresult cuDeviceGet(CUdevice *dev, int ordinal) {
    if (CUresult r = ca::ensureInit()) return r;
    if (!dev) return CUDA_ERROR_INVALID_VALUE;
    return toCuda(ca::backend()->deviceGet(ordinal, *dev));
}

CUresult cuDeviceGetName(char *name, int len, CUdevice) {
    if (!name || len <= 0) return CUDA_ERROR_INVALID_VALUE;
    const std::string n = ca::backend()->deviceName(0);
    const int ncopy = static_cast<int>(n.size()) < len - 1
                          ? static_cast<int>(n.size())
                          : len - 1;
    std::memcpy(name, n.data(), static_cast<size_t>(ncopy));
    name[ncopy] = '\0';
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib,
                              CUdevice dev) {
    if (CUresult r = ca::ensureInit()) return r;
    if (!pi) return CUDA_ERROR_INVALID_VALUE;
    return toCuda(ca::backend()->deviceGetAttribute(static_cast<int>(attrib), pi,
                                                    dev));
}

CUresult cuDeviceTotalMem(size_t *bytes, CUdevice) {
    if (!bytes) return CUDA_ERROR_INVALID_VALUE;
    
    *bytes = size_t(64) << 30;
    return CUDA_SUCCESS;
}

CUresult cuDevicePrimaryCtxRetain(CUcontext *ctx, CUdevice dev) {
    if (CUresult r = ca::ensureInit()) return r;
    void *raw = nullptr;
    if (CUresult r = toCuda(ca::backend()->ctxCreate(0, dev, &raw))) return r;
    CUcontext c = reinterpret_cast<CUcontext>(raw);
    ca::registerContext(c);
    if (ctx) *ctx = c;
    return CUDA_SUCCESS;
}

CUresult cuDevicePrimaryCtxRelease(CUdevice) { return CUDA_SUCCESS; }

CUresult cuCtxCreate(CUcontext *ctx, unsigned flags, CUdevice dev) {
    if (CUresult r = ca::ensureInit()) return r;
    if (!ctx) return CUDA_ERROR_INVALID_VALUE;
    void *raw = nullptr;
    if (CUresult r = toCuda(ca::backend()->ctxCreate(flags, dev, &raw))) return r;
    CUcontext c = reinterpret_cast<CUcontext>(raw);
    ca::registerContext(c);
    *ctx = c;
    return CUDA_SUCCESS;
}

CUresult cuCtxDestroy(CUcontext ctx) {
    if (!ctx) return CUDA_ERROR_INVALID_VALUE;
    ca::unregisterContext(ctx);
    return toCuda(ca::backend()->ctxDestroy(ctx));
}

CUresult cuCtxSetCurrent(CUcontext ctx) {
    return toCuda(ca::backend()->ctxSetCurrent(ctx));
}

CUresult cuCtxGetCurrent(CUcontext *ctx) {
    void *raw = nullptr;
    if (CUresult r = toCuda(ca::backend()->ctxGetCurrent(&raw))) return r;
    if (ctx) *ctx = reinterpret_cast<CUcontext>(raw);
    return CUDA_SUCCESS;
}

CUresult cuCtxGetDevice(CUdevice *dev) {
    if (!dev) return CUDA_ERROR_INVALID_VALUE;
    return toCuda(ca::backend()->ctxGetDevice(dev));
}

CUresult cuCtxGetApiVersion(CUcontext, unsigned int *v) {
    if (!v) return CUDA_ERROR_INVALID_VALUE;
    *v = CUDA_VERSION;
    return CUDA_SUCCESS;
}

CUresult cuCtxSynchronize() {
    
    
    
    if (CUresult r = ca::ensureInit()) return r;
    return toCuda(ca::backend()->streamSynchronize(ca::kNullStream));
}

CUresult cuMemAlloc(CUdeviceptr *ptr, size_t bytes) {
    if (CUresult r = ca::ensureInit()) return r;
    if (!ptr || bytes == 0) return CUDA_ERROR_INVALID_VALUE;
    ca::MemHandle h = 0;
    const CUresult r = toCuda(ca::backend()->memAlloc(&h, bytes));
    if (r) return r;
    *ptr = h;
    return CUDA_SUCCESS;
}

CUresult cuMemAlloc_v2(CUdeviceptr *ptr, size_t bytes) {
    return cuMemAlloc(ptr, bytes);
}

CUresult cuMemFree(CUdeviceptr ptr) {
    if (!ptr) return CUDA_ERROR_INVALID_VALUE;
    return toCuda(ca::backend()->memFree(ptr));
}

CUresult cuMemFree_v2(CUdeviceptr ptr) { return cuMemFree(ptr); }

CUresult cuMemHostAlloc(void **p, size_t bytes, unsigned flags) {
    if (CUresult r = ca::ensureInit()) return r;
    return toCuda(ca::backend()->memHostAlloc(p, bytes, flags));
}

CUresult cuMemHostAlloc_v2(void **p, size_t bytes, unsigned flags) {
    return cuMemHostAlloc(p, bytes, flags);
}

CUresult cuMemFreeHost(void *p) { return toCuda(ca::backend()->memHostFree(p)); }

CUresult cuMemHostRegister(void *p, size_t bytes, unsigned flags) {
    return toCuda(ca::backend()->memHostRegister(p, bytes, flags));
}

CUresult cuMemHostUnregister(void *p) {
    return toCuda(ca::backend()->memHostUnregister(p));
}

CUresult cuMemGetInfo(size_t *free_, size_t *total) {
    if (!free_ || !total) return CUDA_ERROR_INVALID_VALUE;
    *total = size_t(64) << 30;
    *free_ = *total / 2;
    return CUDA_SUCCESS;
}

namespace {

CUresult memsetDevice(CUdeviceptr dst, const void *pattern, size_t patternBytes,
                      size_t count) {
    if (count == 0) return CUDA_SUCCESS;
    const size_t total = count * patternBytes;
    const size_t chunk = 64 * 1024;
    const auto *unit = static_cast<const unsigned char *>(pattern);
    std::vector<unsigned char> buffer;
    for (size_t done = 0; done < total;) {
        const size_t part = std::min(chunk, total - done);
        buffer.assign(part, 0);
        for (size_t i = 0; i < part; ++i)
            buffer[i] = unit[i % patternBytes];
        const CUresult r = toCuda(ca::backend()->memcpy(
            reinterpret_cast<char *>(static_cast<uintptr_t>(dst)) + done,
            buffer.data(), part, 2));
        if (r) return r;
        done += part;
    }
    return CUDA_SUCCESS;
}

} 

CUresult cuMemsetD8(CUdeviceptr dst, unsigned char value, size_t count) {
    if (CUresult r = ca::ensureInit()) return r;
    return memsetDevice(dst, &value, sizeof value, count);
}

CUresult cuMemsetD16(CUdeviceptr dst, unsigned short value, size_t count) {
    if (CUresult r = ca::ensureInit()) return r;
    return memsetDevice(dst, &value, sizeof value, count);
}

CUresult cuMemsetD32(CUdeviceptr dst, unsigned int value, size_t count) {
    if (CUresult r = ca::ensureInit()) return r;
    return memsetDevice(dst, &value, sizeof value, count);
}

CUresult cuStreamCreate(CUstream *s, unsigned flags) {
    if (CUresult r = ca::ensureInit()) return r;
    if (!s) return CUDA_ERROR_INVALID_VALUE;
    ca::StreamHandle h = 0;
    const CUresult r = toCuda(ca::backend()->streamCreate(&h, flags));
    if (r) return r;
    *s = reinterpret_cast<CUstream>(h);
    return CUDA_SUCCESS;
}

CUresult cuStreamCreateWithPriority(CUstream *s, unsigned flags, int) {
    return cuStreamCreate(s, flags);
}

CUresult cuStreamDestroy(CUstream s) {
    return toCuda(ca::backend()->streamDestroy(reinterpret_cast<ca::StreamHandle>(s)));
}

CUresult cuStreamSynchronize(CUstream s) {
    return toCuda(ca::backend()->streamSynchronize(
        reinterpret_cast<ca::StreamHandle>(s)));
}

CUresult cuStreamQuery(CUstream) { return CUDA_SUCCESS; }

CUresult cuStreamWaitEvent(CUstream s, CUevent e, unsigned) {
    return toCuda(ca::backend()->streamWaitEvent(
        reinterpret_cast<ca::StreamHandle>(s),
        reinterpret_cast<ca::EventHandle>(e)));
}

CUresult cuEventCreate(CUevent *e, unsigned flags) {
    if (CUresult r = ca::ensureInit()) return r;
    if (!e) return CUDA_ERROR_INVALID_VALUE;
    ca::EventHandle h = 0;
    const CUresult r = toCuda(ca::backend()->eventCreate(&h, flags));
    if (r) return r;
    *e = reinterpret_cast<CUevent>(h);
    return CUDA_SUCCESS;
}

CUresult cuEventDestroy(CUevent e) {
    return toCuda(ca::backend()->eventDestroy(reinterpret_cast<ca::EventHandle>(e)));
}

CUresult cuEventRecord(CUevent e, CUstream s) {
    return toCuda(ca::backend()->eventRecord(reinterpret_cast<ca::EventHandle>(e),
                                             reinterpret_cast<ca::StreamHandle>(s)));
}

CUresult cuEventSynchronize(CUevent e) {
    return toCuda(ca::backend()->eventSynchronize(
        reinterpret_cast<ca::EventHandle>(e)));
}

CUresult cuEventQuery(CUevent e) {
    return toCuda(ca::backend()->eventQuery(reinterpret_cast<ca::EventHandle>(e)));
}

CUresult cuEventElapsedTime(float *ms, CUevent a, CUevent b) {
    if (!ms) return CUDA_ERROR_INVALID_VALUE;
    *ms = 0.0f;
    return CUDA_SUCCESS;
}

CUresult cuModuleLoadData(CUmodule *mod, const void *image) {
    if (CUresult r = ca::ensureInit()) return r;
    if (!mod || !image) return CUDA_ERROR_INVALID_VALUE;

    ca::shimLog("cuModuleLoadData: image=%p", image);

    
    ca::ModuleHandle h = 0;
    const CUresult r = toCuda(ca::backend()->moduleLoad(&h, image, 0));
    if (r) return r;
    *mod = reinterpret_cast<CUmodule>(h);
    return CUDA_SUCCESS;
}

CUresult cuModuleLoadDataEx(CUmodule *mod, const void *image, unsigned) {
    return cuModuleLoadData(mod, image);
}

CUresult cuModuleUnload(CUmodule m) {
    return toCuda(ca::backend()->moduleUnload(reinterpret_cast<ca::ModuleHandle>(m)));
}

extern "C" const char *cuascendLastError(void) {
    
    static thread_local std::string keep;
    keep = ca::ascend::ascendLastError();
    return keep.c_str();
}

CUresult cuModuleGetFunction(CUfunction *f, CUmodule m, const char *name) {
    if (!f || !name) return CUDA_ERROR_INVALID_VALUE;
    ca::FunctionHandle h = 0;
    const CUresult r = toCuda(ca::backend()->moduleGetFunction(
        &h, reinterpret_cast<ca::ModuleHandle>(m), name));
    if (r) return r;
    *f = reinterpret_cast<CUfunction>(h);
    return CUDA_SUCCESS;
}

CUresult cuFuncSetAttribute(CUfunction f, CUfunc_attribute attr, int value) {
    return toCuda(ca::backend()->functionSetAttribute(
        static_cast<int>(attr), reinterpret_cast<ca::FunctionHandle>(f),
        &value));
}

CUresult cuFuncGetAttribute(int *pi, CUfunc_attribute attr, CUfunction f) {
    if (!pi) return CUDA_ERROR_INVALID_VALUE;
    return toCuda(ca::backend()->functionGetAttribute(
        static_cast<int>(attr), reinterpret_cast<ca::FunctionHandle>(f), pi));
}

CUresult cuMemcpyHtoDAsync(CUdeviceptr dst, const void *src, size_t bytes,
                            CUstream) {
    if (CUresult r = ca::ensureInit()) return r;
    return toCuda(ca::backend()->memcpy(reinterpret_cast<void *>(dst), src, bytes, 2));
}

CUresult cuMemcpyDtoDAsync(CUdeviceptr dst, CUdeviceptr src, size_t bytes,
                            CUstream) {
    if (CUresult r = ca::ensureInit()) return r;
    return toCuda(ca::backend()->memcpy(reinterpret_cast<void *>(dst),
                                        reinterpret_cast<void *>(src), bytes, 4));
}

CUresult cuMemcpy2DUnaligned(const CUDA_MEMCPY2_st *p) {
    if (CUresult r = ca::ensureInit()) return r;
    if (!p) return CUDA_ERROR_INVALID_VALUE;
    
    const size_t rowBytes = p->WidthInBytes;
    for (size_t y = 0; y < p->srcY; ++y) {
        const void *src = nullptr;
        void *dst = nullptr;
        if (p->srcHost)
            src = static_cast<const char *>(p->srcHost) + y * p->srcPitch;
        else
            src = reinterpret_cast<const void *>(p->srcDevice + y * p->srcPitch);
        if (p->dstHost)
            dst = static_cast<char *>(p->dstHost) + y * p->dstPitch;
        else
            dst = reinterpret_cast<void *>(p->dstDevice + y * p->dstPitch);
        const CUresult r = toCuda(ca::backend()->memcpy(dst, src, rowBytes,
                                                        p->srcHost ? 1 : 4));
        if (r) return r;
    }
    return CUDA_SUCCESS;
}

CUresult cuCtxGetFlags(unsigned int *flags) {
    if (!flags) return CUDA_ERROR_INVALID_VALUE;
    *flags = 0;
    return CUDA_SUCCESS;
}

CUresult cuCtxSetFlags(unsigned int) { return CUDA_SUCCESS; }

CUresult cuStreamGetFlags(CUstream, unsigned int *flags) {
    if (!flags) return CUDA_ERROR_INVALID_VALUE;
    *flags = 0;
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetUuid(CUdevice dev, char *uuid) {
    if (!uuid) return CUDA_ERROR_INVALID_VALUE;
    
    
    for (int i = 0; i < 16; ++i) uuid[i] = static_cast<char>(0xA0 + dev + i);
    return CUDA_SUCCESS;
}

CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks,
                                                     CUfunction, int blockSize,
                                                     size_t) {
    if (!numBlocks) return CUDA_ERROR_INVALID_VALUE;
    if (blockSize <= 0) return CUDA_ERROR_INVALID_VALUE;
    
    
    
    *numBlocks = 1024 / blockSize;
    if (*numBlocks < 1) *numBlocks = 1;
    if (*numBlocks > 32) *numBlocks = 32;
    return CUDA_SUCCESS;
}

CUresult cuFuncSetCacheConfig(CUfunction, CUfunc_cache) { return CUDA_SUCCESS; }

CUresult cuFuncSetSharedMemConfig(CUfunction, CUfunc_shared) {
    return CUDA_SUCCESS;
}

CUresult cuLaunchHostFunc(CUstream, CUhostFn fn, void *userData) {
    if (CUresult r = ca::ensureInit()) return r;
    if (!fn) return CUDA_ERROR_INVALID_VALUE;
    
    
    
    fn(userData);
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetDefaultMemPool(CUmempool *pool, CUdevice) {
    if (!pool) return CUDA_ERROR_INVALID_VALUE;
    *pool = reinterpret_cast<CUmempool>(kImplicitPool);
    return CUDA_SUCCESS;
}

CUresult cuDeviceSetMemPool(CUdevice, CUmempool) { return CUDA_SUCCESS; }

CUresult cuMemPoolGetAttribute(CUmempool, int attr, void *value) {
    if (!value) return CUDA_ERROR_INVALID_VALUE;
    switch (attr) {
    case CU_MEMPOOL_ATTR_RELEASE_THRESHOLD:
        *static_cast<uint64_t *>(value) = 0;
        break;
    case CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH:
    case CU_MEMPOOL_ATTR_RESERVED_MEM_LOW:
        *static_cast<uint64_t *>(value) = 0;
        break;
    case CU_MEMPOOL_ATTR_UTILIZATION_CUR:
    case CU_MEMPOOL_ATTR_UTILIZATION_MAX:
        *static_cast<double *>(value) = 0.0;
        break;
    case CU_MEMPOOL_ATTR_TIMEOUT_US:
        *static_cast<uint64_t *>(value) = 0;
        break;
    default:
        return CUDA_ERROR_INVALID_VALUE;
    }
    return CUDA_SUCCESS;
}

CUresult cuMemPoolSetAttribute(CUmempool, int, const void *) {
    return CUDA_SUCCESS;
}

CUresult cuMemPoolTrimTo(CUmempool, size_t) { return CUDA_SUCCESS; }

CUresult cuLaunchKernel(CUfunction f, unsigned gx, unsigned gy, unsigned gz,
                        unsigned bx, unsigned by, unsigned bz,
                        unsigned sharedMem, CUstream stream, void **args,
                        void **extra) {
    if (CUresult r = ca::ensureInit()) return r;
    if (!f) return CUDA_ERROR_INVALID_VALUE;

    ca::shimLog("cuLaunchKernel func=%p grid=(%u,%u,%u) block=(%u,%u,%u) "
                "shared=%u",
                (void *)f, gx, gy, gz, bx, by, bz, sharedMem);

    return toCuda(ca::backend()->launch(reinterpret_cast<ca::FunctionHandle>(f),
                                        gx, gy, gz, bx, by, bz, sharedMem, args,
                                        extra,
                                        reinterpret_cast<ca::StreamHandle>(stream)));
}

CUresult cuMemcpyHtoD(CUdeviceptr dst, const void *src, size_t bytes) {
    return toCuda(ca::backend()->memcpy(reinterpret_cast<void *>(dst), src, bytes, 2));
}

CUresult cuMemcpyDtoH(void *dst, CUdeviceptr src, size_t bytes) {
    return toCuda(ca::backend()->memcpy(dst, reinterpret_cast<void *>(src), bytes, 3));
}

CUresult cuMemcpyDtoD(CUdeviceptr dst, CUdeviceptr src, size_t bytes) {
    return toCuda(ca::backend()->memcpy(reinterpret_cast<void *>(dst),
                                        reinterpret_cast<void *>(src), bytes, 4));
}

CUresult cuMemcpyHtoH(void *dst, const void *src, size_t bytes) {
    return toCuda(ca::backend()->memcpy(dst, src, bytes, 1));
}

CUresult cuMemcpy(CUdeviceptr dst, const void *src, size_t bytes) {
    return toCuda(ca::backend()->memcpy(reinterpret_cast<void *>(dst), src, bytes, 0));
}

CUresult cuMemcpyAsync(CUdeviceptr dst, const void *src, size_t bytes, CUstream s) {
    (void)s;
    return toCuda(ca::backend()->memcpy(reinterpret_cast<void *>(dst), src, bytes, 0));
}

CUresult cuMemcpy2D(const CUDA_MEMCPY2 *p) {
    if (!p) return CUDA_ERROR_INVALID_VALUE;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult cuGetErrorString(CUresult error, const char **str) {
    if (!str) return CUDA_ERROR_INVALID_VALUE;
    *str = ca::shimErrorString(error);
    return CUDA_SUCCESS;
}

CUresult cuGetErrorName(CUresult error, const char **str) {
    return cuGetErrorString(error, str);
}

CUresult cuGetLastError() { return CUDA_SUCCESS; }

CUresult cuCtxGetApiVersionStub() { return CUDA_SUCCESS; }

} 
