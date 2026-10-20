// The libcuda stub.

#include <cstdio>
#include <cstdlib>

extern "C" {

// One function is enough to see whether interception works at all, and this one fails loudly.
#define STUB(name)                                                         \
    typedef int (*name##_t)();                                             \
    extern "C" int name() {                                                \
        std::fprintf(stderr,                                               \
                     "FATAL: %s reached the libcuda stub, not the shim\n",  \
                     #name);                                               \
        std::abort();                                                      \
    }

STUB(cuInit)
STUB(cuDriverGetVersion)
STUB(cuDeviceGetCount)
STUB(cuDeviceGet)
STUB(cuDeviceGetName)
STUB(cuDeviceGetAttribute)
STUB(cuDeviceTotalMem)
STUB(cuDevicePrimaryCtxRetain)
STUB(cuCtxCreate)
STUB(cuCtxDestroy)
STUB(cuCtxSetCurrent)
STUB(cuCtxGetCurrent)
STUB(cuCtxGetDevice)
STUB(cuCtxSynchronize)
STUB(cuMemAlloc_v2)
STUB(cuMemFree_v2)
STUB(cuMemHostAlloc)
STUB(cuMemFreeHost)
STUB(cuMemcpyHtoD)
STUB(cuMemcpyDtoH)
STUB(cuMemcpyDtoD)
STUB(cuMemcpyHtoH)
STUB(cuMemcpyAsync)
STUB(cuModuleLoadData)
STUB(cuModuleGetFunction)
STUB(cuModuleUnload)
STUB(cuLaunchKernel)
STUB(cuFuncSetAttribute)
STUB(cuFuncGetAttribute)
STUB(cuStreamCreate)
STUB(cuStreamDestroy)
STUB(cuStreamSynchronize)
STUB(cuStreamQuery)
STUB(cuGetLastError)
STUB(cuGetErrorString)
STUB(cuGetErrorName)

#undef STUB

} // extern "C"