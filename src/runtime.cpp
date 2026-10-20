// MAIN: the runtime — the backends and the IR executor.
#include "cuascend.h"

// MAIN: the status codes

namespace ca {

CUresult toCuda(BackendStatus s) {
    switch (s) {
    case BackendStatus::Ok:
        return CUDA_SUCCESS;
    case BackendStatus::NotInitialized:
        return CUDA_ERROR_NOT_INITIALIZED;
    case BackendStatus::NoDevice:
        return CUDA_ERROR_NO_DEVICE;
    case BackendStatus::OutOfMemory:
        return CUDA_ERROR_OUT_OF_MEMORY;
    case BackendStatus::Unsupported:
        return CUDA_ERROR_INVALID_IMAGE;
    case BackendStatus::InvalidValue:
        return CUDA_ERROR_INVALID_VALUE;
    case BackendStatus::NotSupported:
        return CUDA_ERROR_NOT_SUPPORTED;
    case BackendStatus::Unknown:
        return CUDA_ERROR_DEINITIALIZED;
    }
    return CUDA_ERROR_DEINITIALIZED;
}

} 

// MAIN: the null backend

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace ca {

class NullBackend : public Backend {
public:
    const char *name() const override { return "null"; }

    BackendStatus init(int) override;
    void shutdown() override;
    bool initialized() const override { return init_; }
    int deviceCount() const override { return 0; }
    BackendStatus deviceGet(int ordinal, int &out) override;
    DeviceCaps deviceCaps(int device) const override;
    BackendStatus deviceGetAttribute(int attr, int *out, int device) override;
    std::string deviceName(int device) const override;

    BackendStatus ctxCreate(unsigned, int, void **outCtx) override;
    BackendStatus ctxDestroy(void *ctx) override;
    BackendStatus ctxSetCurrent(void *ctx) override;
    BackendStatus ctxGetCurrent(void **outCtx) override;
    BackendStatus ctxGetDevice(int *outDevice) override;

    BackendStatus memAlloc(MemHandle *out, size_t bytes) override;
    BackendStatus memFree(MemHandle h) override;
    BackendStatus memHostAlloc(void **out, size_t bytes, unsigned) override;
    BackendStatus memHostFree(void *p) override;
    BackendStatus memHostRegister(void *, size_t, unsigned) override;
    BackendStatus memHostUnregister(void *) override;

    BackendStatus streamCreate(StreamHandle *out, unsigned) override;
    BackendStatus streamDestroy(StreamHandle) override;
    BackendStatus streamSynchronize(StreamHandle) override;
    BackendStatus eventCreate(EventHandle *out, unsigned flags) override;
    BackendStatus eventDestroy(EventHandle) override;
    BackendStatus eventRecord(EventHandle, StreamHandle) override;
    BackendStatus eventSynchronize(EventHandle) override;
    BackendStatus eventQuery(EventHandle) override;
    BackendStatus streamWaitEvent(StreamHandle, EventHandle) override;

    BackendStatus moduleLoad(ModuleHandle *out, const void *image,
                             unsigned) override;
    BackendStatus moduleUnload(ModuleHandle) override;
    BackendStatus moduleGetFunction(FunctionHandle *out, ModuleHandle,
                                    const char *name) override;
    BackendStatus functionSetAttribute(int, FunctionHandle, const void *) override;
    BackendStatus functionGetAttribute(int, FunctionHandle, void *) override;

    BackendStatus launch(FunctionHandle, unsigned, unsigned, unsigned,
                         unsigned, unsigned, unsigned, unsigned, void **,
                         void **, StreamHandle) override;
    BackendStatus memcpy(void *dst, const void *src, size_t bytes,
                         unsigned kind) override;

    
    struct Counters {
        unsigned launches = 0;
        unsigned memAllocs = 0;
        unsigned streamsCreated = 0;
        unsigned modulesLoaded = 0;
        size_t bytesAllocated = 0;
    };
    Counters &counters() { return counters_; }
    const Counters &counters() const { return counters_; }

    ~NullBackend() override;
    void *hostAlloc(size_t bytes);

private:
    bool init_ = false;
    void *context_ = nullptr;
    int currentDevice_ = 0;
    uint64_t nextHandle_ = 0x1000; 
    Counters counters_;
};

BackendStatus NullBackend::init(int) {
    init_ = true;
    return BackendStatus::Ok;
}

void NullBackend::shutdown() {
    init_ = false;
    context_ = nullptr;
}

BackendStatus NullBackend::deviceGet(int ordinal, int &out) {
    if (!init_) return BackendStatus::NotInitialized;
    
    
    
    if (ordinal < 0 || ordinal > 0) return BackendStatus::InvalidValue;
    out = 0;
    return BackendStatus::Ok;
}

DeviceCaps NullBackend::deviceCaps(int) const {
    DeviceCaps caps;
    caps.deviceCount = 0;
    caps.major = 8;
    caps.minor = 0;
    return caps;
}

BackendStatus NullBackend::deviceGetAttribute(int attr, int *out, int) {
    if (!out) return BackendStatus::InvalidValue;
    const DeviceCaps caps = deviceCaps(0);
    switch (attr) {
    case 1: *out = caps.maxThreadsPerBlock; return BackendStatus::Ok;
    case 5: *out = caps.maxThreadsDim[0]; return BackendStatus::Ok;
    case 6: *out = caps.maxThreadsDim[1]; return BackendStatus::Ok;
    case 7: *out = caps.maxThreadsDim[2]; return BackendStatus::Ok;
    case 8: *out = caps.maxSharedMemoryPerBlock; return BackendStatus::Ok;
    case 12: *out = caps.maxRegistersPerBlock; return BackendStatus::Ok;
    case 16: *out = caps.multiprocessorCount; return BackendStatus::Ok;
    case 32: *out = caps.warpSize; return BackendStatus::Ok;
    case 75: *out = caps.major; return BackendStatus::Ok;
    case 76: *out = caps.minor; return BackendStatus::Ok;
    default: break;
    }
    return BackendStatus::NotSupported;
}

std::string NullBackend::deviceName(int) const { return "cuascend null device"; }

BackendStatus NullBackend::ctxCreate(unsigned, int, void **outContext) {
    if (!init_) return BackendStatus::NotInitialized;
    if (!outContext) return BackendStatus::InvalidValue;
    
    
    *outContext = reinterpret_cast<void *>(new int(0));
    context_ = *outContext;
    return BackendStatus::Ok;
}

BackendStatus NullBackend::ctxDestroy(void *ctx) {
    delete reinterpret_cast<int *>(ctx);
    if (context_ == ctx) context_ = nullptr;
    return BackendStatus::Ok;
}

BackendStatus NullBackend::ctxSetCurrent(void *ctx) {
    context_ = ctx;
    return BackendStatus::Ok;
}

BackendStatus NullBackend::ctxGetCurrent(void **outContext) {
    if (!outContext) return BackendStatus::InvalidValue;
    *outContext = context_;
    return context_ ? BackendStatus::Ok : BackendStatus::InvalidValue;
}

BackendStatus NullBackend::ctxGetDevice(int *outDevice) {
    if (!outDevice) return BackendStatus::InvalidValue;
    *outDevice = currentDevice_;
    return BackendStatus::Ok;
}

BackendStatus NullBackend::memAlloc(MemHandle *out, size_t bytes) {
    if (!out || bytes == 0) return BackendStatus::InvalidValue;
    ++counters_.memAllocs;
    counters_.bytesAllocated += bytes;
    *out = (nextHandle_ += 0x40); 
    return BackendStatus::Ok;
}

BackendStatus NullBackend::memFree(MemHandle) {
    return init_ ? BackendStatus::Ok : BackendStatus::NotInitialized;
}

void *NullBackend::hostAlloc(size_t bytes) { return std::calloc(1, bytes); }

BackendStatus NullBackend::memHostAlloc(void **out, size_t bytes, unsigned) {
    if (!out || bytes == 0) return BackendStatus::InvalidValue;
    *out = hostAlloc(bytes);
    return *out ? BackendStatus::Ok : BackendStatus::OutOfMemory;
}

BackendStatus NullBackend::memHostFree(void *p) {
    std::free(p);
    return BackendStatus::Ok;
}

BackendStatus NullBackend::memHostRegister(void *, size_t, unsigned) {
    
    
    return BackendStatus::Ok;
}

BackendStatus NullBackend::memHostUnregister(void *) { return BackendStatus::Ok; }

BackendStatus NullBackend::streamCreate(StreamHandle *out, unsigned) {
    if (!out) return BackendStatus::InvalidValue;
    ++counters_.streamsCreated;
    *out = (nextHandle_ += 0x10);
    return BackendStatus::Ok;
}

BackendStatus NullBackend::streamDestroy(StreamHandle) { return BackendStatus::Ok; }
BackendStatus NullBackend::streamSynchronize(StreamHandle) { return BackendStatus::Ok; }

BackendStatus NullBackend::eventCreate(EventHandle *out, unsigned) {
    if (!out) return BackendStatus::InvalidValue;
    *out = (nextHandle_ += 0x10);
    return BackendStatus::Ok;
}

BackendStatus NullBackend::eventDestroy(EventHandle) { return BackendStatus::Ok; }
BackendStatus NullBackend::eventRecord(EventHandle, StreamHandle) {
    return BackendStatus::Ok;
}
BackendStatus NullBackend::eventSynchronize(EventHandle) { return BackendStatus::Ok; }
BackendStatus NullBackend::eventQuery(EventHandle) { return BackendStatus::Ok; }
BackendStatus NullBackend::streamWaitEvent(StreamHandle, EventHandle) {
    return BackendStatus::Ok;
}

BackendStatus NullBackend::moduleLoad(ModuleHandle *out, const void *image,
                                      unsigned) {
    if (!out || !image) return BackendStatus::InvalidValue;
    ++counters_.modulesLoaded;
    *out = (nextHandle_ += 0x10);
    return BackendStatus::Ok;
}

BackendStatus NullBackend::moduleUnload(ModuleHandle) { return BackendStatus::Ok; }

BackendStatus NullBackend::moduleGetFunction(FunctionHandle *out, ModuleHandle,
                                             const char *name) {
    if (!out || !name) return BackendStatus::InvalidValue;
    *out = (nextHandle_ += 0x10);
    return BackendStatus::Ok;
}

BackendStatus NullBackend::functionSetAttribute(int, FunctionHandle,
                                                const void *) {
    return BackendStatus::Ok;
}

BackendStatus NullBackend::functionGetAttribute(int, FunctionHandle, void *) {
    return BackendStatus::NotSupported;
}

BackendStatus NullBackend::launch(FunctionHandle, unsigned gx, unsigned gy,
                                  unsigned gz, unsigned bx, unsigned by,
                                  unsigned bz, unsigned, void **, void **,
                                  StreamHandle) {
    if (gx == 0 || gy == 0 || gz == 0) return BackendStatus::InvalidValue;
    if (bx == 0 || by == 0 || bz == 0) return BackendStatus::InvalidValue;
    ++counters_.launches;
    
    
    std::fprintf(stderr,
                 "[cuascend:null] launch grid=(%u,%u,%u) block=(%u,%u,%u) "
                 "rejected: null backend does not execute kernels.\n",
                 gx, gy, gz, bx, by, bz);
    return BackendStatus::Unsupported;
}

BackendStatus NullBackend::memcpy(void *dst, const void *src, size_t bytes,
                                  unsigned kind) {
    if (bytes && (!dst || !src)) return BackendStatus::InvalidValue;
    if (!bytes || kind == 1) { 
        if (bytes) std::memcpy(dst, src, bytes);
        return BackendStatus::Ok;
    }
    
    
    
    std::fprintf(stderr,
                 "[cuascend:null] device memcpy of %zu bytes rejected: no "
                 "device memory.\n",
                 bytes);
    return BackendStatus::Unsupported;
}

NullBackend::~NullBackend() = default;

Backend *makeBackend() {
    if (Backend *b = makeAscendBackend()) return b;
    return new NullBackend();
}

} 

// MAIN: the Ascend backend

#if CA_HAS_CANN

#include "acl/acl.h"
#include "acl/acl_op.h"

#include "acl/acl_op_compiler.h"

#include <cstdio>
#include <filesystem>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <map>
#include <iterator>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

namespace ca {

using ascend::CompileError;
using ascend::CompiledOp;
using ascend::OpCache;
using ascend::Toolchain;

namespace {

template <typename H> struct Registry {
    std::unordered_map<uint64_t, H> map;
    uint64_t next = 1;
};

std::mutex g_mu;
bool g_inited = false;

struct Streams {
    std::unordered_map<uint64_t, aclrtStream> map;
    uint64_t next = 1;
};
struct Events {
    std::unordered_map<uint64_t, aclrtEvent> map;
    uint64_t next = 1;
};

Streams g_streams;
Events g_events;

aclDataType toAclType(ascend::ElemType t) {
    switch (t) {
    case ascend::ElemType::Float32: return ACL_FLOAT;
    case ascend::ElemType::Float16: return ACL_FLOAT16;
    case ascend::ElemType::Int8: return ACL_INT8;
    case ascend::ElemType::Int16: return ACL_INT16;
    case ascend::ElemType::Int32: return ACL_INT32;
    case ascend::ElemType::Int64: return ACL_INT64;
    case ascend::ElemType::Uint8: return ACL_UINT8;
    case ascend::ElemType::Uint16: return ACL_UINT16;
    case ascend::ElemType::Uint32: return ACL_UINT32;
    case ascend::ElemType::Uint64: return ACL_UINT64;
    case ascend::ElemType::Unknown: break;
    }
    return ACL_FLOAT;
}

aclrtStream resolveStream(StreamHandle h) {
    if (h == kNullStream) return nullptr; 
    auto it = g_streams.map.find(h);
    return it == g_streams.map.end() ? nullptr : it->second;
}

} 

class AscendBackend : public Backend, public ir::Memory {
public:
    const char *name() const override { return "ascend"; }
    std::string lastError() const override { return ca::ascend::ascendLastError(); }

    BackendStatus init(int) override {
        std::lock_guard<std::mutex> lk(g_mu);
        if (g_inited) return BackendStatus::Ok;
        tc_ = Toolchain::fromEnvironment();
        aclError rc = aclInit(nullptr);
        if (rc != ACL_SUCCESS) {
            std::fprintf(stderr,
                         "[cuascend:ascend] aclInit failed: %s (code %d)\n",
                         aclGetRecentErrMsg(), rc);
            return BackendStatus::NoDevice;
        }
        g_inited = true;
        std::fprintf(stderr, "[cuascend:ascend] CANN initialised\n");
        std::fprintf(stderr, "[cuascend:ascend] toolchain: bisheng=%s msopgen=%s\n",
                     tc_.bisheng.c_str(), tc_.msopgen.c_str());
        return BackendStatus::Ok;
    }

    void shutdown() override {
        std::lock_guard<std::mutex> lk(g_mu);
        if (!g_inited) return;
        aclFinalize();
        g_inited = false;
    }

    bool initialized() const override { return g_inited; }

    int deviceCount() const override { return 1; }

    BackendStatus deviceGet(int ordinal, int &out) override {
        if (!g_inited) return BackendStatus::NotInitialized;
        if (ordinal != 0) return BackendStatus::InvalidValue;
        out = 0;
        return BackendStatus::Ok;
    }

    DeviceCaps deviceCaps(int) const override {
        DeviceCaps c;
        c.deviceCount = 1;
        
        c.warpSize = 32;
        c.maxThreadsPerBlock = 1024;
        c.maxThreadsDim[0] = 1024;
        c.maxThreadsDim[1] = 1024;
        c.maxThreadsDim[2] = 64;
        c.maxSharedMemoryPerBlock = 49152;
        c.unifiedAddressing = true;

        return c;
    }

    BackendStatus deviceGetAttribute(int attr, int *out, int) override {
        if (!out) return BackendStatus::InvalidValue;
        const DeviceCaps c = deviceCaps(0);
        switch (attr) {
        case 1: *out = c.maxThreadsPerBlock; return BackendStatus::Ok;
        case 5: *out = c.maxThreadsDim[0]; return BackendStatus::Ok;
        case 6: *out = c.maxThreadsDim[1]; return BackendStatus::Ok;
        case 7: *out = c.maxThreadsDim[2]; return BackendStatus::Ok;
        case 8: *out = c.maxSharedMemoryPerBlock; return BackendStatus::Ok;
        case 32: *out = c.warpSize; return BackendStatus::Ok;
        
        case 75: *out = c.major; return BackendStatus::Ok;
        case 76: *out = c.minor; return BackendStatus::Ok;
        default: break;
        }
        return BackendStatus::NotSupported;
    }

    std::string deviceName(int) const override { return "Ascend NPU"; }

    BackendStatus ctxCreate(unsigned, int, void **outCtx) override {
        if (!outCtx) return BackendStatus::InvalidValue;
        aclrtContext ctx = nullptr;
        const aclError rc = aclrtCreateContext(&ctx, 0);
        if (rc != ACL_SUCCESS) return BackendStatus::NoDevice;
        
        *outCtx = reinterpret_cast<void *>(ctx);
        return BackendStatus::Ok;
    }

    BackendStatus ctxDestroy(void *ctx) override {
        if (!ctx) return BackendStatus::InvalidValue;
        return toStatus(aclrtDestroyContext(reinterpret_cast<aclrtContext>(ctx)));
    }

    BackendStatus ctxSetCurrent(void *ctx) override {
        return toStatus(
            aclrtSetCurrentContext(reinterpret_cast<aclrtContext>(ctx)));
    }

    BackendStatus ctxGetCurrent(void **outCtx) override {
        if (!outCtx) return BackendStatus::InvalidValue;
        aclrtContext ctx = nullptr;
        const aclError rc = aclrtGetCurrentContext(&ctx);
        if (rc != ACL_SUCCESS) return BackendStatus::InvalidValue;
        *outCtx = reinterpret_cast<void *>(ctx);
        return BackendStatus::Ok;
    }

    BackendStatus ctxGetDevice(int *outDevice) override {
        if (!outDevice) return BackendStatus::InvalidValue;
        int dev = 0;
        const aclError rc = aclrtGetDevice(&dev);
        if (rc != ACL_SUCCESS) return BackendStatus::NoDevice;
        *outDevice = dev;
        return BackendStatus::Ok;
    }

    BackendStatus memAlloc(MemHandle *out, size_t bytes) override {
        if (!out || bytes == 0) return BackendStatus::InvalidValue;
        void *p = nullptr;
        const aclError rc = aclrtMalloc(&p, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (rc != ACL_SUCCESS) return BackendStatus::OutOfMemory;
        
        {
            std::lock_guard<std::mutex> lk(g_mu);
            allocations_[reinterpret_cast<uint64_t>(p)] = bytes;
        }
        *out = reinterpret_cast<MemHandle>(p);
        return BackendStatus::Ok;
    }

    BackendStatus memFree(MemHandle h) override {
        if (!h) return BackendStatus::InvalidValue;
        {
            std::lock_guard<std::mutex> lk(g_mu);
            allocations_.erase(h);
            hostCopies_.erase(h);
        }
        return toStatus(aclrtFree(reinterpret_cast<void *>(h)));
    }

    
    
    
    
    
    bool view(uint64_t address, ir::HostView &out) override {
        std::lock_guard<std::mutex> lk(g_mu);
        uint64_t base = 0;
        size_t size = 0;
        for (const auto &entry : allocations_) {
            if (address >= entry.first && address < entry.first + entry.second) {
                base = entry.first;
                size = entry.second;
                break;
            }
        }
        if (!size) return false;
        auto it = hostCopies_.find(base);
        if (it == hostCopies_.end()) {
            std::vector<uint8_t> copy(size);
            if (aclrtMemcpy(copy.data(), size, reinterpret_cast<void *>(base), size,
                            ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS)
                return false;
            it = hostCopies_.emplace(base, std::move(copy)).first;
        }
        out.base = base;
        out.data = it->second.data();
        out.size = it->second.size();
        return true;
    }

    void flush() override {
        std::lock_guard<std::mutex> lk(g_mu);
        for (auto &entry : hostCopies_) {
            aclrtMemcpy(reinterpret_cast<void *>(entry.first), entry.second.size(),
                        entry.second.data(), entry.second.size(),
                        ACL_MEMCPY_DEVICE_TO_HOST);
        }
    }

    BackendStatus memHostAlloc(void **out, size_t bytes, unsigned) override {
        if (!out || bytes == 0) return BackendStatus::InvalidValue;
        void *p = nullptr;
        
        const aclError rc = aclrtMallocHost(&p, bytes);
        if (rc != ACL_SUCCESS) return BackendStatus::OutOfMemory;
        *out = p;
        return BackendStatus::Ok;
    }

    BackendStatus memHostFree(void *p) override {
        if (!p) return BackendStatus::InvalidValue;
        return toStatus(aclrtFreeHost(p));
    }

    BackendStatus memHostRegister(void *, size_t, unsigned) {
        
        return BackendStatus::NotSupported;
    }
    BackendStatus memHostUnregister(void *) { return BackendStatus::Ok; }

    BackendStatus streamCreate(StreamHandle *out, unsigned) override {
        if (!out) return BackendStatus::InvalidValue;
        aclrtStream s = nullptr;
        const aclError rc = aclrtCreateStream(&s);
        if (rc != ACL_SUCCESS) return BackendStatus::OutOfMemory;
        const uint64_t h = (g_streams.next += 2);
        g_streams.map[h] = s;
        *out = h;
        return BackendStatus::Ok;
    }

    BackendStatus streamDestroy(StreamHandle h) override {
        if (h == kNullStream) return BackendStatus::Ok;
        auto it = g_streams.map.find(h);
        if (it == g_streams.map.end()) return BackendStatus::InvalidValue;
        const aclError rc = aclrtDestroyStream(it->second);
        g_streams.map.erase(it);
        return toStatus(rc);
    }

    BackendStatus streamSynchronize(StreamHandle h) override {
        
        
        
        
        
        if (h == kNullStream) return toStatus(aclrtSynchronizeDevice());
        return toStatus(aclrtSynchronizeStream(resolveStream(h)));
    }

    BackendStatus eventCreate(EventHandle *out, unsigned) override {
        if (!out) return BackendStatus::InvalidValue;
        aclrtEvent e = nullptr;
        const aclError rc = aclrtCreateEvent(&e);
        if (rc != ACL_SUCCESS) return BackendStatus::OutOfMemory;
        const uint64_t h = (g_events.next += 2);
        g_events.map[h] = e;
        *out = h;
        return BackendStatus::Ok;
    }

    BackendStatus eventDestroy(EventHandle h) override {
        auto it = g_events.map.find(h);
        if (it == g_events.map.end()) return BackendStatus::InvalidValue;
        const aclError rc = aclrtDestroyEvent(it->second);
        g_events.map.erase(it);
        return toStatus(rc);
    }

    BackendStatus eventRecord(EventHandle eh, StreamHandle sh) override {
        auto it = g_events.map.find(eh);
        if (it == g_events.map.end()) return BackendStatus::InvalidValue;
        return toStatus(aclrtRecordEvent(it->second, resolveStream(sh)));
    }

    BackendStatus eventSynchronize(EventHandle eh) override {
        auto it = g_events.map.find(eh);
        if (it == g_events.map.end()) return BackendStatus::InvalidValue;
        return toStatus(aclrtSynchronizeEvent(it->second));
    }

    BackendStatus eventQuery(EventHandle eh) override {
        auto it = g_events.map.find(eh);
        if (it == g_events.map.end()) return BackendStatus::InvalidValue;
        aclrtEventWaitStatus st = ACL_EVENT_WAIT_STATUS_COMPLETE;
        const aclError rc = aclrtQueryEventWaitStatus(it->second, &st);
        if (rc != ACL_SUCCESS) return BackendStatus::Unknown;
        
        return BackendStatus::Ok;
    }

    BackendStatus streamWaitEvent(StreamHandle sh, EventHandle eh) override {
        auto it = g_events.map.find(eh);
        if (it == g_events.map.end()) return BackendStatus::InvalidValue;
        return toStatus(aclrtStreamWaitEvent(resolveStream(sh), it->second));
    }

    
    BackendStatus moduleLoad(ModuleHandle *out, const void *image,
                             unsigned) override {
        if (!image) return BackendStatus::InvalidValue;

        
        Image img;
        try {
            const uint8_t *p = static_cast<const uint8_t *>(image);
            const size_t kMaxImage = 512u << 20; 
            size_t len = 0;
            if (!detectImageSize(p, kMaxImage, &len)) {
                std::fprintf(stderr,
                             "[cuascend:ascend] moduleLoad: cannot determine "
                             "image size; the image is neither ELF with a "
                             "section table nor NUL-terminated PTX\n");
                return BackendStatus::InvalidValue;
            }
            if (tc_.verbose)
                std::fprintf(stderr, "[cuascend:ascend] moduleLoad: %zu bytes\n",
                             len);
            img.bytes.assign(p, p + len);
            img.ptx = fatbin::parse(img.bytes.data(), img.bytes.size())
                          .firstPtx();
            if (img.ptx.empty())
                return BackendStatus::InvalidValue;
        } catch (const fatbin::Error &e) {
            std::fprintf(stderr, "[cuascend:ascend] moduleLoad: %s\n", e.what());
            return BackendStatus::InvalidValue;
        }

        const uint64_t h = nextModule_++;
        modules_.emplace(h, std::move(img));
        if (out) *out = h;
        return BackendStatus::Ok;
    }

    BackendStatus moduleUnload(ModuleHandle h) override {
        if (!modules_.erase(h)) return BackendStatus::InvalidValue;
        return BackendStatus::Ok;
    }

    BackendStatus moduleGetFunction(FunctionHandle *out, ModuleHandle m,
                                    const char *name) override {
        if (!out || !name) return BackendStatus::InvalidValue;
        auto it = modules_.find(m);
        if (it == modules_.end()) return BackendStatus::InvalidValue;

        Function f;
        f.module = m;
        f.name = name;
        
        const uint64_t h = nextFunction_++;
        functions_.emplace(h, std::move(f));
        *out = h;
        return BackendStatus::Ok;
    }

    BackendStatus functionSetAttribute(int attr, FunctionHandle f,
                                       const void *value) override {
        auto it = functions_.find(f);
        if (it == functions_.end()) return BackendStatus::InvalidValue;
        if (attr == 1 && value) { 
            it->second.sharedBytes =
                *static_cast<const unsigned *>(value);
        }
        return BackendStatus::Ok;
    }

    BackendStatus functionGetAttribute(int attr, FunctionHandle f,
                                       void *value) override {
        auto it = functions_.find(f);
        if (it == functions_.end()) return BackendStatus::InvalidValue;
        if (!value) return BackendStatus::InvalidValue;
        
        switch (attr) {
        case 0: 
            *static_cast<int *>(value) = 1024;
            return BackendStatus::Ok;
        case 1: 
            *static_cast<int *>(value) = static_cast<int>(it->second.sharedBytes);
            return BackendStatus::Ok;
        case 4: 
            *static_cast<int *>(value) = 0;
            return BackendStatus::Ok;
        default: break;
        }
        return BackendStatus::NotSupported;
    }

    
    BackendStatus launch(FunctionHandle fh, unsigned gx, unsigned gy,
                         unsigned gz, unsigned bx, unsigned by, unsigned bz,
                         unsigned sharedBytes, void **args, void **,
                         StreamHandle sh) override {
        auto fit = functions_.find(fh);
        if (fit == functions_.end()) return BackendStatus::InvalidValue;
        Function &fn = fit->second;

        if (gx == 0 || gy == 0 || gz == 0) return BackendStatus::InvalidValue;
        if (bx == 0 || by == 0 || bz == 0) return BackendStatus::InvalidValue;
        if (!args) return BackendStatus::InvalidValue;

        const unsigned dynShared = sharedBytes;
        if (dynShared > 49152u) {
            std::fprintf(stderr,
                         "[cuascend:ascend] kernel '%s' requests %u bytes of "
                         "dynamic shared memory, limit is 49152\n",
                         fn.name.c_str(), dynShared);
            return BackendStatus::InvalidValue;
        }

        
        
        if (!fn.compiled) {
            auto mit = modules_.find(fn.module);
            if (mit == modules_.end()) return BackendStatus::InvalidValue;
            try {
                fn.op = cache_->OpCache::get(mit->second.ptx, fn.name, tc_);
                if (!fn.op.valid) {
                    std::fprintf(stderr,
                                 "[cuascend:ascend] kernel '%s' compiled to an "
                                 "invalid operation (model missing)\n",
                                 fn.name.c_str());
                    return BackendStatus::InvalidValue;
                }
            } catch (const CompileError &e) {
                
                
                
                
                fn.compileError = e.what();
                if (!e.output.empty())
                    fn.compileError += "\n--- toolchain output ---\n" + e.output;
                fn.op = CompiledOp{};
                try {
                    const ptx::Module pm = ptx::parse(mit->second.ptx, "<fatbin>");
                    const ir::Module im = lower::run(pm).module;
                    for (const auto &f : im.functions)
                        if (f.name == fn.name) {
                            fn.op.ir = std::make_shared<ir::Function>(f);
                            break;
                        }
                } catch (const std::exception &irError) {
                    fn.compileError += std::string("\nIR extraction: ") + irError.what();
                }
                if (!fn.op.ir) {
                    std::fprintf(stderr, "[cuascend:ascend] kernel '%s':\n%s\n",
                                 fn.name.c_str(), fn.compileError.c_str());
                    return BackendStatus::InvalidValue;
                }
            }
            fn.compiled = true;
            if (fn.compileError.empty())
                std::fprintf(stderr, "[cuascend:ascend] kernel '%s' prepared -> %s\n",
                             fn.name.c_str(), fn.op.sourceDir.c_str());
        }

        
        const uint64_t totalThreads =
            static_cast<uint64_t>(gx) * gy * gz * bx * by * bz;

        
        
        
        
        
        const bool oneDimensional =
            gy == 1 && gz == 1 && by == 1 && bz == 1;
        if (oneDimensional && fn.op.ir &&
            ascend::runElementwiseAclnn(*fn.op.ir, args, gx, bx,
                                        resolveStream(sh))) {
            return BackendStatus::Ok;
        }

        
        
        
        
        const size_t n = fn.op.inputCount;
        const bool opDescriptionUsable =
            n != 0 && n <= fn.op.paramTypes.size() &&
            n <= fn.op.paramSizes.size();
        BackendStatus opStatus = BackendStatus::InvalidValue;
        if (opDescriptionUsable) {
            std::vector<aclTensorDesc *> descs;
            std::vector<aclDataBuffer *> bufs;
            std::vector<const aclTensorDesc *> cdescs;
            descs.reserve(n);
            bufs.reserve(n);
            cdescs.reserve(n);

            for (size_t i = 0; i < n; ++i) {
                uint64_t devAddr = 0;
                std::memcpy(&devAddr, args[i], sizeof devAddr);

                const aclDataType dt = toAclType(fn.op.paramTypes[i]);
                const int64_t dims = static_cast<int64_t>(totalThreads);
                aclTensorDesc *d =
                    aclCreateTensorDesc(dt, 1, &dims, ACL_FORMAT_ND);
                if (!d) return BackendStatus::OutOfMemory;

                const size_t bytes =
                    static_cast<size_t>(dims) * fn.op.paramSizes[i];
                aclDataBuffer *b = aclCreateDataBuffer(
                    reinterpret_cast<void *>(devAddr), bytes);
                if (!b) {
                    aclDestroyTensorDesc(d);
                    return BackendStatus::OutOfMemory;
                }
                descs.push_back(d);
                bufs.push_back(b);
                cdescs.push_back(d);
            }

            aclrtStream stream = resolveStream(sh);

            
            if (!fn.registered) {
                
                if (fn.op.modelPath.empty()) {
                    std::fprintf(stderr,
                                 "[cuascend:ascend] kernel '%s' compiled to no "
                                 "model; the op package must be produced\n",
                                 fn.name.c_str());
                    for (auto b : bufs) aclDestroyDataBuffer(b);
                    for (auto d : descs) aclDestroyTensorDesc(d);
                    return BackendStatus::InvalidValue;
                }

                static bool modelDirSet = false;
                
                const char *skip = std::getenv("CUASCEND_MODEL_DIR");
                if (!modelDirSet && !(skip && std::string(skip) == "none")) {
                    
                    const char *root =
                        std::getenv("ASCEND_TOOLKIT_ROOT");
                    const fs::path cannRoot =
                        root ? fs::path(root)
                             : fs::path("/usr/local/Ascend/ascend-toolkit/latest");
                    fs::path modelDir =
                        cannRoot / "opp" / "vendors" / "customize";
                    const char *overrideDir = std::getenv("CUASCEND_MODEL_DIR");
                    if (overrideDir && *overrideDir &&
                        std::string(overrideDir) != "none")
                        modelDir = overrideDir;
                    const aclError sr = aclopSetModelDir(modelDir.c_str());
                    if (sr != ACL_SUCCESS) {
                        std::fprintf(stderr,
                                     "[cuascend:ascend] aclopSetModelDir(%s) "
                                     "failed for '%s' (code %d)\n",
                                     modelDir.c_str(), fn.name.c_str(),
                                     static_cast<int>(sr));
                        for (auto b : bufs) aclDestroyDataBuffer(b);
                        for (auto d : descs) aclDestroyTensorDesc(d);
                        return BackendStatus::InvalidValue;
                    }
                    modelDirSet = true;
                    std::fprintf(stderr,
                                 "[cuascend:ascend] op model dir: %s\n",
                                 modelDir.c_str());
                }
                fn.registered = true;
            }

            
            const size_t nOut = fn.op.outputCount ? fn.op.outputCount : 1;
            const size_t nIn = n > nOut ? n - nOut : n;

            std::vector<const aclTensorDesc *> outDescs;
            std::vector<aclDataBuffer *> outBufs;
            std::vector<aclTensorDesc *> outKeep;
            for (size_t i = nIn; i < n; ++i) {
                uint64_t devAddr = 0;
                std::memcpy(&devAddr, args[i], sizeof devAddr);
                const aclDataType dt = toAclType(fn.op.paramTypes[i]);
                const int64_t dims = static_cast<int64_t>(totalThreads);
                aclTensorDesc *d =
                    aclCreateTensorDesc(dt, 1, &dims, ACL_FORMAT_ND);
                const size_t bytes =
                    static_cast<size_t>(dims) * fn.op.paramSizes[i];
                aclDataBuffer *b = aclCreateDataBuffer(
                    reinterpret_cast<void *>(devAddr), bytes);
                if (!d || !b) {
                    if (d) aclDestroyTensorDesc(d);
                    if (b) aclDestroyDataBuffer(b);
                    for (auto x : descs) aclDestroyTensorDesc(x);
                    for (auto x : bufs) aclDestroyDataBuffer(x);
                    return BackendStatus::OutOfMemory;
                }
                outDescs.push_back(d);
                outBufs.push_back(b);
                outKeep.push_back(d);
            }

            const aclError rc = aclopExecute(
                fn.op.kernelName.c_str(),
                static_cast<int>(nIn), cdescs.data(), bufs.data(),
                static_cast<int>(outDescs.size()), outDescs.data(),
                outBufs.data(), nullptr, stream);

            
            if (rc != ACL_SUCCESS) {
                std::fprintf(stderr,
                             "[cuascend:ascend] aclopExecute('%s', inputs=%d) "
                             "failed: acl error %d\n",
                             fn.op.kernelName.c_str(), static_cast<int>(n),
                             static_cast<int>(rc));
            }

            for (auto b : outBufs) aclDestroyDataBuffer(b);
            for (auto d : outKeep) aclDestroyTensorDesc(d);
            for (auto b : bufs) aclDestroyDataBuffer(b);
            for (auto d : descs) aclDestroyTensorDesc(d);
            opStatus = toStatus(rc);
        } else {
            std::fprintf(stderr,
                         "[cuascend:ascend] kernel '%s' has no usable op "
                         "description (inputs=%zu, types=%zu, sizes=%zu)\n",
                         fn.name.c_str(), n, fn.op.paramTypes.size(),
                         fn.op.paramSizes.size());
        }

        
        
        
        
        if (opStatus != BackendStatus::Ok && fn.op.ir) {
        
        
        
        
        if (fn.op.ir) {
            ir::LaunchConfig config;
            config.gridX = gx;
            config.gridY = gy;
            config.gridZ = gz;
            config.blockX = bx;
            config.blockY = by;
            config.blockZ = bz;
            config.sharedBytes = sharedBytes;
            std::string error;
            if (ir::interpret(*fn.op.ir, config, args, *this, &error)) {
                std::fprintf(stderr,
                             "[cuascend:ascend] kernel '%s' executed by the IR "
                             "interpreter\n",
                             fn.name.c_str());
                return BackendStatus::Ok;
            }
            std::fprintf(stderr,
                         "[cuascend:ascend] kernel '%s' cannot be executed: %s\n",
                         fn.name.c_str(), error.c_str());
            return BackendStatus::Unsupported;
        }

        }

        
        
        if (!fn.compileError.empty()) {
            std::fprintf(stderr, "[cuascend:ascend] kernel '%s' failed:\n%s\n",
                         fn.name.c_str(), fn.compileError.c_str());
            return BackendStatus::InvalidValue;
        }

        return opStatus;
    }

    BackendStatus memcpy(void *dst, const void *src, size_t bytes,
                         unsigned kind) override {
        if (!bytes) return BackendStatus::Ok;
        if (!dst || !src) return BackendStatus::InvalidValue;
        switch (kind) {
        case 1: 
            std::memcpy(dst, src, bytes);
            return BackendStatus::Ok;
        case 2: 
            return toStatus(aclrtMemcpy(dst, bytes, src, bytes,
                                        ACL_MEMCPY_HOST_TO_DEVICE));
        case 3: 
            return toStatus(aclrtMemcpy(dst, bytes, src, bytes,
                                        ACL_MEMCPY_DEVICE_TO_HOST));
        case 4: 
            return toStatus(aclrtMemcpy(dst, bytes, src, bytes,
                                        ACL_MEMCPY_DEVICE_TO_DEVICE));
        case 0: 
            return toStatus(aclrtMemcpy(dst, bytes, src, bytes,
                                        ACL_MEMCPY_DEVICE_TO_DEVICE));
        default: break;
        }
        return BackendStatus::InvalidValue;
    }

public:
    AscendBackend() { cache_ = std::make_unique<OpCache>(tc_.workDir); }

private:
    
    
    std::map<uint64_t, size_t> allocations_;
    std::map<uint64_t, std::vector<uint8_t>> hostCopies_;

    
    struct Image {
        std::vector<uint8_t> bytes;
        std::string ptx;
    };

    
    struct Function {
        uint64_t module = 0;
        std::string name;
        unsigned sharedBytes = 0;
        bool compiled = false;
        bool registered = false; 
        
        
        
        std::string compileError;
        CompiledOp op;
    };

    
    static bool detectImageSize(const uint8_t *p, size_t max, size_t *out) {
        if (max < 64) return false;

        if (std::memcmp(p, "\x7f" "ELF", 4) == 0) {
            const bool is64 = p[4] == 2; 
            uint64_t shoff = 0, phoff = 0;
            uint16_t shnum = 0, phnum = 0;
            uint16_t shentsize = 0, phentsize = 0;
            if (is64) {
                std::memcpy(&shoff, p + 0x28, 8);
                std::memcpy(&phoff, p + 0x20, 8);
                std::memcpy(&shnum, p + 0x3C, 2);
                std::memcpy(&shentsize, p + 0x3A, 2);
                std::memcpy(&phnum, p + 0x38, 2);
                std::memcpy(&phentsize, p + 0x36, 2);
            } else {
                uint32_t t32 = 0;
                std::memcpy(&t32, p + 0x20, 4); phoff = t32;
                std::memcpy(&t32, p + 0x28, 4); shoff = t32;
                std::memcpy(&shnum, p + 0x30, 2);
                std::memcpy(&shentsize, p + 0x2E, 2);
                std::memcpy(&phnum, p + 0x2C, 2);
                std::memcpy(&phentsize, p + 0x2A, 2);
            }
            uint64_t end = 0;
            if (shoff && shnum)
                end = std::max(end, shoff + uint64_t(shnum) * shentsize);
            if (phoff && phnum)
                end = std::max(end, phoff + uint64_t(phnum) * phentsize);
            if (!end || end > max) return false;
            *out = static_cast<size_t>(end);
            return true;
        }

        
        const size_t window = std::min<size_t>(max, 64u << 20);
        const uint8_t *nul =
            static_cast<const uint8_t *>(std::memchr(p, 0, window));
        if (!nul) return false;
        *out = static_cast<size_t>(nul - p);
        return *out > 0;
    }

    static BackendStatus toStatus(aclError rc) {
        if (rc == ACL_SUCCESS) return BackendStatus::Ok;
        if (rc == ACL_ERROR_INVALID_PARAM) return BackendStatus::InvalidValue;
        if (rc == ACL_ERROR_BAD_ALLOC) return BackendStatus::OutOfMemory;
        return BackendStatus::Unknown;
    }

    // The toolchain is captured once, when the runtime starts; after that the environment is ignored.
    Toolchain tc_;
    std::unique_ptr<OpCache> cache_;
    std::unordered_map<uint64_t, Image> modules_;
    std::unordered_map<uint64_t, Function> functions_;
    uint64_t nextModule_ = 1;
    uint64_t nextFunction_ = 1;
};

Backend *makeAscendBackend() { return new AscendBackend(); }

} 

#else 

namespace ca {

Backend *makeAscendBackend() { return nullptr; }

} 

#endif

// MAIN: the aclnn operations

#if CA_HAS_CANN

#include <acl/acl.h>
#include <aclnnop/aclnn_add.h>
#include <aclnnop/aclnn_mul.h>
#include <aclnnop/aclnn_sub.h>

#include <map>
#include <mutex>

namespace ca::aclnnexec {
namespace {

struct TensorKey {
    int64_t elements;
    const void *data;
    bool operator<(const TensorKey &o) const {
        if (elements != o.elements) return elements < o.elements;
        return data < o.data;
    }
};

std::mutex g_mutex;
std::map<TensorKey, aclTensor *> g_tensors;
aclScalar *g_alpha = nullptr;

aclTensor *tensorFor(int64_t elements, void *data) {
    const TensorKey key{elements, data};
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_tensors.find(key);
    if (it != g_tensors.end()) return it->second;

    
    
    int64_t dims[1] = {elements};
    aclTensor *tensor =
        aclCreateTensor(dims, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, nullptr, 0,
                        data);
    if (!tensor) return nullptr;
    g_tensors.emplace(key, tensor);
    return tensor;
}

class ElementwiseOp : public Op {
public:
    enum class Kind { Add, Mul, Sub };

    explicit ElementwiseOp(Kind kind) : kind_(kind) {}

    const char *name() const override {
        switch (kind_) {
        case Kind::Add: return "aclnnAdd";
        case Kind::Mul: return "aclnnMul";
        default: return "aclnnSub";
        }
    }

    bool run(void *a, void *b, void *out, int64_t n, float alpha,
             void *stream) override {
        if (!a || !b || !out || n <= 0) return false;

        aclTensor *lhs = tensorFor(n, a);
        aclTensor *rhs = tensorFor(n, b);
        aclTensor *dst = tensorFor(n, out);
        if (!lhs || !rhs || !dst) return false;

        aclScalar *scale = nullptr;
        if (kind_ != Kind::Mul) {
            std::lock_guard<std::mutex> lock(g_mutex);
            if (!g_alpha) {
                float one = 1.0f;
                g_alpha = aclCreateScalar(&one, ACL_FLOAT);
            }
            scale = g_alpha;
            if (!scale) return false;
        }

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        aclError rc;
        if (kind_ == Kind::Mul)
            rc = aclnnMulGetWorkspaceSize(lhs, rhs, dst, &workspaceSize, &executor);
        else if (kind_ == Kind::Add)
            rc = aclnnAddGetWorkspaceSize(lhs, rhs, scale, dst, &workspaceSize,
                                          &executor);
        else
            rc = aclnnSubGetWorkspaceSize(lhs, rhs, scale, dst, &workspaceSize,
                                          &executor);
        if (rc != ACL_SUCCESS || !executor) return false;

        
        
        
        if (workspaceSize != 0) return false;

        const aclrtStream streamHandle = static_cast<aclrtStream>(stream);
        if (kind_ == Kind::Mul)
            rc = aclnnMul(nullptr, workspaceSize, executor, streamHandle);
        else if (kind_ == Kind::Add)
            rc = aclnnAdd(nullptr, workspaceSize, executor, streamHandle);
        else
            rc = aclnnSub(nullptr, workspaceSize, executor, streamHandle);
        return rc == ACL_SUCCESS;
    }

private:
    Kind kind_;
};

} 

Op *makeAdd() { return new ElementwiseOp(ElementwiseOp::Kind::Add); }
Op *makeMul() { return new ElementwiseOp(ElementwiseOp::Kind::Mul); }
Op *makeSub() { return new ElementwiseOp(ElementwiseOp::Kind::Sub); }

} 

#endif 

// MAIN: the elementwise lowering

#if CA_HAS_CANN

#include "acl/acl.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <vector>

namespace ca::ascend {
namespace {

bool debug() {
    static const bool on = std::getenv("CUASCEND_ACLNN_DEBUG") != nullptr;
    return on;
}

enum class Coord { Thread, Block, WarpLane, Warp };

enum class FloatOp { Add, Mul, Sub };

enum class IntOp { Add, Sub, Mul, DivU, ShrU, Shl };

enum class Pred { Eq, Ne, LtU, LeU, GtU, GeU, LtS, LeS, GtS, GeS, Not, And, Or };

int predicateOf(ir::Op op) {
    switch (op) {
    case ir::Op::CmpEq: return static_cast<int>(Pred::Eq);
    case ir::Op::CmpNe: return static_cast<int>(Pred::Ne);
    case ir::Op::CmpLtU: return static_cast<int>(Pred::LtU);
    case ir::Op::CmpLeU: return static_cast<int>(Pred::LeU);
    case ir::Op::CmpGtU: return static_cast<int>(Pred::GtU);
    case ir::Op::CmpGeU: return static_cast<int>(Pred::GeU);
    case ir::Op::CmpLtS: return static_cast<int>(Pred::LtS);
    case ir::Op::CmpLeS: return static_cast<int>(Pred::LeS);
    case ir::Op::CmpGtS: return static_cast<int>(Pred::GtS);
    case ir::Op::CmpGeS: return static_cast<int>(Pred::GeS);
    default: return -1;
    }
}

int intOpOf(ir::Op op) {
    switch (op) {
    case ir::Op::Add: return static_cast<int>(IntOp::Add);
    case ir::Op::Sub: return static_cast<int>(IntOp::Sub);
    case ir::Op::Mul: return static_cast<int>(IntOp::Mul);
    case ir::Op::DivU: return static_cast<int>(IntOp::DivU);
    case ir::Op::ShrU: return static_cast<int>(IntOp::ShrU);
    case ir::Op::Shl: return static_cast<int>(IntOp::Shl);
    default: return -1;
    }
}

struct Value {
    enum Kind : uint8_t {
        Constant, 
        Pointer,  
        Tensor,   
        Expr,     
        Index,    
        Lane,     
        Mask,     
    } kind = Constant;

    int param = -1;     
    float number = 0;   
    int floatOp = 0;    
    int intOp = 0;      
    int pred = 0;       
    int a = -1, b = -1; 
    bool laneIndexed = false; 
    Coord coord = Coord::Thread;
};

constexpr uint64_t kSyntheticBase = 1 << 20;

std::mutex g_poolMutex;
struct PoolSlot {
    void *ptr;
    size_t capacity;
    bool busy;
};
std::vector<PoolSlot> g_pool;

void *poolAlloc(size_t bytes) {
    std::lock_guard<std::mutex> lock(g_poolMutex);
    for (PoolSlot &slot : g_pool) {
        if (!slot.busy && slot.capacity >= bytes) {
            slot.busy = true;
            return slot.ptr;
        }
    }
    void *ptr = nullptr;
    if (aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS)
        return nullptr;
    g_pool.push_back(PoolSlot{ptr, bytes, true});
    return ptr;
}

void poolRelease(void *ptr) {
    if (!ptr) return;
    std::lock_guard<std::mutex> lock(g_poolMutex);
    for (PoolSlot &slot : g_pool)
        if (slot.ptr == ptr) {
            slot.busy = false;
            return;
        }
}

} 

bool runElementwiseAclnn(const ir::Function &fn, void **args, unsigned gridX,
                         unsigned blockX, void *stream) {
    if (!args || gridX == 0 || blockX == 0) return false;
    const uint64_t laneCount = static_cast<uint64_t>(gridX) * blockX;
    if (debug())
        std::fprintf(stderr, "[aclnn] fn=%s params=%zu lanes=%llu grid=%u block=%u\n",
                     fn.name.c_str(), fn.params.size(),
                     (unsigned long long)laneCount, gridX, blockX);

    
    const size_t paramCount = fn.params.size();
    std::vector<void *> buffers(paramCount, nullptr);
    std::vector<float> scalars(paramCount, 0.0f);
    for (size_t i = 0; i < paramCount; ++i) {
        const ir::Param &p = fn.params[i];
        const bool isAddress =
            p.isPointer || (p.type.isInt() && p.type.width == 64);
        if (p.type.isFloat() && p.type.width != 32) {
            if (debug())
                std::fprintf(stderr, "[aclnn] refuse: param %zu is f%u\n", i,
                             (unsigned)p.type.width);
            return false;
        }
        if (isAddress) {
            uint64_t raw = 0;
            std::memcpy(&raw, args[i], sizeof raw);
            buffers[i] = reinterpret_cast<void *>(raw);
        } else if (p.type.isFloat()) {
            float v = 0;
            std::memcpy(&v, args[i], sizeof v);
            scalars[i] = v;
        } else {
            const size_t bytes = p.type.width / 8;
            if (bytes == 0 || bytes > 8) return false;
            int64_t v = 0;
            std::memcpy(&v, args[i], bytes);
            scalars[i] = static_cast<float>(v);
        }
        if (debug())
            std::fprintf(stderr, "[aclnn]   param %zu %s %s = %s\n", i,
                         p.name.c_str(), isAddress ? "addr" : "value",
                         isAddress ? std::to_string(
                                         reinterpret_cast<uint64_t>(buffers[i]))
                                       .c_str()
                                   : std::to_string(scalars[i]).c_str());
    }

    std::map<uint64_t, Value> values;
    std::map<std::string, int> paramByName;
    for (size_t i = 0; i < paramCount; ++i)
        paramByName[fn.params[i].name] = static_cast<int>(i);

    uint64_t syntheticId = kSyntheticBase;
    auto noteId = [&](ir::ValueId id) {
        if (id != ir::kInvalidValue && static_cast<uint64_t>(id) + 1 > syntheticId)
            syntheticId = static_cast<uint64_t>(id) + 1;
    };
    for (const auto &block : fn.blocks)
        for (const auto &in : block.instrs) {
            noteId(in.result);
            noteId(in.pred);
            for (const auto &o : in.args) noteId(o.value);
            noteId(in.addr.base.value);
            noteId(in.addr.index.value);
        }

    std::map<int64_t, uint64_t> constantIdOf;
    std::vector<uint64_t> parameterId(paramCount, 0);

    auto lookup = [&](int id) -> const Value * {
        if (id < 0) return nullptr;
        auto it = values.find(static_cast<uint64_t>(id));
        return it == values.end() ? nullptr : &it->second;
    };

    auto insert = [&](const Value &v) {
        const uint64_t id = syntheticId++;
        values[id] = v;
        return static_cast<int>(id);
    };

    auto makeConstant = [&](float number) {
        Value v;
        v.kind = Value::Constant;
        v.number = number;
        return insert(v);
    };

    
    
    auto parameterIndex = [&](const ir::Operand &o) -> int {
        if (o.kind != ir::Operand::Kind::ParamRef || o.symbol.empty()) return -1;
        if (o.symbol[0] == '$') {
            const long pos = std::strtol(o.symbol.c_str() + 1, nullptr, 10);
            return pos >= 0 && static_cast<size_t>(pos) < paramCount
                       ? static_cast<int>(pos)
                       : -1;
        }
        auto it = paramByName.find(o.symbol);
        return it == paramByName.end() ? -1 : it->second;
    };

    
    
    
    
    auto operandId = [&](const ir::Operand &o, bool isFloat) -> int {
        switch (o.kind) {
        case ir::Operand::Kind::Value:
            return values.count(o.value) ? static_cast<int>(o.value) : -1;
        case ir::Operand::Kind::Immediate: {
            if (isFloat) return -1;
            auto it = constantIdOf.find(o.imm);
            if (it != constantIdOf.end()) return static_cast<int>(it->second);
            const int id = makeConstant(static_cast<float>(o.imm));
            constantIdOf[o.imm] = static_cast<uint64_t>(id);
            return id;
        }
        case ir::Operand::Kind::ParamRef: {
            const int slot = parameterIndex(o);
            if (slot < 0) return -1;
            if (!parameterId[slot]) {
                Value v;
                if (buffers[slot]) { v.kind = Value::Pointer; v.param = slot; }
                else { v.kind = Value::Constant; v.number = scalars[slot]; }
                parameterId[slot] = static_cast<uint64_t>(insert(v));
            }
            return static_cast<int>(parameterId[slot]);
        }
        default:
            return -1;
        }
    };

    
    
    
    
    

    auto conditionNot = [&](int id) -> int {
        if (id < 0) return makeConstant(0.0f);
        const Value *v = lookup(id);
        if (v && v->kind == Value::Constant)
            return makeConstant(v->number == 0.0f ? 1.0f : 0.0f);
        Value m;
        m.kind = Value::Mask;
        m.pred = static_cast<int>(Pred::Not);
        m.a = id;
        return insert(m);
    };

    auto conditionAnd = [&](int left, int right) -> int {
        if (left < 0) return right;
        if (right < 0) return left;
        const Value *l = lookup(left);
        const Value *r = lookup(right);
        if (l && l->kind == Value::Constant)
            return l->number != 0.0f ? right : makeConstant(0.0f);
        if (r && r->kind == Value::Constant)
            return r->number != 0.0f ? left : makeConstant(0.0f);
        Value m;
        m.kind = Value::Mask;
        m.pred = static_cast<int>(Pred::And);
        m.a = left;
        m.b = right;
        return insert(m);
    };

    auto conditionOr = [&](int left, int right) -> int {
        if (left < 0 || right < 0) return -1;
        const Value *l = lookup(left);
        const Value *r = lookup(right);
        if (l && l->kind == Value::Constant) return l->number != 0.0f ? left : right;
        if (r && r->kind == Value::Constant) return r->number != 0.0f ? right : left;
        Value m;
        m.kind = Value::Mask;
        m.pred = static_cast<int>(Pred::Or);
        m.a = left;
        m.b = right;
        return insert(m);
    };

    
    auto coordinate = [&](Coord coord, uint64_t lane) -> int64_t {
        const uint64_t inBlock = lane % blockX;
        switch (coord) {
        case Coord::Thread: return static_cast<int64_t>(inBlock);
        case Coord::Block: return static_cast<int64_t>(lane / blockX);
        case Coord::WarpLane: return static_cast<int64_t>(inBlock % 32);
        case Coord::Warp: return static_cast<int64_t>(inBlock / 32);
        }
        return 0;
    };

    
    
    std::function<bool(int, uint64_t, int64_t &)> indexValue;
    std::function<bool(int, uint64_t, bool &)> conditionHolds;
    std::function<bool(int)> indexOk;
    std::function<bool(int)> conditionOk;

    
    indexValue = [&](int id, uint64_t lane, int64_t &out) -> bool {
        const Value *v = lookup(id);
        if (!v) return false;
        if (v->kind == Value::Constant) { out = static_cast<int64_t>(v->number); return true; }
        if (v->kind == Value::Lane) { out = coordinate(v->coord, lane); return true; }
        if (v->kind == Value::Mask) {
            bool b = false;
            if (!conditionHolds(id, lane, b)) return false;
            out = b ? 1 : 0;
            return true;
        }
        if (v->kind != Value::Index) return false;
        int64_t x = 0, y = 0;
        if (!indexValue(v->a, lane, x)) return false;
        if (!indexValue(v->b, lane, y)) return false;
        switch (static_cast<IntOp>(v->intOp)) {
        case IntOp::Add: out = x + y; return true;
        case IntOp::Sub: out = x - y; return true;
        case IntOp::Mul: out = x * y; return true;
        case IntOp::DivU: if (y == 0) return false;
            out = static_cast<int64_t>(static_cast<uint64_t>(x) / static_cast<uint64_t>(y));
            return true;
        case IntOp::ShrU: out = static_cast<int64_t>(static_cast<uint64_t>(x) >> (y & 63)); return true;
        case IntOp::Shl: out = static_cast<int64_t>(static_cast<uint64_t>(x) << (y & 63)); return true;
        }
        return false;
    };

    
    conditionHolds = [&](int id, uint64_t lane, bool &out) -> bool {
        const Value *v = lookup(id);
        if (!v) return false;
        if (v->kind == Value::Constant) { out = v->number != 0.0f; return true; }
        if (v->kind != Value::Mask) return false;
        const Pred p = static_cast<Pred>(v->pred);
        if (p == Pred::Not) {
            bool b = false;
            if (!conditionHolds(v->a, lane, b)) return false;
            out = !b;
            return true;
        }
        if (p == Pred::And || p == Pred::Or) {
            bool x = false, y = false;
            if (!conditionHolds(v->a, lane, x) || !conditionHolds(v->b, lane, y))
                return false;
            out = (p == Pred::And) ? (x && y) : (x || y);
            return true;
        }
        int64_t x = 0, y = 0;
        if (!indexValue(v->a, lane, x) || !indexValue(v->b, lane, y)) return false;
        switch (p) {
        case Pred::Eq: out = x == y; break;
        case Pred::Ne: out = x != y; break;
        case Pred::LtU: out = static_cast<uint64_t>(x) < static_cast<uint64_t>(y); break;
        case Pred::LeU: out = static_cast<uint64_t>(x) <= static_cast<uint64_t>(y); break;
        case Pred::GtU: out = static_cast<uint64_t>(x) > static_cast<uint64_t>(y); break;
        case Pred::GeU: out = static_cast<uint64_t>(x) >= static_cast<uint64_t>(y); break;
        case Pred::LtS: out = x < y; break;
        case Pred::LeS: out = x <= y; break;
        case Pred::GtS: out = x > y; break;
        case Pred::GeS: out = x >= y; break;
        default: return false;
        }
        return true;
    };

    
    
    indexOk = [&](int id) -> bool {
        const Value *v = lookup(id);
        if (!v) return false;
        if (v->kind == Value::Constant || v->kind == Value::Lane) return true;
        if (v->kind != Value::Index) return false;
        return indexOk(v->a) && indexOk(v->b);
    };

    
    
    conditionOk = [&](int id) -> bool {
        if (id < 0) return true;
        const Value *v = lookup(id);
        if (!v) return false;
        if (v->kind == Value::Constant) return true;
        if (v->kind != Value::Mask) return false;
        const Pred p = static_cast<Pred>(v->pred);
        if (p == Pred::Not) return conditionOk(v->a);
        if (p == Pred::And || p == Pred::Or)
            return conditionOk(v->a) && conditionOk(v->b);
        auto operandOk = [&](int id) {
            const Value *w = lookup(id);
            if (!w) return false;
            return w->kind == Value::Lane || w->kind == Value::Constant ||
                   (w->kind == Value::Index && indexOk(id)) ||
                   (w->kind == Value::Mask && conditionOk(id));
        };
        return operandOk(v->a) && operandOk(v->b);
    };

    

    
    
    
    std::vector<int> blockGuard(fn.blocks.size(), -1);
    std::vector<bool> reachable(fn.blocks.size(), false);
    if (!fn.blocks.empty()) reachable[0] = true;

    auto addEdge = [&](size_t target, int guard) {
        if (target >= fn.blocks.size()) return;
        blockGuard[target] =
            reachable[target] ? conditionOr(blockGuard[target], guard) : guard;
        reachable[target] = true;
    };

    struct StoreSite {
        int param;        
        int value;        
        int guard;        
        bool laneIndexed; 
        size_t bytes;     
    };
    std::vector<StoreSite> stores;
    bool usesBlockIndex = false;

    
    
    
    auto addressParameter = [&](const ir::Address &addr, bool &laneIndexed) -> int {
        laneIndexed = false;
        if (addr.offset != 0) return -1;
        if (addr.hasIndex) {
            const int index = operandId(addr.index, false);
            const Value *v = index >= 0 ? lookup(index) : nullptr;
            if (!v || v->kind != Value::Lane) return -1;
            if (addr.scale != 0 && addr.scale != 4) return -1; 
            laneIndexed = true;
        }
        const int base = operandId(addr.base, false);
        const Value *v = base >= 0 ? lookup(base) : nullptr;
        if (!v || v->kind != Value::Pointer) return -1;
        laneIndexed = laneIndexed || v->laneIndexed;
        return base;
    };

    for (size_t bi = 0; bi < fn.blocks.size(); ++bi) {
        if (!reachable[bi]) continue; 
        const ir::Block &block = fn.blocks[bi];
        const int guard = blockGuard[bi];
        bool terminated = false;

        for (const ir::Instr &in : block.instrs) {
            if (debug())
                std::fprintf(stderr, "[aclnn]   %zu.%s res=%d args=%zu\n", bi,
                             ir::opName(in.op), static_cast<int>(in.result),
                             in.args.size());

            
            
            
            
            const bool takesPredicate =
                in.op == ir::Op::Store || in.op == ir::Op::Br ||
                in.op == ir::Op::CondBr || in.op == ir::Op::Ret ||
                in.op == ir::Op::Unreachable;
            if (in.hasPred && !takesPredicate) {
                if (debug())
                    std::fprintf(stderr, "[aclnn]   refuse: predicated %s\n",
                                 ir::opName(in.op));
                return false;
            }

            if (in.op == ir::Op::Ret || in.op == ir::Op::Unreachable) {
                terminated = true;
                break;
            }

            
            if (in.op == ir::Op::Br || in.op == ir::Op::CondBr) {
                terminated = true;
                if (in.args.empty() ||
                    in.args[0].kind != ir::Operand::Kind::LabelRef)
                    return false;
                const int target = static_cast<int>(in.args[0].block);
                if (target <= static_cast<int>(bi)) return false; 
                if (static_cast<size_t>(target) >= fn.blocks.size()) return false;
                const bool conditional = in.op == ir::Op::CondBr && in.hasPred &&
                                         in.pred != ir::kInvalidValue;
                if (!conditional) {
                    addEdge(static_cast<size_t>(target), guard);
                    break;
                }
                
                
                const int p = static_cast<int>(in.pred);
                addEdge(static_cast<size_t>(target), conditionAnd(guard, p));
                if (bi + 1 >= fn.blocks.size()) return false;
                addEdge(bi + 1, conditionAnd(guard, conditionNot(p)));
                break;
            }
            if (ir::isTerminator(in.op)) return false; 

            switch (in.op) {
            case ir::Op::Load: {
                
                
                if (in.addr.base.kind == ir::Operand::Kind::ParamRef) {
                    const int id = operandId(in.addr.base, false);
                    const Value *v = id >= 0 ? lookup(id) : nullptr;
                    if (!v) return false;
                    if (in.result != ir::kInvalidValue) values[in.result] = *v;
                    break;
                }
                if (in.space != ir::Space::Global) return false;
                if (!in.type.isFloat() || in.type.width != 32) return false;
                bool indexed = false;
                const int base = addressParameter(in.addr, indexed);
                if (base < 0 || !indexed) {
                    if (debug())
                        std::fprintf(stderr,
                                     "[aclnn]   refuse: load from a fixed address\n");
                    return false;
                }
                Value v;
                v.kind = Value::Tensor;
                v.param = lookup(base)->param;
                if (in.result != ir::kInvalidValue) values[in.result] = v;
                break;
            }

            case ir::Op::Store: {
                if (!in.type.isFloat() || in.type.width != 32) return false;
                if (in.args.empty()) return false;
                bool indexed = false;
                const int base = addressParameter(in.addr, indexed);
                if (base < 0) return false;
                const int dst = lookup(base)->param;
                if (!buffers[dst]) return false;
                const int value = operandId(in.args[0], true);
                if (value < 0) return false;
                int siteGuard = guard;
                if (in.hasPred) {
                    if (in.pred == ir::kInvalidValue) return false;
                    siteGuard = conditionAnd(siteGuard, static_cast<int>(in.pred));
                }
                if (!conditionOk(siteGuard)) return false;
                stores.push_back(
                    StoreSite{dst, value, siteGuard, indexed,
                              static_cast<size_t>(in.type.width) / 8});
                break;
            }

            
            
            case ir::Op::Mov:
            case ir::Op::Cvt: {
                if (in.args.empty()) return false;
                const int src = operandId(in.args[0], false);
                if (src < 0) return false;
                const Value *v = lookup(src);
                if (!v) return false;
                if (in.op == ir::Op::Cvt && v->kind == Value::Constant) return false;
                if (in.result != ir::kInvalidValue) values[in.result] = *v;
                break;
            }

            
            
            case ir::Op::TidX:
            case ir::Op::CtaidX: {
                if (in.op == ir::Op::CtaidX) usesBlockIndex = true;
                Value v;
                v.kind = Value::Lane;
                v.coord = in.op == ir::Op::TidX ? Coord::Thread : Coord::Block;
                if (in.result != ir::kInvalidValue) values[in.result] = v;
                break;
            }
            case ir::Op::LaneId: {
                Value v;
                v.kind = Value::Lane;
                v.coord = Coord::WarpLane;
                if (in.result != ir::kInvalidValue) values[in.result] = v;
                break;
            }
            case ir::Op::WarpId: {
                Value v;
                v.kind = Value::Lane;
                v.coord = Coord::Warp;
                if (in.result != ir::kInvalidValue) values[in.result] = v;
                break;
            }
            case ir::Op::NtidX: {
                Value v;
                v.kind = Value::Constant;
                v.number = static_cast<float>(blockX);
                if (in.result != ir::kInvalidValue) values[in.result] = v;
                break;
            }
            case ir::Op::NtidY:
            case ir::Op::NtidZ:
            case ir::Op::TidY:
            case ir::Op::TidZ:
            case ir::Op::CtaidY:
            case ir::Op::CtaidZ: {
                Value v;
                v.kind = Value::Constant;
                v.number = (in.op == ir::Op::NtidY || in.op == ir::Op::NtidZ)
                               ? 1.0f
                               : 0.0f;
                if (in.result != ir::kInvalidValue) values[in.result] = v;
                break;
            }

            
            
            
            
            
            case ir::Op::Add:
            case ir::Op::Sub:
            case ir::Op::Mul:
            case ir::Op::DivU:
            case ir::Op::ShrU:
            case ir::Op::Shl: {
                if (in.args.size() < 2) return false;
                const int left = operandId(in.args[0], false);
                const int right = operandId(in.args[1], false);
                if (left < 0 || right < 0) return false;
                const Value *l = lookup(left);
                const Value *r = lookup(right);
                if (!l || !r) return false;
                const bool leftIsPointer = l->kind == Value::Pointer;
                const bool rightIsPointer = r->kind == Value::Pointer;
                Value out;
                if (leftIsPointer || rightIsPointer) {
                    if (leftIsPointer && rightIsPointer) return false;
                    if (in.op != ir::Op::Add && in.op != ir::Op::Sub) return false;
                    if (in.op == ir::Op::Sub && leftIsPointer) return false;
                    const Value *base = leftIsPointer ? l : r;
                    const Value *offset = leftIsPointer ? r : l;
                    if (offset->kind == Value::Constant) {
                        if (offset->number != 0.0f) return false;
                    } else if (offset->kind != Value::Lane &&
                               offset->kind != Value::Index) {
                        return false;
                    }
                    out = *base;
                    out.laneIndexed =
                        base->laneIndexed || offset->kind != Value::Constant;
                } else {
                    const int op = intOpOf(in.op);
                    if (op < 0) return false;
                    if (l->kind == Value::Constant && r->kind == Value::Constant) {
                        const int64_t x = static_cast<int64_t>(l->number);
                        const int64_t y = static_cast<int64_t>(r->number);
                        out.kind = Value::Constant;
                        int64_t res = 0;
                        switch (static_cast<IntOp>(op)) {
                        case IntOp::Add: res = x + y; break;
                        case IntOp::Sub: res = x - y; break;
                        case IntOp::Mul: res = x * y; break;
                        case IntOp::DivU: res = y == 0 ? 0 : x / y; break;
                        case IntOp::ShrU: res = x >> (y & 63); break;
                        case IntOp::Shl: res = x << (y & 63); break;
                        }
                        out.number = static_cast<float>(res);
                    } else if (l->kind == Value::Constant || l->kind == Value::Lane ||
                               l->kind == Value::Index) {
                        if (r->kind != Value::Constant && r->kind != Value::Lane &&
                            r->kind != Value::Index)
                            return false;
                        out.kind = Value::Index;
                        out.intOp = op;
                        out.a = left;
                        out.b = right;
                    } else {
                        return false;
                    }
                }
                if (in.result != ir::kInvalidValue) values[in.result] = out;
                break;
            }

            
            
            case ir::Op::FAdd:
            case ir::Op::FSub:
            case ir::Op::FMul: {
                if (in.args.size() < 2) return false;
                const int left = operandId(in.args[0], true);
                const int right = operandId(in.args[1], true);
                if (left < 0 || right < 0) return false;
                const Value *l = lookup(left);
                const Value *r = lookup(right);
                auto usable = [](const Value *v) {
                    return v->kind == Value::Constant || v->kind == Value::Tensor ||
                           v->kind == Value::Expr;
                };
                if (!l || !r || !usable(l) || !usable(r)) return false;
                Value out;
                out.kind = Value::Expr;
                out.floatOp = in.op == ir::Op::FAdd   ? 0
                              : in.op == ir::Op::FMul ? 1
                                                      : 2;
                out.a = left;
                out.b = right;
                if (in.result != ir::kInvalidValue) values[in.result] = out;
                break;
            }

            case ir::Op::CmpEq:
            case ir::Op::CmpNe:
            case ir::Op::CmpLtU:
            case ir::Op::CmpLeU:
            case ir::Op::CmpGtU:
            case ir::Op::CmpGeU:
            case ir::Op::CmpLtS:
            case ir::Op::CmpLeS:
            case ir::Op::CmpGtS:
            case ir::Op::CmpGeS: {
                if (in.args.size() < 2) return false;
                const int left = operandId(in.args[0], false);
                const int right = operandId(in.args[1], false);
                if (left < 0 || right < 0) return false;
                const Value *l = lookup(left);
                const Value *r = lookup(right);
                if (!l || !r) return false;
                Value out;
                if (l->kind == Value::Constant && r->kind == Value::Constant) {
                    const int64_t x = static_cast<int64_t>(l->number);
                    const int64_t y = static_cast<int64_t>(r->number);
                    bool res = false;
                    switch (static_cast<Pred>(predicateOf(in.op))) {
                    case Pred::Eq: res = x == y; break;
                    case Pred::Ne: res = x != y; break;
                    case Pred::LtU: res = static_cast<uint64_t>(x) < static_cast<uint64_t>(y); break;
                    case Pred::LeU: res = static_cast<uint64_t>(x) <= static_cast<uint64_t>(y); break;
                    case Pred::GtU: res = static_cast<uint64_t>(x) > static_cast<uint64_t>(y); break;
                    case Pred::GeU: res = static_cast<uint64_t>(x) >= static_cast<uint64_t>(y); break;
                    case Pred::LtS: res = x < y; break;
                    case Pred::LeS: res = x <= y; break;
                    case Pred::GtS: res = x > y; break;
                    case Pred::GeS: res = x >= y; break;
                    default: return false;
                    }
                    out.kind = Value::Constant;
                    out.number = res ? 1.0f : 0.0f;
                } else {
                    auto laneBased = [](const Value *v) {
                        return v->kind == Value::Lane || v->kind == Value::Constant ||
                               v->kind == Value::Index || v->kind == Value::Mask;
                    };
                    if (!laneBased(l) || !laneBased(r)) return false;
                    out.kind = Value::Mask;
                    out.pred = predicateOf(in.op);
                    out.a = left;
                    out.b = right;
                }
                if (in.result != ir::kInvalidValue) values[in.result] = out;
                break;
            }

            case ir::Op::FCmpEq:
            case ir::Op::FCmpNe:
            case ir::Op::FCmpLt:
            case ir::Op::FCmpLe:
            case ir::Op::FCmpGt:
            case ir::Op::FCmpGe: {
                if (in.args.size() < 2) return false;
                const int left = operandId(in.args[0], true);
                const int right = operandId(in.args[1], true);
                if (left < 0 || right < 0) return false;
                const Value *l = lookup(left);
                const Value *r = lookup(right);
                if (!l || !r || l->kind != Value::Constant || r->kind != Value::Constant)
                    return false;
                bool res = false;
                if (in.op == ir::Op::FCmpEq) res = l->number == r->number;
                else if (in.op == ir::Op::FCmpNe) res = l->number != r->number;
                else if (in.op == ir::Op::FCmpLt) res = l->number < r->number;
                else if (in.op == ir::Op::FCmpLe) res = l->number <= r->number;
                else if (in.op == ir::Op::FCmpGt) res = l->number > r->number;
                else res = l->number >= r->number;
                Value out;
                out.kind = Value::Constant;
                out.number = res ? 1.0f : 0.0f;
                if (in.result != ir::kInvalidValue) values[in.result] = out;
                break;
            }

            case ir::Op::Not:
            case ir::Op::And:
            case ir::Op::Or: {
                if (!in.type.isBool()) return false; 
                auto isCondition = [&](int id) {
                    const Value *v = lookup(id);
                    return v && (v->kind == Value::Mask || v->kind == Value::Constant);
                };
                if (in.op == ir::Op::Not) {
                    if (in.args.empty()) return false;
                    const int src = operandId(in.args[0], false);
                    if (src < 0 || !isCondition(src)) return false;
                    const Value *v = lookup(conditionNot(src));
                    if (!v) return false;
                    if (in.result != ir::kInvalidValue) values[in.result] = *v;
                    break;
                }
                if (in.args.size() < 2) return false;
                const int left = operandId(in.args[0], false);
                const int right = operandId(in.args[1], false);
                if (left < 0 || right < 0 || !isCondition(left) || !isCondition(right))
                    return false;
                const Value *v = lookup(in.op == ir::Op::And
                                            ? conditionAnd(left, right)
                                            : conditionOr(left, right));
                if (!v) return false;
                if (in.result != ir::kInvalidValue) values[in.result] = *v;
                break;
            }

            default:
                if (debug())
                    std::fprintf(stderr, "[aclnn]   refuse: op %s not in model\n",
                                 ir::opName(in.op));
                return false;
            }
        }

        
        if (!terminated && bi + 1 < fn.blocks.size()) addEdge(bi + 1, guard);
    }

    
    
    
    if (gridX > 1 && !usesBlockIndex) {
        if (debug())
            std::fprintf(stderr, "[aclnn]   refuse: grid %u without %%ctaid.x\n",
                         gridX);
        return false;
    }

    if (stores.empty()) {
        if (debug()) std::fprintf(stderr, "[aclnn]   refuse: no store resolved\n");
        return false;
    }
    if (debug()) std::fprintf(stderr, "[aclnn] stores=%zu\n", stores.size());

    
    
    
    
    
    
    

    const size_t bytes = laneCount * 4;
    std::map<int, void *> temporaries;

    std::function<bool(int, void **)> evaluate = [&](int id, void **out) -> bool {
        *out = nullptr;
        if (id < 0) return false;
        auto it = temporaries.find(id);
        if (it != temporaries.end()) { *out = it->second; return true; }
        const Value *v = lookup(id);
        if (!v) return false;
        if (v->kind == Value::Pointer || v->kind == Value::Tensor) {
            if (v->param < 0 || !buffers[v->param]) return false;
            *out = buffers[v->param];
            return true;
        }
        if (v->kind == Value::Constant) {
            void *buffer = poolAlloc(bytes);
            if (!buffer) return false;
            
            
            
            std::vector<float> chunk(
                static_cast<size_t>(std::min<uint64_t>(laneCount, 1u << 16)),
                v->number);
            bool ok = true;
            for (uint64_t done = 0; done < laneCount && ok;) {
                const size_t n = static_cast<size_t>(
                    std::min<uint64_t>(chunk.size(), laneCount - done));
                
                
                
                ok = aclrtMemcpyAsync(static_cast<char *>(buffer) + done * 4,
                                      n * 4, chunk.data(), n * 4,
                                      ACL_MEMCPY_HOST_TO_DEVICE,
                                      static_cast<aclrtStream>(stream)) ==
                     ACL_SUCCESS;
                done += n;
            }
            if (!ok) {
                poolRelease(buffer);
                return false;
            }
            temporaries[id] = buffer;
            *out = buffer;
            return true;
        }
        if (v->kind != Value::Expr) return false;
        void *lhs = nullptr;
        void *rhs = nullptr;
        if (!evaluate(v->a, &lhs) || !evaluate(v->b, &rhs)) return false;
        aclnnexec::Op *op = v->floatOp == 0   ? aclnnexec::makeAdd()
                            : v->floatOp == 1 ? aclnnexec::makeMul()
                                              : aclnnexec::makeSub();
        if (!op) return false;
        void *dst = poolAlloc(bytes);
        if (!dst) {
            delete op;
            return false;
        }
        const bool ok =
            op->run(lhs, rhs, dst, static_cast<int64_t>(laneCount), 1.0f, stream);
        delete op;
        if (!ok) {
            poolRelease(dst);
            return false;
        }
        temporaries[id] = dst;
        *out = dst;
        return true;
    };

    
    
    
    
    std::vector<std::pair<uint64_t, uint64_t>> runs;
    auto activeRuns = [&](int guard) -> bool {
        runs.clear();
        if (guard < 0) { runs.emplace_back(0, laneCount); return true; }
        uint64_t lane = 0;
        while (lane < laneCount) {
            bool active = false;
            if (!conditionHolds(guard, lane, active)) return false;
            if (!active) { ++lane; continue; }
            const uint64_t first = lane;
            while (lane < laneCount) {
                if (!conditionHolds(guard, lane, active)) return false;
                if (!active) break;
                ++lane;
            }
            runs.emplace_back(first, lane);
        }
        return true;
    };

    
    
    
    if (aclrtSynchronizeDevice() != ACL_SUCCESS) {
        if (debug()) std::fprintf(stderr, "[aclnn]   refuse: input sync failed\n");
        return false;
    }

    bool failed = false;
    for (const StoreSite &site : stores) {
        if (!activeRuns(site.guard)) { failed = true; break; }
        if (runs.empty()) continue;
        void *value = nullptr;
        if (!evaluate(site.value, &value)) { failed = true; break; }
        char *dst = static_cast<char *>(buffers[site.param]);
        const char *src = static_cast<const char *>(value);
        if (site.laneIndexed) {
            for (const auto &run : runs) {
                const size_t offset = run.first * site.bytes;
                const size_t length = (run.second - run.first) * site.bytes;
                if (aclrtMemcpyAsync(dst + offset, length, src + offset, length,
                                     ACL_MEMCPY_DEVICE_TO_DEVICE,
                                     static_cast<aclrtStream>(stream)) !=
                    ACL_SUCCESS) {
                    failed = true;
                    break;
                }
            }
        } else {
            
            
            
            const size_t offset = runs.front().first * site.bytes;
            if (aclrtMemcpyAsync(dst, site.bytes, src + offset, site.bytes,
                                 ACL_MEMCPY_DEVICE_TO_DEVICE,
                                 static_cast<aclrtStream>(stream)) != ACL_SUCCESS)
                failed = true;
        }
        if (failed) break;
        if (debug()) {
            
            
            std::vector<float> probe(4, -12345.0f);
            const aclError prc = aclrtMemcpyAsync(
                probe.data(), probe.size() * 4, value, probe.size() * 4,
                ACL_MEMCPY_DEVICE_TO_HOST, static_cast<aclrtStream>(stream));
            if (prc == ACL_SUCCESS)
                std::fprintf(stderr, "[aclnn]   value[0..3] = %g %g %g %g\n",
                             probe[0], probe[1], probe[2], probe[3]);
            const Value *v = lookup(site.value);
            std::fprintf(stderr,
                         "[aclnn]   store param=%d value=%d kind=%d a=%d b=%d "
                         "lanes=[%llu,%llu)%s\n",
                         site.param, site.value, v ? (int)v->kind : -1,
                         v ? v->a : -1, v ? v->b : -1,
                         (unsigned long long)runs.front().first,
                         (unsigned long long)runs.back().second,
                         site.laneIndexed ? "" : " (fixed address)");
        }
    }

    
    
    
    const aclError synced = aclrtSynchronizeDevice();
    if (synced != ACL_SUCCESS) {
        if (debug())
            std::fprintf(stderr, "[aclnn]   sync failed: %d\n",
                         static_cast<int>(synced));
    } else {
        for (auto &entry : temporaries) poolRelease(entry.second);
    }

    if (debug()) std::fprintf(stderr, "[aclnn] %s\n", failed ? "refused" : "lowered");
    return !failed;
}

} 

#endif 

