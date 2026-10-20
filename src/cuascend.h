// MAIN: the interfaces of the CUDA compatibility layer.

// MAIN: the CUDA Driver API ABI

#ifndef CA_CUASCEND_H
#define CA_CUASCEND_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CUDA_VERSION 12040

typedef enum CUresult_enum {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_INVALID_VALUE = 1,
    CUDA_ERROR_OUT_OF_MEMORY = 2,
    CUDA_ERROR_NOT_INITIALIZED = 3,
    CUDA_ERROR_DEINITIALIZED = 4,
    CUDA_ERROR_NO_DEVICE = 100,
    CUDA_ERROR_INVALID_DEVICE = 101,
    CUDA_ERROR_INVALID_IMAGE = 200,
    CUDA_ERROR_INVALID_CONTEXT = 201,
    CUDA_ERROR_MAP_FAILED = 205,
    CUDA_ERROR_OUT_OF_RANGE = 500,
    CUDA_ERROR_NOT_SUPPORTED = 801,
    CUDA_ERROR_LAUNCH_FAILED = 719,

    
    CUDA_ERROR_NOT_FOUND = 500,
} CUresult;

typedef int CUdevice;
typedef struct CUctx_st *CUcontext;
typedef struct CUstream_st *CUstream;
typedef struct CUevent_st *CUevent;
typedef struct CUmodule_st *CUmodule;
typedef struct CUfunction_st *CUfunction;
typedef struct CUarray_st *CUarray;
typedef struct CUmipmappedArray_st *CUmipmappedArray;

typedef uint64_t CUdeviceptr;
typedef void *CUdeviceptr_v;

typedef struct CUDA_MEMCPY2_st {
    CUdeviceptr srcDevice;
    void *srcHost;
    size_t srcPitch;
    size_t srcXInBytes;
    size_t srcY;
    CUdeviceptr dstDevice;
    void *dstHost;
    size_t dstPitch;
    size_t dstXInBytes;
    size_t dstY;
    size_t WidthInBytes;
    size_t Height;
} CUDA_MEMCPY2;

typedef enum CUmemorytype_enum {
    CU_MEMORYTYPE_HOST = 1,
    CU_MEMORYTYPE_DEVICE = 2,
    CU_MEMORYTYPE_ARRAY = 3,
    CU_MEMORYTYPE_UNIFIED = 4,
} CUmemorytype;

typedef enum CUdevice_attribute_enum {
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 5,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 6,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 7,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
    CU_DEVICE_ATTRIBUTE_WARP_SIZE = 32,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
    CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,
    CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36,
    CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37,
    CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40,
    CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 42,
} CUdevice_attribute;

typedef enum CUfunc_attribute_enum {
    CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0,
    CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1,
    CU_FUNC_ATTRIBUTE_NUM_REGS = 4,
    CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3,
    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8,
} CUfunc_attribute;

typedef uint64_t CUdevice_attribute_mask;

typedef struct CUmempool_st *CUmempool;

typedef void (*CUhostFn)(void *userData);

typedef enum CUfunc_cache_enum {
    CU_FUNC_CACHE_PREFER_NONE = 0,
    CU_FUNC_CACHE_PREFER_SHARED = 1,
    CU_FUNC_CACHE_PREFER_L1 = 2,
    CU_FUNC_CACHE_PREFER_EQUAL = 3,
} CUfunc_cache;

typedef enum CUfunc_shared_enum {
    CU_FUNC_SHARED_SIZE_DEFAULT = 0,
    CU_FUNC_SHARED_SIZE_64KB = 1,
    CU_FUNC_SHARED_SIZE_96KB = 2,
    CU_FUNC_SHARED_SIZE_128KB = 3,
    CU_FUNC_SHARED_SIZE_160KB = 4,
    CU_FUNC_SHARED_SIZE_192KB = 5,
    CU_FUNC_SHARED_SIZE_224KB = 6,
    CU_FUNC_SHARED_SIZE_256KB = 7,
} CUfunc_shared;

typedef enum CUmemPoolAttr_enum {
    CU_MEMPOOL_ATTR_RELEASE_THRESHOLD = 0,
    CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH = 1,
    CU_MEMPOOL_ATTR_RESERVED_MEM_LOW = 2,
    CU_MEMPOOL_ATTR_UTILIZATION_CUR = 3,
    CU_MEMPOOL_ATTR_UTILIZATION_MAX = 4,
    CU_MEMPOOL_ATTR_TIMEOUT_US = 5,
} CUmemPoolAttr;

typedef struct CUDA_KERNEL_NODE_PARAMS_st {
    void *func;
    unsigned gridDimX;
    unsigned gridDimY;
    unsigned gridDimZ;
    unsigned blockDimX;
    unsigned blockDimY;
    unsigned blockDimZ;
    unsigned sharedMemBytes;
    void **kernelParams;
    void **extra;
} CUDA_KERNEL_NODE_PARAMS;

#ifdef __cplusplus
}
#endif

// MAIN: the cuBLAS ABI

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CUBLAS_VERSION 12040

typedef enum cublasStatus_t {
    CUBLAS_STATUS_SUCCESS = 0,
    CUBLAS_STATUS_NOT_INITIALIZED = 1,
    CUBLAS_STATUS_ALLOC_FAILED = 3,
    CUBLAS_STATUS_INVALID_VALUE = 7,
    CUBLAS_STATUS_ARCH_MISMATCH = 8,
    CUBLAS_STATUS_NOT_SUPPORTED = 15,
    CUBLAS_STATUS_EXECUTION_FAILED = 13,
} cublasStatus_t;

typedef enum cublasPointerMode_t {
    CUBLAS_POINTER_MODE_HOST = 0,
    CUBLAS_POINTER_MODE_DEVICE = 1,
} cublasPointerMode_t;

typedef enum cublasOperation_t {
    CUBLAS_OP_N = 0,
    CUBLAS_OP_T = 1,
    CUBLAS_OP_C = 2,
} cublasOperation_t;

typedef enum cublasMath_t {
    CUBLAS_DEFAULT_MATH = 0,
    CUBLAS_TENSOR_OP_MATH = 1,
    CUBLAS_PEDANTIC_MATH = 2,
    CUBLAS_TF32_TENSOR_OP_MATH = 3,
} cublasMath_t;

typedef enum cudaDataType_t {
    CUDA_R_16F = 2,
    CUDA_R_32F = 0,
    CUDA_R_64F = 1,
    CUDA_R_8I = 3,
    CUDA_R_32I = 4,
} cudaDataType_t;

typedef struct __half { uint16_t x; } __half;
typedef struct CUstream_st *cudaStream_t;

typedef struct cublasContext *cublasHandle_t;

cublasStatus_t cublasCreate_v2(cublasHandle_t *handle);
cublasStatus_t cublasDestroy_v2(cublasHandle_t handle);
cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, cudaStream_t streamId);
cublasStatus_t cublasGetStream_v2(cublasHandle_t handle, cudaStream_t *streamId);
cublasStatus_t cublasGetVersion_v2(int *version);
cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode);
cublasStatus_t cublasSetPointerMode(cublasHandle_t handle,
                                    cublasPointerMode_t mode);
cublasStatus_t cublasGetPointerMode(cublasHandle_t handle,
                                    cublasPointerMode_t *mode);

cublasStatus_t cublasSgemm(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n, int k,
                           const float *alpha, const float *A, int lda,
                           const float *B, int ldb, const float *beta, float *C,
                           int ldc);
cublasStatus_t cublasDgemm(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n, int k,
                           const double *alpha, const double *A, int lda,
                           const double *B, int ldb, const double *beta,
                           double *C, int ldc);
cublasStatus_t cublasHgemm(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n, int k,
                           const __half *alpha, const __half *A, int lda,
                           const __half *B, int ldb, const __half *beta,
                           __half *C, int ldc);
cublasStatus_t cublasSgemv(cublasHandle_t handle, cublasOperation_t trans,
                           int m, int n, const float *alpha, const float *A,
                           int lda, const float *x, int incx, const float *beta,
                           float *y, int incy);
cublasStatus_t cublasDgemv(cublasHandle_t handle, cublasOperation_t trans,
                           int m, int n, const double *alpha, const double *A,
                           int lda, const double *x, int incx, const double *beta,
                           double *y, int incy);
cublasStatus_t cublasSaxpy(cublasHandle_t handle, int n, const float *alpha,
                            const float *x, int incx, float *y, int incy);
cublasStatus_t cublasDaxpy(cublasHandle_t handle, int n, const double *alpha,
                            const double *x, int incx, double *y, int incy);
cublasStatus_t cublasSscal(cublasHandle_t handle, int n, const float *alpha,
                            float *x, int incx);
cublasStatus_t cublasDscal(cublasHandle_t handle, int n, const double *alpha,
                            double *x, int incx);

#ifdef __cplusplus
}
#endif

// MAIN: the PTX types

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ca::ptx {

class Error : public std::runtime_error {
public:
    explicit Error(const std::string &what) : std::runtime_error(what) {}
    Error(const std::string &what, int line, int col)
        : std::runtime_error(what + " (line " + std::to_string(line) +
                             ", col " + std::to_string(col) + ")") {}
};

enum class TypeKind {
    Void,
    Bool,   
    Int,    
    Float,  
};

enum class StateSpace {
    Generic, 
    Global,
    Shared,
    Local,
    Constant,
    Param,
};

struct Type {
    TypeKind kind = TypeKind::Void;
    unsigned width = 0;   
    unsigned vecLen = 1;  
    StateSpace space = StateSpace::Generic;

    bool isVector() const { return vecLen > 1; }
    bool isPred() const { return kind == TypeKind::Bool && width == 1; }
    bool isFloat() const { return kind == TypeKind::Float; }
    bool isSigned() const { return kind == TypeKind::Int; }

    std::string str() const;
};

enum class OperandKind {
    Immediate,  
    Register,   
    Label,      
    Local,      
    Param,      
    Global,     
    Constant,   
    Addr,       
    Vector,     
    Symbol,     
    Special,    
    FuncRef,    
    TypeRef,    
};

struct Operand {
    OperandKind kind = OperandKind::Immediate;

    
    int64_t imm = 0;
    std::string immText;

    
    std::string name;

    
    struct AddrExpr {
        std::string baseReg;
        std::string indexReg;
        int64_t offset = 0;
        int64_t scale = 0; 
        bool hasBase = false;
        bool hasIndex = false;
        
        bool isParamRef = false;
    } addr;

    
    std::vector<Operand> elements;

    
    StateSpace space = StateSpace::Generic;

    std::string str() const;
};

struct Inst {
    
    bool isLabel = false;
    std::string label;

    std::string opcode;   
    std::vector<std::string> modifiers; 

    bool hasPred = false;
    std::string pred;     

    std::vector<Operand> operands;

    
    std::string loadStoreType;  
    std::string loadStoreSpace; 

    int line = 0;
    int col = 0;

    
    bool hasMod(const std::string &m) const;
    bool isTerm() const { return opcode == "ret" || opcode.rfind("ret.", 0) == 0; }
    std::string modsJoined() const;
};

struct Param {
    Type type;
    std::string name;      
    bool isPointer = false;
    std::vector<unsigned> arrayDims; 
};

struct RegDecl {
    Type type;                    
    std::vector<unsigned> dims;   
    bool isLocal = false;         
    StateSpace space = StateSpace::Generic;
    std::string name;             

    
    unsigned slots() const;
    
    unsigned scalarsPerSlot() const { return type.vecLen; }
};

enum class RegKind { Reg, Local, Shared, Global, Const };

struct Function {
    std::string name;
    bool visible = false;   
    bool external = false; 
    std::vector<Param> params;
    std::vector<Inst> body;
    std::string retType;

    
    std::vector<RegDecl> regs;

    unsigned maxThreads = 0; 
    unsigned minThreads = 0;
    unsigned maxBlocks = 0;

    
    const Type *registerType(const std::string &name) const;
};

struct Module {
    int versionMajor = 0;
    int versionMinor = 0;
    std::string target;        
    int addressSize = 64;
    std::vector<Function> functions;

    const Function *findEntry(const std::string &name) const;
};

Module parse(const std::string &text, const std::string &filename = "<ptx>");

struct Token {
    enum class Kind {
        End,
        Directive,  
        Register,   
        Param,      
        Number,
        Float,
        String,
        Punct,      
        Ident,      
    };
    Kind kind = Kind::End;
    std::string text;
    int line = 1;
    int col = 1;
};

std::vector<Token> lex(const std::string &text);

std::string dump(const Module &m);

} 

// MAIN: the fatbin types

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ca::fatbin {

class Error : public std::runtime_error {
public:
    explicit Error(const std::string &what) : std::runtime_error(what) {}
};

enum class Kind : uint32_t {
    PTX = 1,        
    Cubin = 2,      
    Unknown = 3,
    Compressed = 4, 
};

struct Entry {
    Kind kind;
    uint64_t offset; 
    uint64_t size;   

    
    uint32_t buffer = 0;

    
    uint32_t sm_major = 0;
    uint32_t sm_minor = 0;
};

class Fatbin {
public:
    const std::vector<Entry> &entries() const { return entries_; }

    
    const uint8_t *data(const Entry &e) const;

    
    void addInputEntry(Entry e);

    
    void addOwnedEntry(Entry e);

    
    uint32_t intern(std::vector<uint8_t> buf);

    bool hasPtx() const;
    bool hasCubin() const;

    
    std::string text(const Entry &e) const;

    
    std::string firstPtx() const;

    
    const uint8_t *bufferAt(uint32_t index) const;
    size_t bufferSize(uint32_t index) const;

    
    void setInput(std::vector<uint8_t> bytes);

private:
    std::vector<Entry> entries_;
    std::vector<uint8_t> input_;
    std::vector<std::vector<uint8_t>> owned_;
};

enum class Format {
    Unknown,
    FatbinELF,  
    CubinELF,   
    RawPTX,     
};

Format detect(const void *data, size_t size);

Fatbin parse(const void *data, size_t size);

Fatbin parseFile(const std::string &path);

std::string archName(uint32_t sm_major, uint32_t sm_minor);

} 

// MAIN: the LZ4 decoder

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ca::lz4 {

class Error : public std::runtime_error {
public:
    explicit Error(const std::string &what) : std::runtime_error(what) {}
};

namespace detail {

inline bool readLength(const uint8_t *src, size_t size, size_t &pos, int nibble,
                       size_t &out) {
    out = static_cast<size_t>(nibble);
    if (nibble != 15) return true;
    for (;;) {
        if (pos >= size) return false;
        const uint8_t b = src[pos++];
        out += b;
        if (b != 255) break;
    }
    return true;
}

} 

inline std::vector<uint8_t> decodeBlock(const uint8_t *src, size_t srcSize) {
    std::vector<uint8_t> out;
    
    out.reserve(srcSize * 3);

    size_t pos = 0;
    while (pos < srcSize) {
        const uint8_t token = src[pos++];

        size_t litLen = 0;
        if (!detail::readLength(src, srcSize, pos, token >> 4, litLen))
            throw Error("truncated literal length");
        if (litLen > srcSize - pos)
            throw Error("literal length exceeds input");
        out.insert(out.end(), src + pos, src + pos + litLen);
        pos += litLen;

        
        if (pos == srcSize) break;

        if (pos + 2 > srcSize) throw Error("truncated match offset");
        const uint16_t offset = static_cast<uint16_t>(src[pos] |
                                                     (src[pos + 1] << 8));
        pos += 2;

        if (offset == 0) throw Error("zero match offset");
        if (offset > out.size()) throw Error("match offset before start of output");

        size_t matchLen = 0;
        if (!detail::readLength(src, srcSize, pos, token & 0x0F, matchLen))
            throw Error("truncated match length");
        matchLen += 4; 

        const size_t start = out.size() - offset;
        
        for (size_t i = 0; i < matchLen; ++i) out.push_back(out[start + i]);
    }

    return out;
}

} 

// MAIN: the IR types

#include <cstdint>
#include <string>
#include <vector>

namespace ca::ir {

enum class TypeKind : uint8_t { Void, Bool, Int, Float };
enum class VecLen : uint8_t { Scalar = 1, V2 = 2, V4 = 4 };

struct Type {
    TypeKind kind = TypeKind::Void;
    uint8_t width = 0;    
    VecLen vec = VecLen::Scalar;

    constexpr bool isVoid() const { return kind == TypeKind::Void; }
    bool isBool() const { return kind == TypeKind::Bool && width == 1; }
    bool isInt() const { return kind == TypeKind::Int; }
    bool isFloat() const { return kind == TypeKind::Float; }
    bool isVector() const { return vec != VecLen::Scalar; }
    unsigned components() const { return static_cast<unsigned>(vec); }
    bool isSigned() const;

    std::string str() const;

    
    Type scalar() const { return Type{kind, width, VecLen::Scalar}; }
};

enum class Space : uint8_t {
    Generic,
    Global,   
    Shared,   
    Local,    
    Constant, 
    Param,    
};

const char *spaceName(Space s);

enum class Op : uint16_t {
    
    Ret,
    Br,
    CondBr,
    Call,
    Unreachable,

    
    Add, Sub, Mul, DivS, DivU, RemS, RemU, Neg,
    Shl, ShrS, ShrU, And, Or, Xor, Not,
    MinS, MinU, MaxS, MaxU,
    CmpEq, CmpNe, CmpLtS, CmpLeS, CmpGtS, CmpGeS,
    CmpLtU, CmpLeU, CmpGtU, CmpGeU,

    
    FAdd, FSub, FMul, FDiv, FRem, FFma, FNeg, FAbs,
    FCmpEq, FCmpNe, FCmpLt, FCmpLe, FCmpGt, FCmpGe,
    FMulAdd, 

    
    Exp, Exp2, Log, Log2, Log10, Sqrt, RSqrt,
    Sin, Cos, Tan, SinCos, Pow, Floor, Ceil, Round, Trunc, Fma3,
    MinF, MaxF,

    
    Cvt,     
    Selp,    
    Mov,
    Set,

    
    Load,
    Store,

    
    BarrierThreads,  
    BarrierWarp,     

    
    ShflIdx,   
    ShflUp,
    ShflDown,
    ShflXor,
    ShflPred,  
    VoteAll, VoteAny, VoteBallot, Ballot, Popc,
    ActiveMask,
    LaneId, WarpId, NumWarps,

    
    TidX, TidY, TidZ, CtaidX, CtaidY, CtaidZ, NtidX, NtidY, NtidZ,

    
    AtomicRMW,  
    AtomicCAS,

    
    Unknown,
};

const char *opName(Op o);
bool isWarpOp(Op o);      
bool isTerminator(Op o);
bool isFloatOp(Op o);
bool isComparison(Op o);

using ValueId = uint32_t;
using BlockId = uint32_t;
using InstrId = uint32_t;
static constexpr ValueId kInvalidValue = ~0u;
static constexpr BlockId kInvalidBlock = ~0u;

struct Operand {
    enum class Kind : uint8_t {
        Value,     
        Immediate, 
        GlobalSym, 
        ParamRef,  
        LabelRef,  
    };

    Kind kind = Kind::Immediate;

    ValueId value = kInvalidValue;
    BlockId block = kInvalidBlock;

    
    int64_t imm = 0;
    uint64_t immBits = 0;
    std::string symbol;

    static Operand val(ValueId v) {
        Operand o;
        o.kind = Kind::Value;
        o.value = v;
        return o;
    }
    static Operand lit(int64_t v) {
        Operand o;
        o.kind = Kind::Immediate;
        o.imm = v;
        o.immBits = static_cast<uint64_t>(v);
        return o;
    }
    static Operand sym(std::string s) {
        Operand o;
        o.kind = Kind::GlobalSym;
        o.symbol = std::move(s);
        return o;
    }
    static Operand param(std::string n) {
        Operand o;
        o.kind = Kind::ParamRef;
        o.symbol = std::move(n);
        return o;
    }
    static Operand branch(BlockId b) {
        Operand o;
        o.kind = Kind::LabelRef;
        o.block = b;
        return o;
    }

    bool isValue() const { return kind == Kind::Value; }
    bool isImm() const { return kind == Kind::Immediate; }
    bool isSym() const { return kind == Kind::GlobalSym; }
    bool isBranch() const { return kind == Kind::LabelRef; }
};

struct Address {
    
    Operand base;
    
    Operand index;
    int64_t scale = 0;
    int64_t offset = 0;
    bool hasIndex = false;
};

struct Instr {
    Op op = Op::Unknown;

    
    Type type;

    
    ValueId result = kInvalidValue;

    std::vector<Operand> args;

    Space space = Space::Generic;

    
    bool hasAddr = false;
    Address addr;

    
    bool hasPred = false;
    ValueId pred = kInvalidValue;

    
    std::vector<std::string> mods;

    
    std::string origText;

    
    int line = 0;

    bool hasResult() const { return result != kInvalidValue; }
};

struct Block {
    std::string label;
    std::vector<Instr> instrs;

    
    bool terminated() const {
        return !instrs.empty() && isTerminator(instrs.back().op);
    }
    const Instr *terminator() const {
        return terminated() ? &instrs.back() : nullptr;
    }
};

struct Param {
    std::string name;   
    Type type;
    Space space = Space::Param;
    bool isPointer = false;
    std::vector<uint32_t> arrayDims;
};

struct Function {
    std::string name;
    bool isEntry = false;    
    std::vector<Param> params;

    
    uint32_t reqThreads = 0;

    std::vector<Block> blocks;

    
    std::vector<ValueId> declared;

    const Block *entryBlock() const { return blocks.empty() ? nullptr : &blocks[0]; }
    Block *entryBlock() { return blocks.empty() ? nullptr : &blocks[0]; }

    size_t instrCount() const;
};

struct Module {
    std::string name;
    std::vector<Function> functions;

    
    std::vector<std::string> notes;

    const Function *find(const std::string &n) const;
    Function *find(const std::string &n);
};

std::string print(const Module &m);
std::string print(const Function &f);

std::vector<std::string> verify(const Module &m);

} 

// MAIN: the lowering interface

namespace ca::lower {

struct Options {
    
    bool strict = false;
};

struct Result {
    ir::Module module;

    
    size_t lowered = 0;     
    size_t unknown = 0;     
    size_t warpOps = 0;     
    size_t bytes = 0;       

    double coverage() const {
        const size_t total = lowered + unknown;
        return total ? static_cast<double>(lowered) / static_cast<double>(total)
                     : 1.0;
    }
};

Result run(const ptx::Module &in, const Options &opt = {});

Result fromText(const std::string &ptxText, const Options &opt = {});

} 

// MAIN: the executor interface

#include <cstdint>
#include <string>

namespace ca::ir {

struct HostView {
    uint64_t base = 0;    
    uint8_t *data = nullptr;
    size_t size = 0;      
};

class Memory {
public:
    virtual ~Memory() = default;
    
    virtual bool view(uint64_t address, HostView &out) = 0;
    
    virtual void flush() = 0;
};

struct LaunchConfig {
    unsigned gridX = 1, gridY = 1, gridZ = 1;
    unsigned blockX = 1, blockY = 1, blockZ = 1;
    unsigned sharedBytes = 0;
};

bool interpret(const Function &fn, const LaunchConfig &config, void **args,
               Memory &memory, std::string *error);

} 

// MAIN: the generator interface

#include <stdexcept>
#include <string>
#include <vector>

namespace ca::codegen {

class CodegenError : public std::runtime_error {
public:
    CodegenError(std::string fn, std::string op, std::string why)
        : std::runtime_error(fn + ": operation '" + op + "' cannot be lowered "
                             "to AscendC: " + why),
          function(std::move(fn)), operation(std::move(op)),
          reason(std::move(why)) {}

    std::string function;
    std::string operation;
    std::string reason;
};

struct TargetInfo {
    
    std::string socVersion = "Ascend910B1";

    
    int coreCount = 0;

    
    unsigned repeatWidth = 32;
};

struct Output {
    std::string source;        
    std::string kernelName;    
    std::vector<ir::Param> params;
    TargetInfo target;

    
    unsigned long long linearThreads = 0;
};

Output generate(const ir::Function &fn, const TargetInfo &target = {});

std::vector<std::string> unsupported(const ir::Function &fn);

} 

// MAIN: the backend interface

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ca {

enum class BackendStatus {
    Ok,
    NotInitialized,
    NoDevice,
    OutOfMemory,
    Unsupported,
    InvalidValue,
    NotSupported,
    Unknown,
};

CUresult toCuda(BackendStatus s);

using MemHandle = uint64_t;
using StreamHandle = uint64_t;
using EventHandle = uint64_t;
using ModuleHandle = uint64_t;
using FunctionHandle = uint64_t;

constexpr MemHandle kNullMem = 0;
constexpr StreamHandle kNullStream = 0;

struct DeviceCaps {
    int deviceCount = 1;
    int major = 8; 
    int minor = 0; 
    int maxThreadsPerBlock = 1024;
    int maxThreadsDim[3] = {1024, 1024, 64};
    int maxSharedMemoryPerBlock = 49152;
    int maxRegistersPerBlock = 65536;
    int multiprocessorCount = 0; 
    int warpSize = 32;
    int clockRateKHz = 0;
    int memoryClockRateKHz = 0;
    int memoryBusWidthBits = 0;
    int asyncEngineCount = 1;
    bool unifiedAddressing = true;
    bool concurrentKernels = true;
};

class Backend {
public:
    virtual ~Backend() = default;

    
    virtual std::string lastError() const { return {}; }

    virtual const char *name() const = 0;

    
    virtual BackendStatus init(int flags = 0) = 0;
    virtual void shutdown() = 0;
    virtual bool initialized() const = 0;

    
    virtual int deviceCount() const = 0;
    virtual BackendStatus deviceGet(int ordinal, int &out) = 0;
    virtual DeviceCaps deviceCaps(int device) const = 0;
    virtual BackendStatus deviceGetAttribute(int attr, int *out, int device) = 0;
    virtual std::string deviceName(int device) const = 0;

    
    virtual BackendStatus ctxCreate(unsigned flags, int device, void **outCtx) = 0;
    virtual BackendStatus ctxDestroy(void *ctx) = 0;
    virtual BackendStatus ctxSetCurrent(void *ctx) = 0;
    virtual BackendStatus ctxGetCurrent(void **outCtx) = 0;
    virtual BackendStatus ctxGetDevice(int *outDevice) = 0;

    
    virtual BackendStatus memAlloc(MemHandle *out, size_t bytes) = 0;
    virtual BackendStatus memFree(MemHandle h) = 0;
    virtual BackendStatus memHostAlloc(void **out, size_t bytes, unsigned flags) = 0;
    virtual BackendStatus memHostFree(void *p) = 0;
    virtual BackendStatus memHostRegister(void *p, size_t bytes, unsigned flags) = 0;
    virtual BackendStatus memHostUnregister(void *p) = 0;

    
    virtual BackendStatus streamCreate(StreamHandle *out, unsigned flags) = 0;
    virtual BackendStatus streamDestroy(StreamHandle h) = 0;
    virtual BackendStatus streamSynchronize(StreamHandle h) = 0;
    virtual BackendStatus eventCreate(EventHandle *out, unsigned flags) = 0;
    virtual BackendStatus eventDestroy(EventHandle h) = 0;
    virtual BackendStatus eventRecord(EventHandle ev, StreamHandle s) = 0;
    virtual BackendStatus eventSynchronize(EventHandle ev) = 0;
    virtual BackendStatus eventQuery(EventHandle ev) = 0;
    virtual BackendStatus streamWaitEvent(StreamHandle s, EventHandle ev) = 0;

    
    virtual BackendStatus moduleLoad(ModuleHandle *out, const void *image,
                                      unsigned flags) = 0;
    virtual BackendStatus moduleUnload(ModuleHandle h) = 0;
    virtual BackendStatus moduleGetFunction(FunctionHandle *out, ModuleHandle m,
                                            const char *name) = 0;
    virtual BackendStatus functionSetAttribute(int attr, FunctionHandle f,
                                               const void *value) = 0;
    virtual BackendStatus functionGetAttribute(int attr, FunctionHandle f,
                                               void *value) = 0;

    
    virtual BackendStatus launch(FunctionHandle f, unsigned gx, unsigned gy,
                                 unsigned gz, unsigned bx, unsigned by,
                                 unsigned bz, unsigned sharedBytes,
                                 void **args, void **extra,
                                 StreamHandle stream) = 0;

    
    virtual BackendStatus memcpy(void *dst, const void *src, size_t bytes,
                                 unsigned kind) = 0;
};

Backend *makeBackend();

Backend *makeAscendBackend();

} 

// MAIN: the compile pipeline interface

#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include <memory>
#include <string>
#include <vector>

namespace ca::ascend {

std::string ascendLastError();

class CompileError : public std::runtime_error {
public:
    CompileError(std::string what, std::string toolOutput)
        : std::runtime_error(std::move(what)),
          output(std::move(toolOutput)) {}

    std::string output; 
};

struct Toolchain {
    
    std::string root;
    
    std::string bisheng;
    
    std::string msopgen;
    
    std::string socVersion = "Ascend910B1";
    
    std::string workDir;
    
    bool verbose = false;

    
    static Toolchain fromEnvironment();
};

enum class ElemType {
    Unknown,
    Float32,
    Float16,
    Int8,
    Int16,
    Int32,
    Int64,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
};

struct CompiledOp {
    
    std::shared_ptr<const ir::Function> ir;
    bool valid = false;
    std::string kernelName;   
    std::string modelPath;    
    std::string binPath;      
    std::string sourceDir;    
                              
    std::string sourceHash;   

    size_t inputCount = 0;
    size_t outputCount = 0;

    
    std::vector<ElemType> paramTypes;
    std::vector<uint32_t> paramSizes; 
};

CompiledOp compileKernel(const std::string &ptxSource,
                         const std::string &kernelName,
                         const std::string &socVersion,
                         const Toolchain &tc);

class OpCache {
public:
    explicit OpCache(std::string dir) : dir_(std::move(dir)) {}

    
    CompiledOp get(const std::string &ptxSource, const std::string &kernelName,
                   const Toolchain &tc);

    size_t size() const { return entries_.size(); }
    size_t hits() const { return hits_; }
    size_t misses() const { return misses_; }

private:
    std::string dir_;
    std::vector<std::pair<std::string, CompiledOp>> entries_;
    size_t hits_ = 0;
    size_t misses_ = 0;
};

std::string hashSource(const std::string &s);

std::string runCommand(const std::vector<std::string> &argv, bool verbose);

struct TensorDesc {
    std::string name;               
    std::string paramType;          
    std::vector<int64_t> shape;
    int format = -1;                
    std::string dtype;              
    std::string value;              
};

std::string cannDtype(const ir::Type &t);

std::string canonicalOpName(const std::string &fnName);

// An identifier that is safe to put into generated sources and into file names: the names come from the image, which is data.
std::string sanitizeIdentifier(const std::string &name);

std::string generateOpInfo(const ir::Function &fn, size_t inputCount,
                           size_t outputCount, int format = -1);

std::string generateOpInfo(const ir::Function &fn,
                           const std::vector<TensorDesc> &inputs,
                           const std::vector<TensorDesc> &outputs,
                           int format = -1);

std::string generateOpInfo(const ir::Function &fn,
                           const std::vector<TensorDesc> &inputs,
                           const std::vector<TensorDesc> &outputs,
                           const std::vector<TensorDesc> &attrs,
                           int format = -1);

} 

// MAIN: the aclnn operation interface

#include <cstddef>
#include <cstdint>

namespace ca::aclnnexec {

class Op {
public:
    virtual ~Op() = default;
    
    virtual bool run(void *a, void *b, void *out, int64_t n, float alpha,
                     void *stream) = 0;
    virtual const char *name() const = 0;
};

Op *makeAdd();
Op *makeMul();
Op *makeSub();

} 

// MAIN: the lowering interface

namespace ca::ascend {

bool runElementwiseAclnn(const ir::Function &fn, void **args, unsigned gridX,
                         unsigned blockX, void *stream);

} 

// MAIN: the interception interface

namespace ca {

Backend *backend();

void shimLog(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
bool shimVerbose();

const char *shimErrorString(CUresult r);

CUresult ensureInit();

CUcontext contextFrom(void *p);
void registerContext(CUcontext ctx);
void unregisterContext(CUcontext ctx);

CUstream streamFrom(CUstream s);
CUevent eventFrom(CUevent e);

} 

#endif // CA_CUASCEND_H
