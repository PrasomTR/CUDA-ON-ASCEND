// MAIN: the cuBLAS compatibility layer.
#include "cuascend.h"

// MAIN: the cuBLAS layer

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>
#include <new>
#include <unordered_map>

#include <acl/ops/acl_cblas.h>

namespace {

struct CublasHandle {
    aclrtStream stream = nullptr;
    int pointerMode = 0; // 0: the scalars live in host memory
    int mathMode = 0;
    
    struct Key {
        int transA, transB, transC;
        int m, n, k;
        int dtA, dtB, dtC, compute;
        bool operator==(const Key &o) const {
            return transA == o.transA && transB == o.transB &&
                   transC == o.transC && m == o.m && n == o.n && k == o.k &&
                   dtA == o.dtA && dtB == o.dtB && dtC == o.dtC &&
                   compute == o.compute;
        }
    };
    struct KeyHash {
        size_t operator()(const Key &x) const {
            size_t h = 1469598103934665603ull;
            const int vals[] = {x.transA, x.transB, x.transC, x.m, x.n,
                                x.k,     x.dtA,   x.dtB,  x.dtC, x.compute};
            for (int v : vals) {
                h ^= static_cast<size_t>(v + 1);
                h *= 1099511628211ull;
            }
            return h;
        }
    };
    std::unordered_map<Key, aclopHandle *, KeyHash> handles;
};

std::mutex g_mu;

aclTransType toTrans(int t) {
    switch (t) {
    case 0: return ACL_TRANS_N; 
    case 1: return ACL_TRANS_T; 
    default:
        
        return static_cast<aclTransType>(-1);
    }
}

bool transSupported(int t) { return t == 0 || t == 1; }

aclComputeType computeOf(aclDataType dt) {
    return dt == ACL_FLOAT16 ? ACL_COMPUTE_LOW_PRECISION
                             : ACL_COMPUTE_HIGH_PRECISION;
}

cublasStatus_t toCublas(aclError rc) {
    return rc == ACL_SUCCESS ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_NOT_SUPPORTED;
}

CublasHandle::Key makeKey(int ta, int tb, int tc, int m, int n, int k,
                          aclDataType dtA, aclDataType dtB, aclDataType dtC,
                          aclComputeType compute) {
    return CublasHandle::Key{static_cast<int>(ta), static_cast<int>(tb),
                             static_cast<int>(tc), m, n, k, static_cast<int>(dtA),
                             static_cast<int>(dtB), static_cast<int>(dtC),
                             static_cast<int>(compute)};
}

aclopHandle *acquireHandle(CublasHandle *self, int ta, int tb, int tc, int m,
                           int n, int k, aclDataType dtA, aclDataType dtB,
                           aclDataType dtC, aclComputeType compute) {
    const auto key = makeKey(ta, tb, tc, m, n, k, dtA, dtB, dtC, compute);

    std::lock_guard<std::mutex> lk(g_mu);
    auto it = self->handles.find(key);
    if (it != self->handles.end()) return it->second;

    aclopHandle *h = nullptr;
    const aclError rc = aclblasCreateHandleForGemmEx(
        toTrans(ta), toTrans(tb), toTrans(tc), m, n, k, dtA, dtB, dtC,
        compute, &h);
    if (rc != ACL_SUCCESS || !h) {
        
        std::fprintf(stderr,
                     "[cubascend] aclblasCreateHandleForGemmEx failed: "
                     "code=%d transA=%d transB=%d m=%d n=%d k=%d "
                     "type=%d compute=%d\n",
                     static_cast<int>(rc), ta, tb, m, n, k,
                     static_cast<int>(dtA), static_cast<int>(compute));
        return nullptr;
    }

    self->handles.emplace(key, h);
    return h;
}

// A scalar of the element type, as a number.
double readScalar(const void *pointer, aclDataType dt) {
    if (dt == ACL_DOUBLE) {
        double value = 0;
        std::memcpy(&value, pointer, sizeof value);
        return value;
    }
    float value = 0;
    std::memcpy(&value, pointer, sizeof value);
    return value;
}

// The matrix multiplication on the host: the operations of the BLAS layer live in the operator registry, and on an image without the operator packages the registry is empty.
cublasStatus_t gemmOnHost(int ta, int tb, int m, int n, int k,
                           const void *alpha, const void *A, int lda,
                           const void *B, int ldb, const void *beta, void *C,
                           int ldc, aclDataType dt) {
    const size_t element = dt == ACL_DOUBLE ? 8 : 4;
    const bool isDouble = dt == ACL_DOUBLE;
    const size_t aCount = (ta == 0) ? size_t(m) * k : size_t(k) * m;
    const size_t bCount = (tb == 0) ? size_t(k) * n : size_t(n) * k;
    const size_t cCount = size_t(m) * n;

    std::vector<unsigned char> hostA(aCount * element);
    std::vector<unsigned char> hostB(bCount * element);
    std::vector<unsigned char> hostC(cCount * element);
    if (aclrtMemcpy(hostA.data(), hostA.size(), A, hostA.size(),
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
        aclrtMemcpy(hostB.data(), hostB.size(), B, hostB.size(),
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
        aclrtMemcpy(hostC.data(), hostC.size(), C, hostC.size(),
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS)
        return CUBLAS_STATUS_EXECUTION_FAILED;

    const double scale = readScalar(alpha, dt);
    const double betaValue = readScalar(beta, dt);

    auto elementAt = [&](std::vector<unsigned char> &host, size_t index) -> double {
        if (element == 8) {
            double value = 0;
            std::memcpy(&value, &host[index * 8], 8);
            return value;
        }
        float value = 0;
        std::memcpy(&value, &host[index * 4], 4);
        return value;
    };
    auto storeAt = [&](std::vector<unsigned char> &host, size_t index,
                       double value) {
        if (element == 8) {
            std::memcpy(&host[index * 8], &value, 8);
            return;
        }
        const float narrow = static_cast<float>(value);
        std::memcpy(&host[index * 4], &narrow, 4);
    };

    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            double sum = 0;
            for (int p = 0; p < k; ++p) {
                const size_t ai = ta == 0 ? size_t(i) * k + p : size_t(p) * m + i;
                const size_t bi = tb == 0 ? size_t(p) * n + j : size_t(j) * k + p;
                sum += elementAt(hostA, ai) * elementAt(hostB, bi);
            }
            const size_t ci = size_t(i) * n + j;
            storeAt(hostC, ci, betaValue * elementAt(hostC, ci) + scale * sum);
        }
    (void)lda;
    (void)ldb;
    (void)ldc;
    (void)isDouble;

    if (aclrtMemcpy(C, hostC.size(), hostC.data(), hostC.size(),
                    ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS)
        return CUBLAS_STATUS_EXECUTION_FAILED;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t gemmCommon(cublasHandle_t handle, int ta, int tb, int m, int n,
                          int k, const void *alpha, const void *A, int lda,
                          const void *B, int ldb, const void *beta, void *C,
                          int ldc, aclDataType dt) {
    const aclComputeType compute = computeOf(dt);
    if (!handle || !A || !B || !C) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (m < 0 || n < 0 || k < 0) return CUBLAS_STATUS_INVALID_VALUE;
    
    if (!transSupported(ta) || !transSupported(tb))
        return CUBLAS_STATUS_NOT_SUPPORTED;

    CublasHandle *self = reinterpret_cast<CublasHandle *>(handle);

    
    aclopHandle *h =
        acquireHandle(self, ta, tb, 0, m, n, k, dt, dt, dt, compute);
    if (!h)
        return gemmOnHost(ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc,
                          dt);

    return toCublas(aclblasGemmEx(toTrans(ta), toTrans(tb), ACL_TRANS_N, m, n,
                                  k, alpha, A, lda, dt, B, ldb, dt, beta, C,
                                  ldc, dt, compute, self->stream));
}

// The level-1 operations have no counterpart in the CBLAS interface of this CANN, so they are computed on the host: correct, and slow.
cublasStatus_t level1(cublasHandle_t handle, int64_t n, const void *alpha,
                      const void *x, int64_t incX, void *y, int64_t incY,
                      aclDataType dt, int operation) {
    if (!handle || !x || !y || n < 0) return CUBLAS_STATUS_INVALID_VALUE;
    if (incX == 0 || incY == 0) return CUBLAS_STATUS_INVALID_VALUE;
    CublasHandle *self = reinterpret_cast<CublasHandle *>(handle);
    const size_t element = dt == ACL_DOUBLE ? 8 : (dt == ACL_FLOAT16 ? 2 : 4);

    // The scalar is a host pointer or a device one, depending on the pointer mode of the handle.
    double scale = 0;
    if (alpha) {
        if (self->pointerMode == 0) {
            scale = readScalar(alpha, dt);
        } else {
            std::vector<unsigned char> raw(element);
            if (aclrtMemcpy(raw.data(), element, alpha, element,
                            ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS)
                return CUBLAS_STATUS_EXECUTION_FAILED;
            scale = readScalar(raw.data(), dt);
        }
    }
    if (dt == ACL_FLOAT16) {
        const uint16_t raw = static_cast<uint16_t>(scale);
        float value = 0;
        std::memcpy(&value, &raw, sizeof value);
        scale = value;
    }

    const size_t count = static_cast<size_t>(n);
    std::vector<unsigned char> hostX(count * element);
    std::vector<unsigned char> hostY(count * element);
    if (aclrtMemcpy(hostX.data(), hostX.size(), x, hostX.size(),
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS)
        return CUBLAS_STATUS_EXECUTION_FAILED;
    if (aclrtMemcpy(hostY.data(), hostY.size(), y, hostY.size(),
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS)
        return CUBLAS_STATUS_EXECUTION_FAILED;

    for (int64_t i = 0; i < n; ++i) {
        const size_t xi = static_cast<size_t>(i * incX) * element;
        const size_t yi = static_cast<size_t>(i * incY) * element;
        if (xi + element > hostX.size() || yi + element > hostY.size())
            return CUBLAS_STATUS_INVALID_VALUE;
        if (operation == 0) { // y = alpha * x + y
            if (dt == ACL_DOUBLE) {
                double a = 0, b = 0;
                std::memcpy(&a, &hostX[xi], sizeof a);
                std::memcpy(&b, &hostY[yi], sizeof b);
                b += scale * a;
                std::memcpy(&hostY[yi], &b, sizeof b);
            } else {
                float a = 0, b = 0;
                std::memcpy(&a, &hostX[xi], sizeof a);
                std::memcpy(&b, &hostY[yi], sizeof b);
                b += static_cast<float>(scale) * a;
                std::memcpy(&hostY[yi], &b, sizeof b);
            }
        } else { // y = alpha * y
            if (dt == ACL_DOUBLE) {
                double b = 0;
                std::memcpy(&b, &hostY[yi], sizeof b);
                b *= scale;
                std::memcpy(&hostY[yi], &b, sizeof b);
            } else {
                float b = 0;
                std::memcpy(&b, &hostY[yi], sizeof b);
                b *= static_cast<float>(scale);
                std::memcpy(&hostY[yi], &b, sizeof b);
            }
        }
    }
    if (aclrtMemcpy(y, hostY.size(), hostY.data(), hostY.size(),
                    ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS)
        return CUBLAS_STATUS_EXECUTION_FAILED;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t gemvCommon(cublasHandle_t handle, int trans, int m, int n,
                          const void *alpha, const void *A, int lda,
                          const void *x, int incx, const void *beta, void *y,
                          int incy, aclDataType dt) {
    if (!handle || !A || !x || !y) return CUBLAS_STATUS_INVALID_VALUE;
    if (m < 0 || n < 0) return CUBLAS_STATUS_INVALID_VALUE;
    if (!transSupported(trans)) return CUBLAS_STATUS_NOT_SUPPORTED;
    CublasHandle *self = reinterpret_cast<CublasHandle *>(handle);
    const aclError rc =
        aclblasGemvEx(toTrans(trans), m, n, alpha, A, lda, dt, x, incx, dt,
                      beta, y, incy, dt, computeOf(dt), self->stream);
    if (rc == ACL_SUCCESS) return CUBLAS_STATUS_SUCCESS;

    // The same story as the matrix multiplication: the operation is in the registry, and the registry is empty on an image without the operator packages.
    const size_t element = dt == ACL_DOUBLE ? 8 : 4;
    const size_t rows = trans == 0 ? size_t(m) : size_t(n);
    const size_t cols = trans == 0 ? size_t(n) : size_t(m);
    std::vector<unsigned char> hostA(rows * cols * element);
    std::vector<unsigned char> hostX(cols * element);
    std::vector<unsigned char> hostY(rows * element);
    if (aclrtMemcpy(hostA.data(), hostA.size(), A, hostA.size(),
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
        aclrtMemcpy(hostX.data(), hostX.size(), x, hostX.size(),
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
        aclrtMemcpy(hostY.data(), hostY.size(), y, hostY.size(),
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS)
        return CUBLAS_STATUS_EXECUTION_FAILED;
    const double scale = readScalar(alpha, dt);
    const double betaValue = readScalar(beta, dt);
    for (size_t i = 0; i < rows; ++i) {
        double sum = 0;
        for (size_t j = 0; j < cols; ++j) {
            double a = 0, xv = 0;
            if (element == 8) {
                std::memcpy(&a, &hostA[(i * cols + j) * 8], 8);
                std::memcpy(&xv, &hostX[j * 8], 8);
            } else {
                float af = 0, xf = 0;
                std::memcpy(&af, &hostA[(i * cols + j) * 4], 4);
                std::memcpy(&xf, &hostX[j * 4], 4);
                a = af;
                xv = xf;
            }
            sum += a * xv;
        }
        double yv = 0;
        std::memcpy(&yv, &hostY[i * element], element);
        const double result = betaValue * yv + scale * sum;
        if (element == 8) {
            std::memcpy(&hostY[i * 8], &result, 8);
        } else {
            const float narrow = static_cast<float>(result);
            std::memcpy(&hostY[i * 4], &narrow, 4);
        }
    }
    if (aclrtMemcpy(y, hostY.size(), hostY.data(), hostY.size(),
                    ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS)
        return CUBLAS_STATUS_EXECUTION_FAILED;
    return CUBLAS_STATUS_SUCCESS;
}

} 

extern "C" {

cublasStatus_t cublasCreate_v2(cublasHandle_t *out) {
    if (!out) return CUBLAS_STATUS_INVALID_VALUE;
    *out = reinterpret_cast<cublasHandle_t>(new (std::nothrow) CublasHandle());
    return *out ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_ALLOC_FAILED;
}

cublasStatus_t cublasDestroy_v2(cublasHandle_t handle) {
    if (!handle) return CUBLAS_STATUS_INVALID_VALUE;
    CublasHandle *self = reinterpret_cast<CublasHandle *>(handle);
    
    delete self;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, cudaStream_t stream) {
    if (!handle) return CUBLAS_STATUS_INVALID_VALUE;
    reinterpret_cast<CublasHandle *>(handle)->stream =
        reinterpret_cast<aclrtStream>(stream);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetStream_v2(cublasHandle_t handle, cudaStream_t *out) {
    if (!handle || !out) return CUBLAS_STATUS_INVALID_VALUE;
    *out = reinterpret_cast<cudaStream_t>(
        reinterpret_cast<CublasHandle *>(handle)->stream);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSgemm(cublasHandle_t handle, cublasOperation_t ta,
                           cublasOperation_t tb, int m, int n, int k,
                           const float *alpha, const float *A, int lda,
                           const float *B, int ldb, const float *beta,
                           float *C, int ldc) {
    return gemmCommon(handle, static_cast<int>(ta), static_cast<int>(tb), m, n,
                      k, alpha, A, lda, B, ldb, beta, C, ldc, ACL_FLOAT);
}

cublasStatus_t cublasHgemm(cublasHandle_t handle, cublasOperation_t ta,
                           cublasOperation_t tb, int m, int n, int k,
                           const __half *alpha, const __half *A, int lda,
                           const __half *B, int ldb, const __half *beta,
                           __half *C, int ldc) {
    return gemmCommon(handle, static_cast<int>(ta), static_cast<int>(tb), m, n,
                      k, alpha, A, lda, B, ldb, beta, C, ldc, ACL_FLOAT16);
}

cublasStatus_t cublasDgemm(cublasHandle_t handle, cublasOperation_t ta,
                           cublasOperation_t tb, int m, int n, int k,
                           const double *alpha, const double *A, int lda,
                           const double *B, int ldb, const double *beta,
                           double *C, int ldc) {
    
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasSetPointerMode(cublasHandle_t handle,
                                    cublasPointerMode_t mode) {
    if (!handle) return CUBLAS_STATUS_INVALID_VALUE;
    reinterpret_cast<CublasHandle *>(handle)->pointerMode =
        mode == CUBLAS_POINTER_MODE_DEVICE ? 1 : 0;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetPointerMode(cublasHandle_t handle,
                                    cublasPointerMode_t *out) {
    if (!handle || !out) return CUBLAS_STATUS_INVALID_VALUE;
    *out = reinterpret_cast<CublasHandle *>(handle)->pointerMode
               ? CUBLAS_POINTER_MODE_DEVICE
               : CUBLAS_POINTER_MODE_HOST;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSaxpy(cublasHandle_t handle, int n, const float *alpha,
                            const float *x, int incx, float *y, int incy) {
    return level1(handle, n, alpha, x, incx, y, incy, ACL_FLOAT, 0);
}

cublasStatus_t cublasDaxpy(cublasHandle_t handle, int n, const double *alpha,
                            const double *x, int incx, double *y, int incy) {
    return level1(handle, n, alpha, x, incx, y, incy, ACL_DOUBLE, 0);
}

cublasStatus_t cublasSscal(cublasHandle_t handle, int n, const float *alpha,
                            float *x, int incx) {
    return level1(handle, n, alpha, x, incx, x, incx, ACL_FLOAT, 1);
}

cublasStatus_t cublasDscal(cublasHandle_t handle, int n, const double *alpha,
                            double *x, int incx) {
    return level1(handle, n, alpha, x, incx, x, incx, ACL_DOUBLE, 1);
}

cublasStatus_t cublasSgemv(cublasHandle_t handle, cublasOperation_t trans,
                           int m, int n, const float *alpha, const float *A,
                           int lda, const float *x, int incx, const float *beta,
                           float *y, int incy) {
    return gemvCommon(handle, static_cast<int>(trans), m, n, alpha, A, lda, x,
                       incx, beta, y, incy, ACL_FLOAT);
}

cublasStatus_t cublasDgemv(cublasHandle_t handle, cublasOperation_t trans,
                           int m, int n, const double *alpha, const double *A,
                           int lda, const double *x, int incx, const double *beta,
                           double *y, int incy) {
    return gemvCommon(handle, static_cast<int>(trans), m, n, alpha, A, lda, x,
                       incx, beta, y, incy, ACL_DOUBLE);
}

cublasStatus_t cublasGetVersion_v2(int *out) {
    if (!out) return CUBLAS_STATUS_INVALID_VALUE;
    
    *out = CUBLAS_VERSION;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetMathMode(cublasHandle_t handle,
                                 cublasMath_t mode) {
    
    if (mode == CUBLAS_TF32_TENSOR_OP_MATH ||
        mode == CUBLAS_PEDANTIC_MATH)
        return CUBLAS_STATUS_NOT_SUPPORTED;
    return CUBLAS_STATUS_SUCCESS;
}

} 
