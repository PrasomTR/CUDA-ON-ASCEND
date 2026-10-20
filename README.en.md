# cuascend

Run CUDA applications on Huawei Ascend NPUs, unchanged: the application
keeps its CUDA Driver API calls, and a preloaded library answers them.

```
  your CUDA application
        |  cuInit, cuModuleLoadData, cuLaunchKernel, cuMemcpy ...
        v
  libcuascend.so            <- LD_PRELOAD
        |
        v
  PTX -> IR -> AscendC -> bisheng -> op package
        |
        v
  Ascend 910B / 910C
```

The application is not recompiled and not modified. A CUDA application
built against NVIDIA's headers links against `libcuda.so.1`; on a machine
without the NVIDIA driver that library is replaced by a stub that exports
the symbols the application imports, and the real work is done by the
preloaded shim.

## What works today

* The CUDA Driver API is intercepted: devices, contexts, memory, streams,
  events, modules, functions, launches and copies.
* Kernels are parsed from PTX, lowered to a hardware-independent IR and
  turned into AscendC, which is compiled and packaged into an operator
  model.
* Elementwise kernels run on the NPU itself, at native speed, through
  the prebuilt `aclnn` operations.
* Everything else — warp shuffles, atomics, loops, barriers — runs through
  an IR executor on the host, with the data staying in the application's
  own buffers. The numbers are right, the speed is not.
* The cuBLAS layer (`cublasSgemm`, `cublasSgemv`, `cublasSaxpy`,
  `cublasSscal` and their double variants) answers with numbers that match
  a CPU reference.

What does not work is refused, loudly, instead of being computed wrongly:
an operation the models do not cover is reported with its reason, and the
kernel does not run.

## Quick start

```bash
cmake -S . -B build && cmake --build build -j
cd build && ctest --output-on-failure
```

On a machine with CANN and an Ascend card:

```bash
cmake -S . -B build && cmake --build build -j
cd build && ctest --output-on-failure
```

`ctest` reports four tests: the unit tests, the interception of a real CUDA
application, the full stack on the card, and the cuBLAS layer.

To run an application of your own, build the stub for it and preload the
shim:

```bash
./tools/make_stub.py /path/to/your/app      # a libcuda.so.1 with the symbols it imports
LD_PRELOAD=/path/to/libcuascend.so \
LD_LIBRARY_PATH=/path/to/your/app.libcuda:$LD_LIBRARY_PATH \
  /path/to/your/app
```

The toolchain — the AscendC compiler and the operator package generator —
is found through the CANN installation, captured once when the runtime
starts, and checked for executability.

## Layout

| File | What is in it |
|---|---|
| `src/cuascend.h` | the whole API: the CUDA ABI, the IR, the backend, the compiler |
| `src/cuascend.cpp` | the frontend: PTX, the IR, the AscendC generator, the compile pipeline, the IR executor |
| `src/runtime.cpp` | the execution: the Ascend backend, the stub, the lowering to `aclnn` |
| `src/shim.cpp` | the interception of the CUDA Driver API |
| `src/cublas.cpp` | the cuBLAS layer |
| `tests/` | the unit tests and the interception application |
| `tools/` | `captx`, the verification tools, the stub generator, `deploy.sh` |

Every section of a file starts with a `// MAIN:` line, and the code
carries no commentary beyond that: the reasoning lives in README.md.

## Requirements

* A C++20 compiler and CMake 3.20.
* CANN 8.3 and an Ascend card for the hardware tests. Without them the
  project still builds, and the hardware tests report a skip.
* Python 3 for `tools/make_stub.py`.

## Documentation

README.md is the full description in Russian: how each part works, what was
found on the way, what is still open and why.

## License

MIT. See LICENSE.
