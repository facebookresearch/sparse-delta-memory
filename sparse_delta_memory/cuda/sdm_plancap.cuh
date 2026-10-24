// sdm_plancap.cuh: launch capture for the backward's per-step plans (host code only: no device code).
//
// A plan builder runs a per-step launcher's own host function ONCE, at the step loop's largest step, with this thread's
// capture slot set: every check of the host function runs as for a real call, and its single kernel launch is RECORDED
// instead of issued -- the kernel (its host stub address), grid, block, dynamic shared memory and every parameter by
// value, converted to the kernel's own parameter type as <<<>>> and cudaLaunchKernelEx(&cfg, kernel, args...) convert the
// arguments. A plan's step launch patches only the parameters that vary per step (the step index, row offsets, valid
// tokens) and issues the launch with cudaLaunchKernel (what <<<>>> issues) or, with PDL, with cudaLaunchKernelExC and
// programmatic stream serialization (what cudaLaunchKernelEx(&cfg, kernel, args...) issues with that attribute): the
// same kernel, configuration, parameters and stream as the host function's own launch.
#pragma once
#include <ATen/ATen.h>
#include <cuda_runtime.h>
#include <c10/util/Exception.h>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <utility>

namespace sdm_plancap {

constexpr int kMaxParams = 32;
constexpr int kBufBytes = 768;

struct Launch {
    int32_t set;                      // 1 once the planned call recorded its launch (0: it launched nothing)
    int32_t nparams;
    const void* fn;                   // the kernel's host stub address (what <<<>>> passes to cudaLaunchKernel)
    uint32_t grid[3];
    uint32_t block[3];
    uint64_t smem;
    int32_t off[kMaxParams];          // byte offset of parameter i in buf
    int32_t size[kMaxParams];         // sizeof(parameter i's type)
    alignas(16) unsigned char buf[kBufBytes];
};

// This thread's capture slot: non-null only while a plan builder runs its host function.
inline Launch*& slot() {
    static thread_local Launch* p = nullptr;
    return p;
}

template <typename T>
inline void put(Launch& c, int& pos, const T& v) {
    static_assert(std::is_trivially_copyable<T>::value, "plan capture: kernel parameters must be trivially copyable");
    constexpr int al = alignof(T) < 16 ? static_cast<int>(alignof(T)) : 16;
    pos = (pos + al - 1) / al * al;
    TORCH_CHECK(c.nparams < kMaxParams && pos + static_cast<int>(sizeof(T)) <= kBufBytes,
                "plan capture: too many kernel parameter bytes");
    std::memcpy(c.buf + pos, &v, sizeof(T));
    c.off[c.nparams] = pos;
    c.size[c.nparams] = static_cast<int32_t>(sizeof(T));
    ++c.nparams;
    pos += static_cast<int>(sizeof(T));
}

// Inside a plan builder: record the launch (each argument converted to the kernel's parameter type) and return true; the
// host function then returns without launching. Outside one: false (the host function launches as it always did).
template <typename... P, typename... A>
inline bool take(void (*kernel)(P...), dim3 grid, dim3 block, size_t smem, A&&... args) {
    static_assert(sizeof...(P) == sizeof...(A), "plan capture: argument count != kernel parameter count");
    Launch* c = slot();
    if (c == nullptr) return false;
    TORCH_CHECK(c->set == 0, "plan capture: the planned call launched more than one kernel");
    c->set = 1;
    c->nparams = 0;
    c->fn = reinterpret_cast<const void*>(kernel);
    c->grid[0] = grid.x; c->grid[1] = grid.y; c->grid[2] = grid.z;
    c->block[0] = block.x; c->block[1] = block.y; c->block[2] = block.z;
    c->smem = static_cast<uint64_t>(smem);
    int pos = 0;
    (put<typename std::decay<P>::type>(*c, pos, static_cast<typename std::decay<P>::type>(std::forward<A>(args))), ...);
    return true;
}

// Parameter i of a captured launch as type T (nullptr when the recorded size differs: the caller refuses the plan).
template <typename T>
inline T* param(Launch& c, int i) {
    if (i < 0 || i >= c.nparams || c.size[i] != static_cast<int32_t>(sizeof(T))) return nullptr;
    return reinterpret_cast<T*>(c.buf + c.off[i]);
}

// Issue a captured launch on `stream`: cudaLaunchKernel (= <<<>>>), or with pdl cudaLaunchKernelExC + programmatic stream
// serialization (= cudaLaunchKernelEx(&cfg, kernel, args...) with that attribute). Returns 0 or a cudaError_t (the
// launch's, else cudaGetLastError()'s right after it, as C10_CUDA_KERNEL_LAUNCH_CHECK probes); an error state is consumed.
inline int64_t launch(const Launch& c, bool pdl, cudaStream_t stream) {
    void* params[kMaxParams];
    for (int i = 0; i < c.nparams; ++i) params[i] = const_cast<unsigned char*>(c.buf) + c.off[i];
    const dim3 grid(c.grid[0], c.grid[1], c.grid[2]);
    const dim3 block(c.block[0], c.block[1], c.block[2]);
    cudaError_t e;
    if (pdl) {
        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[0].val.programmaticStreamSerializationAllowed = 1;
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = grid;
        cfg.blockDim = block;
        cfg.dynamicSmemBytes = static_cast<size_t>(c.smem);
        cfg.stream = stream;
        cfg.attrs = attr;
        cfg.numAttrs = 1;
        e = cudaLaunchKernelExC(&cfg, c.fn, params);
    } else {
        e = cudaLaunchKernel(c.fn, grid, block, params, static_cast<size_t>(c.smem), stream);
    }
    if (e == cudaSuccess) e = cudaGetLastError();
    else (void)cudaGetLastError();
    return static_cast<int64_t>(e);
}

// RAII: set this thread's slot to `c` for one planned call (refuses nesting).
struct Scope {
    explicit Scope(Launch* c) {
        TORCH_CHECK(slot() == nullptr, "plan capture: nested plan builders");
        std::memset(c, 0, sizeof(Launch));
        slot() = c;
    }
    ~Scope() { slot() = nullptr; }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
};

// The current device, or -1 (error state consumed).
inline int current_device() {
    int dev = -1;
    if (cudaGetDevice(&dev) != cudaSuccess) {
        (void)cudaGetLastError();
        return -1;
    }
    return dev;
}

// A plan as a CPU uint8 tensor (the bytes of the trivially copyable plan struct), and back.
template <typename T>
inline at::Tensor to_tensor(const T& pl) {
    static_assert(std::is_trivially_copyable<T>::value, "plan: trivially copyable");
    at::Tensor t = at::empty({static_cast<int64_t>(sizeof(T))}, at::TensorOptions().dtype(at::kByte));
    std::memcpy(t.data_ptr<uint8_t>(), &pl, sizeof(T));
    return t;
}

template <typename T>
inline bool from_bytes(T& pl, const void* bytes, int64_t n, uint64_t magic) {
    if (n != static_cast<int64_t>(sizeof(T))) return false;
    std::memcpy(&pl, bytes, sizeof(T));
    return pl.magic == magic;
}

}  // namespace sdm_plancap
