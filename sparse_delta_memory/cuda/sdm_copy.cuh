// sdm_copy.cuh: the routing predicate's device mirror and the portable copy layer of the fast kernels.
//
// SDM_PORTABLE_ARCH is defined in every device pass except the modern ones (sm_90 / sm_100 / sm_103: H100, B200,
// GB300), mirroring cuda/_device.py backend(): the modern passes keep their preprocessed text, every other device pass
// (sm_80, 86, 89, 120, ...) compiles the portable side of the kernels' arch guards, so an sm_120 build runs the same
// device branches as sm_86 / 89. Guards: `#ifdef SDM_PORTABLE_ARCH` (a portable device pass) and
// `#if defined(__CUDA_ARCH__) && !defined(SDM_PORTABLE_ARCH)` (a modern device pass); the host pass defines neither.
//
// sdmcp: cooperative global -> shared copies with cp.async for the portable side. Every copying thread issues its pieces
// and commits its own groups; a thread's cp.async wait covers its own copies only, so the bytes reach the other threads
// through a barrier passed AFTER every copier waited (publish_cta / publish_named), or through an mbarrier the copies
// arrive on (arrive_on). No bulk copies, no transaction counts, no PDL, no clusters. Every call is made by all threads of
// the copying group at the same program point, never from a thread-0 branch (the barriers would deadlock). Only portable
// device passes see this code (no modern pass gets a token of it); below sm_80 (no cp.async) every call traps, so
// multi-architecture builds keep compiling.
#pragma once
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 900 && __CUDA_ARCH__ != 1000 && __CUDA_ARCH__ != 1030
#define SDM_PORTABLE_ARCH 1
#endif

#ifdef SDM_PORTABLE_ARCH
#include <cstdint>
#if __CUDA_ARCH__ >= 800
#define SDMCP_ASM(...) asm volatile(__VA_ARGS__)
#else
#define SDMCP_ASM(...) __trap()
#endif
namespace sdmcp {
__device__ __forceinline__ uint32_t saddr(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
// 16 bytes through L2 only (coherent with what the previous kernel wrote); the bytes past src_bytes (0..16) are
// zero-filled (src_bytes 0 reads nothing, but `g` must still be a valid global address).
__device__ __forceinline__ void cp16(void* s, const void* g, int src_bytes = 16) {
    SDMCP_ASM("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(saddr(s)), "l"(g), "r"(src_bytes) : "memory");
}
// 8 / 4 bytes (rows that are not whole 16-byte vectors), through L1 and L2; zero fill past src_bytes.
__device__ __forceinline__ void cp8(void* s, const void* g, int src_bytes = 8) {
    SDMCP_ASM("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(saddr(s)), "l"(g), "r"(src_bytes) : "memory");
}
__device__ __forceinline__ void cp4(void* s, const void* g, int src_bytes = 4) {
    SDMCP_ASM("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(saddr(s)), "l"(g), "r"(src_bytes) : "memory");
}
__device__ __forceinline__ void commit() { SDMCP_ASM("cp.async.commit_group;\n" ::: "memory"); }
// This thread's groups: at most N of the newest still pending (wait<N>), or none (wait_all).
template <int N>
__device__ __forceinline__ void wait() { SDMCP_ASM("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }
__device__ __forceinline__ void wait_all() { SDMCP_ASM("cp.async.wait_all;\n" ::: "memory"); }
// Cooperative copy of `rows` rows of `row_bytes` bytes (a multiple of 16; every row start 16-byte aligned on both
// sides): row r from g + r * g_stride to s + r * s_stride (bytes), the 16-byte pieces dealt round-robin over the threads
// tid = 0 .. nthr - 1 of the copying group. Issues only: the caller commits.
__device__ __forceinline__ void copy_rows16(void* s, int s_stride, const void* g, int64_t g_stride, int rows,
                                            int row_bytes, int tid, int nthr) {
    const int per = row_bytes >> 4;
    for (int i = tid; i < rows * per; i += nthr) {
        const int r = i / per;
        const int c = (i - r * per) << 4;
        cp16(static_cast<char*>(s) + r * s_stride + c, static_cast<const char*>(g) + r * g_stride + c);
    }
}
// Every thread of the CTA: its own groups have landed, then the CTA barrier publishes every copier's bytes.
__device__ __forceinline__ void publish_cta() {
    wait_all();
    __syncthreads();
}
// The same for the nthr threads (a multiple of 32) that use named barrier `id` (1..15; 0 is __syncthreads').
__device__ __forceinline__ void publish_named(int id, int nthr) {
    wait_all();
    asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(nthr) : "memory");
}
// mbarrier forms (sm_80): init with the number of copying threads; arrive_on = one arrival of this thread that the
// system makes when ALL of its prior cp.async copies have landed (.noinc: it counts against the init count); consumers
// spin on test_parity(bar, phase & 1).
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
    SDMCP_ASM("mbarrier.init.shared.b64 [%0], %1;\n" ::"r"(saddr(bar)), "r"(count) : "memory");
}
__device__ __forceinline__ void arrive_on(uint64_t* bar) {
    SDMCP_ASM("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" ::"r"(saddr(bar)) : "memory");
}
__device__ __forceinline__ bool test_parity(uint64_t* bar, uint32_t parity) {
    uint32_t done = 0u;
#if __CUDA_ARCH__ >= 800
    asm volatile("{\n .reg .pred p;\n mbarrier.test_wait.parity.shared.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(done) : "r"(saddr(bar)), "r"(parity) : "memory");
#else
    __trap();
#endif
    return done != 0u;
}
}  // namespace sdmcp
#undef SDMCP_ASM
#endif  // SDM_PORTABLE_ARCH
