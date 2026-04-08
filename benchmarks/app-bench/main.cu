/**
 * main.cu — HOL Blocking Benchmark Suite  (AGILE-framework edition)
 * Target : NVIDIA RTX A5000 (Ampere sm_86, 64 SMs, 24 GB VRAM)
 * Proves : GDS Host-Initiated HOL blocking < BaM < AGILE GPU-Initiated I/O
 *
 * Architecture map
 * ─────────────────────────────────────────────────────────────────────────
 *  Arch 1 — GDS  : CPU calls cuFileRead() in a serial batch loop.
 *                  cudaDeviceSynchronize() after every batch = HOL blocking.
 *                  No AGILE framework involvement.
 *
 *  Arch 2 — BaM  : Persistent GPU threads own a ticket dispenser.
 *                  Each warp calls ctrl->getArrayWrap<int>(chain) (demand
 *                  fetch, no prefetch) via the real AGILE framework.
 *                  AgileLockChain is declared per-iteration so the slot
 *                  lock is released after each page, mirroring BaM eviction.
 *
 *  Arch 3 — AGILE: Same persistent-thread model as BaM.
 *                  Lane 0 additionally calls ctrl->prefetch(0, page_id+d, chain)
 *                  for d = 1..PREFETCH_DEPTH before the demand access.
 *                  GPUClockReplacementCache retains prefetched pages so that
 *                  the subsequent demand access is a guaranteed cache hit.
 * ─────────────────────────────────────────────────────────────────────────
 *
 * Build  : see Makefile  (sm_86, -lcufile, AGILE SDK on include/link path)
 * Usage  : ./main [config.h flags]
 *   Important flags (see config.h / -h for full list):
 *     -gsn  gpu_slot_num    GPU cache slots  (controls effective VRAM budget)
 *     -ss   slot_size       bytes per slot   (keep at 4096)
 *     -bar  nvme_bar        PCIe BAR address of the target SSD
 *     -qn   queue_num       NVMe queue pairs
 *     -qd   queue_depth     commands per queue (increase for AGILE prefetch)
 *     -sbn  ssd_block_num   logical SSD blocks (must be >= TOTAL_PAGES)
 *     -ad   agile_dim       AGILE internal parallelism dimension
 *
 * Key constants
 *   PAGE_SIZE_BYTES = 4096   one NVMe LBA = 1024 ints
 *   DATASET_BYTES   = 96 GB
 *   PREFETCH_DEPTH  = 8      oracle look-ahead window
 *   DUMMY_ITERS     = 50     xorshift-32 rounds (simulates real compute)
 *   PT_BLOCKS/THREADS        64 SMs x 8 warps = 3072 concurrent warps
 */

/* ─── Standard + CUDA headers ─────────────────────────────────────────────── */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>

#include <cuda_runtime.h>
#include <cuda.h>
#include <cufile.h>

/* ─── AGILE framework headers ─────────────────────────────────────────────── */
/*
 * Adjust -I paths in Makefile so these resolve from the AGILE SDK tree.
 *   agile_host.h  : AgileHost, the AGILE device API (AgileCtrl,
 *                   AgileLockChain, getArrayWrap, prefetch, asyncRead, …)
 *   cache_impl.h  : GPUClockReplacementCache, DisableCPUCache
 *   table_impl.h  : SimpleShareTable
 */
#include "agile_host.h"
#include "config.h"
#include "../common/cache_impl.h"
#include "../common/table_impl.h"

/* ─── AGILE type aliases (match main_1_.cu convention) ───────────────────── */
#define CPU_CACHE_IMPL   DisableCPUCache
#define SHARE_TABLE_IMPL SimpleShareTable
#define GPU_CACHE_IMPL   GPUClockReplacementCache<CPU_CACHE_IMPL, SHARE_TABLE_IMPL>
#define AGILE_CTRL       AgileCtrl<GPU_CACHE_IMPL, CPU_CACHE_IMPL, SHARE_TABLE_IMPL>
#define AGILE_HOST       AgileHost<GPU_CACHE_IMPL, CPU_CACHE_IMPL, SHARE_TABLE_IMPL>

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  Constants                                                                  */
/* ═══════════════════════════════════════════════════════════════════════════ */

static constexpr size_t PAGE_SIZE_BYTES = 4096ULL;
static constexpr size_t PAGE_SIZE_INTS  = PAGE_SIZE_BYTES / sizeof(int); /* 1024 */
static constexpr size_t DATASET_BYTES   = 96ULL * 1024ULL * 1024ULL * 1024ULL;
static constexpr size_t TOTAL_PAGES     = DATASET_BYTES / PAGE_SIZE_BYTES;
static constexpr int    WARP_SIZE       = 32;
static constexpr int    PREFETCH_DEPTH  = 8;
static constexpr int    DUMMY_ITERS     = 50;

/* Persistent-thread grid: 64 SMs × 8 warps/SM × 32 threads */
static constexpr int PT_BLOCKS  = 64;
static constexpr int PT_THREADS = 256;

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  Error macros                                                               */
/* ═══════════════════════════════════════════════════════════════════════════ */

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "[CUDA] %s:%d  %s\n",                             \
                    __FILE__, __LINE__, cudaGetErrorString(_e));               \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CUFILE_CHECK(call)                                                     \
    do {                                                                       \
        CUfileError_t _e = (call);                                             \
        if (_e.err != CU_FILE_SUCCESS) {                                       \
            fprintf(stderr, "[cuFile] %s:%d  err=%d\n",                       \
                    __FILE__, __LINE__, (int)_e.err);                          \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

/* Alias expected by AGILE framework internals */
#ifndef cuda_err_chk
#define cuda_err_chk(call) CUDA_CHECK(call)
#endif

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  Timing                                                                     */
/* ═══════════════════════════════════════════════════════════════════════════ */

static double wall_seconds()
{
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  dummy_compute                                                              */
/*                                                                             */
/*  50 rounds of xorshift-32 + LCG over one input word.  Forces real ALU      */
/*  activity so the benchmark measures I/O-compute overlap, not just I/O.     */
/* ═══════════════════════════════════════════════════════════════════════════ */

__device__ __forceinline__
uint32_t dummy_compute(uint32_t seed)
{
    uint32_t v = seed;
    #pragma unroll 5
    for (int i = 0; i < DUMMY_ITERS; ++i) {
        v ^= v << 13;
        v ^= v >> 7;
        v ^= v << 17;
        v  = v * 1664525u + 1013904223u;
    }
    return v;
}

/*
 * dummy_compute_agile — reads one int from the AGILE array-wrap and hashes it.
 *
 * AgileArrayWrap<int> supports arr[dev_idx][flat_element_idx].
 *   dev_idx  = 0 (single SSD)
 *   flat_idx = page_id * PAGE_SIZE_INTS + lane  (element in logical address space)
 */
template<typename AgileArray>
__device__ __forceinline__
uint32_t dummy_compute_agile(AgileArray& arr, int dev_idx, long long flat_idx)
{
    return dummy_compute((uint32_t)arr[dev_idx][flat_idx]);
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/*                                                                             */
/*  Architecture 1 — GDS  (Host-Initiated I/O, HOL Blocking baseline)         */
/*                                                                             */
/*  CPU loop:  cuFileRead (blocks) → launch kernel → cudaDeviceSynchronize    */
/*  The sync before the next read is the HOL point: I/O and compute are       */
/*  fully serialised; no AGILE framework is used.                             */
/*                                                                             */
/* ═══════════════════════════════════════════════════════════════════════════ */

__global__
void gds_compute_kernel(const int* __restrict__ batch,
                         size_t      batch_pages,
                         uint64_t*   checksum)
{
    size_t gid   = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch_pages * PAGE_SIZE_INTS;
    uint64_t acc = 0;
    unsigned long long int tmp = 0;

    for (size_t i = gid; i < total; i += (size_t)gridDim.x * blockDim.x)
        acc += dummy_compute((uint32_t)batch[i]);

    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, off);
    if (threadIdx.x % WARP_SIZE == 0) {
        atomicAdd((unsigned long long int *)checksum, (unsigned long long int)acc);
    }
}

static double run_gds(CUfileHandle_t cfh,
                      int*           d_buf,
                      size_t         vram_budget_bytes)
{
    size_t batch_pages = vram_budget_bytes / PAGE_SIZE_BYTES;
    if (batch_pages == 0) batch_pages = 1;

    uint64_t* d_checksum;
    CUDA_CHECK(cudaMalloc(&d_checksum, sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_checksum, 0, sizeof(uint64_t)));

    /* Register VRAM buffer for zero-copy GDS DMA */
    CUFILE_CHECK(cuFileBufRegister(d_buf, batch_pages * PAGE_SIZE_BYTES, 0));

    size_t pages_done = 0;
    double t0 = wall_seconds();

    while (pages_done < TOTAL_PAGES) {
        size_t chunk = batch_pages;
        if (pages_done + chunk > TOTAL_PAGES)
            chunk = TOTAL_PAGES - pages_done;

        off_t  file_off   = (off_t)(pages_done * PAGE_SIZE_BYTES);
        size_t read_bytes = chunk * PAGE_SIZE_BYTES;

        /* ── HOL blocking point: CPU blocks until NVMe DMA finishes ── */
        ssize_t ret = cuFileRead(cfh, d_buf, read_bytes, file_off, 0);
        if (ret < 0) {
            fprintf(stderr, "[GDS] cuFileRead error at page %zu\n", pages_done);
            break;
        }

        /* ── Compute kernel + sync: next read cannot start until sync ── */
        int grid = (int)((chunk * PAGE_SIZE_INTS + 255) / 256);
        gds_compute_kernel<<<grid, 256>>>(d_buf, chunk, d_checksum);
        CUDA_CHECK(cudaDeviceSynchronize()); /* ← explicit HOL */

        pages_done += chunk;
    }

    double elapsed = wall_seconds() - t0;
    CUFILE_CHECK(cuFileBufDeregister(d_buf));
    cudaFree(d_checksum);
    return elapsed;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/*                                                                             */
/*  Architecture 2 — BaM  (GPU-Initiated I/O, demand-only)                    */
/*                                                                             */
/*  Each warp claims pages via atomicAdd (ticket dispenser) and issues a       */
/*  synchronous demand fetch through the real AGILE framework:                 */
/*                                                                             */
/*    AgileLockChain chain;                                                    */
/*    auto agileArr = ctrl->getArrayWrap<int>(chain);                          */
/*                                                                             */
/*  On a cache miss the AGILE service kernel issues an NVMe command and the   */
/*  user warp spins until the slot is filled — the GPU stalls on I/O but the  */
/*  HOST is never involved, so HOL blocking is eliminated at the system level. */
/*                                                                             */
/*  AgileLockChain is declared inside the while-loop body so it is destroyed  */
/*  (and the slot lock released) at the end of each page's scope — this       */
/*  reproduces BaM's "instant evict after use" policy, preventing cache        */
/*  pollution and forcing a fresh fetch for every access.                      */
/*                                                                             */
/* ═══════════════════════════════════════════════════════════════════════════ */

__global__
void bam_bench_kernel(AGILE_CTRL* ctrl,
                      long long   total_pages,
                      uint64_t*   checksum_out,
                      long long *  ticket)
{
    const unsigned wmask = 0xFFFFFFFFu;
    int lane = threadIdx.x % WARP_SIZE;
    unsigned long long int tmp_chk = 0, tmp_tkt = 0;

    while (true) {
        /* One atomicAdd per warp via lane 0; broadcast page_id */
        long long page_id = 0;
        if (lane == 0) {
            page_id = atomicAdd((unsigned long long int *)ticket, 1ULL);
        }
        page_id = __shfl_sync(wmask, page_id, 0);
        if (page_id >= total_pages) break;

        /*
         * Fresh AgileLockChain per page.
         * Destructor at end of this block releases the cache slot lock,
         * making the slot immediately reclaimable — mirroring BaM eviction.
         */
        {
            AgileLockChain chain;

            /*
             * Synchronous demand access.
             * Hit  → returns immediately (page already in GPU cache).
             * Miss → AGILE service kernel issues NVMe read; this warp spins
             *        until the DMA completes and the slot is marked valid.
             *        No prefetch has been issued, so misses are frequent when
             *        the VRAM budget is smaller than the working set.
             */
            auto agileArr = ctrl->getArrayWrap<int>(chain);

            /* Warp-centric compute: each lane processes its own word */
            long long flat_base = page_id * (long long)PAGE_SIZE_INTS;
            uint32_t v = dummy_compute_agile(agileArr, 0, flat_base + lane);

            for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
                v += __shfl_xor_sync(wmask, v, off);
            if (lane == 0) {
                atomicAdd((unsigned long long int *)checksum_out, (unsigned long long int)v);
            }

        } /* chain destructs here: slot lock released */
    }
}

/* ─────────────────────────────────────────────────────────────────────────── */

static double run_bam(const Configs& cfg)
{
    AGILE_HOST host(0, cfg.slot_size);

    CPU_CACHE_IMPL   c_cache(0, cfg.slot_size);
    SHARE_TABLE_IMPL s_table(cfg.gpu_slot_num / 4);
    GPU_CACHE_IMPL   g_cache(cfg.gpu_slot_num, cfg.slot_size, cfg.ssd_block_num);

    host.setGPUCache(g_cache);
    host.setCPUCache(c_cache);
    host.setShareTable(s_table);

    host.addNvmeDev(cfg.nvme_bar, cfg.bar_size,
                    cfg.ssd_blk_offset, cfg.queue_num, cfg.queue_depth);
    host.initNvme();

    host.configParallelism(PT_BLOCKS, PT_THREADS, cfg.agile_dim);

    /* Report occupancy of both the service kernel and the user kernel */
    int occ_svc = 0, occ_usr = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occ_svc,
        start_agile_cq_service<GPU_CACHE_IMPL, CPU_CACHE_IMPL, SHARE_TABLE_IMPL>,
        PT_THREADS, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occ_usr, bam_bench_kernel, PT_THREADS, 0);
    fprintf(stderr, "[BaM]  service %d blk/SM   user %d blk/SM\n",
            occ_svc, occ_usr);

    host.initializeAgile();
    auto* ctrl = host.getAgileCtrlDevicePtr();

    long long* d_ticket;   CUDA_CHECK(cudaMalloc(&d_ticket,   sizeof(long long)));
    uint64_t*  d_chk;      CUDA_CHECK(cudaMalloc(&d_chk,      sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_ticket, 0, sizeof(long long)));
    CUDA_CHECK(cudaMemset(d_chk,   0, sizeof(uint64_t)));

    host.startAgile();
    double t0 = wall_seconds();

    /*
     * host.runKernel launches bam_bench_kernel on the configured grid,
     * then blocks until both the user kernel and the AGILE service kernel
     * have returned.  Equivalent to:
     *   bam_bench_kernel<<<PT_BLOCKS, PT_THREADS>>>(...);
     *   cudaDeviceSynchronize();
     * but with the service kernel co-scheduled alongside.
     */
    host.runKernel(bam_bench_kernel,
                   ctrl, (long long)TOTAL_PAGES, d_chk, d_ticket);

    double elapsed = wall_seconds() - t0;
    host.stopAgile();
    host.closeNvme();

    cudaFree(d_ticket);
    cudaFree(d_chk);
    return elapsed;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/*                                                                             */
/*  Architecture 3 — AGILE  (GPU-Initiated I/O, oracle prefetch depth 8)      */
/*                                                                             */
/*  Same persistent-thread + ticket-dispenser structure as BaM.               */
/*  The critical addition: lane 0 calls                                        */
/*                                                                             */
/*      ctrl->prefetch(dev_idx, blk_idx, chain)                               */
/*                                                                             */
/*  for pages [page_id+1 .. page_id+PREFETCH_DEPTH] BEFORE the demand access. */
/*  The AGILE service kernel drains these prefetch commands in the background  */
/*  via NVMe completion queues.  When a later warp demands one of those pages  */
/*  the slot is already warm → getArrayWrap returns immediately → zero stall.  */
/*                                                                             */
/*  GPUClockReplacementCache (not the instant-evict policy used in BaM)       */
/*  retains recently-prefetched pages until evicted by capacity pressure,      */
/*  ensuring the prefetch window stays warm.                                   */
/*                                                                             */
/*  AgileLockChain chain is declared once per iteration and covers both the   */
/*  prefetch calls and the demand access within that scope, letting AGILE      */
/*  track all in-flight I/O associated with this warp's transaction.           */
/*                                                                             */
/* ═══════════════════════════════════════════════════════════════════════════ */

__global__
void agile_bench_kernel(AGILE_CTRL* ctrl,
                        long long   total_pages,
                        uint64_t*   checksum_out,
                        long long*  ticket)
{
    const unsigned wmask = 0xFFFFFFFFu;
    int lane = threadIdx.x % WARP_SIZE;
    unsigned long long int tmp_chk = 0, tmp_tkt = 0;

    while (true) {
        long long page_id = 0;
        if (lane == 0) {
            page_id = atomicAdd((unsigned long long int *)ticket, 1ULL);
        }
        page_id = __shfl_sync(wmask, page_id, 0);
        if (page_id >= total_pages) break;

        /*
         * One AgileLockChain per iteration, shared across prefetch + demand.
         * This lets AGILE associate the prefetch completions with the correct
         * lock context so in-flight I/O is tracked and released together.
         */
        AgileLockChain chain;

        /*
         * ── Oracle prefetch (lane 0 only) ───────────────────────────────────
         *
         * ctrl->prefetch(dev_idx, blk_idx, chain)
         *   dev_idx = 0            — single SSD in this benchmark
         *   blk_idx = page_id + d  — logical SSD block to pre-load
         *   chain                  — lock context for this transaction
         *
         * prefetch() enqueues a non-blocking NVMe read into the AGILE
         * completion queue without stalling this warp.  The service kernel
         * services the queue in parallel and marks the target cache slot
         * valid when the DMA completes.
         *
         * "Oracle" = we know the exact sequential access pattern, so every
         * prefetched page IS the next demand — perfect hit rate.
         */
        if (lane == 0) {
            for (int d = 1; d <= PREFETCH_DEPTH; ++d) {
                long long pf = page_id + (long long)d;
                if (pf < total_pages)
                    ctrl->prefetch(0, (unsigned int)pf);
            }
        }
        __syncwarp(); /* ensure prefetch cmds are visible before the demand */

        /*
         * ── Demand access ───────────────────────────────────────────────────
         *
         * Because ctrl->prefetch() was called PREFETCH_DEPTH ticks ago
         * (by a prior warp's iteration), this page's slot is already filled.
         * getArrayWrap<int>(chain) finds a hit and returns immediately.
         * NVMe latency is fully hidden behind computation.
         */
        auto agileArr = ctrl->getArrayWrap<int>(chain);

        long long flat_base = page_id * (long long)PAGE_SIZE_INTS;
        uint32_t v = dummy_compute_agile(agileArr, 0, flat_base + lane);

        for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
            v += __shfl_xor_sync(wmask, v, off);
        if (lane == 0) {
            atomicAdd((unsigned long long int *)checksum_out, (unsigned long long int)v);
        }

        /*
         * chain destructs here — releases the demand slot lock.
         * GPUClockReplacementCache keeps the page alive in the cache until
         * evicted by capacity pressure, so prefetched-but-not-yet-demanded
         * pages remain valid for future warp accesses.
         */
    }
}

/* ─────────────────────────────────────────────────────────────────────────── */

static double run_agile(const Configs& cfg)
{
    AGILE_HOST host(0, cfg.slot_size);

    CPU_CACHE_IMPL   c_cache(0, cfg.slot_size);
    SHARE_TABLE_IMPL s_table(cfg.gpu_slot_num / 4);

    /*
     * AGILE with a PREFETCH_DEPTH-8 window needs enough cache slots to hold
     * (active_warps × PREFETCH_DEPTH) pages simultaneously without evicting
     * in-flight prefetches.  With 3072 concurrent warps × 8 = 24576 slots
     * minimum.  gpu_slot_num in config.h defaults to 65536×8 which is ample.
     */
    GPU_CACHE_IMPL g_cache(cfg.gpu_slot_num, cfg.slot_size, cfg.ssd_block_num);

    host.setGPUCache(g_cache);
    host.setCPUCache(c_cache);
    host.setShareTable(s_table);

    /*
     * Increase queue_depth (config flag -qd) to keep multiple NVMe commands
     * in flight simultaneously.  The AGILE service kernel can saturate the
     * NVMe bandwidth only when queue depth >= number of concurrent warps
     * × average outstanding prefetches.
     */
    host.addNvmeDev(cfg.nvme_bar, cfg.bar_size,
                    cfg.ssd_blk_offset, cfg.queue_num, cfg.queue_depth);
    host.initNvme();

    host.configParallelism(PT_BLOCKS, PT_THREADS, cfg.agile_dim);

    int occ_svc = 0, occ_usr = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occ_svc,
        start_agile_cq_service<GPU_CACHE_IMPL, CPU_CACHE_IMPL, SHARE_TABLE_IMPL>,
        PT_THREADS, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occ_usr, agile_bench_kernel, PT_THREADS, 0);
    fprintf(stderr, "[AGILE] service %d blk/SM   user %d blk/SM\n",
            occ_svc, occ_usr);

    host.initializeAgile();
    auto* ctrl = host.getAgileCtrlDevicePtr();

    long long* d_ticket; CUDA_CHECK(cudaMalloc(&d_ticket, sizeof(long long)));
    uint64_t*  d_chk;    CUDA_CHECK(cudaMalloc(&d_chk,    sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_ticket, 0, sizeof(long long)));
    CUDA_CHECK(cudaMemset(d_chk,   0, sizeof(uint64_t)));

    host.startAgile();
    double t0 = wall_seconds();

    host.runKernel(agile_bench_kernel,
                   ctrl, (long long)TOTAL_PAGES, d_chk, d_ticket);

    double elapsed = wall_seconds() - t0;
    host.stopAgile();
    host.closeNvme();

    cudaFree(d_ticket);
    cudaFree(d_chk);
    return elapsed;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  main                                                                       */
/* ═══════════════════════════════════════════════════════════════════════════ */

int main(int argc, char** argv)
{
    Configs cfg(argc, argv);
    cfg.show_settings();

    /* Effective VRAM budget = gpu_slot_num × slot_size bytes */
    size_t vram_budget_bytes = (size_t)cfg.gpu_slot_num * cfg.slot_size;
    size_t vram_budget_mb    = vram_budget_bytes / (1024ULL * 1024ULL);

    if ((size_t)cfg.ssd_block_num < TOTAL_PAGES)
        fprintf(stderr,
            "[WARN] ssd_block_num (%u) < TOTAL_PAGES (%zu); "
            "results may be cache-inflated due to address wrap.\n",
            cfg.ssd_block_num, TOTAL_PAGES);

    printf("=================================================================\n");
    printf("HOL Blocking Benchmark Suite — RTX A5000 (sm_86)\n");
    printf("VRAM budget  : %zu MB  (%u slots x %u B)\n",
           vram_budget_mb, cfg.gpu_slot_num, cfg.slot_size);
    printf("Dataset      : %.1f GB  (%zu pages x %zu B)\n",
           (double)DATASET_BYTES / (1024.0*1024.0*1024.0),
           TOTAL_PAGES, PAGE_SIZE_BYTES);
    printf("Prefetch depth (AGILE): %d\n", PREFETCH_DEPTH);
    printf("NVMe queue depth      : %u\n", cfg.queue_depth);
    printf("=================================================================\n\n");
    fflush(stdout);

    /* ── Open file + cuFile init (Architecture 1 only) ───────────────────── */
    const char* data_path = cfg.offset_file.empty()
                            ? "/tmp/hol_bench_96gb.bin"
                            : cfg.offset_file.c_str();

    int posix_fd = open(data_path, O_RDWR | O_CREAT, 0644);
    if (posix_fd < 0) { perror("open"); return EXIT_FAILURE; }
    if (ftruncate(posix_fd, (off_t)DATASET_BYTES) != 0)
        perror("ftruncate (non-fatal)");

    cuFileDriverOpen();
    CUfileDescr_t  cf_descr  = {};
    CUfileHandle_t cf_handle = {};
    cf_descr.handle.fd = posix_fd;
    cf_descr.type      = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    CUFILE_CHECK(cuFileHandleRegister(&cf_handle, &cf_descr));

    /* ── Architecture 1: GDS ──────────────────────────────────────────────── */
    if (cfg.mode == 1) {
    printf("[1/3] GDS — Host-Initiated I/O (HOL Blocking baseline)\n");
    fflush(stdout);

    int* d_gds_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_gds_buf, vram_budget_bytes));

    double gds_time = run_gds(cf_handle, d_gds_buf, vram_budget_bytes);
    cudaFree(d_gds_buf);

    /* Tear down cuFile — not needed by BaM or AGILE */
    cuFileHandleDeregister(cf_handle);
    cuFileDriverClose();
    close(posix_fd);

    double gds_bw = (double)DATASET_BYTES / gds_time / (1024.0*1024.0*1024.0);
    printf("    Time      : %.3f s\n", gds_time);
    printf("    Throughput: %.3f GB/s\n", gds_bw);
    printf("GDS_TIME=%.6f\n", gds_time);   /* token for run_ablation.py */
    fflush(stdout);

    /* ── Summary ──────────────────────────────────────────────────────────── */
    printf("\n=================================================================\n");
    printf("Summary  (VRAM budget = %zu MB)\n", vram_budget_mb);
    printf("-----------------------------------------------------------------\n");

    printf("  %-24s  %8.3f s   %8.3f GB/s\n",
           "GDS (HOL baseline)", gds_time, gds_bw);

    }
    else if (cfg.mode == 2) {

    /* ── Architecture 2: BaM ──────────────────────────────────────────────── */
    printf("\n[2/3] BaM — GPU-Initiated I/O via AGILE (demand-only, no prefetch)\n");
    fflush(stdout);

    double bam_time = run_bam(cfg);

    double bam_bw = (double)DATASET_BYTES / bam_time / (1024.0*1024.0*1024.0);
    printf("    Time      : %.3f s\n", bam_time);
    printf("    Throughput: %.3f GB/s\n", bam_bw);
    printf("BAM_TIME=%.6f\n", bam_time);
    fflush(stdout);

    /* ── Summary ──────────────────────────────────────────────────────────── */
    printf("\n=================================================================\n");
    printf("Summary  (VRAM budget = %zu MB)\n", vram_budget_mb);
    printf("-----------------------------------------------------------------\n");

    printf("  %-24s  %8.3f s   %8.3f GB/s  \n",
           "BaM (demand-only)", bam_time, bam_bw);

    }
    else if (cfg.mode == 3) {
    /* ── Architecture 3: AGILE ────────────────────────────────────────────── */
    printf("\n[3/3] AGILE — GPU-Initiated I/O via AGILE (prefetch depth=%d)\n",
           PREFETCH_DEPTH);
    fflush(stdout);

    double agile_time = run_agile(cfg);

    double agile_bw = (double)DATASET_BYTES / agile_time / (1024.0*1024.0*1024.0);
    printf("    Time      : %.3f s\n", agile_time);
    printf("    Throughput: %.3f GB/s\n", agile_bw);
    printf("AGILE_TIME=%.6f\n", agile_time);
    fflush(stdout);

    /* ── Summary ──────────────────────────────────────────────────────────── */
    printf("\n=================================================================\n");
    printf("Summary  (VRAM budget = %zu MB)\n", vram_budget_mb);
    printf("-----------------------------------------------------------------\n");

    printf("  %-24s  %8.3f s   %8.3f GB/s  \n",
           "AGILE (prefetch x8)", agile_time, agile_bw);
    }

    printf("=================================================================\n");

    return EXIT_SUCCESS;
}
