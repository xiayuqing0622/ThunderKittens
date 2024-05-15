#define TORCH_COMPILE // defined by default for PyTorch bindings - to use cpp harness, comment this out

#ifdef TORCH_COMPILE
#include "src/kittens.cuh"
#else
#include "../../src/kittens.cuh"
#endif

#include <cooperative_groups.h>

#define NUM_WORKERS (8)
#define NUM_WARPGROUPS (NUM_WORKERS/(kittens::WARPGROUP_WARPS))
#define NUM_WORKERS_KV (1)

using namespace kittens;

template<ducks::rt::row_layout RT>
__device__ static inline void wg_make_causal(RT &dst, const RT &src, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    const typename RT::dtype packed_val = base_types::packing<typename RT::dtype>::pack(val);
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {

            if(j < ((warpid() % kittens::WARPGROUP_WARPS) * dst.height) + i) { // below the diagonal, copy
                #pragma unroll
                for(int k = 0; k < dst.packed_per_tile; k++) {
                    dst.tiles[i][j].data[k] = src.tiles[i][j].data[k];
                }
            }
            else if(j > ((warpid() % kittens::WARPGROUP_WARPS) * dst.height) + i) { // above the diagonal, zero
                #pragma unroll
                for(int k = 0; k < dst.packed_per_tile; k++) {
                    dst.tiles[i][j].data[k] = packed_val;
                }
            }
            else { // on the diagonal, interesting!
                constexpr uint32_t MASK_X = 0xFF773311, MASK_Y = 0xF7733110; // magic numbers for on-diagonal core matrices
                dst.tiles[i][j].data[1] = src.tiles[i][j].data[1]; // below diagonal, copy
                dst.tiles[i][j].data[2] = packed_val; // above diagonal, zero
                if((MASK_X >> laneid()) & 1) {
                    dst.tiles[i][j].data[0].x = src.tiles[i][j].data[0].x;
                    dst.tiles[i][j].data[3].x = src.tiles[i][j].data[3].x;
                }
                else {
                    dst.tiles[i][j].data[0].x = val;
                    dst.tiles[i][j].data[3].x = val;
                }
                if((MASK_Y >> laneid()) & 1) {
                    dst.tiles[i][j].data[0].y = src.tiles[i][j].data[0].y;
                    dst.tiles[i][j].data[3].y = src.tiles[i][j].data[3].y;
                }
                else {
                    dst.tiles[i][j].data[0].y = val;
                    dst.tiles[i][j].data[3].y = val;
                }
            }
        }
    }
}

using layout_q = kittens::ducks::st_layout::wgmma_swizzle; 
using layout_k = kittens::ducks::st_layout::wgmma_swizzle; 
using layout_v = kittens::ducks::st_layout::wgmma_interleave; 
using layout_o = kittens::ducks::st_layout::swizzle;

__global__  __launch_bounds__((NUM_WORKERS)*kittens::WARP_THREADS, 2)
void attend_ker_fwd_train(const int N, CUtensorMap* tma_q, CUtensorMap* tma_k, CUtensorMap* tma_v, CUtensorMap* tma_o, CUtensorMap* tma_l) {
    extern __shared__ int __shm[]; // this is the CUDA shared memory
    tma_swizzle_allocator al((int*)&__shm[0]);

    st_bf<4, 4, layout_q>          (&q_smem)   [NUM_WARPGROUPS] = al.allocate<st_bf<4, 4, layout_q>,          NUM_WARPGROUPS>();
    st_bf<4, 4, layout_k>          (&k_smem)[2][NUM_WORKERS_KV] = al.allocate<st_bf<4, 4, layout_k>, 2,       NUM_WORKERS_KV>();
    st_bf<4, 4, layout_v>          (&v_smem)[2][NUM_WORKERS_KV] = al.allocate<st_bf<4, 4, layout_v>, 2,       NUM_WORKERS_KV>();
    st_bf<4, 4, layout_o>::col_vec (&l_smem)   [NUM_WARPGROUPS] = al.allocate<st_bf<4, 4, layout_o>::col_vec, NUM_WARPGROUPS>();

    int tic = 0, toc = 1;
 
    rt_fl<1, 4> att_block;
    rt_bf<1, 4> att_block_mma;
    rt_fl<1, 4> o_prev;
    rt_fl<1, 4>::col_vec max_vec_last, max_vec;
    rt_fl<1, 4>::col_vec norm_vec_last, norm_vec;

    int warpid      = kittens::warpid();
    int warpgroupid = warpid/kittens::WARPGROUP_WARPS;

    int qo_index    = (blockIdx.x * NUM_WARPGROUPS) + warpgroupid;

    int kv_blocks = N / (NUM_WORKERS_KV*k_smem[0][0].rows);

    __shared__ uint64_t qsmem_barrier, kvsmem_barrier;//, vsmem_barrier;

    int q_phasebit = 0;
    int kv_phasebit = 0;

    if (threadIdx.x == 0) {
        tma::init_barrier<st_bf<4, 4, layout_q>, NUM_WARPGROUPS>(qsmem_barrier, 1);
        tma::init_barrier<st_bf<4, 4, layout_k>, NUM_WORKERS_KV*2>(kvsmem_barrier, 1); 
    }

    if (warpid == 0) {
        for (int wg = 0; wg < NUM_WORKERS/kittens::WARPGROUP_WARPS; wg++) { // load q
            int tile_idx = (blockIdx.y * NUM_WARPGROUPS * gridDim.x) + (blockIdx.x * NUM_WARPGROUPS) + wg;
            tma::load_async((q_smem[wg]), tma_q, qsmem_barrier, tile_idx); 
        }
        for (int w = 0; w < NUM_WORKERS_KV; w++) { // load k, v      
            int tile_idx = (blockIdx.y * NUM_WORKERS_KV * kv_blocks) + (0 * NUM_WORKERS_KV) + w; 
            tma::load_async((k_smem[tic][w]), tma_k, kvsmem_barrier, tile_idx); 
            tma::load_async((v_smem[tic][w]), tma_v, kvsmem_barrier, tile_idx); 
        }
    }

    neg_infty(max_vec); // zero registers for the Q chunk
    zero(norm_vec);
    zero(o_prev);
    __syncthreads();

    tma::arrive_and_wait(qsmem_barrier, q_phasebit);
    q_phasebit ^= 1;

    warpgroup::mul(q_smem[warpgroupid], q_smem[warpgroupid], __float2bfloat16(0.125f));

    for (auto kv_idx = 0; kv_idx <= qo_index; kv_idx++, tic ^= 1, toc ^= 1) {
        tma::arrive_and_wait(kvsmem_barrier, kv_phasebit);
        kv_phasebit ^= 1;

        __syncthreads();
        if (warpid == 0) {
            if (kv_idx + 1 < kv_blocks) {
                tma::set_bytes(kvsmem_barrier, 2 * NUM_WORKERS_KV * k_smem[0][0].num_elements * sizeof(bf16));

                for (int w = 0; w < NUM_WORKERS_KV; w++) {        
                    int tile_idx = (blockIdx.y * NUM_WORKERS_KV * kv_blocks) + ((kv_idx + 1) * NUM_WORKERS_KV) + w; 
                    tma::load_async((k_smem[toc][w]), tma_k, kvsmem_barrier, tile_idx); 
                    tma::load_async((v_smem[toc][w]), tma_v, kvsmem_barrier, tile_idx);
                }
            }
        }

        warpgroup::mma_fence(att_block);
        warpgroup::mm_ABt(att_block, q_smem[warpgroupid], k_smem[tic][0]);
        warpgroup::mma_commit_group();

        copy(norm_vec_last, norm_vec);
        copy(max_vec_last,  max_vec);

        warpgroup::mma_async_wait();

        if (kv_idx == qo_index) {
            wg_make_causal(att_block, att_block, -INFINITY); 
        }

        row_max(max_vec, att_block, max_vec); // accumulate onto the max_vec
        sub_row(att_block, att_block, max_vec);
        exp(att_block, att_block);

        sub(max_vec_last, max_vec_last, max_vec);
        exp(max_vec_last, max_vec_last);
        mul(norm_vec, norm_vec, max_vec_last);

        row_sum(norm_vec, att_block, norm_vec); // accumulate onto the norm_vec
        div_row(att_block, att_block, norm_vec);

        mul(norm_vec_last, norm_vec_last, max_vec_last);
        div(norm_vec_last, norm_vec_last, norm_vec);

        copy(att_block_mma, att_block); // convert to bf16 for mma
        mul_row(o_prev, o_prev, norm_vec_last); // normalize o_prev in advance of mma'ing onto it

        warpgroup::mma_fence(o_prev);
        warpgroup::mma_AB(o_prev, att_block_mma, v_smem[tic][0]);
        warpgroup::mma_commit_group();
    }

    auto (*o_smem) = reinterpret_cast<st_bf<4, 4, layout_o>(*)>(q_smem); // reuse q memory
    warpgroup::store(o_smem[warpgroupid], o_prev); 
    __syncthreads();
    
    if (warpid % 4 == 0) { // store o
        int tile_idx = (blockIdx.y * NUM_WARPGROUPS * gridDim.x) + (blockIdx.x * NUM_WARPGROUPS) + warpgroupid;
        tma::store_async(tma_o, (o_smem[warpgroupid]), tile_idx); 
        tma::store_commit_group(); 
    }

    log(norm_vec, norm_vec);
    add(norm_vec, norm_vec, max_vec);
    __syncthreads();

    warpgroup::store(l_smem[warpgroupid], norm_vec);
    __syncthreads();
    if (warpid % 4 == 0) {
        int tile_idx = (blockIdx.y * NUM_WARPGROUPS * gridDim.x) + (blockIdx.x * NUM_WARPGROUPS) + warpgroupid; 
        tma::store_async(tma_l, (l_smem[warpgroupid]), tile_idx); 
        tma::store_commit_group(); 
    }

    tma::store_async_wait();
}

#define WORKERS 4

using layout_nrow = ducks::st_layout::swizzle;

__global__  __launch_bounds__(WORKERS*kittens::WARP_THREADS, 2)
void attend_ker_prep_train(CUtensorMap* tma_o, CUtensorMap* tma_d, CUtensorMap* tma_o_grad) {
    extern __shared__ int __shm[]; // this is the CUDA shared memory
    tma_swizzle_allocator al((int*)&__shm[0]);

    int warpid = kittens::warpid();

    st_bf<4, 4, layout_nrow>          (&og_smem)[WORKERS] = al.allocate<st_bf<4, 4, layout_nrow>, WORKERS>();
    st_bf<4, 4, layout_nrow>          (&o_smem) [WORKERS] = al.allocate<st_bf<4, 4, layout_nrow>, WORKERS>();
    st_bf<4, 4, layout_nrow>::col_vec (&d_smem) [WORKERS] = al.allocate<st_bf<4, 4, layout_nrow>::col_vec, WORKERS>();

    rt_fl<4, 4> og_reg;
    rt_fl<4, 4> o_reg; 
    rt_fl<4, 4>::col_vec d_reg;

    __shared__ uint64_t smem_barrier;
    int o_phasebit = 0; 

    if (threadIdx.x == 0) {
        tma::init_barrier<st_bf<4, 4, layout_o>, WORKERS * 2>(smem_barrier, 1);
    }

    if (warpid == 0) {
        for (int w = 0; w < WORKERS; w++) { // load o, o_grad
            int tile_idx = (blockIdx.y * WORKERS * gridDim.x) + (blockIdx.x * WORKERS) + w; 
            tma::load_async((o_smem[w]),  tma_o,      smem_barrier, tile_idx); 
            tma::load_async((og_smem[w]), tma_o_grad, smem_barrier, tile_idx); 
        }
    }
    __syncthreads();

    tma::arrive_and_wait(smem_barrier, o_phasebit);
    o_phasebit ^= 1;

    load(o_reg, o_smem[warpid]);
    load(og_reg, og_smem[warpid]);

    mul(og_reg, og_reg, o_reg);
    row_sum(d_reg, og_reg);
    
    store(d_smem[warpid], d_reg);
    __syncthreads(); 

    if (warpid == 0) {
        for (int w = 0; w < WORKERS; w++) {
            int tile_idx = (blockIdx.y * WORKERS * gridDim.x) + (blockIdx.x * WORKERS) + w; 
            tma::store_async(tma_d, (d_smem[w]), tile_idx); 
        }
        tma::store_commit_group();
    }

    tma::store_async_wait();
}

constexpr int WORKERS_BWD    = 4;  
constexpr int WORKERS_BWD_QO = 4; 

constexpr int NUM_WARPGROUPS_BWD    = (WORKERS_BWD/(kittens::WARPGROUP_WARPS));
constexpr int NUM_WARPGROUPS_BWD_QO = (WORKERS_BWD_QO/(kittens::WARPGROUP_WARPS));

constexpr int tile_h = 4;   
constexpr int tile_h_qo = 4;

constexpr int tile_w = 64/16;

using layout_wgmma     = ducks::st_layout::wgmma_swizzle;
using layout_wgmma_itl = ducks::st_layout::wgmma_interleave;
using layout_tma_swi   = ducks::st_layout::swizzle; 

#define k_smem_tile  st_bf<tile_h, tile_w, layout_wgmma_itl>
#define v_smem_tile  st_bf<tile_h, tile_w, layout_wgmma>

#define q_smem_tile  st_bf<tile_h_qo, tile_w, layout_wgmma_itl>
#define og_smem_tile st_bf<tile_h_qo, tile_w, layout_wgmma_itl>
#define qg_smem_tile st_bf<tile_h_qo, tile_w, layout_tma_swi>
#define l_smem_tile  st_bf<tile_h_qo, tile_w, layout_tma_swi>::col_vec
#define d_smem_tile  st_bf<tile_h_qo, tile_w, layout_tma_swi>::col_vec

using namespace cooperative_groups;
namespace cg = cooperative_groups;

__global__ __launch_bounds__(WORKERS_BWD*kittens::WARP_THREADS, 2)
void attend_ker_bwd_train(const int N, CUtensorMap* tma_q, CUtensorMap* tma_k, CUtensorMap* tma_v, 
                            CUtensorMap* tma_l_vec, CUtensorMap* tma_d_vec, 
                            CUtensorMap* tma_og, CUtensorMap* tma_qg, CUtensorMap* tma_kg, CUtensorMap* tma_vg)
{
    extern __shared__ int __shm[]; // this is the CUDA shared memory
    tma_swizzle_allocator al((int*)&__shm[0]);

    k_smem_tile  (&k_smem) [NUM_WARPGROUPS_BWD] = al.allocate<k_smem_tile, NUM_WARPGROUPS_BWD>();
    v_smem_tile  (&v_smem) [NUM_WARPGROUPS_BWD] = al.allocate<v_smem_tile, NUM_WARPGROUPS_BWD>();

    q_smem_tile  (&q_smem) [2][NUM_WARPGROUPS_BWD_QO]                     = al.allocate<q_smem_tile,  2, NUM_WARPGROUPS_BWD_QO>();
    og_smem_tile (&og_smem)[2][NUM_WARPGROUPS_BWD_QO]                     = al.allocate<og_smem_tile, 2, NUM_WARPGROUPS_BWD_QO>();
    qg_smem_tile (&qg_smem)[2][NUM_WARPGROUPS_BWD_QO][NUM_WARPGROUPS_BWD] = al.allocate<qg_smem_tile, 2, NUM_WARPGROUPS_BWD_QO, NUM_WARPGROUPS_BWD>();
    
    l_smem_tile  (&l_smem) [2][NUM_WARPGROUPS_BWD_QO]                     = al.allocate<l_smem_tile,  2, NUM_WARPGROUPS_BWD_QO>();
    d_smem_tile  (&d_smem) [2][NUM_WARPGROUPS_BWD_QO]                     = al.allocate<d_smem_tile,  2, NUM_WARPGROUPS_BWD_QO>();

    rt_fl<tile_h/kittens::WARPGROUP_WARPS, tile_w> kg_reg;
    rt_fl<tile_h/kittens::WARPGROUP_WARPS, tile_w> vg_reg;

    rt_fl<tile_h_qo/kittens::WARPGROUP_WARPS, tile_w> qg_reg;

    rt_bf<tile_h_qo/kittens::WARPGROUP_WARPS, tile_h>::col_vec l_reg_bf; 
    rt_bf<tile_h_qo/kittens::WARPGROUP_WARPS, tile_h>::col_vec d_reg_bf;
    rt_fl<tile_h_qo/kittens::WARPGROUP_WARPS, tile_h>::col_vec l_reg_fl; 
    rt_fl<tile_h_qo/kittens::WARPGROUP_WARPS, tile_h>::col_vec d_reg_fl;

    rt_fl<tile_h_qo/kittens::WARPGROUP_WARPS, tile_h> att_block; 
    rt_bf<tile_h_qo/kittens::WARPGROUP_WARPS, tile_h> att_block_mma;
    rt_fl<tile_h_qo/kittens::WARPGROUP_WARPS, tile_h> temp_block;

    int warpid = kittens::warpid();
    int warpgroupid = warpid/kittens::WARPGROUP_WARPS;

    const int qo_blocks = N / (tile_h_qo * kittens::TILE_DIM * NUM_WARPGROUPS_BWD_QO);

    __shared__ uint64_t kv_b, qo_b, vec_b;

    int tic = 0, toc = 1;

    int kv_phasebit  = 0;
    int qo_phasebit  = 0;
    int vec_phasebit = 0;

    if (threadIdx.x == 0) {
        tma::init_barrier<q_smem_tile,  NUM_WARPGROUPS_BWD_QO * 2>(qo_b,  1); // q, og
        tma::init_barrier<k_smem_tile,  NUM_WARPGROUPS_BWD    * 2>(kv_b,  1); // k, v
        tma::init_barrier<l_smem_tile,  NUM_WARPGROUPS_BWD_QO * 2>(vec_b, 1); // l, d
    } 

    if (warpid == 0) {
        for (int w = 0; w < NUM_WARPGROUPS_BWD_QO; w++) {
            int tile_idx = (blockIdx.y * NUM_WARPGROUPS_BWD_QO * qo_blocks) + (blockIdx.x * NUM_WARPGROUPS_BWD_QO) + w;

            tma::load_async((q_smem [tic][w]),    tma_q,     qo_b, tile_idx); 
            tma::load_async((og_smem[tic][w]),    tma_og,    qo_b, tile_idx);

            tma::load_async((l_smem[tic][w]),     tma_l_vec, vec_b, tile_idx);
            tma::load_async((d_smem[tic][w]),     tma_d_vec, vec_b, tile_idx);
        } 
    }
    
    if (warpid == 0) {
        // load k and v
        for (int w = 0; w < NUM_WARPGROUPS_BWD; w++) {
            int tile_idx = (blockIdx.y * NUM_WARPGROUPS_BWD * gridDim.x) + (blockIdx.x * NUM_WARPGROUPS_BWD) + (0 * NUM_WARPGROUPS_BWD) + w;
            tma::load_async((k_smem[w]), tma_k, kv_b, tile_idx); 
            tma::load_async((v_smem[w]), tma_v, kv_b, tile_idx); 
        }
    }

    zero(kg_reg);
    zero(vg_reg);

    for (int qo_idx = blockIdx.x; qo_idx < qo_blocks; qo_idx++, tic ^= 1, toc ^= 1) {

        if (qo_idx == blockIdx.x) {
            tma::arrive_and_wait(kv_b, kv_phasebit);
            kv_phasebit ^= 1;
        }

        tma::arrive_and_wait(vec_b, vec_phasebit);
        tma::arrive_and_wait(qo_b,  qo_phasebit);
        vec_phasebit ^= 1;
        qo_phasebit  ^= 1;

        if (qo_idx + 1 < qo_blocks) {
            if (warpid == 0) {
                tma::set_bytes(qo_b,  NUM_WARPGROUPS_BWD_QO * sizeof(bf16) * q_smem[0][0].num_elements * 2); 
                tma::set_bytes(vec_b, NUM_WARPGROUPS_BWD_QO * sizeof(bf16) * l_smem[0][0].length       * 2);

                for (int w = 0; w < NUM_WARPGROUPS_BWD_QO; w++) {
                    int tile_idx = (blockIdx.y * NUM_WARPGROUPS_BWD_QO * qo_blocks) + ((qo_idx + 1) * NUM_WARPGROUPS_BWD_QO) + w;
                    tma::load_async((q_smem [toc][w]),    tma_q,   qo_b, tile_idx); 
                    tma::load_async((og_smem[toc][w]),    tma_og,  qo_b, tile_idx);

                    tma::load_async((l_smem[toc][w]),     tma_l_vec, vec_b, tile_idx);
                    tma::load_async((d_smem[toc][w]),     tma_d_vec, vec_b, tile_idx);
                }
            }
        }

        if (qo_idx > 0) {
            tma::store_async_wait(); 
        }

        warpgroup::mma_fence(att_block);
        warpgroup::mm_ABt(att_block, q_smem[tic][0], k_smem[warpgroupid]);
        warpgroup::mma_commit_group();

        warpgroup::load(l_reg_bf, l_smem[tic][0]);
        copy(l_reg_fl, l_reg_bf);
        
        warpgroup::mma_async_wait();
        mul(att_block, att_block, 0.125f);

        if (blockIdx.x == qo_idx) {
            wg_make_causal(att_block, att_block, -INFINITY); 
        }

        sub_row(att_block, att_block, l_reg_fl);
        exp(att_block, att_block);
        copy(temp_block, att_block);
        copy(att_block_mma, att_block);

        auto (*att_smem)[NUM_WARPGROUPS_BWD_QO][NUM_WARPGROUPS_BWD] = reinterpret_cast<st_bf<tile_h_qo, tile_w, layout_wgmma_itl> (*)[NUM_WARPGROUPS_BWD_QO][NUM_WARPGROUPS_BWD]>(qg_smem); 

        warpgroup::store(att_smem[tic][0][warpgroupid], att_block_mma);
    
        warpgroup::mma_fence(att_block);
        warpgroup::mm_ABt(att_block, og_smem[tic][0], v_smem[warpgroupid]);
        warpgroup::mma_commit_group();

        warpgroup::load(d_reg_bf, d_smem[tic][0]);
        copy(d_reg_fl, d_reg_bf);

        warpgroup::mma_fence(vg_reg);
        warpgroup::mma_AtB(vg_reg, att_smem[tic][0][warpgroupid], og_smem[tic][0]);
        warpgroup::mma_commit_group();

        warpgroup::mma_async_wait<1>();
        sub_row(att_block, att_block, d_reg_fl);
        mul(temp_block, temp_block, att_block);
        mul(temp_block, temp_block, 0.125f);
        copy(att_block_mma, temp_block);

        warpgroup::mma_async_wait(); 
        warpgroup::store(att_smem[tic][0][warpgroupid], att_block_mma);

        zero(qg_reg);
        warpgroup::mma_fence(qg_reg);
        warpgroup::mma_AB(qg_reg, att_block_mma, k_smem[warpgroupid]);
        warpgroup::mma_commit_group(); 

        warpgroup::mma_fence(kg_reg);
        warpgroup::mma_AtB(kg_reg, att_smem[tic][0][warpgroupid], q_smem[tic][0]);
        warpgroup::mma_commit_group();
            
        warpgroup::mma_async_wait();
        warpgroup::store(qg_smem[tic][0][warpgroupid], qg_reg);

        if (warpid % 4 == 0) {
            int tile_idx = (blockIdx.y * NUM_WARPGROUPS_BWD_QO * qo_blocks) + (qo_idx * NUM_WARPGROUPS_BWD_QO) + warpgroupid; 
            for (int idx = 0; idx < NUM_WARPGROUPS_BWD; idx++) {
                tma::store_sum_async(tma_qg, (qg_smem[tic][warpgroupid][idx]), tile_idx); 
            }
            tma::store_commit_group();
        }
    }

    warpgroup::store(k_smem[warpgroupid], kg_reg);
    warpgroup::store(v_smem[warpgroupid], vg_reg);
    __syncthreads();

    if (warpid % 4 == 0) {
        int tile_idx = (blockIdx.y * NUM_WARPGROUPS_BWD * gridDim.x) + (blockIdx.x * NUM_WARPGROUPS_BWD) + (0 * NUM_WARPGROUPS_BWD) + warpgroupid; 
        tma::store_async(tma_kg, (k_smem[warpgroupid]), tile_idx);
        tma::store_async(tma_vg, (v_smem[warpgroupid]), tile_idx);
        tma::store_commit_group();
    }
    
    tma::store_async_wait();
}

#ifdef TORCH_COMPILE    
#include "src/common/pyutils/torch_helpers.cuh"
#include <iostream>

void attention_train_forward_causal(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o, torch::Tensor l) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(o);
    CHECK_INPUT(l);

    auto batch = q.size(0);
    auto heads = q.size(1);
    auto N     = q.size(2);
    auto D     = q.size(3);

    TORCH_CHECK(D == 64, "Train functions do not support DIM=128 yet"); 

    auto threads = NUM_WORKERS * kittens::WARP_THREADS;

    // sequence length must be divisible by 256 for bwd_prep kernel!!!
    TORCH_CHECK(N % (NUM_WORKERS * kittens::TILE_DIM) == 0, "For training, please pad sequence length to be divisible by 256");

    // convert to bf16
    c10::BFloat16 *q_ptr = q.data_ptr<c10::BFloat16>();
    c10::BFloat16 *k_ptr = k.data_ptr<c10::BFloat16>();
    c10::BFloat16 *v_ptr = v.data_ptr<c10::BFloat16>();
    c10::BFloat16 *o_ptr = o.data_ptr<c10::BFloat16>();
    c10::BFloat16 *l_ptr = l.data_ptr<c10::BFloat16>();

    bf16* q_bf = reinterpret_cast<bf16*>(q_ptr);
    bf16* k_bf = reinterpret_cast<bf16*>(k_ptr);
    bf16* v_bf = reinterpret_cast<bf16*>(v_ptr);
    bf16* o_bf = reinterpret_cast<bf16*>(o_ptr);
    bf16* l_bf = reinterpret_cast<bf16*>(l_ptr);

    CUtensorMap* tma_q_d = tma::allocate_and_create_tensor_map<kittens::st_bf<4, 4, layout_q>         >(q_bf, (batch*heads*N)/(4 * 16));
    CUtensorMap* tma_k_d = tma::allocate_and_create_tensor_map<kittens::st_bf<4, 4, layout_k>         >(k_bf, (batch*heads*N)/(4 * 16));
    CUtensorMap* tma_v_d = tma::allocate_and_create_tensor_map<kittens::st_bf<4, 4, layout_v>         >(v_bf, (batch*heads*N)/(4 * 16));
    CUtensorMap* tma_o_d = tma::allocate_and_create_tensor_map<kittens::st_bf<4, 4, layout_o>         >(o_bf, (batch*heads*N)/(4 * 16));
    CUtensorMap* tma_l_d = tma::allocate_and_create_tensor_map<kittens::st_bf<4, 4, layout_q>::col_vec>(l_bf, (batch*heads*N)/(4 * 16));

    cudaFuncSetAttribute(attend_ker_fwd_train, cudaFuncAttributeMaxDynamicSharedMemorySize, 112000);

    dim3 grid(N/(NUM_WORKERS*kittens::TILE_DIM), batch*heads, 1);
    attend_ker_fwd_train<<<grid, threads, 112000>>>(N, tma_q_d, tma_k_d, tma_v_d, tma_o_d, tma_l_d);

    CHECK_CUDA_ERROR(cudaGetLastError());
}

void attention_train_backward_causal(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o, torch::Tensor l_vec, torch::Tensor d_vec, torch::Tensor og, torch::Tensor qg, torch::Tensor kg, torch::Tensor vg) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(o);
    CHECK_INPUT(l_vec);
    CHECK_INPUT(d_vec);
    CHECK_INPUT(og);
    CHECK_INPUT(qg);
    CHECK_INPUT(kg);
    CHECK_INPUT(vg);

    auto batch = q.size(0);
    auto heads = q.size(1);
    auto N     = q.size(2);
    auto D     = q.size(3);

    TORCH_CHECK(D == 64, "Train functions do not support DIM=128 yet");

    // sequence length must be divisible by 256 for bwd_prep kernel!!!
    TORCH_CHECK(N % (WORKERS * kittens::TILE_DIM * 4) == 0, "For training, please pad sequence length to be divisible by 256");
    TORCH_CHECK(N % (WORKERS_BWD * kittens::TILE_DIM) == 0, "For training, please pad sequence length to be divisible by 256");

    // convert to bf16
    c10::BFloat16 *q_ptr  = q.data_ptr<c10::BFloat16>();
    c10::BFloat16 *k_ptr  = k.data_ptr<c10::BFloat16>();
    c10::BFloat16 *v_ptr  = v.data_ptr<c10::BFloat16>();
    c10::BFloat16 *o_ptr  = o.data_ptr<c10::BFloat16>();
    c10::BFloat16 *l_ptr  = l_vec.data_ptr<c10::BFloat16>();
    c10::BFloat16 *og_ptr = og.data_ptr<c10::BFloat16>();
    c10::BFloat16 *d_ptr  = d_vec.data_ptr<c10::BFloat16>();
    c10::BFloat16 *qg_ptr = qg.data_ptr<c10::BFloat16>();
    c10::BFloat16 *kg_ptr = kg.data_ptr<c10::BFloat16>();
    c10::BFloat16 *vg_ptr = vg.data_ptr<c10::BFloat16>();

    const bf16* q_bf  = reinterpret_cast<const bf16*>(q_ptr);
    const bf16* k_bf  = reinterpret_cast<const bf16*>(k_ptr);
    const bf16* v_bf  = reinterpret_cast<const bf16*>(v_ptr);
    const bf16* o_bf  = reinterpret_cast<const bf16*>(o_ptr);
    const bf16* l_bf  = reinterpret_cast<const bf16*>(l_ptr);
    const bf16* og_bf = reinterpret_cast<const bf16*>(og_ptr);
    bf16* d_bf        = reinterpret_cast<bf16*>(d_ptr);
    bf16* qg_bf       = reinterpret_cast<bf16*>(qg_ptr);
    bf16* kg_bf       = reinterpret_cast<bf16*>(kg_ptr);
    bf16* vg_bf       = reinterpret_cast<bf16*>(vg_ptr);

    CUtensorMap* tma_o_d_pre  = tma::allocate_and_create_tensor_map<kittens::st_bf<4, 4, layout_nrow>           >(o_bf, (batch*heads*N)/(4*16)); 
    CUtensorMap* tma_d_d_pre  = tma::allocate_and_create_tensor_map<kittens::st_bf<4, 4, layout_nrow>::col_vec  >(d_bf, (batch*heads*N)/(4*16));
    CUtensorMap* tma_og_d_pre = tma::allocate_and_create_tensor_map<kittens::st_bf<4, 4, layout_nrow>           >(og_bf,(batch*heads*N)/(4*16));

    unsigned long mem_size = 227000;
    cudaFuncSetAttribute(attend_ker_prep_train, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);

    dim3 grid_1(N/(WORKERS*kittens::TILE_DIM*4), batch*heads, 1);
    auto threads_1 = WORKERS * kittens::WARP_THREADS;
    attend_ker_prep_train<<<grid_1, threads_1, mem_size>>>(tma_o_d_pre, tma_d_d_pre, tma_og_d_pre);

    CUtensorMap* tma_b_q_d  = tma::allocate_and_create_tensor_map<q_smem_tile> (q_bf,  (batch*heads*N)/(tile_h_qo * 16)); 
    CUtensorMap* tma_b_k_d  = tma::allocate_and_create_tensor_map<k_smem_tile> (k_bf,  (batch*heads*N)/(tile_h    * 16));
    CUtensorMap* tma_b_v_d  = tma::allocate_and_create_tensor_map<v_smem_tile> (v_bf,  (batch*heads*N)/(tile_h    * 16));
    CUtensorMap* tma_b_l_d  = tma::allocate_and_create_tensor_map<l_smem_tile> (l_bf,  (batch*heads*N)/(tile_h_qo * 16));
    CUtensorMap* tma_n_d_d  = tma::allocate_and_create_tensor_map<d_smem_tile> (d_bf,  (batch*heads*N)/(tile_h_qo * 16));
    CUtensorMap* tma_n_og_d = tma::allocate_and_create_tensor_map<og_smem_tile>(og_bf, (batch*heads*N)/(tile_h_qo * 16));
    CUtensorMap* tma_b_qg_d = tma::allocate_and_create_tensor_map<qg_smem_tile>(qg_bf, (batch*heads*N)/(tile_h_qo * 16));
    CUtensorMap* tma_b_kg_d = tma::allocate_and_create_tensor_map<k_smem_tile> (kg_bf, (batch*heads*N)/(tile_h    * 16));
    CUtensorMap* tma_b_vg_d = tma::allocate_and_create_tensor_map<v_smem_tile> (vg_bf, (batch*heads*N)/(tile_h    * 16));

    cudaFuncSetAttribute(
        attend_ker_bwd_train,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        112000
    );

    dim3 grid_2(N/(WORKERS_BWD*kittens::TILE_DIM), batch*heads, 1);
    attend_ker_bwd_train<<<grid_2, (kittens::WARP_THREADS*WORKERS_BWD), 112000>>>(N, tma_b_q_d, tma_b_k_d, tma_b_v_d, tma_b_l_d, tma_n_d_d, tma_n_og_d, tma_b_qg_d, tma_b_kg_d, tma_b_vg_d); 

    CHECK_CUDA_ERROR(cudaGetLastError());
}
#else 
#include "harness_h100_train.impl"
#endif