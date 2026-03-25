import torch
import torch.utils.cpp_extension
import time

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// 【微观优化 1】：彻底剔除耗时的 `%` 取模指令，换用极速的加减法比较！
__device__ inline void get_pair_padded(int k, int phase, int N_pad, int &p, int &q) {
    if (k == 0) {
        p = 0; q = phase + 1;
    } else {
        int M = N_pad - 1;
        
        // 原逻辑: p = (phase + k) % M + 1;
        int val_p = phase + k;
        p = (val_p >= M ? val_p - M : val_p) + 1;
        
        // 原逻辑: q = (phase - k + M) % M + 1;
        int val_q = phase - k + M;
        q = (val_q >= M ? val_q - M : val_q) + 1;
    }
    if (p > q) { int temp = p; p = q; q = temp; }
}

// =========================================================================
// 终极真理版: 2x2 独立战区 + 1-Pass + 动态 Occupancy 控制
// =========================================================================
__global__ void jacobi_h800_tile_fused_kernel(double* A_global, double* Vt_global, 
                                              int N_actual, int N_pad, 
                                              int pre_sweeps, int standard_sweeps, double initial_thresh) {
    int b = blockIdx.x;        
    int tid = threadIdx.x;     
    int bdim = blockDim.x;     

    double* A_b = A_global + b * N_actual * N_actual;
    double* Vt_b = Vt_global + b * N_actual * N_actual;

    int pitch = N_pad / 2;
    int num_phases = N_pad - 1;

    extern __shared__ double smem[];
    double* A_s = smem;                                  
    double* C_s = smem + N_pad * N_pad;            
    double* S_s = smem + N_pad * N_pad + pitch;    

    for (int idx = tid; idx < N_pad * N_pad; idx += bdim) {
        int r = idx / N_pad;
        int c = idx % N_pad;
        if (r < N_actual && c < N_actual) {
            A_s[r * N_pad + c] = A_b[r * N_actual + c];
        } else {
            A_s[r * N_pad + c] = (r == c) ? 1.0 : 0.0;
        }
    }
    __syncthreads();

    int total_sweeps = pre_sweeps + standard_sweeps;

    for (int s = 0; s < total_sweeps; ++s) {

        double current_thresh = 0.0;
        if (s < pre_sweeps) {
            current_thresh = initial_thresh / pow(10.0, s); 
        }

        for (int phase = 0; phase < num_phases; ++phase) {
            
            // --- 步骤 1: 计算角度 ---
            for (int k = tid; k < pitch; k += bdim) {
                int p, q;
                get_pair_padded(k, phase, N_pad, p, q);
                double app = A_s[p * N_pad + p];
                double aqq = A_s[q * N_pad + q];
                double apq = A_s[p * N_pad + q];
                
                double c = 1.0, s = 0.0;
                if (fabs(apq) >= current_thresh && fabs(apq) > 1e-15) { 
                    double tau = (aqq - app) / (2.0 * apq);
                    double t = (tau == 0.0) ? 1.0 : copysign(1.0, tau) / (fabs(tau) + sqrt(1.0 + tau * tau));
                    c = 1.0 / sqrt(1.0 + t * t);
                    s = c * t;
                }
                C_s[k] = c; S_s[k] = s;
            }
            __syncthreads(); 

            // --- 步骤 2: 2x2 独立战区 ---
            int total_blocks = pitch * (pitch + 1) / 2;
            for (int idx = tid; idx < total_blocks; idx += bdim) {
                
                int J = (int)( (sqrtf(8.0f * idx + 1.0f) - 1.0f) / 2.0f ); 
                int I = idx - J * (J + 1) / 2;                             

                int p_I, q_I; get_pair_padded(I, phase, N_pad, p_I, q_I);
                int p_J, q_J; get_pair_padded(J, phase, N_pad, p_J, q_J);

                double c_I = C_s[I], s_I = S_s[I];
                double c_J = C_s[J], s_J = S_s[J];

                if (I == J) { 
                    if (c_I != 1.0) {
                        double app = A_s[p_I * N_pad + p_I];
                        double aqq = A_s[q_I * N_pad + q_I];
                        double apq = A_s[p_I * N_pad + q_I];

                        A_s[p_I * N_pad + p_I] = c_I*c_I*app + s_I*s_I*aqq - 2.0*s_I*c_I*apq;
                        A_s[q_I * N_pad + q_I] = s_I*s_I*app + c_I*c_I*aqq + 2.0*s_I*c_I*apq;
                        double new_apq = s_I*c_I*(app - aqq) + (c_I*c_I - s_I*s_I)*apq;
                        
                        A_s[p_I * N_pad + q_I] = new_apq;
                        A_s[q_I * N_pad + p_I] = new_apq; 
                    }
                } else {
                    if (c_I != 1.0 || c_J != 1.0) {
                        double E00 = A_s[p_I * N_pad + p_J];
                        double E01 = A_s[p_I * N_pad + q_J];
                        double E10 = A_s[q_I * N_pad + p_J];
                        double E11 = A_s[q_I * N_pad + q_J];

                        double T00 = c_I * E00 - s_I * E10;
                        double T01 = c_I * E01 - s_I * E11;
                        double T10 = s_I * E00 + c_I * E10;
                        double T11 = s_I * E01 + c_I * E11;

                        double out00 = T00 * c_J - T01 * s_J;
                        double out01 = T00 * s_J + T01 * c_J;
                        double out10 = T10 * c_J - T11 * s_J;
                        double out11 = T10 * s_J + T11 * c_J;

                        A_s[p_I * N_pad + p_J] = out00;
                        A_s[p_I * N_pad + q_J] = out01;
                        A_s[q_I * N_pad + p_J] = out10;
                        A_s[q_I * N_pad + q_J] = out11;

                        A_s[p_J * N_pad + p_I] = out00;
                        A_s[q_J * N_pad + p_I] = out01;
                        A_s[p_J * N_pad + q_I] = out10;
                        A_s[q_J * N_pad + q_I] = out11;
                    }
                }
            }
            
            // --- 步骤 3: 特征向量 V_T 更新 ---
            int total_v_tasks = pitch * N_actual;
            for (int idx = tid; idx < total_v_tasks; idx += bdim) {
                int k = idx / N_actual;
                int j = idx % N_actual;
                int p, q;
                get_pair_padded(k, phase, N_pad, p, q);
                
                double c = C_s[k];
                double s = S_s[k];
                if (c != 1.0 && q < N_actual) { 
                    double vp = Vt_b[p * N_actual + j];
                    double vq = Vt_b[q * N_actual + j];
                    Vt_b[p * N_actual + j] = c * vp - s * vq;
                    Vt_b[q * N_actual + j] = s * vp + c * vq;
                }
            }
            __syncthreads(); 
        }
    }

    // 3. 将对角化后的矩阵写回全局
    for (int idx = tid; idx < N_actual * N_actual; idx += bdim) {
        int r = idx / N_actual;
        int c = idx % N_actual;
        A_b[r * N_actual + c] = A_s[r * N_pad + c];
    }
}

// --------------------------------------------------------
// C++ 接口
// --------------------------------------------------------
std::vector<torch::Tensor> jacobi_batched_cuda(torch::Tensor A, int pre_sweeps, int standard_sweeps, double initial_thresh) {
    int batch_size = A.size(0);
    int N_actual = A.size(1);
    
    int N_pad = (N_actual % 2 == 0) ? N_actual : N_actual + 1;
    int pitch = N_pad / 2;
    
    auto Vt = torch::eye(N_actual, A.options()).unsqueeze(0).repeat({batch_size, 1, 1});

    // 动态共享内存总大小
    size_t shared_mem_bytes = (N_pad * N_pad + 2 * pitch) * sizeof(double);

    auto kernel = jacobi_h800_tile_fused_kernel;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_bytes);

    // --- 【宏观优化 2：动态感知硬件，极限压榨 Occupancy】 ---
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    // H800 通常是 227KB (232448 bytes) 和 2048 threads
    int max_smem_per_sm = prop.sharedMemPerMultiprocessor; 
    int max_threads_per_sm = prop.maxThreadsPerMultiProcessor; 
    
    if (max_smem_per_sm == 0) max_smem_per_sm = 227 * 1024; 
    if (max_threads_per_sm == 0) max_threads_per_sm = 2048;

    // 1. 根据共享内存，算出一个 SM 最多能塞下几个矩阵？
    int max_blocks_by_smem = max_smem_per_sm / shared_mem_bytes;
    if (max_blocks_by_smem < 1) max_blocks_by_smem = 1; 
    if (max_blocks_by_smem > 32) max_blocks_by_smem = 32; // Hopper 架构单一 SM 最多 32 个 Block

    // 2. 为了榨干 2048 个线程，平均每个矩阵该分多少个线程？
    int ideal_threads = max_threads_per_sm / max_blocks_by_smem;
    
    // 3. 将线程数卡在合法范围内 [32, 1024]
    if (ideal_threads > 1024) ideal_threads = 1024;
    if (ideal_threads < 32) ideal_threads = 32;

    // 4. 对齐到最近的 2 的幂次方 (GPU Warp 调度最舒服的格式)
    int threads_per_block = 1;
    while (threads_per_block * 2 <= ideal_threads) {
        threads_per_block *= 2;
    }

    int blocks = batch_size;

    kernel<<<blocks, threads_per_block, shared_mem_bytes>>>(
        A.data_ptr<double>(), Vt.data_ptr<double>(), N_actual, N_pad, pre_sweeps, standard_sweeps, initial_thresh
    );

    return {A, Vt.transpose(-1, -2).contiguous()};
}
"""

print("正在编译 [H800 动态算力占据版] Kernel ...")
batched_jacobi = torch.utils.cpp_extension.load_inline(
    name="batched_jacobi_cuda_h800_dynamic",
    cpp_sources="std::vector<torch::Tensor> jacobi_batched_cuda(torch::Tensor A, int pre_sweeps, int standard_sweeps, double initial_thresh);",
    cuda_sources=cuda_source,
    functions=["jacobi_batched_cuda"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_90", "-Xptxas", "-v"]
)
print("编译完成！\n")

def eigh_block_jacobi(A_input, pre_sweeps=2, standard_sweeps=4, initial_thresh=0.1):
    A_out, V_out = batched_jacobi.jacobi_batched_cuda(A_input.clone(), pre_sweeps, standard_sweeps, initial_thresh)
    vals_unsorted = torch.diagonal(A_out, dim1=-2, dim2=-1)
    vals_sorted, indices = torch.sort(vals_unsorted, dim=-1)
    N = A_input.size(1)
    indices_expanded = indices.unsqueeze(1).expand(-1, N, -1)
    vecs_sorted = torch.gather(V_out, dim=-1, index=indices_expanded)
    return vals_sorted, vecs_sorted

def test_block_dominance():
    # 既然有 114 个 SM，我们直接喂给它 20 波的矩阵，让它彻底吃满！
    batch_size = 100 #114 * 20 
    N = 64
    pre_sweeps = 3
    standard_sweeps = 5
    retry = 10
    
    print(f"--- 测试 Block Jacobi 动态占据架构: Batch={batch_size}, N={N} ---")
    M = torch.randn(batch_size, N, N, dtype=torch.float64, device='cuda')
    A_input = (M + M.transpose(-1, -2)) / 2.0 

    for retry_id in range(0, retry):
        torch.cuda.synchronize()
        start = time.perf_counter()
        vals_torch, vecs_torch = torch.linalg.eigh(A_input)
        torch.cuda.synchronize()
        time_torch = (time.perf_counter() - start) * 1000
        if retry_id == retry - 1:
            print(f"torch_eigh time: {time_torch:.4f} ms")

    for retry_id in range(0, retry):
        torch.cuda.synchronize()
        start = time.perf_counter()
        vals_jacobi, vecs_jacobi = eigh_block_jacobi(A_input, pre_sweeps, standard_sweeps)
        torch.cuda.synchronize()
        time_jacobi = (time.perf_counter() - start) * 1000
        if retry_id == retry - 1:
            print(f"Block jacobi time: {time_jacobi:.4f} ms")

    print("\n--- 精度验证 ---")
    vals_error = torch.max(torch.abs(vals_jacobi - vals_torch))
    print(f"特征值误差: {vals_error.item():.4e}")

if __name__ == "__main__":
    test_block_dominance()