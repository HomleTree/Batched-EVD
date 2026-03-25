import torch
import torch.utils.cpp_extension
import time
import numpy as np

torch.manual_seed(42)

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// =========================================================================
// Sameh 调度器 (Sameh's Cyclic Ordering)
// 保证旋转像“波前”一样推进，完美保全矩阵的结构特性
// =========================================================================
__device__ inline bool get_pair_sameh(int thread_task_id, int phase, int N, int &p, int &q) {
    // phase 的范围是 0 到 2N - 4
    int min_p = (phase < N - 1) ? 0 : phase - (N - 2);
    int max_p = phase / 2;
    int num_pairs = max_p - min_p + 1;

    // 如果当前线程分配到的任务号超出了本 Phase 的最大任务数，则挂起
    if (thread_task_id >= num_pairs) return false;

    p = min_p + thread_task_id;
    q = phase + 1 - p;

    return true;
}

// =========================================================================
// 1. 矩阵 A 全量驻留 SRAM，内部列更新几乎 0 惩罚。
// 2. 特征向量 V_T (V的转置) 驻留全局显存，行更新 100% 内存合并！
// 3. 核心调度替换为 Sameh 波前调度
// =========================================================================
__global__ void jacobi_h800_sameh_fused_kernel(double* A_global, double* Vt_global, int N_actual, int sweeps) {
    int b = blockIdx.x;        
    int tid = threadIdx.x;     
    int bdim = blockDim.x;     

    double* A_b = A_global + b * N_actual * N_actual;
    double* Vt_b = Vt_global + b * N_actual * N_actual;

    // Sameh 调度下的每个 Phase 最大并发任务数依然是 N/2
    int pitch = N_actual / 2;
    // Sameh 调度的总 Phase 数量为 2N - 3
    int num_phases = 2 * N_actual - 3;

    // 获取解锁后的超大动态共享内存
    extern __shared__ double smem[];
    double* A_s = smem;                                  // Size: N * N
    double* C_s = smem + N_actual * N_actual;            // Size: pitch
    double* S_s = smem + N_actual * N_actual + pitch;    // Size: pitch

    // 1. 将 A 全量加载进 Shared Memory (合并读取，极限带宽)
    for (int idx = tid; idx < N_actual * N_actual; idx += bdim) {
        A_s[idx] = A_b[idx];
    }
    __syncthreads();

    // 2. 在芯片内部疯狂迭代 (完全切断与全局慢速显存的 A 通信)
    for (int s = 0; s < sweeps; ++s) {
        for (int phase = 0; phase < num_phases; ++phase) {
            
            // --- 步骤 1: 计算旋转角度 ---
            for (int k = tid; k < pitch; k += bdim) {
                int p, q;
                if (!get_pair_sameh(k, phase, N_actual, p, q)) {
                    C_s[k] = 1.0; S_s[k] = 0.0;
                } else {
                    double app = A_s[p * N_actual + p];
                    double aqq = A_s[q * N_actual + q];
                    double apq = A_s[p * N_actual + q];
                    
                    double c = 1.0, s = 0.0;
                    if (fabs(apq) > 1e-15) { 
                        double tau = (aqq - app) / (2.0 * apq);
                        double t;
                        if (tau == 0.0) t = 1.0;
                        else t = copysign(1.0, tau) / (fabs(tau) + sqrt(1.0 + tau * tau));
                        c = 1.0 / sqrt(1.0 + t * t);
                        s = c * t;
                    }
                    C_s[k] = c; S_s[k] = s;
                }
            }
            __syncthreads(); 

            // --- 步骤 2: A 的行更新 (Shared Memory) ---
            int total_tasks = pitch * N_actual;
            for (int idx = tid; idx < total_tasks; idx += bdim) {
                int k = idx / N_actual;
                int j = idx % N_actual;
                int p, q;
                if (get_pair_sameh(k, phase, N_actual, p, q)) {
                    double c = C_s[k];  
                    double s = S_s[k];
                    if (c != 1.0) {
                        double apj = A_s[p * N_actual + j];
                        double aqj = A_s[q * N_actual + j];
                        A_s[p * N_actual + j] = c * apj - s * aqj;
                        A_s[q * N_actual + j] = s * apj + c * aqj;
                    }
                }
            }
            __syncthreads(); 

            // --- 步骤 3: A 的列更新 (Shared Memory) ---
            for (int idx = tid; idx < total_tasks; idx += bdim) {
                int k = idx / N_actual;
                int i = idx % N_actual; // i 为行索引
                int p, q;
                if (get_pair_sameh(k, phase, N_actual, p, q)) {
                    double c = C_s[k];
                    double s = S_s[k];
                    if (c != 1.0) {
                        double aip = A_s[i * N_actual + p];
                        double aiq = A_s[i * N_actual + q];
                        A_s[i * N_actual + p] = aip * c - aiq * s;
                        A_s[i * N_actual + q] = aip * s + aiq * c;
                    }
                }
            }
            __syncthreads(); 

            // --- 步骤 4: 特征向量 V_T 的行更新 (Global Memory, 100% 合并访存!) ---
            for (int idx = tid; idx < total_tasks; idx += bdim) {
                int k = idx / N_actual;
                int j = idx % N_actual;
                int p, q;
                if (get_pair_sameh(k, phase, N_actual, p, q)) {
                    double c = C_s[k];
                    double s = S_s[k];
                    if (c != 1.0) {
                        double v_p = Vt_b[p * N_actual + j];
                        double v_q = Vt_b[q * N_actual + j];
                        Vt_b[p * N_actual + j] = c * v_p - s * v_q;
                        Vt_b[q * N_actual + j] = s * v_p + c * v_q;
                    }
                }
            }
            __syncthreads(); 
        }
    }

    // 3. 将对角化后的 A 搬回全局显存 (合并写入)
    for (int idx = tid; idx < N_actual * N_actual; idx += bdim) {
        A_b[idx] = A_s[idx];
    }
}

// --------------------------------------------------------
// C++ 接口
// --------------------------------------------------------
std::vector<torch::Tensor> jacobi_batched_cuda(torch::Tensor A, int standard_sweeps) {
    int batch_size = A.size(0);
    int N_actual = A.size(1);
    
    // Sameh 调度最大并发对数为 N/2（向下取整）
    int pitch = N_actual / 2;
    
    // 初始化 V 的转置 (Vt) 为单位矩阵
    auto Vt = torch::eye(N_actual, A.options()).unsqueeze(0).repeat({batch_size, 1, 1});

    // 计算所需的动态共享内存总大小 (A矩阵 + C向量 + S向量)
    size_t shared_mem_bytes = (N_actual * N_actual + 2 * pitch) * sizeof(double);

    auto kernel = jacobi_h800_sameh_fused_kernel;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_bytes);

    int threads_per_block = 512; 
    int blocks = batch_size;

    kernel<<<blocks, threads_per_block, shared_mem_bytes>>>(
        A.data_ptr<double>(), Vt.data_ptr<double>(), N_actual, standard_sweeps
    );

    // 计算完成后，将 Vt 转置回正常的 V 形状
    return {A, Vt.transpose(-1, -2).contiguous()};
}
"""

print("正在编译...")
batched_jacobi = torch.utils.cpp_extension.load_inline(
    name="batched_jacobi_cuda_h800_sameh",
    cpp_sources="std::vector<torch::Tensor> jacobi_batched_cuda(torch::Tensor A, int standard_sweeps);",
    cuda_sources=cuda_source,
    functions=["jacobi_batched_cuda"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_90"] # 强制指明 Hopper 架构以开启硬件特性
)
print("编译完成！\n")

def eigh_jacobi_custom(A_input, standard_sweeps):
    A_out, V_out = batched_jacobi.jacobi_batched_cuda(A_input.clone(), standard_sweeps)
    
    vals_unsorted = torch.diagonal(A_out, dim1=-2, dim2=-1)
    vals_sorted, indices = torch.sort(vals_unsorted, dim=-1)
    
    N = A_input.size(1)
    indices_expanded = indices.unsqueeze(1).expand(-1, N, -1)
    vecs_sorted = torch.gather(V_out, dim=-1, index=indices_expanded)
    
    return vals_sorted, vecs_sorted

def test_batch_dominance():
    batch_size = 100
    N = 64
    standard_sweeps = 6
    retry = 10
    
    print(f"--- 测试 Sameh 调度: Batch={batch_size}, N={N} ---")
    # 生成随机对称矩阵
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
        vals_jacobi, vecs_jacobi = eigh_jacobi_custom(A_input, standard_sweeps)
        torch.cuda.synchronize()
        time_jacobi = (time.perf_counter() - start) * 1000

        if retry_id == retry - 1:
            print(f"Sameh jacobi time: {time_jacobi:.4f} ms")

    print("\n--- 精度验证 ---")
    vals_error = torch.max(torch.abs(vals_jacobi - vals_torch))
    print(f"特征值误差: {vals_error.item():.4e}")

if __name__ == "__main__":
    test_batch_dominance()
