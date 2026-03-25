import torch
import torch.utils.cpp_extension
import time

torch.manual_seed(42)

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ inline bool get_pair(int k, int phase, int N_actual, int N_pad, int &p, int &q) {
    if (k == 0) {
        p = 0; q = phase + 1;
    } else {
        int M = N_pad - 1;
        p = (phase + k) % M + 1;
        q = (phase - k + M) % M + 1;
    }
    if (p > q) { int temp = p; p = q; q = temp; }
    if (q >= N_actual) return false;
    return true;
}

// =========================================================================
// 带阈值预处理的全缓存 Jacobi
// 引入 Pre-sweeps: 只猎杀大元素
// =========================================================================
__global__ void jacobi_h800_precond_kernel(double* A_global, double* Vt_global, 
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
    double* C_s = smem + N_actual * N_actual;            
    double* S_s = smem + N_actual * N_actual + pitch;    

    // 1. 将 A 全量加载进 Shared Memory
    for (int idx = tid; idx < N_actual * N_actual; idx += bdim) {
        A_s[idx] = A_b[idx];
    }
    __syncthreads();

    int total_sweeps = pre_sweeps + standard_sweeps;

    // 2. 芯片内部迭代 (包含预处理阶段和标准阶段)
    for (int s = 0; s < total_sweeps; ++s) {
        
        double current_thresh = 0.0;
        if (s < pre_sweeps) {
            current_thresh = initial_thresh / pow(10.0, s); 
        }

        for (int phase = 0; phase < num_phases; ++phase) {
            
            // --- 步骤 1: 计算旋转角度 (带阈值过滤) ---
            for (int k = tid; k < pitch; k += bdim) {
                int p, q;
                if (!get_pair(k, phase, N_actual, N_pad, p, q)) {
                    C_s[k] = 1.0; S_s[k] = 0.0;
                } else {
                    double app = A_s[p * N_actual + p];
                    double aqq = A_s[q * N_actual + q];
                    double apq = A_s[p * N_actual + q];
                    
                    double c = 1.0, s = 0.0;
                    
                    // 【预处理拦截】：只有绝对值大于当前阈值，才进行旋转计算
                    if (fabs(apq) >= current_thresh && fabs(apq) > 1e-15) { 
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

            // --- 步骤 2: A 的行更新 ---
            int total_tasks = pitch * N_actual;
            for (int idx = tid; idx < total_tasks; idx += bdim) {
                int k = idx / N_actual;
                int j = idx % N_actual;
                int p, q;
                if (get_pair(k, phase, N_actual, N_pad, p, q)) {
                    double c = C_s[k];  
                    double s = S_s[k];
                    // 如果被预处理拦截 (c==1.0)，则这几百个线程直接跳过乘加运算，极速放行！
                    if (c != 1.0) {
                        double apj = A_s[p * N_actual + j];
                        double aqj = A_s[q * N_actual + j];
                        A_s[p * N_actual + j] = c * apj - s * aqj;
                        A_s[q * N_actual + j] = s * apj + c * aqj;
                    }
                }
            }
            __syncthreads(); 

            // --- 步骤 3: A 的列更新 ---
            for (int idx = tid; idx < total_tasks; idx += bdim) {
                int k = idx / N_actual;
                int i = idx % N_actual; 
                int p, q;
                if (get_pair(k, phase, N_actual, N_pad, p, q)) {
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

            // --- 步骤 4: 特征向量 V_T 的行更新 (Global Memory) ---
            for (int idx = tid; idx < total_tasks; idx += bdim) {
                int k = idx / N_actual;
                int j = idx % N_actual;
                int p, q;
                if (get_pair(k, phase, N_actual, N_pad, p, q)) {
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

    // 3. 将对角化后的 A 搬回全局显存
    for (int idx = tid; idx < N_actual * N_actual; idx += bdim) {
        A_b[idx] = A_s[idx];
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
    size_t shared_mem_bytes = (N_actual * N_actual + 2 * pitch) * sizeof(double);

    auto kernel = jacobi_h800_precond_kernel;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_bytes);

    int threads_per_block = 512; 
    int blocks = batch_size;

    kernel<<<blocks, threads_per_block, shared_mem_bytes>>>(
        A.data_ptr<double>(), Vt.data_ptr<double>(), N_actual, N_pad, pre_sweeps, standard_sweeps, initial_thresh
    );

    return {A, Vt.transpose(-1, -2).contiguous()};
}
"""

print("正在编译...")
batched_jacobi = torch.utils.cpp_extension.load_inline(
    name="batched_jacobi_h800_precond",
    cpp_sources="std::vector<torch::Tensor> jacobi_batched_cuda(torch::Tensor A, int pre_sweeps, int standard_sweeps, double initial_thresh);",
    cuda_sources=cuda_source,
    functions=["jacobi_batched_cuda"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_90"] 
)
print("编译完成！\n")

def eigh_jacobi_custom(A_input, pre_sweeps=2, standard_sweeps=4, initial_thresh=0.1):
    # 现在你可以通过组合 pre_sweeps 和 standard_sweeps 来减少总迭代次数！
    A_out, V_out = batched_jacobi.jacobi_batched_cuda(A_input.clone(), pre_sweeps, standard_sweeps, initial_thresh)
    
    vals_unsorted = torch.diagonal(A_out, dim1=-2, dim2=-1)
    vals_sorted, indices = torch.sort(vals_unsorted, dim=-1)
    
    N = A_input.size(1)
    indices_expanded = indices.unsqueeze(1).expand(-1, N, -1)
    vecs_sorted = torch.gather(V_out, dim=-1, index=indices_expanded)
    
    return vals_sorted, vecs_sorted

def test_batch_dominance():
    batch_size = 1000
    N = 147

    pre_sweeps = 3
    standard_sweeps = 4
    retry = 10
    
    print(f"--- 测试 Batch={batch_size}, N={N} ---")
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
        vals_jacobi, vecs_jacobi = eigh_jacobi_custom(A_input, pre_sweeps, standard_sweeps)
        # A_out, V_out = batched_jacobi.jacobi_batched_cuda(A_input.clone(), standard_sweeps)
        torch.cuda.synchronize()
        time_jacobi = (time.perf_counter() - start) * 1000

        if retry_id == retry - 1:
            print(f"jacobi time: {time_jacobi:.4f} ms")

    print("\n--- 精度验证 ---")
    vals_error = torch.max(torch.abs(vals_jacobi - vals_torch))
    print(f"特征值误差: {vals_error.item():.4e}")

if __name__ == "__main__":
    test_batch_dominance()
