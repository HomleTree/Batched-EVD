# import torch
# import torch.utils.cpp_extension
# import time

# cuda_source = """
# #include <torch/extension.h>
# #include <cuda_runtime.h>
# #include <math.h>

# // ==========================================
# // 圆桌调度算法: 确保(p,q)互斥, 且每轮更改
# // ==========================================
# __device__ inline bool get_pair(int k, int phase, int N_actual, int N_pad, int &p, int &q) {
#     if (k == 0) {
#         p = 0; q = phase + 1;
#     } else {
#         int M = N_pad - 1;
#         p = (phase + k) % M + 1;
#         q = (phase - k + M) % M + 1;
#     }
#     if (p > q) { int temp = p; p = q; q = temp; }
#     if (q >= N_actual) return false;
#     return true;
# }

# // ==========================================
# // 计算角度: 将全部c、s先分别存储到C、S中
# // ==========================================
# __global__ void compute_angles_kernel(double* A, double* C, double* S, int N_actual, int N_pad, int phase, int batch_size, int pitch) {
#     int k = blockIdx.x * blockDim.x + threadIdx.x; 
#     int b = blockIdx.y; 
#     if (b >= batch_size || k >= pitch) return;
    
#     int p, q;
#     if (!get_pair(k, phase, N_actual, N_pad, p, q)) {
#         C[b * pitch + k] = 1.0; S[b * pitch + k] = 0.0; return;
#     }

#     double* A_batch = A + b * N_actual * N_actual;
#     double app = A_batch[p * N_actual + p];
#     double aqq = A_batch[q * N_actual + q];
#     double apq = A_batch[p * N_actual + q];

#     double c = 1.0, s = 0.0;
#     if (fabs(apq) > 1e-15) { 
#         double tau = (aqq - app) / (2.0 * apq);
#         double t;
#         if (tau == 0.0) t = 1.0;
#         else t = copysign(1.0, tau) / (fabs(tau) + sqrt(1.0 + tau * tau));
#         c = 1.0 / sqrt(1.0 + t * t);
#         s = c * t;
#     }
#     C[b * pitch + k] = c; S[b * pitch + k] = s;
# }

# // ==========================================
# // 行列更新: 一次性更新完毕
# // ==========================================
# __global__ void update_rows_kernel(double* A, double* C, double* S, int N_actual, int N_pad, int phase, int batch_size, int pitch) {
#     int j = blockIdx.x * blockDim.x + threadIdx.x; 
#     int k = blockIdx.y * blockDim.y + threadIdx.y; 
#     int b = blockIdx.z;

#     if (b >= batch_size || j >= N_actual || k >= pitch) return;
#     int p, q;
#     if (!get_pair(k, phase, N_actual, N_pad, p, q)) return;

#     double c = C[b * pitch + k];
#     double s = S[b * pitch + k];

#     if (c != 1.0) {
#         double* A_batch = A + b * N_actual * N_actual;
#         double apj = A_batch[p * N_actual + j];
#         double aqj = A_batch[q * N_actual + j];
#         A_batch[p * N_actual + j] = c * apj - s * aqj;
#         A_batch[q * N_actual + j] = s * apj + c * aqj;
#     }
# }

# __global__ void update_cols_and_vecs_kernel(double* A, double* V, double* C, double* S, int N_actual, int N_pad, int phase, int batch_size, int pitch) {
#     int i = blockIdx.x * blockDim.x + threadIdx.x; 
#     int k = blockIdx.y * blockDim.y + threadIdx.y; 
#     int b = blockIdx.z;

#     if (b >= batch_size || i >= N_actual || k >= pitch) return;
#     int p, q;
#     if (!get_pair(k, phase, N_actual, N_pad, p, q)) return;

#     double c = C[b * pitch + k];
#     double s = S[b * pitch + k];

#     if (c != 1.0) {
#         double* V_batch = V + b * N_actual * N_actual;
#         double vip = V_batch[i * N_actual + p];
#         double viq = V_batch[i * N_actual + q];
#         V_batch[i * N_actual + p] = vip * c - viq * s;
#         V_batch[i * N_actual + q] = vip * s + viq * c;

#         double* A_batch = A + b * N_actual * N_actual;
#         double aip = A_batch[i * N_actual + p];
#         double aiq = A_batch[i * N_actual + q];
#         A_batch[i * N_actual + p] = aip * c - aiq * s;
#         A_batch[i * N_actual + q] = aip * s + aiq * c;
#     }
# }

# // --------------------------------------------------------
# // C++ 接口包装函数
# // --------------------------------------------------------
# std::vector<torch::Tensor> jacobi_batched_cuda(torch::Tensor A, int standard_sweeps) {
#     int batch_size = A.size(0);
#     int N_actual = A.size(1);
    
#     int N_pad = (N_actual % 2 == 0) ? N_actual : N_actual + 1;
#     int pitch = N_pad / 2;
#     int num_phases = N_pad - 1; 
    
#     auto V = torch::eye(N_actual, A.options()).unsqueeze(0).repeat({batch_size, 1, 1});
#     auto C = torch::zeros({batch_size, pitch}, A.options());
#     auto S = torch::zeros({batch_size, pitch}, A.options());

#     dim3 threads_1d(256);
#     dim3 blocks_1d((pitch + 255) / 256, batch_size);

#     dim3 threads_2d(16, 16);
#     dim3 blocks_2d((N_actual + 15) / 16, (pitch + 15) / 16, batch_size);

#     for (int s = 0; s < standard_sweeps; ++s) {
#         for (int phase = 0; phase < num_phases; ++phase) { 
#             compute_angles_kernel<<<blocks_1d, threads_1d>>>(
#                 A.data_ptr<double>(), C.data_ptr<double>(), S.data_ptr<double>(), N_actual, N_pad, phase, batch_size, pitch
#             );
#             update_rows_kernel<<<blocks_2d, threads_2d>>>(
#                 A.data_ptr<double>(), C.data_ptr<double>(), S.data_ptr<double>(), N_actual, N_pad, phase, batch_size, pitch
#             );
#             update_cols_and_vecs_kernel<<<blocks_2d, threads_2d>>>(
#                 A.data_ptr<double>(), V.data_ptr<double>(), C.data_ptr<double>(), S.data_ptr<double>(), N_actual, N_pad, phase, batch_size, pitch
#             );
#         }
#     }
#     return {A, V};
# }
# """

# print("正在编译 Kernel ...")
# batched_jacobi = torch.utils.cpp_extension.load_inline(
#     name="batched_jacobi_cuda_pure",
#     cpp_sources="std::vector<torch::Tensor> jacobi_batched_cuda(torch::Tensor A, int standard_sweeps);",
#     cuda_sources=cuda_source,
#     functions=["jacobi_batched_cuda"],
#     extra_cflags=["-O3"],
#     extra_cuda_cflags=["-O3", "--use_fast_math"]
# )
# print("编译完成！\n")

# def eigh_jacobi_custom(A_input, standard_sweeps):

#     A_out, V_out = batched_jacobi.jacobi_batched_cuda(A_input.clone(), standard_sweeps)
    
#     # 特征值排序
#     vals_unsorted = torch.diagonal(A_out, dim1=-2, dim2=-1)
#     vals_sorted, indices = torch.sort(vals_unsorted, dim=-1)
    
#     # 特征向量排序
#     N = A_input.size(1)
#     indices_expanded = indices.unsqueeze(1).expand(-1, N, -1)
#     vecs_sorted = torch.gather(V_out, dim=-1, index=indices_expanded)
    
#     return vals_sorted, vecs_sorted


# def test_batch_dominance():
    
#     # 数据
#     batch_size = 1000
#     N = 147
#     standard_sweeps = 8
    
#     print(f"--- Batch={batch_size}, N={N} ---")
#     M = torch.randn(batch_size, N, N, dtype=torch.float64, device='cuda')
#     A_input = (M + M.transpose(-1, -2)) / 2.0 

#     # 预热
#     _ = eigh_jacobi_custom(A_input[:2], 5)
#     _ = torch.linalg.eigh(A_input[:2])

#     # PyTorch
#     torch.cuda.synchronize()
#     start = time.perf_counter()
#     vals_torch, vecs_torch = torch.linalg.eigh(A_input)
#     torch.cuda.synchronize()
#     time_torch = (time.perf_counter() - start) * 1000
#     print(f"PyTorch 原生 eigh 耗时: {time_torch:.4f} ms")
    
#     # Fused Jacobi
#     torch.cuda.synchronize()
#     start = time.perf_counter()
#     vals_custom, vecs_custom = eigh_jacobi_custom(A_input, standard_sweeps)
#     torch.cuda.synchronize()
#     time_custom = (time.perf_counter() - start) * 1000
#     print(f"Fused Jacobi 耗时: {time_custom:.4f} ms")
    
#     # Error
#     print("\n--- 精度验证 ---")
#     vals_error = torch.max(torch.abs(vals_custom - vals_torch))
#     print(f"特征值误差: {vals_error.item():.4e}")

#     # vecs_error = torch.max(torch.abs(vecs_custom - vecs_torch))
#     # print(f"特征向量误差: {vecs_error.item():.4e}")
    
# if __name__ == "__main__":
#     test_batch_dominance()

import torch
from torch.profiler import profile, record_function, ProfilerActivity
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
// 终极融合内核 (Fused Single-Launch Kernel)
// 一个 Block 处理一个 Batch。循环、同步、计算全部在 GPU 芯片内部完成！
// =========================================================================
__global__ void jacobi_fused_single_launch(double* A, double* V, int N_actual, int N_pad, int sweeps) {
    int b = blockIdx.x;        // 当前 Block 负责的 Batch
    int tid = threadIdx.x;     // 线程 ID
    int bdim = blockDim.x;     // Block 总线程数

    // 指向当前 Batch 的全局显存指针 (极大概率命中 L1/L2 Cache)
    double* A_b = A + b * N_actual * N_actual;
    double* V_b = V + b * N_actual * N_actual;

    int pitch = N_pad / 2;
    int num_phases = N_pad - 1;

    // 动态分配共享内存：仅存储旋转角度 C 和 S (体积极小，速度极快)
    extern __shared__ double smem[];
    double* C_s = smem;
    double* S_s = smem + pitch;

    // 所有的 Sweep 和 Phase 都在设备端循环, 0 Launch 开销！
    for (int s = 0; s < sweeps; ++s) {
        for (int phase = 0; phase < num_phases; ++phase) {
            
            // ---------------------------------------------------
            // 步骤 1: 计算角度 (结果写入 Shared Memory)
            // ---------------------------------------------------
            for (int k = tid; k < pitch; k += bdim) {
                int p, q;
                if (!get_pair(k, phase, N_actual, N_pad, p, q)) {
                    C_s[k] = 1.0; S_s[k] = 0.0;
                } else {
                    double app = A_b[p * N_actual + p];
                    double aqq = A_b[q * N_actual + q];
                    double apq = A_b[p * N_actual + q];
                    
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
            __syncthreads(); // 必须等待所有角度计算完毕

            // ---------------------------------------------------
            // 步骤 2: 行更新 (A = R^T * A)
            // 利用网格步幅循环，数千个独立任务被线程池完美消化
            // ---------------------------------------------------
            int total_row_tasks = pitch * N_actual;
            for (int idx = tid; idx < total_row_tasks; idx += bdim) {
                int k = idx / N_actual;
                int j = idx % N_actual;
                int p, q;
                if (get_pair(k, phase, N_actual, N_pad, p, q)) {
                    double c = C_s[k];  // 从 Shared Memory 极速读取
                    double s = S_s[k];
                    if (c != 1.0) {
                        double apj = A_b[p * N_actual + j];
                        double aqj = A_b[q * N_actual + j];
                        A_b[p * N_actual + j] = c * apj - s * aqj;
                        A_b[q * N_actual + j] = s * apj + c * aqj;
                    }
                }
            }
            __syncthreads(); // 必须等待行更新完毕，防止读写竞争

            // ---------------------------------------------------
            // 步骤 3: 列与特征向量更新 (A = A * R, V = V * R)
            // ---------------------------------------------------
            int total_col_tasks = N_actual * pitch;
            for (int idx = tid; idx < total_col_tasks; idx += bdim) {
                int i = idx / pitch;
                int k = idx % pitch;
                int p, q;
                if (get_pair(k, phase, N_actual, N_pad, p, q)) {
                    double c = C_s[k];
                    double s = S_s[k];
                    if (c != 1.0) {
                        // 利用寄存器 (Registers) 暂存数据，实现一读两算两写
                        double aip = A_b[i * N_actual + p];
                        double aiq = A_b[i * N_actual + q];
                        A_b[i * N_actual + p] = aip * c - aiq * s;
                        A_b[i * N_actual + q] = aip * s + aiq * c;

                        double vip = V_b[i * N_actual + p];
                        double viq = V_b[i * N_actual + q];
                        V_b[i * N_actual + p] = vip * c - viq * s;
                        V_b[i * N_actual + q] = vip * s + viq * c;
                    }
                }
            }
            __syncthreads(); // 等待本 Phase 全部结束，再进入下一 Phase
        }
    }
}

// --------------------------------------------------------
// C++ 接口包装函数
// --------------------------------------------------------
std::vector<torch::Tensor> jacobi_batched_cuda(torch::Tensor A, int standard_sweeps) {
    int batch_size = A.size(0);
    int N_actual = A.size(1);
    
    int N_pad = (N_actual % 2 == 0) ? N_actual : N_actual + 1;
    int pitch = N_pad / 2;
    
    auto V = torch::eye(N_actual, A.options()).unsqueeze(0).repeat({batch_size, 1, 1});

    // 为 Shared Memory 分配空间：只存 C 和 S 数组
    size_t shared_mem_bytes = 2 * pitch * sizeof(double);

    // 采用极限性能配置：每个 Batch 1个 Block, 使用 512 或 1024 个线程并发
    int threads_per_block = 256; 
    int blocks = batch_size;

    jacobi_fused_single_launch<<<blocks, threads_per_block, shared_mem_bytes>>>(
        A.data_ptr<double>(), V.data_ptr<double>(), N_actual, N_pad, standard_sweeps
    );

    return {A, V};
}
"""

print("正在编译...")
batched_jacobi = torch.utils.cpp_extension.load_inline(
    name="batched_jacobi_cuda_device_fused",
    cpp_sources="std::vector<torch::Tensor> jacobi_batched_cuda(torch::Tensor A, int standard_sweeps);",
    cuda_sources=cuda_source,
    functions=["jacobi_batched_cuda"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"]
)
print("编译完成！\n")

def eigh_jacobi(A_input, standard_sweeps):
    A_out, V_out = batched_jacobi.jacobi_batched_cuda(A_input.clone(), standard_sweeps)
    
    vals_unsorted = torch.diagonal(A_out, dim1=-2, dim2=-1)
    vals_sorted, indices = torch.sort(vals_unsorted, dim=-1)
    
    N = A_input.size(1)
    indices_expanded = indices.unsqueeze(1).expand(-1, N, -1)
    vecs_sorted = torch.gather(V_out, dim=-1, index=indices_expanded)
    
    return vals_sorted, vecs_sorted

def test_batch_dominance():
    # 模拟真实工作流中的批量处理
    batch_size = 1000
    N = 147
    standard_sweeps = 8 
    
    print(f"--- 测试 Batch={batch_size}, N={N} ---")
    M = torch.randn(batch_size, N, N, dtype=torch.float64, device='cuda')
    A_input = (M + M.transpose(-1, -2)) / 2.0 

    # 预热
    # print("正在预热并排除冷启动开销...")
    # for _ in range(3):
    #     _ = eigh_jacobi_custom(A_input, 2)
    #     _ = torch.linalg.eigh(A_input)

    # 预热后测试耗时
    retry = 10
    for retry_id in range(0, retry):

        torch.cuda.synchronize()
        start = time.perf_counter()
        vals_torch, vecs_torch = torch.linalg.eigh(A_input)
        torch.cuda.synchronize()
        time_torch = (time.perf_counter() - start) * 1000

        if retry_id == retry - 1:
            print(f"torch_eigh time: {time_torch:.4f} ms")

    # for retry_id in range(0, retry):

    #     streams = [torch.cuda.Stream() for _ in range(batch_size)]
    #     vals_stream = [None] * batch_size
    #     vecs_stream = [None] * batch_size

    #     torch.cuda.synchronize()
    #     start = time.perf_counter()
    #     for i in range(batch_size):
    #         with torch.cuda.stream(streams[i]):
    #             vals_stream[i], vecs_stream[i] = torch.linalg.eigh(A_input[i])

    #     vals_stream_stacked = torch.stack(vals_stream)
    #     vecs_stream_stacked = torch.stack(vecs_stream)

    #     torch.cuda.synchronize()
    #     time_torch_stream = (time.perf_counter() - start) * 1000

    #     if retry_id == retry - 1:
    #         print(f"torch_eigh_stream time: {time_torch_stream:.4f} ms")

    for retry_id in range(0, retry):

        torch.cuda.synchronize()
        start = time.perf_counter()
        vals_jacobi, vecs_jacobi = eigh_jacobi(A_input, standard_sweeps)
        # A_out, V_out = batched_jacobi.jacobi_batched_cuda(A_input.clone(), standard_sweeps)
        torch.cuda.synchronize()
        time_jacobi = (time.perf_counter() - start) * 1000

        if retry_id == retry - 1:
            print(f"jacobi time: {time_jacobi:.4f} ms")


    # # torch.linalg.eigh
    # torch.cuda.synchronize()
    # start = time.perf_counter()
    # vals_torch, vecs_torch = torch.linalg.eigh(A_input)
    # torch.cuda.synchronize()
    # time_torch = (time.perf_counter() - start) * 1000
    # print(f"PyTorch 原生 eigh 耗时: {time_torch:.4f} ms")

    # # CUDA Stream并发
    # streams = [torch.cuda.Stream() for _ in range(batch_size)]
    # vals_stream = [None] * batch_size
    # vecs_stream = [None] * batch_size

    # torch.cuda.synchronize()
    # start = time.perf_counter()
    # for i in range(batch_size):
    #     with torch.cuda.stream(streams[i]):
    #         vals_stream[i], vecs_stream[i] = torch.linalg.eigh(A_input[i])

    # vals_stream_stacked = torch.stack(vals_stream)
    # vecs_stream_stacked = torch.stack(vecs_stream)

    # torch.cuda.synchronize()
    # time_stream = (time.perf_counter() - start) * 1000
    # print(f"CUDA Streams 并发 eigh 耗时: {time_stream:.4f} ms")
    # error = torch.max(torch.abs(vals_torch - vals_stream_stacked))
    # print(f"\n特征值对齐误差: {error.item():.4e}")

    # # Fused_Jacobi
    # torch.cuda.synchronize()
    # start = time.perf_counter()
    # with profile(
    #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #     record_shapes=True,               # 记录张量形状
    #     profile_memory=True,              # 记录显存
    #     with_stack=True                   # 记录 Python/C++ 调用栈
    # ) as prof:
    #     vals_custom, vecs_custom = eigh_jacobi_custom(A_input, standard_sweeps)

    # # 直接查看结果
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    # prof.export_chrome_trace("trace.json")

    # torch.cuda.synchronize()
    # time_custom = (time.perf_counter() - start) * 1000
    # print(f"Fused Jacobi (单启动) 耗时: {time_custom:.4f} ms")
    
    print("\n--- 精度验证 ---")
    vals_error = torch.max(torch.abs(vals_jacobi - vals_torch))
    print(f"特征值误差: {vals_error.item():.4e}")
    
    # D_custom = torch.diag_embed(vals_jacobi)
    # A_recon_custom = torch.bmm(vecs_jacobi, torch.bmm(D_custom, vecs_jacobi.transpose(1, 2)))
    # recon_error = torch.max(torch.abs(A_recon_custom - A_input))
    # print(f"重构误差 (|V D V^T - A|): {recon_error.item():.4e}")

if __name__ == "__main__":
    test_batch_dominance()