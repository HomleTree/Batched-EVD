import torch
import time

# =========================================================================
# 阶段 1：Householder 变换 (将稠密对称阵 A 转化为三对角阵 T)
# =========================================================================
def householder_tridiagonalization(A):
    """
    输入: 稠密对称矩阵 A (N x N)
    输出: 主对角线 d, 副对角线 e, 以及正交变换矩阵 Q (满足 A = Q * T * Q^T)
    """
    N = A.size(0)
    Q = torch.eye(N, dtype=A.dtype, device=A.device)
    A_work = A.clone()
    
    d = torch.zeros(N, dtype=A.dtype, device=A.device)
    e = torch.zeros(N - 1, dtype=A.dtype, device=A.device)
    
    for i in range(N - 1):
        x = A_work[i+1:, i]
        norm_x = torch.norm(x)
        
        if norm_x < 1e-15:
            e[i] = 0.0
        else:
            # 构造 Householder 反射向量 v
            alpha = -torch.sign(x[0]) * norm_x if x[0] != 0 else -norm_x
            e[i] = alpha
            v = x.clone()
            v[0] -= alpha
            v = v / torch.norm(v)
            
            # 对子矩阵应用变换: A_sub = (I - 2vv^T) A_sub (I - 2vv^T)
            A_sub = A_work[i+1:, i+1:]
            v = v.unsqueeze(1)
            
            # 优化计算: p = 2 * A_sub * v, w = p - (v^T * p) * v
            p = 2.0 * torch.matmul(A_sub, v)
            w = p - torch.matmul(v.t(), p) * v
            
            A_work[i+1:, i+1:] -= (torch.matmul(v, w.t()) + torch.matmul(w, v.t()))
            
            # 累积正交矩阵 Q
            Q_sub = Q[:, i+1:]
            Q[:, i+1:] -= 2.0 * torch.matmul(torch.matmul(Q_sub, v), v.t())
            
    d = torch.diag(A_work)
    return d, e, Q

# =========================================================================
# 阶段 2：久期方程求解器 (Secular Equation)
# =========================================================================
def solve_secular(d, v, rho):
    """求解 f(x) = 1 + rho * sum( v_i^2 / (d_i - x) ) = 0"""
    N = d.size(0)
    lam = torch.zeros_like(d)
    
    # 将 d 严格升序排序，并同步打乱 v (这是合并两个子问题后的必要操作)
    d_sorted, indices = torch.sort(d)
    v_sorted = v[indices]
    
    for i in range(N):
        # 确定每个根的严格上下界
        if rho > 0:
            left = d_sorted[i] + 1e-14
            right = d_sorted[i+1] - 1e-14 if i < N - 1 else d_sorted[i] + rho * torch.dot(v_sorted, v_sorted) + 1e-14
        else:
            right = d_sorted[i] - 1e-14
            left = d_sorted[i-1] + 1e-14 if i > 0 else d_sorted[i] + rho * torch.dot(v_sorted, v_sorted) - 1e-14

        # 暴力且极其稳定的二分查找 (FP64下60次迭代足以收敛到机器精度)
        for _ in range(60):
            mid = (left + right) / 2.0
            f_mid = 1.0 + rho * torch.sum((v_sorted ** 2) / (d_sorted - mid))
            
            # --- 核心修复区：严格遵循久期方程的单调性 ---
            if rho > 0:
                # rho > 0 时，f(x) 是严格递增的 (从 -inf 到 +inf)
                if f_mid > 0: 
                    right = mid # 已经大于0了，说明根在左边
                else:         
                    left = mid  # 小于0，说明根在右边
            else:
                # rho < 0 时，f(x) 是严格递减的 (从 +inf 到 -inf)
                if f_mid > 0: 
                    left = mid  # 大于0，说明根在右边
                else:         
                    right = mid # 小于0，说明根在左边
                
        lam[i] = (left + right) / 2.0
        
    # 计算新特征向量矩阵 U
    U = torch.zeros((N, N), dtype=d.dtype, device=d.device)
    for i in range(N):
        u_i = v_sorted / (d_sorted - lam[i])
        U[:, i] = u_i / torch.norm(u_i)
        
    # 逆向映射回排序前的空间
    U_original = torch.zeros_like(U)
    U_original[indices, :] = U
    
    return lam, U_original

# =========================================================================
# 阶段 3：Cuppen 分治法核心递归
# =========================================================================
def dc_tridiagonal(d, e):
    """递归求解三对角矩阵的特征系统"""
    N = d.size(0)
    if N == 1:
        return d, torch.ones((1, 1), dtype=d.dtype, device=d.device)
        
    # 1. 切分 (Divide)
    mid = N // 2
    rho = e[mid - 1]
    
    d1, d2 = d[:mid].clone(), d[mid:].clone()
    d1[-1] -= rho
    d2[0]  -= rho
    
    e1, e2 = e[:mid - 1], e[mid:]
    
    # 2. 递归求解子问题 (Conquer)
    lam1, Q1 = dc_tridiagonal(d1, e1)
    lam2, Q2 = dc_tridiagonal(d2, e2)
    
    # 3. 合并 (Merge)
    d_sub = torch.cat([lam1, lam2])
    Q_sub = torch.zeros((N, N), dtype=d.dtype, device=d.device)
    Q_sub[:mid, :mid] = Q1
    Q_sub[mid:, mid:] = Q2
    
    v = torch.cat([Q1[-1, :], Q2[0, :]])
    
    lam_new, U = solve_secular(d_sub, v, rho)
    Q_new = torch.matmul(Q_sub, U)
    
    return lam_new, Q_new

# =========================================================================
# 终极封装与测试
# =========================================================================
def eigh_divide_and_conquer(A):
    """端到端对称矩阵特征值分解"""
    # 1. 稠密转三对角
    d, e, Q_householder = householder_tridiagonalization(A)
    # 2. 分治法解三对角
    lam, Q_tridiag = dc_tridiagonal(d, e)
    # 3. 乘回原空间: V = Q_householder * Q_tridiag
    V = torch.matmul(Q_householder, Q_tridiag)
    return lam, V

if __name__ == "__main__":
    N = 32 # 测试维度
    torch.manual_seed(42)
    retry = 10
    
    # 构造一个随机对称矩阵，强制使用 FP64 以验证极限精度
    X = torch.randn(N, N, dtype=torch.float64)
    A = X + X.t()
    
    print(f"--- 正在测试 N={N} 的对称矩阵 ---")
    
    # 1. PyTorch 原生标准库 (底层调用 cuSOLVER / LAPACK)
    # lam_torch, V_torch = torch.linalg.eigh(A)

    for retry_id in range(0, retry):
        torch.cuda.synchronize()
        start = time.perf_counter()
        lam_torch, V_torch = torch.linalg.eigh(A)
        torch.cuda.synchronize()
        time_torch = (time.perf_counter() - start) * 1000
        if retry_id == retry - 1:
            print(f"torch_eigh time: {time_torch:.4f} ms")
    
    for retry_id in range(0, retry):
        torch.cuda.synchronize()
        start = time.perf_counter()
        lam_dc, V_dc = eigh_divide_and_conquer(A)
        torch.cuda.synchronize()
        time_torch = (time.perf_counter() - start) * 1000
        if retry_id == retry - 1:
            print(f"D&C time: {time_torch:.4f} ms")
    # 2. 自研完全分治法
    # lam_dc, V_dc = eigh_divide_and_conquer(A)
    
    # 3. 精度对比 (需要对齐排序)
    lam_dc_sorted, indices = torch.sort(lam_dc)
    V_dc_sorted = V_dc[:, indices]
    
    # 检查特征值绝对误差
    val_error = torch.max(torch.abs(lam_dc_sorted - lam_torch))
    print(f"1. 特征值最大绝对误差: {val_error.item():.4e}")
    
    # 检查重构误差 A - V * D * V^T
    A_reconstructed = torch.matmul(V_dc_sorted, torch.matmul(torch.diag(lam_dc_sorted), V_dc_sorted.t()))
    recon_error = torch.max(torch.abs(A - A_reconstructed))
    print(f"2. 矩阵重构绝对误差: {recon_error.item():.4e}")
    
    # 检查特征向量正交性 V^T * V - I
    I = torch.eye(N, dtype=torch.float64)
    ortho_error = torch.max(torch.abs(torch.matmul(V_dc_sorted.t(), V_dc_sorted) - I))
    print(f"3. 正交性绝对误差:   {ortho_error.item():.4e}")