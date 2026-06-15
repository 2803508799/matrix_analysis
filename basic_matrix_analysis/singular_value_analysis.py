import numpy as np


def analyze_singular_values(A, tol=1e-10, logger=None):
    """
    奇异值分析：完整 SVD 分解及衍生指标。

    包含：
    - 完整奇异值谱
    - 矩阵的四个基本子空间信息
    - 低秩近似误差界
    - 奇异值分布统计

    Parameters
    ----------
    A : ndarray (m, n)
        输入矩阵。
    tol : float
        截断容差。
    logger : logging.Logger, optional
        日志记录器。

    Returns
    -------
    info : dict
        - 'U', 's', 'Vt'           : SVD 分解 A = U @ diag(s) @ Vt
        - 'singular_values'        : 奇异值数组
        - 'spectral_norm'          : ||A||_2
        - 'frobenius_norm'         : ||A||_F
        - 'nuclear_norm'           : ||A||_*
        - 'rank'                   : 数值秩
        - 'condition_number'       : s_max / s_min_nonzero
        - 'min_singular_value'     : 最小奇异值
        - 'max_singular_value'     : 最大奇异值
        - 'sv_mean', 'sv_median', 'sv_std'
        - 'sv_skewness', 'sv_kurtosis'
        - 'range_dim'              : 列空间维数
        - 'nullspace_dim'          : 零空间维数
        - 'domain_range_dim'       : 行空间维数
        - 'domain_nullspace_dim'   : 左零空间维数
    """
    m, n = A.shape
    U, s, Vt = np.linalg.svd(A, full_matrices=True)

    nonzero_s = s[s > tol]
    rank = len(nonzero_s)

    s_max = float(s[0])
    s_min_nonzero = float(nonzero_s[-1]) if rank > 0 else 0.0
    s_min = float(s[-1])

    from scipy.stats import skew, kurtosis
    sv_skew = float(skew(s)) if len(s) > 2 else 0.0
    sv_kurt = float(kurtosis(s)) if len(s) > 3 else 0.0

    info = {
        'U': U,
        's': s,
        'Vt': Vt,
        'singular_values': s,
        'spectral_norm': s_max,
        'frobenius_norm': float(np.sqrt(np.sum(s ** 2))),
        'nuclear_norm': float(np.sum(s)),
        'rank': rank,
        'condition_number': s_max / s_min_nonzero if s_min_nonzero > tol else float('inf'),
        'min_singular_value': s_min,
        'max_singular_value': s_max,
        'min_nonzero_singular_value': s_min_nonzero,
        'sv_mean': float(np.mean(s)),
        'sv_median': float(np.median(s)),
        'sv_std': float(np.std(s)),
        'sv_skewness': sv_skew,
        'sv_kurtosis': sv_kurt,
        'range_dim': rank,
        'nullspace_dim': n - rank,
        'domain_range_dim': rank,
        'domain_nullspace_dim': m - rank,
        'sv_range_ratio': s_max / (s_min_nonzero + tol) if rank > 0 else float('inf'),
    }

    # 低秩近似误差界 (Eckart-Young)
    info['rank_k_errors'] = {}
    for k_target in [1, 2, 5, 10, 20, 50]:
        if k_target < rank:
            error_fro = float(np.sqrt(np.sum(s[k_target:] ** 2)))
            error_spec = float(s[k_target]) if k_target < len(s) else 0.0
            info['rank_k_errors'][k_target] = {
                'frobenius_error': error_fro,
                'spectral_error': error_spec,
                'relative_fro_error': error_fro / info['frobenius_norm'] if info['frobenius_norm'] > tol else 0.0,
            }

    return info


def print_singular_value_report(info, logger=None):
    """
    格式化输出奇异值分析报告。

    Parameters
    ----------
    info : dict
        analyze_singular_values 的返回值。
    logger : logging.Logger, optional
        日志记录器。
    """
    out = logger.info if logger else print
    out("\n" + "=" * 60)
    out("奇异值分析报告")
    out("=" * 60)
    out(f"  数值秩:            {info['rank']}")
    out(f"  谱范数 (||A||_2):  {info['spectral_norm']:.6e}")
    out(f"  Frobenius 范数:    {info['frobenius_norm']:.6e}")
    out(f"  核范数 (||A||_*):  {info['nuclear_norm']:.6e}")
    out(f"  条件数:            {info['condition_number']:.6e}")
    out(f"  最大奇异值:        {info['max_singular_value']:.6e}")
    out(f"  最小非零奇异值:    {info['min_nonzero_singular_value']:.6e}")
    out(f"  最小奇异值:        {info['min_singular_value']:.6e}")
    out(f"  奇异值均值:        {info['sv_mean']:.6e}")
    out(f"  奇异值中位数:      {info['sv_median']:.6e}")
    out(f"  奇异值标准差:      {info['sv_std']:.6e}")
    out(f"  奇异值偏度:        {info['sv_skewness']:.4f}")
    out(f"  奇异值峰度:        {info['sv_kurtosis']:.4f}")
    out(f"  列空间维数:        {info['range_dim']}")
    out(f"  零空间维数:        {info['nullspace_dim']}")
    out(f"  左零空间维数:      {info['domain_nullspace_dim']}")
    out(f"  最大/最小奇异值比:  {info['sv_range_ratio']:.6e}")

    if info['rank_k_errors']:
        out("\n[低秩近似误差]")
        for k, err in info['rank_k_errors'].items():
            out(f"  rank={k:>3d}: Fro 误差={err['frobenius_error']:.6e}, "
                f"相对={err['relative_fro_error']:.4e}")
    out("=" * 60)
