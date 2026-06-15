import numpy as np
from scipy.linalg import null_space


def analyze_rank(A, tol=1e-10, logger=None):
    """
    矩阵秩分析：计算秩、零空间维数、秩亏等信息。

    Parameters
    ----------
    A : ndarray (m, n)
        输入矩阵。
    tol : float
        奇异值截断阈值，小于此值的奇异值视为零。
    logger : logging.Logger, optional
        日志记录器。

    Returns
    -------
    info : dict
        - 'rank'            : 数值秩
        - 'nullity'         : 零空间维数
        - 'min_dim'         : min(m, n)
        - 'is_full_rank'    : 是否为满秩
        - 'is_rank_deficient': 是否秩亏
        - 'rank_deficiency' : 秩亏量
        - 'effective_rank'  : 相对有效秩（rank / min(m,n)）
        - 'singular_values' : 全部奇异值（降序）
        - 'sv_gap'          : 第 rank 与 rank+1 个奇异值之间的间隙
    """
    m, n = A.shape
    min_dim = min(m, n)
    s = np.linalg.svd(A, compute_uv=False)
    rank = int(np.sum(s > tol))
    nullity = n - rank

    sv_gap = (s[rank - 1] / s[rank]) if 0 < rank < len(s) and s[rank] > 0 else float('inf')

    info = {
        'rank': rank,
        'nullity': nullity,
        'min_dim': min_dim,
        'is_full_rank': rank == min_dim,
        'is_rank_deficient': rank < min_dim,
        'rank_deficiency': min_dim - rank,
        'effective_rank': rank / min_dim if min_dim > 0 else 0.0,
        'singular_values': s,
        'sv_gap': sv_gap,
    }

    if rank < n and m >= n:
        ns = null_space(A, rcond=tol)
        info['null_space_basis'] = ns
    else:
        info['null_space_basis'] = None

    return info


def print_rank_report(info, logger=None):
    """
    格式化输出秩分析报告。

    Parameters
    ----------
    info : dict
        analyze_rank 的返回值。
    logger : logging.Logger, optional
        日志记录器。
    """
    out = logger.info if logger else print
    out("\n" + "=" * 60)
    out("矩阵秩分析报告")
    out("=" * 60)
    out(f"  数值秩:          {info['rank']}")
    out(f"  零空间维数:      {info['nullity']}")
    out(f"  是否满秩:        {info['is_full_rank']}")
    out(f"  是否秩亏:        {info['is_rank_deficient']}")
    out(f"  秩亏量:          {info['rank_deficiency']}")
    out(f"  有效秩比例:      {info['effective_rank']:.4f}")
    out(f"  最大奇异值:      {info['singular_values'][0]:.6e}")
    out(f"  最小奇异值:      {info['singular_values'][-1]:.6e}")
    if info['rank_deficiency'] > 0:
        out(f"  奇异值间隙:      {info['sv_gap']:.6e}")
    if info['null_space_basis'] is not None:
        ns_dim = info['null_space_basis'].shape[1]
        out(f"  零空间基维度:    {ns_dim}")
    out("=" * 60)
