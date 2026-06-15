import numpy as np


def analyze_sparsity(A, tol=1e-10, logger=None):
    """
    矩阵稀疏性分析。

    Parameters
    ----------
    A : ndarray (m, n)
        输入矩阵。
    tol : float
        判断元素为零的阈值。
    logger : logging.Logger, optional
        日志记录器。

    Returns
    -------
    info : dict
        - 'total_elements'        : 总元素数
        - 'nonzero_elements'      : 非零元素数
        - 'zero_elements'         : 零元素数
        - 'sparsity_ratio'        : 零元素比例
        - 'density_ratio'         : 非零元素比例
        - 'nnz_per_row'           : 每行非零元数量
        - 'nnz_per_col'           : 每列非零元数量
        - 'max_nnz_per_row'       : 单行最大非零元数
        - 'min_nnz_per_row'       : 单行最小非零元数
        - 'max_nnz_per_col'       : 单列最大非零元数
        - 'min_nnz_per_col'       : 单列最小非零元数
        - 'bandwidth_lower'       : 下半带宽
        - 'bandwidth_upper'       : 上半带宽
        - 'bandwidth_total'       : 总带宽
        - 'is_banded'             : 是否为带状矩阵
        - 'is_diagonal'           : 是否为对角矩阵
        - 'is_tridiagonal'        : 是否为三对角矩阵
        - 'is_upper_triangular'   : 是否为上三角
        - 'is_lower_triangular'   : 是否为下三角
        - 'is_block_diagonal'     : 是否为块对角（简易判断）
        - 'sparsity_pattern'      : 稀疏模式描述
    """
    m, n = A.shape
    total = m * n
    mask = np.abs(A) > tol
    nnz = int(np.sum(mask))
    nz = total - nnz
    sparsity = nz / total if total > 0 else 1.0

    # 每行/列非零元统计
    nnz_per_row = np.sum(mask, axis=1)
    nnz_per_col = np.sum(mask, axis=0)

    # 带宽分析（仅方阵）
    if m == n:
        bw_lower, bw_upper = _compute_bandwidth(mask)
        bandwidth_total = bw_lower + bw_upper + 1
        is_tridiag = bw_lower <= 1 and bw_upper <= 1 and nnz > 0
    else:
        bw_lower = bw_upper = bandwidth_total = None
        is_tridiag = False

    # 结构类型判断
    diag_mask = np.eye(m, n, dtype=bool)
    off_diag_mask = ~diag_mask
    has_off_diag_nonzero = np.any(mask & off_diag_mask)

    is_diagonal = not has_off_diag_nonzero and m == n
    is_upper_tri = not np.any(np.tril(mask, k=-1)) if m == n else False
    is_lower_tri = not np.any(np.triu(mask, k=1)) if m == n else False
    is_banded = (bw_lower is not None and bw_lower < n * 0.5 and bw_upper < n * 0.5)

    # 稀疏模式描述
    if sparsity > 0.99:
        pattern = "extremely sparse"
    elif sparsity > 0.9:
        pattern = "highly sparse"
    elif sparsity > 0.7:
        pattern = "moderately sparse"
    elif sparsity > 0.5:
        pattern = "slightly sparse"
    elif sparsity > 0.3:
        pattern = "moderately dense"
    elif sparsity > 0.1:
        pattern = "dense"
    else:
        pattern = "fully dense"

    info = {
        'total_elements': total,
        'nonzero_elements': nnz,
        'zero_elements': nz,
        'sparsity_ratio': sparsity,
        'density_ratio': nnz / total if total > 0 else 0.0,
        'nnz_per_row': nnz_per_row,
        'nnz_per_col': nnz_per_col,
        'max_nnz_per_row': int(np.max(nnz_per_row)),
        'min_nnz_per_row': int(np.min(nnz_per_row)),
        'max_nnz_per_col': int(np.max(nnz_per_col)),
        'min_nnz_per_col': int(np.min(nnz_per_col)),
        'bandwidth_lower': bw_lower,
        'bandwidth_upper': bw_upper,
        'bandwidth_total': bandwidth_total,
        'is_banded': is_banded,
        'is_diagonal': is_diagonal,
        'is_tridiagonal': is_tridiag,
        'is_upper_triangular': is_upper_tri,
        'is_lower_triangular': is_lower_tri,
        'sparsity_pattern': pattern,
        'empty_rows': np.where(nnz_per_row == 0)[0].tolist(),
        'empty_cols': np.where(nnz_per_col == 0)[0].tolist(),
    }

    return info


def _compute_bandwidth(mask):
    """计算矩阵的上下带宽。"""
    n = mask.shape[0]
    bw_lower = 0
    bw_upper = 0
    for i in range(n):
        row_nnz = np.where(mask[i])[0]
        if len(row_nnz) > 0:
            lower = max(0, i - row_nnz[0])
            upper = max(0, row_nnz[-1] - i)
            bw_lower = max(bw_lower, lower)
            bw_upper = max(bw_upper, upper)
    return bw_lower, bw_upper


def print_sparsity_report(info, logger=None):
    """
    格式化输出稀疏性分析报告。

    Parameters
    ----------
    info : dict
        analyze_sparsity 的返回值。
    logger : logging.Logger, optional
        日志记录器。
    """
    out = logger.info if logger else print
    out("\n" + "=" * 60)
    out("矩阵稀疏性分析报告")
    out("=" * 60)
    out(f"  总元素数:          {info['total_elements']}")
    out(f"  非零元素数:        {info['nonzero_elements']}")
    out(f"  零元素数:          {info['zero_elements']}")
    out(f"  稀疏度:            {info['sparsity_ratio']:.4%}")
    out(f"  稠密度:            {info['density_ratio']:.4%}")
    out(f"  稀疏模式:          {info['sparsity_pattern']}")
    out(f"  每行非零元范围:    [{info['min_nnz_per_row']}, {info['max_nnz_per_row']}]")
    out(f"  每列非零元范围:    [{info['min_nnz_per_col']}, {info['max_nnz_per_col']}]")
    out(f"  是否对角矩阵:      {info['is_diagonal']}")
    out(f"  是否三对角矩阵:    {info['is_tridiagonal']}")
    out(f"  是否上三角:        {info['is_upper_triangular']}")
    out(f"  是否下三角:        {info['is_lower_triangular']}")
    out(f"  是否带状:          {info['is_banded']}")
    if info['bandwidth_lower'] is not None:
        out(f"  下半带宽:          {info['bandwidth_lower']}")
        out(f"  上半带宽:          {info['bandwidth_upper']}")
        out(f"  总带宽:            {info['bandwidth_total']}")
    if info['empty_rows']:
        out(f"  全零行索引:        {info['empty_rows'][:10]}{' ...' if len(info['empty_rows']) > 10 else ''}")
    if info['empty_cols']:
        out(f"  全零列索引:        {info['empty_cols'][:10]}{' ...' if len(info['empty_cols']) > 10 else ''}")
    out("=" * 60)
