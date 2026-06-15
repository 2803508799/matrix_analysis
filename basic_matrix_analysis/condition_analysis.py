import numpy as np


def analyze_condition(A, tol=1e-10, logger=None):
    """
    矩阵病态分析。

    从多个角度评估矩阵的数值稳定性：
    - 条件数（2-范数、Frobenius 范数）
    - 行列式与近奇异程度
    - 矩阵平衡性
    - 病态来源诊断

    Parameters
    ----------
    A : ndarray (n, n)
        输入方阵。
    tol : float
        数值容差。
    logger : logging.Logger, optional
        日志记录器。

    Returns
    -------
    info : dict
        包含各种条件数与病态程度评估。
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError("病态分析仅适用于方阵")

    n = A.shape[0]
    s = np.linalg.svd(A, compute_uv=False)
    s_max = float(s[0])
    s_min = float(s[-1])

    # 2-范数条件数
    cond2 = s_max / s_min if s_min > tol else float('inf')

    # Frobenius 范数条件数
    frob_norm = float(np.sqrt(np.sum(s ** 2)))
    try:
        A_inv = np.linalg.inv(A)
        frob_norm_inv = float(np.linalg.norm(A_inv, 'fro'))
        cond_fro = frob_norm * frob_norm_inv
    except np.linalg.LinAlgError:
        cond_fro = float('inf')

    # 行列式
    sign, logdet = np.linalg.slogdet(A)
    det_info = {'sign': sign, 'logdet': logdet, 'det': sign * np.exp(logdet) if logdet > -700 else 0.0}

    # 病态程度分级
    if cond2 < 10:
        ill_condition_level = "well-conditioned"
    elif cond2 < 1e3:
        ill_condition_level = "mildly ill-conditioned"
    elif cond2 < 1e7:
        ill_condition_level = "moderately ill-conditioned"
    elif cond2 < 1e12:
        ill_condition_level = "severely ill-conditioned"
    else:
        ill_condition_level = "extremely ill-conditioned (nearly singular)"

    # 近零奇异值数量
    tiny_sv_count = int(np.sum(s < tol * s_max))

    # 矩阵行/列范数均衡性
    row_norms = np.linalg.norm(A, axis=1)
    col_norms = np.linalg.norm(A, axis=0)
    row_balance = float(np.min(row_norms) / (np.max(row_norms) + tol))
    col_balance = float(np.min(col_norms) / (np.max(col_norms) + tol))

    # 通过最大元缩放后的条件数估计
    scaled = A / (np.max(np.abs(A)) + tol)
    s_scaled = np.linalg.svd(scaled, compute_uv=False)
    cond_scaled = float(s_scaled[0] / (s_scaled[-1] + tol))

    info = {
        'condition_number_2': cond2,
        'condition_number_fro': cond_fro,
        'log10_condition_number': float(np.log10(cond2)) if cond2 != float('inf') else float('inf'),
        'ill_condition_level': ill_condition_level,
        'is_singular': s_min < tol,
        'is_nearly_singular': cond2 > 1e12,
        'singular_values': s,
        'max_singular_value': s_max,
        'min_singular_value': s_min,
        'tiny_singular_value_count': tiny_sv_count,
        'determinant': det_info['det'],
        'log_abs_determinant': det_info['logdet'],
        'row_balance': row_balance,
        'col_balance': col_balance,
        'scaled_condition_number': cond_scaled,
    }

    return info


def print_condition_report(info, logger=None):
    """
    格式化输出病态分析报告。

    Parameters
    ----------
    info : dict
        analyze_condition 的返回值。
    logger : logging.Logger, optional
        日志记录器。
    """
    out = logger.info if logger else print
    out("\n" + "=" * 60)
    out("矩阵病态分析报告")
    out("=" * 60)
    out(f"  2-范数条件数:     {info['condition_number_2']:.6e}")
    out(f"  log10(条件数):    {info['log10_condition_number']:.4f}")
    out(f"  Frobenius 条件数: {info['condition_number_fro']:.6e}")
    out(f"  病态等级:         {info['ill_condition_level']}")
    out(f"  是否奇异:         {info['is_singular']}")
    out(f"  是否近奇异:       {info['is_nearly_singular']}")
    out(f"  最大奇异值:       {info['max_singular_value']:.6e}")
    out(f"  最小奇异值:       {info['min_singular_value']:.6e}")
    out(f"  近零奇异值数:     {info['tiny_singular_value_count']}")
    out(f"  行列式:           {info['determinant']:.6e}")
    out(f"  log|det(A)|:      {info['log_abs_determinant']:.6f}")
    out(f"  行范数均衡比:     {info['row_balance']:.6f}")
    out(f"  列范数均衡比:     {info['col_balance']:.6f}")
    out(f"  缩放后条件数:     {info['scaled_condition_number']:.6e}")
    out("=" * 60)
