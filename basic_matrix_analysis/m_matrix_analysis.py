import numpy as np


def analyze_m_matrix(A, tol=1e-10, logger=None):
    """
    M 矩阵性质检查。

    检查项：
    - Z 矩阵：所有非对角线元素 ≤ 0
    - M 矩阵：Z 矩阵 + 所有特征值实部 ≥ 0（或逆矩阵非负）
    - 非奇异 M 矩阵：Z 矩阵 + 所有特征值实部 > 0
    - 逆矩阵的非负性
    - 对角元正性（M 矩阵必要条件）

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
        包含各项 M 矩阵相关的判断结果。
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError("M 矩阵性质检查仅适用于方阵")

    n = A.shape[0]

    # Z 矩阵检查：非对角线元素 ≤ 0
    off_diag_mask = ~np.eye(n, dtype=bool)
    off_diag_values = A[off_diag_mask]
    is_z_matrix = bool(np.all(off_diag_values <= tol))
    max_positive_off_diag = float(np.max(off_diag_values))
    positive_off_diag_count = int(np.sum(off_diag_values > tol))

    # 特征值
    eigenvals = np.linalg.eigvals(A)
    real_parts = np.real(eigenvals)
    min_real_part = float(np.min(real_parts))

    is_nonsingular_m = is_z_matrix and min_real_part > tol
    is_singular_m = is_z_matrix and min_real_part >= -tol and np.any(np.abs(real_parts) < tol)
    is_m_matrix = is_nonsingular_m or is_singular_m

    # 对角线元素正性
    diag_values = np.diag(A)
    all_diag_positive = bool(np.all(diag_values > tol))
    zero_or_neg_diag = np.where(diag_values <= tol)[0].tolist()

    info = {
        'is_z_matrix': is_z_matrix,
        'is_m_matrix': is_m_matrix,
        'is_nonsingular_m_matrix': is_nonsingular_m,
        'is_singular_m_matrix': is_singular_m,
        'all_diag_positive': all_diag_positive,
        'zero_or_neg_diag_indices': zero_or_neg_diag,
        'max_positive_off_diag': max_positive_off_diag,
        'positive_off_diag_count': positive_off_diag_count,
        'min_eigenvalue_real': min_real_part,
        'eigenvalues': eigenvals,
    }

    # 对非奇异 M 矩阵，检查逆矩阵非负性
    if is_nonsingular_m and n <= 1000:
        try:
            A_inv = np.linalg.inv(A)
            is_inv_nonnegative = bool(np.all(A_inv >= -tol))
            min_inv_element = float(np.min(A_inv))
        except np.linalg.LinAlgError:
            is_inv_nonnegative = False
            min_inv_element = float('-inf')
        info['is_inverse_nonnegative'] = is_inv_nonnegative
        info['min_inverse_element'] = min_inv_element
    else:
        info['is_inverse_nonnegative'] = None
        info['min_inverse_element'] = None

    return info


def print_m_matrix_report(info, logger=None):
    """
    格式化输出 M 矩阵性质检查报告。

    Parameters
    ----------
    info : dict
        analyze_m_matrix 的返回值。
    logger : logging.Logger, optional
        日志记录器。
    """
    out = logger.info if logger else print
    out("\n" + "=" * 60)
    out("M 矩阵性质检查报告")
    out("=" * 60)
    out(f"  是否 Z 矩阵:          {info['is_z_matrix']}")
    out(f"  是否 M 矩阵:          {info['is_m_matrix']}")
    out(f"  是否非奇异 M 矩阵:    {info['is_nonsingular_m_matrix']}")
    out(f"  是否奇异 M 矩阵:      {info['is_singular_m_matrix']}")
    out(f"  所有对角元 > 0:       {info['all_diag_positive']}")
    if info['zero_or_neg_diag_indices']:
        out(f"  非正对角元索引:       {info['zero_or_neg_diag_indices'][:10]}{' ...' if len(info['zero_or_neg_diag_indices']) > 10 else ''}")
    out(f"  最大正非对角元:       {info['max_positive_off_diag']:.6e}")
    out(f"  正非对角元个数:       {info['positive_off_diag_count']}")
    out(f"  最小特征值实部:       {info['min_eigenvalue_real']:.6e}")
    if info['is_inverse_nonnegative'] is not None:
        out(f"  逆矩阵非负:           {info['is_inverse_nonnegative']}")
        out(f"  逆矩阵最小元素:       {info['min_inverse_element']:.6e}")
    out("=" * 60)
