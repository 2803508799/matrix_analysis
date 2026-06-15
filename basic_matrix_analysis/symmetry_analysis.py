import numpy as np


def analyze_symmetry(A, tol=1e-10, logger=None):
    """
    矩阵对称性分析：检查对称性、反对称性，给出非对称度量。

    Parameters
    ----------
    A : ndarray (n, n)
        输入方阵。
    tol : float
        判断对称性的容差。
    logger : logging.Logger, optional
        日志记录器。

    Returns
    -------
    info : dict
        - 'is_symmetric'       : 是否对称 (A ≈ A^T)
        - 'is_skew_symmetric'  : 是否反对称 (A ≈ -A^T)
        - 'is_hermitian'       : 是否 Hermitian (仅实矩阵等价于对称)
        - 'symmetry_error'     : ||A - A^T||_F / ||A||_F
        - 'skew_error'         : ||A + A^T||_F / ||A||_F
        - 'max_asymmetry'      : max|A_ij - A_ji|
        - 'asymmetry_matrix'   : A - A^T (非对称部分)
        - 'symmetric_part'     : (A + A^T) / 2
        - 'skew_symmetric_part': (A - A^T) / 2
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError("对称性分析仅适用于方阵")

    n = A.shape[0]
    A_norm = np.linalg.norm(A, 'fro')
    if A_norm < tol:
        raise ValueError("矩阵范数接近零，无法进行对称性分析")

    sym_part = (A + A.T) / 2
    skew_part = (A - A.T) / 2
    asym_matrix = A - A.T

    sym_error = np.linalg.norm(asym_matrix, 'fro') / A_norm
    skew_error = np.linalg.norm(A + A.T, 'fro') / A_norm
    max_asymmetry = np.max(np.abs(asym_matrix))
    is_symmetric = sym_error < tol
    is_skew_symmetric = skew_error < tol

    info = {
        'is_symmetric': is_symmetric,
        'is_skew_symmetric': is_skew_symmetric,
        'is_hermitian': is_symmetric and np.allclose(A.imag if np.iscomplexobj(A) else 0, 0),
        'symmetry_error': sym_error,
        'skew_error': skew_error,
        'max_asymmetry': max_asymmetry,
        'asymmetry_matrix': asym_matrix,
        'symmetric_part': sym_part,
        'skew_symmetric_part': skew_part,
    }

    # 非对称元素定位
    if not is_symmetric and n <= 500:
        row, col = np.unravel_index(np.argmax(np.abs(asym_matrix)), asym_matrix.shape)
        info['max_asym_loc'] = (int(row), int(col))
        info['max_asym_values'] = (A[row, col], A[col, row])

    return info


def print_symmetry_report(info, logger=None):
    """
    格式化输出对称性分析报告。

    Parameters
    ----------
    info : dict
        analyze_symmetry 的返回值。
    logger : logging.Logger, optional
        日志记录器。
    """
    out = logger.info if logger else print
    out("\n" + "=" * 60)
    out("矩阵对称性分析报告")
    out("=" * 60)
    out(f"  是否对称:         {info['is_symmetric']}")
    out(f"  是否反对称:       {info['is_skew_symmetric']}")
    out(f"  对称性误差:       {info['symmetry_error']:.6e}")
    out(f"  反对称性误差:     {info['skew_error']:.6e}")
    out(f"  最大非对称元素差: {info['max_asymmetry']:.6e}")
    if 'max_asym_loc' in info:
        r, c = info['max_asym_loc']
        v1, v2 = info['max_asym_values']
        out(f"  最大非对称位置:   ({r}, {c}) — A[{r},{c}]={v1:.6e}, A[{c},{r}]={v2:.6e}")
    out("=" * 60)
