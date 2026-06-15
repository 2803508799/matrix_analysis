import numpy as np
from scipy.linalg import eigh


def analyze_inertia(A, tol=1e-10, logger=None):
    """
    惯性指数分析 (仅适用于对称/Hermitian 矩阵)。

    惯性指数 (π, ν, ζ):
    - π: 正特征值个数
    - ν: 负特征值个数
    - ζ: 零特征值个数
    满足 π + ν + ζ = n

    同时计算正负惯性商 (π - ν) 和 signature。

    Parameters
    ----------
    A : ndarray (n, n)
        输入对称/Hermitian 方阵。若非对称，将自动使用 (A+A^T)/2。
    tol : float
        特征值为零的判定阈值。
    logger : logging.Logger, optional
        日志记录器。

    Returns
    -------
    info : dict
        - 'positive_inertia'  : π
        - 'negative_inertia'  : ν
        - 'zero_inertia'      : ζ
        - 'signature'         : π - ν
        - 'is_positive_definite'   : bool
        - 'is_negative_definite'   : bool
        - 'is_positive_semidefinite': bool
        - 'is_negative_semidefinite': bool
        - 'is_indefinite'          : bool
        - 'eigenvalues'            : 升序排列的特征值
        - 'used_symmetric_part'    : 是否使用了对称部分
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError("惯性指数分析仅适用于方阵")

    n = A.shape[0]
    sym_error = np.linalg.norm(A - A.T, 'fro')
    used_sym_part = False

    if sym_error > tol * np.linalg.norm(A, 'fro'):
        if logger:
            logger.warning("矩阵非对称，将使用 (A+A^T)/2 进行分析")
        A_use = (A + A.T) / 2
        used_sym_part = True
    else:
        A_use = A

    eigenvals = eigh(A_use, eigvals_only=True)

    pi = int(np.sum(eigenvals > tol))
    nu = int(np.sum(eigenvals < -tol))
    zeta = n - pi - nu

    info = {
        'positive_inertia': pi,
        'negative_inertia': nu,
        'zero_inertia': zeta,
        'signature': pi - nu,
        'is_positive_definite': pi == n,
        'is_negative_definite': nu == n,
        'is_positive_semidefinite': nu == 0 and zeta >= 0 and pi > 0,
        'is_negative_semidefinite': pi == 0 and zeta >= 0 and nu > 0,
        'is_indefinite': pi > 0 and nu > 0,
        'eigenvalues': eigenvals,
        'used_symmetric_part': used_sym_part,
    }

    return info


def print_inertia_report(info, logger=None):
    """
    格式化输出惯性指数分析报告。

    Parameters
    ----------
    info : dict
        analyze_inertia 的返回值。
    logger : logging.Logger, optional
        日志记录器。
    """
    out = logger.info if logger else print
    out("\n" + "=" * 60)
    out("矩阵惯性指数分析报告")
    out("=" * 60)
    out(f"  正惯性指数 (π):   {info['positive_inertia']}")
    out(f"  负惯性指数 (ν):   {info['negative_inertia']}")
    out(f"  零惯性指数 (ζ):   {info['zero_inertia']}")
    out(f"  签名 (π-ν):       {info['signature']}")
    out(f"  是否正定:         {info['is_positive_definite']}")
    out(f"  是否负定:         {info['is_negative_definite']}")
    out(f"  是否半正定:       {info['is_positive_semidefinite']}")
    out(f"  是否半负定:       {info['is_negative_semidefinite']}")
    out(f"  是否不定:         {info['is_indefinite']}")
    if info['used_symmetric_part']:
        out("  [注意] 输入矩阵非对称，已自动使用对称部分分析")
    ev = info['eigenvalues']
    out(f"  特征值范围:       [{ev[0]:.6e}, {ev[-1]:.6e}]")
    out("=" * 60)
