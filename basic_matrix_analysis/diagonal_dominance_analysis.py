import numpy as np


def analyze_diagonal_dominance(A, tol=1e-10, logger=None):
    """
    对角占优性分析。

    检查：
    - 严格行对角占优: |a_ii| > sum_{j≠i} |a_ij| for all i
    - 弱行对角占优:   |a_ii| >= sum_{j≠i} |a_ij| for all i, 且至少一个严格
    - 不可约对角占优: 矩阵不可约且弱对角占优，且至少一个严格
    - 列对角占优类似。

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
        包含多种对角占优判断结果和逐行/列详细数据。
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError("对角占优分析仅适用于方阵")

    n = A.shape[0]
    abs_A = np.abs(A)
    diag_abs = np.abs(np.diag(A))

    # 行对角占优
    row_sums = abs_A.sum(axis=1) - diag_abs
    row_strict = diag_abs > row_sums + tol
    row_weak = diag_abs >= row_sums - tol
    row_strict_count = int(np.sum(row_strict))
    row_weak_count = int(np.sum(row_weak))
    row_violation_count = int(np.sum(~row_weak))

    # 列对角占优
    col_sums = abs_A.sum(axis=0) - diag_abs
    col_strict = diag_abs > col_sums + tol
    col_weak = diag_abs >= col_sums - tol
    col_strict_count = int(np.sum(col_strict))
    col_weak_count = int(np.sum(col_weak))
    col_violation_count = int(np.sum(~col_weak))

    info = {
        # 行对角占优
        'row_strictly_diag_dominant': bool(np.all(row_strict)),
        'row_weakly_diag_dominant': bool(np.all(row_weak)) and row_strict_count > 0,
        'row_diag_dominance_ratios': diag_abs / (row_sums + tol),
        'row_violation_indices': np.where(~row_weak)[0].tolist(),
        'row_strict_count': row_strict_count,
        'row_weak_count': row_weak_count,
        'row_violation_count': row_violation_count,

        # 列对角占优
        'col_strictly_diag_dominant': bool(np.all(col_strict)),
        'col_weakly_diag_dominant': bool(np.all(col_weak)) and col_strict_count > 0,
        'col_diag_dominance_ratios': diag_abs / (col_sums + tol),
        'col_violation_indices': np.where(~col_weak)[0].tolist(),
        'col_strict_count': col_strict_count,
        'col_weak_count': col_weak_count,
        'col_violation_count': col_violation_count,

        # 对有对角线元素为零的行/列额外标记
        'zero_diag_indices': np.where(diag_abs < tol)[0].tolist(),
    }

    # 不可约性检查（仅当弱对角占优时才继续）
    if info['row_weakly_diag_dominant'] or info['col_weakly_diag_dominant']:
        info['is_irreducible'] = _is_irreducible(A, tol)
        info['is_irreducibly_diag_dominant'] = (
            info['is_irreducible'] and
            (info['row_weakly_diag_dominant'] or info['col_weakly_diag_dominant'])
        )
    else:
        info['is_irreducible'] = None
        info['is_irreducibly_diag_dominant'] = False

    return info


def _is_irreducible(A, tol=1e-10):
    """检查矩阵是否不可约（基于强连通有向图）。"""
    n = A.shape[0]
    adj = np.abs(A) > tol
    visited = np.zeros(n, dtype=bool)
    stack = [0]
    visited[0] = True
    while stack:
        u = stack.pop()
        for v in range(n):
            if adj[u, v] and not visited[v]:
                visited[v] = True
                stack.append(v)
    if not np.all(visited):
        return False

    adj_t = adj.T
    visited = np.zeros(n, dtype=bool)
    stack = [0]
    visited[0] = True
    while stack:
        u = stack.pop()
        for v in range(n):
            if adj_t[u, v] and not visited[v]:
                visited[v] = True
                stack.append(v)
    return bool(np.all(visited))


def print_diagonal_dominance_report(info, logger=None):
    """
    格式化输出对角占优性分析报告。

    Parameters
    ----------
    info : dict
        analyze_diagonal_dominance 的返回值。
    logger : logging.Logger, optional
        日志记录器。
    """
    out = logger.info if logger else print
    out("\n" + "=" * 60)
    out("对角占优性分析报告")
    out("=" * 60)

    out("\n[行对角占优]")
    out(f"  严格行对角占优:   {info['row_strictly_diag_dominant']}")
    out(f"  弱行对角占优:     {info['row_weakly_diag_dominant']}")
    out(f"  严格占优行数:     {info['row_strict_count']}")
    out(f"  弱占优行数:       {info['row_weak_count']}")
    out(f"  违反行数:         {info['row_violation_count']}")
    if info['row_violation_indices']:
        out(f"  违反行索引:       {info['row_violation_indices'][:10]}{' ...' if len(info['row_violation_indices']) > 10 else ''}")

    out("\n[列对角占优]")
    out(f"  严格列对角占优:   {info['col_strictly_diag_dominant']}")
    out(f"  弱列对角占优:     {info['col_weakly_diag_dominant']}")
    out(f"  严格占优列数:     {info['col_strict_count']}")
    out(f"  弱占优列数:       {info['col_weak_count']}")
    out(f"  违反列数:         {info['col_violation_count']}")
    if info['col_violation_indices']:
        out(f"  违反列索引:       {info['col_violation_indices'][:10]}{' ...' if len(info['col_violation_indices']) > 10 else ''}")

    out("\n[其他]")
    out(f"  不可约对角占优:   {info['is_irreducibly_diag_dominant']}")
    if info['zero_diag_indices']:
        out(f"  零对角元索引:     {info['zero_diag_indices'][:10]}{' ...' if len(info['zero_diag_indices']) > 10 else ''}")
    else:
        out("  零对角元:         无")
    out("=" * 60)
