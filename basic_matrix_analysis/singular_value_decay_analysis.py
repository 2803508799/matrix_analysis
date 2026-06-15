import numpy as np


def analyze_singular_value_decay(A, tol=1e-10, logger=None):
    """
    奇异值衰减分析。

    分析奇异值的衰减速度，用于判断矩阵是否适合低秩近似，
    以及是否存在快速衰减的低维结构。

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
        - 'singular_values'        : 全部奇异值（降序）
        - 'normalized_sv'          : s / s_max
        - 'cumulative_energy'      : 累计能量比 Σ_{i≤k} s_i² / Σ s_i²
        - 'decay_rates'            : s_{i+1} / s_i 逐项衰减比
        - 'ranks_at_thresholds'    : 不同能量阈值下的有效秩
        - 'approximate_rank_90'    : 90% 能量所需的秩
        - 'approximate_rank_95'    : 95% 能量所需的秩
        - 'approximate_rank_99'    : 99% 能量所需的秩
        - 'approximate_rank_999'   : 99.9% 能量所需的秩
        - 'sv_half_life'           : 奇异值衰减到一半时的索引
        - 'tail_weight'            : 最小 10% 奇异值占总能量的比例
        - 'spectral_norm'          : 最大奇异值
        - 'nuclear_norm'           : 奇异值之和
        - 'effective_rank'         : exp(-Σ p_i log p_i), p_i = s_i / Σ s_j
    """
    s = np.linalg.svd(A, compute_uv=False)
    s_sorted = s[s > tol] if len(s[s > tol]) > 0 else s[:1]

    n_sv = len(s_sorted)
    s1 = s_sorted[0]
    normalized = s_sorted / s1 if s1 > tol else np.zeros_like(s_sorted)

    sq_sum = np.sum(s_sorted ** 2)
    cum_energy = np.cumsum(s_sorted ** 2) / sq_sum if sq_sum > tol else np.ones_like(s_sorted)

    # 各阈值下的有效秩
    ranks = {}
    for threshold in [0.5, 0.8, 0.9, 0.95, 0.99, 0.999]:
        ranks[f'rank_{int(threshold*100)}'] = int(np.searchsorted(cum_energy, threshold) + 1)

    # 半衰期
    half_life_idx = np.searchsorted(normalized[::-1], 0.5)
    half_life = max(1, n_sv - half_life_idx)

    # 尾部权重
    tail_start = max(0, int(n_sv * 0.9))
    tail_weight = (np.sum(s_sorted[tail_start:] ** 2) / sq_sum) if sq_sum > tol and tail_start < n_sv else 0.0

    # 有效秩（基于熵）
    p = s_sorted / (np.sum(s_sorted) + tol)
    p = p[p > tol]
    entropy = -np.sum(p * np.log(p))
    effective_rank = int(np.ceil(np.exp(entropy))) if entropy > 0 else 1

    # 衰减率
    decay_rates = s_sorted[1:] / (s_sorted[:-1] + tol)

    info = {
        'singular_values': s_sorted,
        'normalized_sv': normalized,
        'cumulative_energy': cum_energy,
        'decay_rates': decay_rates,
        'ranks_at_thresholds': ranks,
        'approximate_rank_90': ranks.get('rank_90', n_sv),
        'approximate_rank_95': ranks.get('rank_95', n_sv),
        'approximate_rank_99': ranks.get('rank_99', n_sv),
        'approximate_rank_999': ranks.get('rank_999', n_sv),
        'sv_half_life': half_life,
        'tail_weight': tail_weight,
        'spectral_norm': float(s1),
        'nuclear_norm': float(np.sum(s_sorted)),
        'effective_rank': effective_rank,
        'total_sv_count': len(s),
        'nonzero_sv_count': n_sv,
    }

    return info


def print_sv_decay_report(info, logger=None):
    """
    格式化输出奇异值衰减分析报告。

    Parameters
    ----------
    info : dict
        analyze_singular_value_decay 的返回值。
    logger : logging.Logger, optional
        日志记录器。
    """
    out = logger.info if logger else print
    out("\n" + "=" * 60)
    out("奇异值衰减分析报告")
    out("=" * 60)
    out(f"  总奇异值数:        {info['total_sv_count']}")
    out(f"  非零奇异值数:      {info['nonzero_sv_count']}")
    out(f"  谱范数:            {info['spectral_norm']:.6e}")
    out(f"  核范数:            {info['nuclear_norm']:.6e}")
    out(f"  有效秩 (熵):       {info['effective_rank']}")
    out(f"  衰减半衰期索引:    {info['sv_half_life']}")
    out(f"  尾部能量占比:      {info['tail_weight']:.6e}")
    out(f"  90% 能量所需秩:    {info['approximate_rank_90']}")
    out(f"  95% 能量所需秩:    {info['approximate_rank_95']}")
    out(f"  99% 能量所需秩:    {info['approximate_rank_99']}")
    out(f"  99.9% 能量所需秩:  {info['approximate_rank_999']}")
    if len(info['decay_rates']) > 0:
        out(f"  最大衰减率:        {np.max(info['decay_rates']):.6f}")
        out(f"  最小衰减率:        {np.min(info['decay_rates']):.6f}")
        out(f"  平均衰减率:        {np.mean(info['decay_rates']):.6f}")
    out("=" * 60)
