# -*- coding: utf-8 -*-
"""
矩阵分析主程序：
  1. 从 CSV 文件导入矩阵数据
  2. 对矩阵执行基础分析（秩、对称性、对角占优、惯性指数、M矩阵、
     奇异值衰减、奇异值分析、病态分析、稀疏性分析）和连通性分析
  3. 所有结果写入日志
"""

import numpy as np

# 必须在导入 logger 之前调用，否则 logger 会以默认路径自动初始化并锁定
from log.my_log import setup_log_path
setup_log_path(log_dir="logs")

from log.my_log import logger
from file_operation.read_matrix import load_matrix_from_csv
from basic_matrix_analysis import (
    analyze_rank, print_rank_report,
    analyze_symmetry, print_symmetry_report,
    analyze_diagonal_dominance, print_diagonal_dominance_report,
    analyze_inertia, print_inertia_report,
    analyze_m_matrix, print_m_matrix_report,
    analyze_singular_value_decay, print_sv_decay_report,
    analyze_singular_values, print_singular_value_report,
    analyze_condition, print_condition_report,
    analyze_sparsity, print_sparsity_report,
)
from matrix_topology_analysis.connectivity_analysis import (
    analyze_matrix_connectivity, print_connectivity_report,
)

# ============================================================
# 用户配置区域 —— 请根据实际 CSV 文件结构调整
# ============================================================
CSV_PATH = "data/matrix_2.csv"          # CSV 文件路径

# 列索引（0-based）：节点名称列、矩阵起始列、结束列、源向量列
NAME_COL = 0                          # 第0列为节点名称，设为 None 表示无名称列
MATRIX_START_COL = 1                  # 矩阵从第1列开始
MATRIX_END_COL = -2                   # 矩阵到倒数第2列（不含最后一列向量）
VECTOR_COL = -1                       # 最后一列为源向量，设为 None 表示无源向量
SKIP_ROWS = 1                         # 跳过第0行（信息行）

# 数值容差
TOL = 1e-10
# ============================================================


def main():
    # ---- 1. 初始化日志（已在模块导入时完成） ----
    logger.info("=" * 60)
    logger.info("矩阵分析程序启动")
    logger.info("=" * 60)

    # ---- 2. 导入矩阵数据 ----
    logger.info(f"正在读取矩阵文件: {CSV_PATH}")
    A, b, node_names, n_rows, n_cols = load_matrix_from_csv(
        csv_path=CSV_PATH,
        name_col=NAME_COL,
        matrix_start_col=MATRIX_START_COL,
        matrix_end_col=MATRIX_END_COL,
        vector_col=VECTOR_COL,
        skip_rows=SKIP_ROWS,
        logger=logger,
    )
    logger.info(f"矩阵规模: {n_rows} 行 × {n_cols} 列")
    if b is not None:
        logger.info(f"源向量: 长度 {len(b)}, 非零元 {int(np.sum(np.abs(b) > TOL))}")
    if node_names is not None:
        logger.info(f"节点名称: {len(node_names)} 个")

    # ---- 3. 基础矩阵分析 ----
    logger.info("\n" + "=" * 60)
    logger.info("开始基础矩阵分析")
    logger.info("=" * 60)

    # 3.1 秩分析
    logger.info("[1/9] 秩分析 ...")
    rank_info = analyze_rank(A, tol=TOL, logger=logger)
    print_rank_report(rank_info, logger=logger)

    # 3.2 对称性分析
    logger.info("[2/9] 对称性分析 ...")
    sym_info = analyze_symmetry(A, tol=TOL, logger=logger)
    print_symmetry_report(sym_info, logger=logger)

    # 3.3 对角占优性分析
    logger.info("[3/9] 对角占优性分析 ...")
    dd_info = analyze_diagonal_dominance(A, tol=TOL, logger=logger)
    print_diagonal_dominance_report(dd_info, logger=logger)

    # 3.4 惯性指数
    logger.info("[4/9] 惯性指数分析 ...")
    inertia_info = analyze_inertia(A, tol=TOL, logger=logger)
    print_inertia_report(inertia_info, logger=logger)

    # 3.5 M 矩阵性质检查
    logger.info("[5/9] M 矩阵性质检查 ...")
    m_info = analyze_m_matrix(A, tol=TOL, logger=logger)
    print_m_matrix_report(m_info, logger=logger)

    # 3.6 奇异值衰减分析
    logger.info("[6/9] 奇异值衰减分析 ...")
    sv_decay_info = analyze_singular_value_decay(A, tol=TOL, logger=logger)
    print_sv_decay_report(sv_decay_info, logger=logger)

    # 3.7 奇异值分析
    logger.info("[7/9] 奇异值分析 ...")
    sv_info = analyze_singular_values(A, tol=TOL, logger=logger)
    print_singular_value_report(sv_info, logger=logger)

    # 3.8 病态分析
    logger.info("[8/9] 病态分析 ...")
    cond_info = analyze_condition(A, tol=TOL, logger=logger)
    print_condition_report(cond_info, logger=logger)

    # 3.9 稀疏性分析
    logger.info("[9/9] 稀疏性分析 ...")
    sparsity_info = analyze_sparsity(A, tol=TOL, logger=logger)
    print_sparsity_report(sparsity_info, logger=logger)

    # ---- 4. 连通性分析 ----
    logger.info("\n" + "=" * 60)
    logger.info("开始连通性分析")
    logger.info("=" * 60)
    labels, components_info = analyze_matrix_connectivity(
        A, node_names=node_names, source_vector=b, tol=TOL,
    )
    print_connectivity_report(labels, components_info, node_names=node_names, tol=TOL, logger=logger)

    # ---- 5. 完成 ----
    logger.info("\n" + "=" * 60)
    logger.info("矩阵分析完成")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
