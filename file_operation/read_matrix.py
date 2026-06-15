import pandas as pd
import numpy as np

def load_matrix_from_csv(
        csv_path,
        name_col=1,
        matrix_start_col=2,
        matrix_end_col=-2,
        vector_col=-1,
        skip_rows=0,
        verbose=True,
        logger=None          # 新增参数：用于日志输出
):
    """
    从CSV文件读取矩阵、节点名称与源向量，并返回矩阵的行列数。

    Parameters
    ----------
    csv_path : str
        CSV文件路径。
    name_col : int or None
        节点名称所在列的索引（0‑based），设为 None 则不读取名称。
    matrix_start_col : int
        矩阵起始列索引（包含）。
    matrix_end_col : int
        矩阵结束列索引（包含），负数表示从最后列倒数。
    vector_col : int or None
        源向量所在列索引，None 表示无向量。
    skip_rows : int
        文件开头跳过的行数（如标题行）。
    verbose : bool
        是否输出信息（若为 True，则根据 logger 决定输出到日志或屏幕）。
    logger : logging.Logger or None
        若提供，则使用 logger.info 输出；否则使用 print。

    Returns
    -------
    A : ndarray (n, n)
        提取的二维方阵。
    b : ndarray (n,) or None
        源向量，若 vector_col=None 则返回 None。
    node_names : list of str or None
        节点名称列表，若 name_col=None 则返回 None。
    n_rows : int
        矩阵行数。
    n_cols : int
        矩阵列数。
    """
    df = pd.read_csv(csv_path, skiprows=skip_rows, header=None)

    # 提取节点名称（可选）
    if name_col is not None:
        node_names = df.iloc[:, name_col].astype(str).tolist()
    else:
        node_names = None

    # 提取矩阵A
    total_cols = df.shape[1]
    start = matrix_start_col
    end = matrix_end_col if matrix_end_col >= 0 else total_cols + matrix_end_col + 1
    A = df.iloc[:, start:end].values.astype(np.float64)

    n_rows, n_cols = A.shape

    # 提取源向量b
    if vector_col is not None:
        col_idx = vector_col if vector_col >= 0 else total_cols + vector_col
        b = df.iloc[:, col_idx].values.astype(np.float64)
    else:
        b = None

    # 输出信息（可选）
    if verbose:
        out = logger.info if logger else print
        out(f"读取矩阵: {n_rows} 行 × {n_cols} 列")
        if b is not None:
            out(f"源向量长度: {len(b)}")
        if node_names is not None:
            out(f"节点名称数量: {len(node_names)}")

    return A, b, node_names, n_rows, n_cols