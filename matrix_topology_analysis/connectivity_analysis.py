import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

def analyze_matrix_connectivity(A, node_names=None, source_vector=None, tol=1e-10):
    """
    分析矩阵的连通性（识别独立热岛）

    Parameters
    ----------
    A : ndarray (n, n)
        系统矩阵（如热导矩阵），需为对称或可对称化，非零元素表示连接。
    node_names : list of str, optional
        节点名称列表，长度为 n，用于返回结果中的可读信息。
    source_vector : ndarray (n,), optional
        源项向量（如热源），用于统计各分量内的热源分布。
    tol : float
        判断矩阵元素非零的阈值，绝对值大于 tol 视为存在连接。

    Returns
    -------
    labels : ndarray (n,)
        每个节点所属的连通分量编号（从 0 开始）。
    components_info : list of dict
        每个分量的详细信息，字典包含：
            - 'id'       : 分量编号
            - 'size'     : 节点数量
            - 'nodes'    : 该分量包含的节点索引列表
            - 'sources'  : 仅当 source_vector 提供时存在，
                           为 (节点索引, 节点名, 源值) 的列表
    """
    n = A.shape[0]

    # 构建布尔邻接矩阵（忽略主对角线）
    adjacency = np.abs(A) > tol
    np.fill_diagonal(adjacency, False)

    # 计算无向连通分量
    n_components, labels = connected_components(
        csr_matrix(adjacency), directed=False
    )

    # 整理每个分量的信息
    components_info = []
    for comp_id in range(n_components):
        node_indices = np.where(labels == comp_id)[0]
        info = {
            'id': comp_id,
            'size': len(node_indices),
            'nodes': node_indices.tolist(),
        }

        if source_vector is not None:
            sources = []
            for idx in node_indices:
                if np.abs(source_vector[idx]) > tol:
                    name = node_names[idx] if node_names is not None else f"node_{idx}"
                    sources.append((idx, name, source_vector[idx]))
            info['sources'] = sources

        components_info.append(info)

    return labels, components_info


def print_connectivity_report(labels, components_info, node_names=None, tol=1e-10, logger=None):
    """
    格式化输出矩阵连通性分析报告

    Parameters
    ----------
    labels : ndarray (n,)
        每个节点的分量标签（由 analyze_matrix_connectivity 返回）。
    components_info : list of dict
        每个分量的详细信息，格式同 analyze_matrix_connectivity 的返回值。
    node_names : list of str, optional
        节点名称，用于可读性输出。
    tol : float
        判断源项非零的阈值，与 analyze_matrix_connectivity 保持一致。
    logger : logging.Logger, optional
        若提供日志记录器，报告将写入日志（INFO 级别）；否则打印到控制台。
    """
    n_components = len(components_info)

    # 根据是否提供 logger 选择输出方式
    out = logger.info if logger else print

    out("\n" + "=" * 80)
    out("矩阵连通性分析报告")
    out("=" * 80)
    out(f"独立连通分量总数: {n_components}")

    for comp in components_info:
        comp_id = comp['id']
        size = comp['size']
        out(f"\n分量 {comp_id}: 包含 {size} 个节点")

        # 展示部分节点名称
        if node_names is not None:
            nodes = comp['nodes']
            show_ids = nodes[:5]
            name_list = [str(node_names[i]) for i in show_ids]
            if len(nodes) > 5:
                name_list.append(f"... (共{len(nodes)}个)")
            out(f"  节点示例: {', '.join(name_list)}")

        # 显示热源信息
        sources = comp.get('sources', [])
        if sources:
            out(f"  内部热源数量: {len(sources)}")
            for idx, name, val in sources[:5]:  # 最多显示5个
                out(f"    - {name}: {val:.4e} W")
            if len(sources) > 5:
                out(f"    ... 以及其他 {len(sources)-5} 个热源")
        else:
            out("  内部热源: 无")

    out("\n" + "=" * 80)
