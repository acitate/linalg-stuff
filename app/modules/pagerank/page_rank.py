import networkx as nx
import numpy as np


def get_transition_matrix(graph: nx.DiGraph) -> tuple[list, int, np.ndarray]:
    """
    Returns (sorted list of pages, number of pages, transition matrix)
    """
    nodes = sorted(graph.nodes())
    adjacency_matrix = nx.to_numpy_array(graph, nodelist=nodes, dtype=float)
    transition_matrix = list()

    for row in adjacency_matrix:
        outbound = row.sum()
        transition_row = (
            (row / outbound) if outbound != 0 else (np.ones(row.shape) / len(row))
        )
        transition_matrix.append(transition_row)

    return nodes, len(nodes), np.array(transition_matrix, dtype=float).transpose()


def page_rank(
    graph: nx.DiGraph, damping_factor: float, tolerance: float
) -> dict[str, float]:

    pages, page_count, transition_matrix = get_transition_matrix(graph)
    ones = np.ones(page_count, dtype=float)
    rank = ones / page_count

    for _ in range(100):
        old_rank = rank.copy()
        rank = (
            damping_factor * transition_matrix @ rank
            + ((1 - damping_factor) / page_count) * ones
        )

        if np.linalg.norm(rank - old_rank, 1) < tolerance:
            break

    ranks = dict(zip(pages, np.round(rank, 5).tolist()))

    return ranks