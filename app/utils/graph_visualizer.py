import networkx as nx
import matplotlib.pyplot as plt


def plot_digraph(
    G: nx.DiGraph,
    figsize=(5, 4),
    layout="shell",
    seed=42,
):
    """
    Create a matplotlib visualization of a NetworkX DiGraph.

    Parameters
    ----------
    G : nx.DiGraph
        Directed graph to visualize.
    figsize : tuple
        Figure size in inches.
    layout : str
        One of: "spring", "kamada_kawai", "circular", "shell", "planar".
    seed : int
        Random seed for reproducible layouts.

    Returns
    -------
    matplotlib.figure.Figure
    """

    layouts = {
        "spring": lambda g: nx.spring_layout(g, seed=seed),
        "kamada_kawai": nx.kamada_kawai_layout,
        "circular": nx.circular_layout,
        "shell": nx.shell_layout,
        "planar": nx.planar_layout,
    }

    if layout not in layouts:
        raise ValueError(f"Unknown layout '{layout}'")

    pos = layouts[layout](G)

    fig, ax = plt.subplots(figsize=figsize)

    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_size=800,
        node_color="pink",
    )

    nx.draw_networkx_labels(
        G,
        pos,
        ax=ax,
        font_size=10,
    )

    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=22,
        connectionstyle="arc3,rad=0.1",  # separates opposite edges
    )

    ax.set_axis_off()
    fig.tight_layout()

    return fig
