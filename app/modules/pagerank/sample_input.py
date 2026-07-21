from networkx import DiGraph


DAMPING_FACTOR = 0.85
TOLERANCE = 1e-6
GRAPH = DiGraph()
graph_data = [
    ("1", "2"),
    ("1", "4"),
    ("2", "4"),
    ("3", "1"),
    ("2", "5"),
    ("5", "6"),
    ("6", "4"),
    ("3", "5"),
    ("5", "4"),
    ("2", "1"),
    ("6", "2"),
    ("1", "5"),
    ("3", "4"),
]
GRAPH.add_edges_from(graph_data)