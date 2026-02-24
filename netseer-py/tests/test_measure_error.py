import igraph as ig
from netseer.measure_error import measure_error


def sample_graph_1():
    return ig.Graph(n=3, edges=[[0, 1], [1, 2]])


def sample_graph_2():
    return ig.Graph(n=6, edges=[[0, 1]])


def test_measure_error_1():
    graph_actual = sample_graph_1()
    graph_predicted = sample_graph_1()

    vertex_error, edge_error = measure_error(graph_actual, graph_predicted)
    assert vertex_error == 0.0
    assert edge_error == 0.0


def test_measure_error_2():
    graph_actual = sample_graph_1()
    graph_predicted = sample_graph_2()

    vertex_error, edge_error = measure_error(graph_actual, graph_predicted)
    assert vertex_error == 1.0
    assert edge_error == 0.5
