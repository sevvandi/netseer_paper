"""Prediction Module:"""

from collections.abc import Iterable

import igraph as ig

import netseer.functions as functions


def predict_graph(
    graph_list: Iterable[ig.Graph],
    h: int = 5,
    conf_nodes=None,
    conf_degree: int = 90,
    weights_option: int = 4,
    weights_param=0.001,
) -> ig.Graph:
    """From a list of graphs, predicts a new graph

    Args:
        graph_list: A list of graphs to be predicted on: See generate_graph_list()
        h: How many steps into the future the prediction should be.
        conf_nodes: -- Not Implemented.
        conf_degree: -- Not Implemented.
        weights_option: Int between 1-7 determines edge weight schemes.
            1: Uniform Weight. 1 for all Edges.
            2: Binary Weight. 1 for all Edges.
            3: Binary Weight. 1 for most connected vertices.
            4: Proportional Weights according to history.
            5: Proportional Weight. Linear scaling weights for new edges based on time series (Time Step Index+ 1 ). E.g. 1, 2, 3
            6: Proportional Weights. Decaying weights based on time series. (1/(Number of graphs - Time Step Index + 1)) E.g. 1, 1/2, 1/3
            7: Weight 0 for all edges, except for last edge which is 1.
            8: Not Implemented.
        weights_param: The weight given for possible edges from new vertices. Default is 0.001

    Returns:
        Predicted Graph.
    """
    if not 1 <= weights_option <= 7:
        raise ValueError("weights_option must be between 1 and 7.")

    print("Forecasting properties of the new graph")
    # predict the number of nodes for the predicted graph
    new_nodes, new_nodes_lower, new_nodes_upper = functions.predict_num_nodes(
        graph_list, h=h, conf_level=conf_nodes
    )
    # predict the degrees for the new nodes
    new_nodes_degrees = functions.predict_new_nodes_degrees(graph_list)

    # predict the degrees of the existing nodes
    existing_nodes_degrees = functions.predict_existing_nodes_degrees(
        graph_list, h=h, conf_level=conf_degree
    )

    total_edges = functions.predict_total_edges_counts(
        graph_list, h=h, conf_level=conf_degree
    )

    EDGES_CONF_LEVEL = "conf_high"  # "mean"
    print("Using forecasts to predict the new graph")
    mean_graph = functions.predict_graph_from_forecasts(
        graph_list,
        new_nodes,
        existing_nodes_degrees[EDGES_CONF_LEVEL],
        new_nodes_degrees,
        total_edges[EDGES_CONF_LEVEL],
        conf_nodes,
        conf_degree,
        weights_option,
        weights_param,
    )

    print("Prediction Completed")
    return mean_graph
