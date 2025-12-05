"""Prediction Module:

Typical Usage Example:
    ``` Python
    from netseer import utils, generate_graph, prediction
    from pathlib import Path

    # Get the directory/folder with all the graphs.
    directory_path = Path.cwd() / Path("path/to/data")
    # Use glob to get a list of absolute paths for .gml files.
    graph_files = directory_path.glob("*.gml"))

    # Load the graphs into igraph using the list of gml files.
    graph_list = utils.read_graph_list(graph_files)
    # Predict the next graph.
    predict = prediction.predict_graph(graph_list, h=1)
    ```

"""

from collections.abc import Iterable
from typing import Optional

import numpy as np
import igraph as ig
import pulp
import pulp.constants

import netseer.functions as functions


def predict_graph(
    graph_list: Iterable[ig.Graph],
    h: int = 5,
    conf_nodes=None,
    conf_degree: int = 90,
    weights_option: int = 4,
) -> ig.Graph:
    """From a list of graphs, predicts a new graph

    Args:
        graph_list: A list of graphs to be predicted on: See generate_graph_list()
        h: How many steps into the future the prediction should be.
        conf_nodes: -- TODO
        conf_degree: -- TODO
        weights_option: Int between 1-7 determines the weight of newly created edges.
            1-3: Weight 0 for all edges.
            4: Weight 1 for all edges.
            5: Linear scaling weights for new edges (x+1). E.g. 1, 2, 3
            6: Proportional Weights.
            7: Weight 0 for all edges, except for last edge which is 1.

    Returns:
        Predicted Graph.
    """

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
    )

    print("done")
    return mean_graph
