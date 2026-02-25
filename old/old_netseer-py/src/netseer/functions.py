"""Internal: Supplimentary functions for predictions."""

from collections.abc import Iterable
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels
import statsmodels.api as sm
import statsmodels.tools.tools
import statsforecast as sf
import statsforecast.models as sfm

import pulp
import pulp.constants

import igraph.operators.functions as igf
import igraph as ig


WEIGHTS = "weights"


def predict_num_nodes(
    graph_list: Iterable[ig.Graph],
    h: int = 5,
    conf_level: Optional[int] = None,
    conf_lower=90,
):
    num_graphs = len(graph_list)
    node_counts = np.array([gr.vcount() for gr in graph_list])
    print(f"{node_counts=}")
    conf_lower = None
    conf_upper = None
    if num_graphs < 10:
        # r: number of nodes = m * time step
        # python: number of nodes = m * time stop + c
        exogenous = statsmodels.tools.tools.add_constant(
            np.arange(num_graphs).reshape((num_graphs, 1))
        )
        linear_model = sm.OLS(node_counts, exog=exogenous).fit()
        prediction = linear_model.predict(exog=[1, h + num_graphs])
        print(f"short node count prediction: {prediction}")
    else:
        model = sfm.AutoETS()
        model.fit(y=node_counts, X=range(num_graphs))
        prediction = model.predict(
            h=h, level=[conf_level] if conf_level is not None else None
        )
        if conf_level is not None:
            conf_lower = prediction[f"lo-{conf_level}"]
            conf_upper = prediction[f"hi-{conf_level}"]
        print(f"longer node count prediction: {prediction}")
        prediction = np.ceil(prediction["mean"]).astype(int)
    # account for the possibility that the number of nodes may decrease
    new_nodes = np.hstack(([prediction[0] - node_counts[-1]], np.diff(prediction)))
    removed_nodes = 0
    for i in range(h):
        if new_nodes[i] < 0:
            removed_nodes += np.abs(new_nodes[i])
            new_nodes[i] = 0
        if removed_nodes <= 0:
            continue
        this_removed_nodes = min(removed_nodes, new_nodes[i])
        removed_nodes -= this_removed_nodes
        new_nodes[i] -= this_removed_nodes
    if removed_nodes > 0:
        new_nodes[-1] -= removed_nodes
        if conf_lower is not None:
            conf_lower[-1] -= removed_nodes
            conf_upper[-1] -= removed_nodes
    # return results
    new_nodes = new_nodes[-1]
    conf_lower = conf_lower[-1] if conf_lower is not None else None
    conf_upper = conf_upper[-1] if conf_upper is not None else None
    return (
        new_nodes,
        conf_lower,
        conf_upper,
    )  # {"new_nodes":, "conf_lower":, "conf_upper":}


def predict_existing_nodes_degrees(
    graph_list: Iterable[ig.Graph], h: int = 5, conf_level: Optional[int] = None
):
    num_graphs = len(graph_list)
    num_nodes = [gr.vcount() for gr in graph_list]
    total_nodes = sum(num_nodes)
    data = np.ndarray((total_nodes, 3), dtype=int)
    data_start = 0
    for graph_idx, gr in enumerate(graph_list):
        this_end = data_start + num_nodes[graph_idx]
        data[data_start:this_end, 0] = range(num_nodes[graph_idx])  # node index
        data[data_start:this_end, 1] = graph_idx  # time step
        data[data_start:this_end, 2] = gr.degree()  # degree of each node
        data_start = this_end

    df = pd.DataFrame(data=data, columns=["node_index", "time_step", "degree"])
    model = sf.StatsForecast(models=[sfm.AutoARIMA()], freq=1)
    model_str = "AutoARIMA"
    rename_dict = {model_str: "mean"}
    if conf_level is not None:
        rename_dict.update(
            {
                f"{model_str}-lo-{conf_level}": "conf_low",
                f"{model_str}-hi-{conf_level}": "conf_high",
            }
        )
    prediction = model.fit_predict(
        df=df,
        id_col="node_index",
        time_col="time_step",
        target_col="degree",
        h=h,
        level=[conf_level],
    )
    print(f"first degree prediction:\n{prediction}")
    print(type(prediction))
    print(f"rename dict: {rename_dict}")
    prediction.rename(columns=rename_dict, inplace=True)
    prediction = prediction[prediction["time_step"] == (num_graphs + h - 1)]
    for prediction_name in rename_dict.values():
        prediction[prediction_name] = np.ceil(prediction[prediction_name]).astype(int)
    # print(f"degree prediction:\n{prediction}")
    return prediction  #


def predict_total_edges_counts(
    graph_list: Iterable[ig.Graph], h: int = 5, conf_level: Optional[int] = None
):
    # predict the total number of edges in the final graph
    model = sfm.AutoARIMA()
    total_edges = np.array([gr.ecount() for gr in graph_list])
    time_steps = np.arange(len(graph_list))
    level = [conf_level] if conf_level is not None else None
    prediction = model.forecast(y=total_edges, h=h, level=level)
    results = {"mean": np.ceil(prediction["mean"][-1]).astype(int)}
    if conf_level is not None:
        results.update(
            {
                "conf_low": np.ceil(prediction[f"lo-{conf_level}"].iloc[-1]).astype(
                    int
                ),
                "conf_high": np.ceil(prediction[f"hi-{conf_level}"].iloc[-1]).astype(
                    int
                ),
            }
        )
    return results


def predict_new_nodes_degrees(graph_list: Iterable[ig.Graph]):
    # predict the degrees for new nodes (nodes that are in the predicted graph but not the original graph list)
    mean_new_degrees = np.zeros((len(graph_list)), dtype=float)
    for graph_idx in range(1, len(graph_list)):
        last_node = graph_list[graph_idx - 1].vcount()
        this_new_nodes_degrees = graph_list[graph_idx].degree()[last_node:]
        this_mean_new_nodes = (
            np.mean(this_new_nodes_degrees) if len(this_new_nodes_degrees) > 0 else 0
        )
        mean_new_degrees[graph_idx] = this_mean_new_nodes
    model = sfm.AutoARIMA()
    prediction = model.forecast(y=mean_new_degrees, h=1)
    # print(f"new nodes degree = {int(np.ceil(prediction['mean'].item()))}")
    return int(np.ceil(prediction["mean"].item()))


def construct_union_graph(
    graph_list: list[ig.Graph],
    num_new_nodes: int,
    new_nodes_degrees,
    remove_vertices: Optional[int] = 0,
    weights_option: int = 2,
    weights_param=0.001,
    num_most_connected_nodes: int = 10,
):
    def weights_value(weights_option: int, index: int, num_graphs: int) -> float:
        # different options for assigning weights to each edge
        if weights_option <= 3:
            return 0
        if weights_option == 4:
            return 1
        if weights_option == 5:
            return index + 1
        if weights_option == 6:
            return 1 / (num_graphs - index)
        if weights_option == 7:
            return 1 if index == num_graphs - 1 else 0
        return 0

    num_graphs = len(graph_list)

    def this_weights_key(graph_idx):
        return f"{WEIGHTS}_{graph_idx}"

    # set the weights for each edge
    for graph_idx in range(num_graphs):
        graph_list[graph_idx].es[this_weights_key(graph_idx)] = weights_value(
            weights_option, graph_idx, num_graphs
        )
        graph_list[graph_idx].vs["name"] = list(range(graph_list[graph_idx].vcount()))
    # create the graph that is the union of every graph in the graphlist
    union_graph = igf.union(graph_list, byname=False)
    # combine the weights of the components of the union graph
    weights = np.zeros((union_graph.ecount(),), dtype=float)
    for graph_idx in range(num_graphs):
        this_weights_str = this_weights_key(graph_idx)
        this_weights = np.array(union_graph.es[this_weights_str], dtype=float)
        del union_graph.es[this_weights_str]
        np.nan_to_num(
            this_weights, copy=False, nan=0.0
        )  # this is for edges that exist somewhere in the union graph, but not in this graph
        if any(np.isnan(this_weights)):
            print(f"Error: nan weights for graph {graph_idx}")
        weights += this_weights
    weights /= np.max(weights)
    union_graph.es[WEIGHTS] = weights
    # TODO: Unused
    num_existing_edges = union_graph.ecount()

    # add potential new edges and their weights
    non_edges = get_neighbours_of_neighbours(union_graph)
    if weights_option in [4, 5, 6]:
        non_edges_weight = np.quantile(weights, q=weights_param)  #
    elif weights_option == 7:
        non_edges_weight = weights_param
    else:
        # TODO: Ask if typo - was nonedges_weight
        non_edges_weight = 0
    union_graph.add_edges(non_edges, {WEIGHTS: non_edges_weight})

    if num_new_nodes > 0:
        # Add new nodes
        num_existing_nodes = union_graph.vcount()

        existing_node_degrees = np.array(union_graph.degree(), dtype=int)
        # Use only a fixed number of old vertices
        # Which vertices have the highest degree
        if weights_option == 3:
            num_most_connected_nodes *= 2
        num_most_connected_nodes = min(num_most_connected_nodes, union_graph.vcount())
        most_connected_nodes = np.argpartition(
            existing_node_degrees, kth=-num_most_connected_nodes
        )[
            -num_most_connected_nodes:
        ]  # attach potential edges to this number of nodes in the union graph, with the highest degrees
        union_graph.add_vertices(num_new_nodes)
        new_nodes = union_graph.vs[num_existing_nodes:].indices

        new_nodes_edges_weights = 0  # []*len(new_node_edges) #weight for the new node
        new_node_edges = [
            (old_node, new_node)
            for old_node in most_connected_nodes
            for new_node in new_nodes
        ]
        if weights_option in [1, 2]:
            # Binary weights - new nodes connected to all old nodes
            pass
        elif weights_option == 3:
            # Binary weights - new nodes connected to most connected old nodes
            pass
        elif weights_option in [4, 5, 6, 7]:
            # Proportional weights - new nodes connected to all old nodes
            # But the weights will be much smaller
            existing_nodes_quantiles = existing_node_degrees / (
                sum(existing_node_degrees)
            )
            # TODO: Unused
            new_nodes_edges_weights = np.quantile(a=weights, q=existing_nodes_quantiles)

        union_graph.add_edges(
            new_node_edges, attributes={WEIGHTS: new_nodes_edges_weights}
        )  #

        # use uniform weights for weights options 1, 2, and 3
        if weights_option in (1, 2, 3):
            union_graph.es[WEIGHTS] = 1

    return union_graph


def setup_lp_solver(
    union_graph: ig.Graph, max_node_degrees: list[int], total_edges: int
) -> pulp.LpProblem:
    # create object to run the integer linear programming solver
    problem = pulp.LpProblem(name="edge_solver", sense=pulp.constants.LpMaximize)
    weights = union_graph.es[WEIGHTS]
    variables = [
        pulp.LpVariable(str(e_idx), cat="Binary") for e_idx in range(len(weights))
    ]
    problem.setObjective(pulp.LpAffineExpression(zip(variables, weights)))
    # degree constraints for each node
    for node_idx in range(union_graph.vcount()):
        this_edges = union_graph.es.select(_source=node_idx).indices
        this_expression = pulp.LpAffineExpression(
            [(variables[edge], 1) for edge in this_edges]
        )
        this_constraint = pulp.LpConstraint(
            e=this_expression,
            sense=pulp.LpConstraintLE,
            rhs=max_node_degrees[node_idx],
            name=str(node_idx),
        )
        problem.addConstraint(this_constraint)
    # total edges constraint
    total_edges_expression = pulp.LpAffineExpression((v, 1) for v in variables)
    total_edges_constraint = pulp.LpConstraint(
        e=total_edges_expression,
        sense=pulp.LpConstraintLE,
        rhs=total_edges,
        name="total_edges",
    )
    problem.addConstraint(total_edges_constraint)

    # print(problem)

    return problem


def run_lp_solver(problem: pulp.LpProblem) -> list[int]:
    # run the linear programming solver, and extract the solution
    # this argument stops the solver from printing output - the relevant command(s) will change depending on which solver is used
    status = problem.solve(pulp.PULP_CBC_CMD(msg=False))  # , gapAbs=0.0001
    if status != pulp.constants.LpSolutionOptimal:
        print(
            f"Error running solver - {status=} - {pulp.constants.LpSolution[status]}"
        )  #
    vars = problem.variables()
    solution_edge_idxs = [int(edge.name) for edge in vars if edge.varValue > 0]
    return solution_edge_idxs


def make_solution_graph(
    union_graph: ig.Graph, solution_edge_idxs: list[int]
) -> ig.Graph:
    # solution_edges = []
    return union_graph.subgraph_edges(edges=solution_edge_idxs, delete_vertices=False)


def predict_graph_from_forecasts(
    graph_list: list[ig.Graph],
    new_nodes: int,
    existing_nodes_degrees,
    new_nodes_degrees: int,
    total_edges,
    conf_nodes: Optional[int] = None,
    conf_degree: Optional[int] = None,
    weights_option: int = 1,
    weights_param=0.001,
) -> ig.Graph:
    # the maximum degree for each node
    max_node_degrees = None  # the maximim degree of each node in the solution graph
    if new_nodes > 0:
        num_existing_nodes = len(existing_nodes_degrees)
        max_node_degrees = np.ndarray((num_existing_nodes + new_nodes), dtype=int)
        max_node_degrees[0:num_existing_nodes] = existing_nodes_degrees
        max_node_degrees[num_existing_nodes:] = new_nodes_degrees
        pass
    else:
        max_node_degrees = existing_nodes_degrees

    # construction a graph that is the union of the graphs in the graph list
    print("Creating union graph for solver")
    union_graph = construct_union_graph(
        graph_list,
        new_nodes,
        new_nodes_degrees,
        weights_option=weights_option,
        weights_param=weights_param,
    )

    # setup linear programming solver
    problem = setup_lp_solver(union_graph, max_node_degrees, total_edges)

    # solve problem
    print("Running LP Solver")
    solution_edge_idxs = run_lp_solver(problem)

    # turn the solution into a graph
    solution_graph = make_solution_graph(union_graph, solution_edge_idxs)

    # check that the solution meets the constraints
    solution_ok = check_solution_against_constraints(
        solution_graph, union_graph, max_node_degrees, total_edges
    )
    print(f"Does the solution follow the constraints? {'yes' if solution_ok else 'no'}")

    # done
    return solution_graph


def get_neighbours_of_neighbours(gr: ig.Graph) -> np.ndarray:
    # get neighbours-of-neighours of a graph - edges where there's an edge from A to B and from B to C, but not A to C
    # returns an (n, 2) ndarray of integers that is a list of (source node, target node) pairs
    adj1 = gr.get_adjacency_sparse()
    adj2 = adj1 * adj1  # neighbours of neighbours
    non = np.vstack(
        np.nonzero((adj2 - adj1) > 0)
    ).T  # neighbours of neighbours, not including existing edges
    non = non[non[:, 0] < non[:, 1]]  # get only one triangle of edges

    return non


def check_solution_against_constraints(
    solution_graph: ig.Graph,
    union_graph: ig.Graph,
    max_node_degrees: list[int],
    total_edges_constraint: int,
) -> bool:
    all_ok = True
    # check total number of edges
    num_solution_edges = solution_graph.ecount()
    if num_solution_edges > int(total_edges_constraint):
        print(
            f"num solution edges = {num_solution_edges}, total_edges constraint = {total_edges_constraint}, union_graph edges= {union_graph.ecount()}"
        )
        all_ok = False

    # check the degree constraints of each node
    over_degree_capacity = [
        n
        for n in range(solution_graph.vcount())
        if solution_graph.degree()[n] > int(max_node_degrees[n])
    ]
    if len(over_degree_capacity) > 0:
        for n in over_degree_capacity:
            print(
                f"degree of Node {n}: {solution_graph.degree()[n]} is larger than its constraint: {int(max_node_degrees[n])}"
            )
            all_ok = False
            continue

    if num_solution_edges >= int(total_edges_constraint):
        return all_ok

    # if not all edges were used, check for any edges that could have been added to the graph
    print(f"Nodes with spare degrees:")
    spare_degrees = []
    for n in range(union_graph.vcount()):
        solution_degree = solution_graph.degree()[n]
        degree_constraint = int(max_node_degrees[n])
        if solution_degree == degree_constraint:
            continue
        for e in union_graph.es(_source=n):
            other = e.source if e.target == n else e.target
            if solution_graph.degree()[other] >= int(max_node_degrees[other]):
                continue
            if solution_graph.are_connected(n, other):
                continue
            spare_degrees.append((n, other, e))
            print(
                f"Spare connection from {n}: {solution_degree} < {degree_constraint} to {other}: {solution_graph.degree()[other]} < {int(max_node_degrees[other])} through edge {e.index}"
            )
            all_ok = False

    return all_ok


def triangle_density(gr: ig.Graph) -> float:
    def count_triangles(gr: ig.Graph) -> np.ndarray:
        triangles = gr.list_triangles()
        counts = np.zeros(gr.vcount(), dtype=int)
        for triangle in triangles:
            for vertex in triangle:
                counts[vertex] += 1
        return counts

    return sum(count_triangles(gr)) / (
        gr.vcount() * (gr.vcount() - 1) * (gr.vcount() - 2) / 6
    )


def evaluation(actual_graph: ig.Graph, predicted_graph: ig.Graph) -> dict:
    # evaluate the performance of the prdiction by comparing a gredicted graph to an actual graph

    # Prediction stats for pred
    vcount_err = abs(predicted_graph.vcount() / actual_graph.vcount() - 1)
    ecount_err = abs(predicted_graph.ecount() / actual_graph.ecount() - 1)

    def edge_density(gr: ig.Graph) -> float:
        return gr.ecount() / (gr.vcount**2)

    edge_density_err = abs(predicted_graph.density() - actual_graph.density())
    triangle_density_err = abs(
        triangle_density(predicted_graph) - triangle_density(actual_graph)
    )
    max_degree_err = abs(max(predicted_graph.degree()) / max(actual_graph.degree()) - 1)

    d = dict(
        vcount_err=vcount_err,
        ecount_err=ecount_err,
        edge_density_err=edge_density_err,
        triangle_density_err=triangle_density_err,
        max_degree_err=max_degree_err,
    )
    return d


def arrange_matrix(mat: np.ndarray) -> np.ndarray:
    rowsums = np.sum(mat, axis=1)
    order_row = np.flip(np.argsort(rowsums))
    mat2 = mat[order_row, :]

    colsums = np.sum(mat, axis=0)
    order_col = np.flip(np.argsort(colsums))
    mat3 = mat2[:, order_col]

    return mat3


def diff_metrics(act_adj: np.ndarray, pred_adj: np.ndarray) -> dict:
    # positives to be denoted by 1 and negatives with 0
    if act_adj.shape != pred_adj.shape:
        raise ValueError(
            "The actual and predicted adjacency lists must have the same shapes"
        )
    n = act_adj.size
    tp = np.sum((act_adj == 1) & (pred_adj == 1))
    tn = np.sum((act_adj == 0) & (pred_adj == 0))
    fp = np.sum((act_adj == 0) & (pred_adj == 1))
    fn = np.sum((act_adj == 1) & (pred_adj == 0))
    prec = (tp + tn) / n
    sn = tp / (tp + fn)
    sp = tn / (tn + fp)
    precision = (tp / (tp + fp)) if ((tp + fp) != 0) else 0
    recall = tp / (tp + fn)
    fmeasure = (
        (2 * precision * recall / (precision + recall))
        if ((precision + recall) != 0)
        else 0
    )
    return dict(
        N=n,
        true_pos=tp,
        true_neg=tn,
        false_pos=fp,
        false_neg=fn,
        accuracy=prec,
        sensitivity=sn,
        specificity=sp,
        gmean=math.sqrt(sn * sp),
        precision=precision,
        recall=recall,
        fmeasure=fmeasure,
    )


def eval_metrics2(actual_graph: ig.Graph, predicted_graph: ig.Graph) -> dict:
    new_vert = None
    node_error = (
        predicted_graph.vcount() - actual_graph.vcount()
    ) / actual_graph.vcount()

    if node_error > 0:
        new_vert = predicted_graph.vcount() - actual_graph.vcount()
        actual_graph.add_vertices(new_vert)
    elif node_error < 0:
        new_vert = actual_graph.vcount() - predicted_graph.vcount()
        predicted_graph.add_vertices(new_vert)

    adj_act = np.array(actual_graph.get_adjacency())
    adj_pred = np.array(predicted_graph.get_adjacency())

    def norm(array: np.ndarray) -> float:
        ord = "fro" if array.ndim > 1 else 2
        return npla.norm(array, ord=ord)

    # arrange rows by row sums and columns by column sums,
    adj_act2 = arrange_matrix(adj_act)
    adj_pred2 = arrange_matrix(adj_pred)
    adjacency_error = norm(adj_pred2 - adj_act2) / norm(adj_act2)

    deg_act = np.array(actual_graph.degree())
    deg_prd = np.array(predicted_graph.degree())
    deg_cosine_sim = sum(deg_act * deg_prd) / (norm(deg_act) * norm(deg_prd))

    lap_act = actual_graph.laplacian()
    lap_prd = predicted_graph.laplacian()

    eig_act = npla.eigvals(lap_act)
    eig_prd = npla.eigvals(lap_prd)
    eig_cosine_sim = sum(eig_act * eig_prd) / (norm(eig_act) * norm(eig_prd))

    evaluation_metrics = dict(
        node_error=node_error,
        adjacency_error=adjacency_error,
        degree_cosine=deg_cosine_sim,
        lap_eigen_cosine=eig_cosine_sim,
    )

    difference_metric = diff_metrics(adj_act, adj_pred)
    evaluation_metrics.update(difference_metric)

    return evaluation_metrics
