# Netseer

## Predicting graph structure from a time series of graphs

This is a Python implementation of [*netseer*](https://arxiv.org/abs/2507.05806).  
Netseer is a tool that outputs a predicted graph based on a time series graph sequence

## Purpose

The goal of netseer is to predict the graph structure including new nodes and edges from a time series of graphs.  
The methodology is explained in the preprint (Kandanaarachchi et al. 2025).

![Image of a time-series list of graphs and a predicted graph.](img/netseer.svg "netseer")

## Authors

Stefan Westerlund: Created netseer code.  
Brodie Oldfield: Packaging and docs.

## Installation

This package is available on PyPI, and can be installed with PIP or with a Package Manager:

``` Bash
pip install netseer # or uv add netseer
```

## Quick Example

Generating an example graph list:

``` Python
import netseer as ns

graph_list = ns.generate_graph_linear(num_iters = 15)
```

The `generate_graph_list()` function has parameters for templating what types of graphs to generate. Information about these can be found in the reference docs.  
The `num_iters` parameter sets how many graphs to generate.  

Predicting on that graph:

``` Python

predict = ns.predict_graph(graph_list, h=1)
```

Increasing the `h` parameter increases how many steps into the future the prediction is, with `h=1` being 1 step in the graph sequence.

After creating a new predicted graph, you can compare it to the original graph list.  
As the original graph list has 15 graphs, we can compare the 15th original graph to the predicted 16th graph.

``` Python
vertex_error, edge_error = ns.measure_error(graph_list[14], predict)
print(vertex_error)
print(edge_error)
```
