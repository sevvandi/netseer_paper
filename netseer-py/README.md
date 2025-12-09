**TODO:** after redoing netseer-r/README.md, redo netseer-py/README.md in the same fashion


# Netseer

_Netseer_ is a software package for predicting new graphs from a given time series of graphs.

The underlying prediction algorithm combines time series modelling
with an adapted form of Flux Balance Analysis, 
an approach widely used in biochemistry for reconstructing metabolic networks from partial information.
A comprehensive description of the algorithm is given in:
* Predicting Graph Structure via Adapted Flux Balance Analysis.  
  Lecture Notes in Computer Science (LNCS), Vol.&nbsp;16370, 2026.  
  DOI: [10.1007/978-981-95-4969-6_27](https://doi.org/10.1007/978-981-95-4969-6_27); arXiv: [2507.05806](https://arxiv.org/abs/2507.05806)


## Installation - TODO

This package is available for installation on PyPI:

``` Bash
pip install netseer
```

**TODO:** show how to install from GitHub (this repo)


## Available Functions

* `read_graph_list()`  - load user provded graphs
* `predict_graph()` - predict the next graph in a sequence
* `measure_error()`  - return the vertex error and edge error between two graphs
* `generate_graph_linear()`  - generate a time series of random graphs that grow linearly
* `generate_graph_exp()` - generate a time series of random graphs that grow exponentially


## Example - TODO

**TODO:** reduce example to only the following: (i) load 20 pre-generated graphs, (ii) use first 19 to predict the 20th graph, (iii) measure error between real and predicted 20th graph

**TODO:** the pre-generated graphs should be in a standard format, preferably not specific to R.  a format that can also be opened in Python or other tools.


Generating an example graph list:

The `generate_graph_list()` function has parameters for templating what types of graphs to generate.  

``` Python
import netseer as ns

# generate 20 graphs.
graph_list = ns.generate_graph_linear(num_iters = 20)
```

Predicting on that graph:

For predicting using the `predict_graph()` function, increasing the `h` parameter increases how many steps into the future the prediction is, with `h=1` being 1 step in the graph sequence. For the below example, we're using the first 19th graphs to predict the 20th.  

The `weights_option` parameter changes the method used to predict the graph. A `weights_option` of 5 causes older edge weights from earlier graphs to have less weight in the prediction.  

``` Python
# Using the first 19 graphs, predict the 20th.
predicted_graph = ns.predict_graph(
                                graph_list[0:18], 
                                h=1, 
                                weights_option = 5)
```

Now use `measure_error` to compare the 20th Actual graph with the Predicted 20th graph by generating metrics. The metrics are vertex_error and edge_error, which shows how close the number of vertices and edges are between graphs. The closer to zero the better performing the prediction was.  

``` Python
vertex_error, edge_error = ns.measure_error(graph_list[19], predicted_graph)
```

