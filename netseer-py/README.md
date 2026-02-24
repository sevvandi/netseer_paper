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

## Installation

This package is available for installation on PyPI:

``` Bash
pip install netseer
```

Alternatively, `netseer` can be built from source from GitHub:  

``` Bash
pip install "git+https://github.com/sevvandi/netseer_paper.git#subdirectory=netseer-py"
```

## Available Functions - TODO

| Function                  | Summary                                                         |
| ---                       | ---                                                             |
| `read_graph_list()`       | Load user provided graphs alphanumerically                                        |
| `predict_graph()`         | Predict the next graph in a sequence                            |
| `measure_error()`         | Return the vertex error and edge error between two graphs       |
| `generate_graph_linear()` | Generate a time series of random graphs that grow linearly      |
| `generate_graph_exp()`    | Generate a time series of random graphs that grow exponentially |

Documentation for the above functions is available in the [Documentation PDF](./documentation/netseer.pdf).

## Example

Goal:

* Load 20 graphs from the file system.
* Use graphs 1 to 19 to predict the 20th graph.
* Compare the actual 20th graph to the newly predicted 20th graph.

Before starting, download the [example_graphs.zip](./example_graphs.zip) and extract the zip to your project root.
The zip contains 20 example graphs.

Information about specific functions can be found in the [Documentation PDF](./documentation/netseer.pdf), such as what the `weights_option` parameter changes in `predict_graph()`.

**TODO:** ensure that the value for the `weights_option` is the same as the R version

---

**TODO:** ensure functionality is the same as the R version

**TODO:** ensure the errors are the same (or very close) for the same weights_option for the R and Python versions

``` Python
import netseer as ns

# Load the graphs from the file system into a list of iGraphs.
## Replace "example_graphs" with the relative path from the project root to your graphs directory.
graph_list = ns.read_graph_list(filepath = "example_graphs")

# Predict the 20th graph using graphs 1 to 19.
## h=1 means predict 1 step into the future.
## weights_opt=7 sets the edge weight of the last seen graph as 1, otherwise 0.
predicted_graph = ns.predict_graph(graph_list[0:19], h=1, weights_option = 7)

# Compare the 20th actual graph and the predicted 20th graph by checking the vertex and edge error.
vertex_error, edge_error = ns.measure_error(graph_list[19], predicted_graph)
print(f"Vertex Error: {vertex_error} |  Edge Error: {edge_error}")
```
