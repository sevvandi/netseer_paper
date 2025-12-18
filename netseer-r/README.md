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

The `netseer` package is available for installation via CRAN:

``` r
install_packages("netseer")
```

Alternatively, `netseer` can be built from source from GitHub:  

``` r
library(remotes)
remotes::install_github("sevvandi/netseer_paper/netseer-r")
```

## Available Functions

| Function                  | Summary                                                         |
| ---                       | ---                                                             |
| `read_graph_list()`       | Load user provided graphs.                                        |
| `predict_graph()`         | Predict the next graph in a sequence.                            |
| `measure_error()`         | Return the vertex error and edge error between two graphs.       |
| `generate_graph_linear()` | Generate a time series of random graphs that grow linearly.      |
| `generate_graph_exp()`    | Generate a time series of random graphs that grow exponentially. |

Documentation for the above functions is available in the [Documentation PDF](./docs/netseer.pdf)

## Example

Goal:

* Load 20 graphs from the file system.
* Use graphs 1 to 19 to predict the 20th graph.
* Compare the actual 20th graph to the newly predicted 20th graph.

Before starting, download the `/data/` directory under `/netseer-paper/netseer-r/`. This directory contains 20 example graphs.  

``` r
library("netseer")

# Load 20 graphs from the /data/ directory.
## Replace ./data/ in file.path with a path to the /data/ directory.
path_to_graphs <- file.path("./data/")
graph_list <- netseer::read_graph_list(path_to_graphs = path_to_graphs, format = "gml")

# Predict the 20th graph using graphs 1 to 19.  
## A h value of 1 predicts 1 step into the future.
predicted_graph <- netseer::predict_graph(graph_list[1:19], weights_opt = 8, h=1)

# Compare the 20th actual graph and the predicted 20th graph by checking the vertex and edge error.
vertex_err, edge_err <- netseer::measure_error(graph_list[[20]], predicted_graph[[1]])
print(vertex_err)
print(edge_err)
```
