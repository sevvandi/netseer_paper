# Netseer

## Predicting graph structure from a time series of graphs

`netseer` predicts the graph structure including new vertices and edges from a time series of graphs.
It adapts Flux Balance Analysis, a method used in metabolic network reconstruction
to predict the structure of future graphs.
Technical details of the approach are given in **TODO:** add ref.

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

- `read_graph_list()`  
  Load into memory user defined graphs.  
- `predict_graph()`  
  Predict the next graph in a sequence.  
- `measure_error()`  
  Returns a Tuple with the vertex error and edge error of two graphs.  
- `generate_graph_linear()`  
  Randomly generate a set of time series graphs that grow linearly.  
- `generate_graph_exp()`  
  Randomly generate a set of time series graphs that grow exponentially.  

Full documentation for the above functions is available in the [Documentation PDF.](./docs/netseer.pdf)

## Examples

Example Goal:

- Load 20 graphs from the file system.
- Use graphs 1 to 19 to predict the 20th graph.
- Compare the actual 20th graph to the newly predicted 20th graph.

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
