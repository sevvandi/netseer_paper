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

**TODO:** show how to install from GitHub

## Available Functions

- `read_graph_list()`  
  Load into memory user defined graphs.  
- `predict_graph()`  
  Predict the next graph in a sequence.  
- `measure_error()`  
  Returns the vertex error and edge error of two graphs.  
- `generate_graph_linear()`  
  Randomly generate a set of time series graphs that grow linearly.  
- `generate_graph_exp()`  
  Randomly generate a set of time series graphs that grow exponentially.  

**TODO:** Full documentation for the above functions is available in [netseer.pdf](TODO:netseer.pdf) (TODO: update link to netseer.pdf)

## Examples

**TODO:** reduce example to only the following: (i) load 20 pre-generated graphs, (ii) use first 19 to predict the 20th graph, (iii) measure error between real and predicted 20th graph

**TODO:** the pre-generated graphs should be in a standard format, preferably not specific to R.  a format that can also be opened in Python or other tools.

Example Goal:

- Load 20 graphs from the file system.
- Use graphs 1 to 19 to predict the 20th graph.
- Compare the actual 20th graph to the newly predicted 20th graph.

Before starting, download the `/data/` directory under `/netseer-paper/netseer-r/`. This directory contains 20 example graphs.  

``` r
library(netseer)

%% Create an Absolute or Relative path to the /data/ directory.
%% Replace ./data/ with the path to the /data/ directory.
path_to_graphs <- system.file("./data/", package = "netseer")
graph_list <- read_graph_list(path_to_graphs = path_to_graphs, format = "gml")

%% Predict the 20th graph using graphs 1 to 19.  
%% A h value of 1 predicts 1 step into the future.
predicted_graph <- predict_graph(
                                graph_list[1:19], 
                                h=1,
                                weight_opt = 5)

%% Compare the 20th actual graph and the predicted 20th graph by checking the vertex and edge error.
vertex_error, edge_error <- measure_error(graph_list[[20]], predicted_graph)
print(vertex_error)
print(edge_error)

```
