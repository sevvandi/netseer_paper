# Netseer

## Predicting graph structure from a time series of graphs

Netseer predicts the graph structure including new nodes and edges from
a time series of graphs. It adapts Flux Balance Analysis, a method used
in metabolic network reconstruction to predict the structue of future
graphs. The methodology is explained in the preprint (Kandanaarachchi et
al. 2025).

## Purpose

If you have a time series of dynamic graphs with changing structure, how
would you predict future graphs? This is the goal of netseer.

## Installation

The algorithm is available in both R and Python. The R package `netseer`
in on CRAN and can be installed as follows:

``` r
install_packages("netseer")
```

The vignette for the R package is available under \[Get Started\] at
<https://sevvandi.github.io/netseer/>

## Quick Example

To get started with Netseer, there are 3 ways of loading graphs for processing.

- Option 1: Loading pre-generated graphs as a data source.
- Option 2: Randomly generating a set number of graphs.
- Option 3: Loading user supplied graphs.

``` r
library(netseer)
% Option 1
data(syngraphs) %% Loads 20 pre-generated time series graphs.


% Option 2
%% Create the graph-list and first graph.
graphlist <- list()
graphlist[[1]] <- gr <-  igraph::sample_pa(5, directed = FALSE)

%% Generate 19 graphs using the first graph.
for(i in 2:20){
 gr <-  generate_graph_exp(gr, 
  del_edge = 0.1, 
  new_nodes = 0.1, 
  edge_increase = 0.1)
 graphlist[[i]] <- gr
}


% Option 3
%% Create a path to your graph file location.
path_to_graphs <- system.file("./path/to/graphs", package = "netseer")
graph_list <- read_graph_list(path_to_graphs = path_to_graphs, format = "gml")


```

Then, use either syngraph(Option 1), or graph_list(Option 2, 3) for predictions.  
Set `h` for how many steps into the future you want the predicted graph to be.  
In this example, we are using graphs 1 to 19 to predict 1 step into the future, therefore predicted graph 20.

``` r
% Replace 'syngraphs' with 'graph_list' if using Options 2 and 3.
predicted_graph <- predict_graph(syngraphs[1:19], h=1)
```

Generate metrics to compare graphs. The closer to zero the better.  
Here we are comparing the 20th Actual graph with the Predicted 20th.

```r
% Replace 'syngraphs' with 'graph_list' if using Options 2 and 3.
node_error, edge_error <- measure_error(syngraphs[[20]], predicted_graph)

```

## References

### network_prediction

- predict_graph()

### graph_generation

- generate_graph_linear()
- generate_graph_exp()

### read_graphs

- read_graph_list()
- read_pickled_list()

### measure_error

- measure_error()

### functions

## Citation

<div id="refs" class="references csl-bib-body hanging-indent"
entry-spacing="0">

<div id="ref-kand2025graphpred" class="csl-entry">

Kandanaarachchi, Sevvandi, Ziqi Xu, Stefan Westerlund, and Conrad
Sanderson. 2025. “Predicting Graph Structure via Adapted Flux Balance
Analysis.” <https://arxiv.org/abs/2507.05806>.

</div>

</div>
