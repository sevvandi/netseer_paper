# Netseer

_Netseer_ is a software package for predicting new graphs from a given time series of graphs.

The underlying prediction algorithm is comprised of time series modelling
combined with an adapted form of Flux Balance Analysis, 
an approach widely used in biochemistry for reconstructing metabolic networks from partial information.
A comprehensive description of the algorithm is given in:
* Sevvandi et al.
  Predicting Graph Structure via Adapted Flux Balance Analysis.
  Lecture Notes in Computer Science (LNCS), Vol. 16370, 2026.
  [DOI: 10.1007/978-981-95-4969-6_27](https://doi.org/10.1007/978-981-95-4969-6_27)

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

To get started with `netseer`, there are 3 ways of loading graphs for processing.

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

The `weights_opt` parameter changes the method used to predict the graph. A `weights_opt` of 5 causes older edge weights from earlier graphs to have less weight in the prediction.  

``` r
% Replace 'syngraphs' with 'graph_list' if using Options 2 and 3.
predicted_graph <- predict_graph(
                                syngraphs[1:19], 
                                h=1,
                                weight_opt = 5)
```

Now use `measure_error` to compare the 20th Actual graph with the Predicted 20th graph by generating metrics. The metrics are vertex_error and edge_error, which shows how close the number of vertices and edges are between graphs. The closer to zero the better performing the prediction was.  

```r
% Replace 'syngraphs' with 'graph_list' if using Options 2 and 3.
node_error, edge_error <- measure_error(syngraphs[[20]], predicted_graph)

```

