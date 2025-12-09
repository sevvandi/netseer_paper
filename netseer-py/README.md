# Netseer

## Predicting graph structure from a time series of graphs

`netseer` predicts the graph structure including new nodes and edges from
a time series of graphs. It adapts Flux Balance Analysis, a method used
in metabolic network reconstruction to predict the structure of future
graphs.

## Installation

This package is available for installation on PyPI:

``` Bash
pip install netseer
```

## Quick Example

Note: Function descriptions are in under [Available Functions](#available-functions).  
Comprehensive Function descriptions can be found under TODO  

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

## Available Functions

- [generate_graph_linear()](./src/netseer/graph_generation.R)  
  Randomly generate a set of time series graphs that grow linearly.  
- [generate_graph_exp()](./src/netseer/graph_generation.R)  
  Randomly generate a set of time series graphs that grow exponentially.  
- [predict_graph()](./src/netseer/network_prediction.py)  
  Predict the next graph in a sequence.  
- [read_graph_list()](./src/netseer/read_graphs.R)  
  Load into memory user defined graphs.  
- [measure_error()](./src/netseer/measure_error.R)  
  Returns the vertex error and edge error of two graphs.  
