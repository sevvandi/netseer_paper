---
title: 'Netseer: A Package for Predicting Graph Structure via Adapted Flux Balance Analysis in R and Python'
tags:
  - R
  - Python
authors:
  - name: Brodie Oldfield
    orcid: 0009-0000-4500-5006
    affiliation: 1
  - name: Sevvandi Kandanaarachchi
    orcid: 0000-0002-0337-0395
    affiliation: 1
    corresponding: true # (This is how to denote the corresponding author)
  - name: Stefan Westerlund
    affiliation: 1
  - name: Conrad Sanderson
    orcid: 0000-0002-0049-4501
    affiliation: [1,2]
affiliations:
 - name: CSIRO, Australia
   index: 1
 - name: Griffith University, Australia
   index: 2
date: xx Feburary 2026
bibliography: paper.bib
---


# Summary

_Netseer_ is an open-source package for both R and Python that models a temporal sequence of graphs and predicts graph structures at future time steps.
The underlying algorithm is comprised of time series modelling combined with an adapted form of Flux Balance Analysis (FBA),
a technique originating from biochemistry used for reconstructing metabolic networks from partial information [@Orth_2010].
_Netseer_ is able to predict both vertices and edges in the context of growing graphs
while having low computational intensity and data requirements.


# Statement of need

Many dynamic processes such as transport, electricity, telecommunication, and social networks
can be represented through an ordered sequence of graphs, known as dynamic graphs.
The graphs evolve over time through the addition and deletion of vertices (nodes) and/or edges (links between nodes) [@repLearning].
Modelling the dynamics can be used for predicting graphs at future time steps,
which in turn facilitates applications such as the detection of anomalous graphs via differences between observed and predicted graphs.
The anomalous graphs may represent events of interest, including network overloads, cyber attacks, and car accidents.

Current approaches to graph prediction have notable limitations
such as assuming that vertices do not to change between consecutive graphs and that only the edges change [@Kumar_2020],
or employ computationally expensive deep generative models that may require large amounts of training data [@Clarkson_2022,@Fan_2020].
In many practical situations access to high-performance computational resources can be limited
and large amounts of training data may be infeasible to obtain.

The `Netseer` package aims to be both computationally efficient and have low training data requirements,
allowing execution on standard desktop computers.
This is achieved via exploiting time series modelling in conjunction with an adapted form of FBA [@Orth_2010,@Sahu_2021].
A comprehensive description of the prediction algorithm is given in [@Kandan_2026].

<!--
Existing approaches related to graph prediction have notable shortcomings,
including limitations in processing of vertices and edges,
requiring large amounts of training data,
or being computationally expensive.
-->


# Functionality

_Netseer_ is provided as two separate implementations in R and Python, available on CRAN and PyPI, respectively.
The package provides functions for loading a time series (set) of graphs,
predicting a graph from a given time series, and measuring the error between graphs.
There are also functions for generating synthetic graphs.

Graphs are predicted in two steps.
In the first step,
standard time series methods are used to model and predict the evolution of vertex degrees
(TODO: add brief explanation of what is a vertex degree).
The degree predictions include the degrees of new, unseen vertices.
In the second step,
the predicted degrees, which correspond to edges, are allocated to the vertices using FBA.
There are various options, including the time step of the prediction and selection of weight methods,
which are used to emphasise or de-emphasise certain edges.
For example, older edges can be assigned a lower weight and hence reduce their influence during prediction.
An explanation of the weight options is given in the associated documentation/

A conceptual example of graph prediction is shown in [@fig:graph_grow].

![Example of a time series with growing graphs, followed by a 1 step prediction by Netseer.\label{fig:graph_grow}](assets/graphs_1_to_15.pdf)


# Example in R


**TODO:** need to show a self-contained program that does the following:
* load pre-generated graphs that are available with the package
* let's say the number of pre-generated graphs is N
* using graphs 1 to N-1, predict graph N
* measure error between predicted graph N and real graph N
* include comments to aid interpretation of the code

``` R
graphlist <- list()
%% Create the first graph.
graphlist[[1]] <- gr <-  igraph::sample_pa(5, directed = FALSE)

%% Generate 19 graphs using the first graph. Resulting in 20 graphs total.
for(i in 2:20){
 gr <-  generate_graph_exp(gr, 
  del_edge = 0.1, 
  new_nodes = 0.1, 
  edge_increase = 0.1)
 graphlist[[i]] <- gr
}

%% Predict graph one step into the future. Excluding the 20th graph for comparison.
%% weights_opt 5 causes older edges to have less weight.
pred_1 <- predict_graph(graphlist[1:19], h = 1, weights_opt = 5)

%% Compare the predicted 20th graph with the actual 20th graph.
measure_error(graphlist[[20]], pred_1)
```

# Example in Python

**TODO:** adapt the self-contained example in R code into Python code

``` Python
# load graphs
graph_list = utils.read_graph_list(path_to_graphs)

# predict graph one step ahead
predicted_graph_1 = predict.predict_graph(graph_list, h=1)

# predict graph five steps ahead
predicted_graph_5 = predict.predict_graph(graph_list, h=5)

```

`path_to_graphs` is a list of paths to be loaded into memory.  
Increase `h` to predict more steps into the future.


# Licensing and Availability

The _Netseer_ package is licensed under the **TODO** license,
with the source code available at GitHub (<https://github.com/sevvandi/netseer>).
We are open to feature requests and bug reports, as well as questions and concerns.

# Acknowledgements

This work has been supported in part by the Australian Research Council (ARC) Industrial Transformation Training Centre
in Optimisation Technologies, Integrated Methodologies, and Applications (OPTIMA), Project ID IC200100009.

# References
