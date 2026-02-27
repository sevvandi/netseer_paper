---
title: 'Netseer: An R Package for Predicting Dynamic Graphs via Adapted Flux Balance Analysis'
tags:
  - R
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
date: xx March 2026
bibliography: paper.bib
---


# Summary

_Netseer_ is an open-source package for R that models a temporal sequence of graphs and predicts new graphs at future time steps.
The underlying algorithm is comprised of time series modelling combined with an adapted form of Flux Balance Analysis (FBA),
a technique originating from biochemistry used for reconstructing metabolic networks from partial information [@Orth_2010].
_Netseer_ is able to predict both vertices and edges while having low computational intensity and data requirements.


# Statement of need

Many dynamic processes such as transport, electricity, telecommunication, and social networks
can be represented through an ordered sequence of graphs, known as dynamic graphs.
The graphs evolve over time through the addition and deletion of vertices (nodes) and/or edges (links between nodes) [@Kazemi_2020].
Modelling the dynamics can be used for predicting graphs at future time steps,
which in turn facilitates applications such as the detection of anomalous graphs via differences between observed and predicted graphs.
The anomalous graphs may represent events of interest, including network overloads, cyber attacks, and car accidents.

Current approaches to graph prediction have notable limitations
such as assuming that vertices do not to change between consecutive graphs and that only the edges change [@Kumar_2020],
or employ computationally expensive deep generative models that may require large amounts of training data [@Clarkson_2022] [@Fan_2020].
In many practical situations access to high-performance computational resources can be limited
and large amounts of training data may be infeasible to obtain.

The `Netseer` package aims to be both computationally efficient and have low training data requirements,
allowing execution on standard desktop computers.
This is achieved via exploiting time series modelling in conjunction with an adapted form of FBA [@Orth_2010] [@Sahu_2021].
A comprehensive description of the prediction algorithm is given in [@Kandan_2026].

<!--
Existing approaches related to graph prediction have notable shortcomings,
including limitations in processing of vertices and edges,
requiring large amounts of training data,
or being computationally expensive.
-->


# Functionality

_Netseer_ is provided as a package for R, available on CRAN and GitHub.
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

![Example of a time series with growing graphs. Graphs 1 to 14 are used for learning the dynamics, followed by predicting graph 15.  TODO: remove "1 step" from the rightmost panel. \label{fig:graph_grow}](assets/graphs_1_to_15.pdf)


# Example Usage

The following example code shows how _Netseer_ can be used in R.
The code uses 20 example graphs provided online as [example_graphs.zip](https://github.com/sevvandi/netseer_paper/blob/main/netseer-r/example_graphs.zip).

## TODO: discuss possibly embedding the example_graphs directory as part of the package

``` R
library("netseer")

# load 20 graphs from the example_graphs directory
path_to_graphs <- file.path("./example_graphs/")
graph_list <- netseer::read_graph_list(path_to_graphs = path_to_graphs, format = "gml")

# predict graph 20 using graphs 1 to 19
# h=1 means predict 1 step into the future
# weights_opt=7 sets the edge weight of the last seen graph as 1, otherwise 0.
predicted_graph <- netseer::predict_graph(graph_list[1:19], weights_opt=7, h=1)

# compare real graph 20 and predicted graph 20 by measuring vertex and edge errors
output <- netseer::measure_error(graph_list[[20]], predicted_graph[[1]])
print(output$vertex_err)
print(output$edge_err)
```

# Licensing and Availability

The _Netseer_ package is licensed under the **TODO** license,
with the source code available at GitHub (<https://github.com/sevvandi/netseer>).
We are open to feature requests and bug reports, as well as questions and concerns.

# Acknowledgements

This work has been supported in part by the Australian Research Council (ARC) Industrial Transformation Training Centre
in Optimisation Technologies, Integrated Methodologies, and Applications (OPTIMA), Project ID IC200100009.

# References
