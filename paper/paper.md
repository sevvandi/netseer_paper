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
iGraph compatible graphs are loaded into memory as an ordered list, representing a time series of graphs.
The given graph list is then used for generating predicted graphs at future time steps in the time series.
Internally, the graph structure is predicted in two steps.
In the first step,
standard time series methods are used to model and predict the evolution of vertex degrees (TODO: add brief explanation of what is a vertex degree).
The degree predictions include the degrees of new, unseen vertices.
In the second step,
the predicted degrees, which correspond to edges, are allocated to the vertices using FBA.

There are various options, including the time step of the prediction and selection of weight methods,
which are used to emphasise or de-emphasise certain edges.
For example, older edges can be assigned a lower weight and hence reduce their influence during prediction.
An explanation of the weight options is given in the associated documentation **TODO: rephrase** 


<!-- The weight methods affect how edge weights are calculated. -->

Both the Python and R implementations have methods for loading user supplied data from the filesystem and generating synthetic graphs.
The dummy data can be generated with various constraints,
such as exponential growth between time-series steps, or linear growth.
(TODO: HUH??? where did this come from?)  
TODO: Brodie: In R there is 2 generator functions. generate_graph_linear() and generate_graph_exp(). I think in Python only one (Linear) was implemented due to time constraints.  

![Example of a time series with growing graphs, followed by a 1 step prediction by Netseer.\label{fig:graph_grow}](assets/graphs_1_to_15.pdf)


# Example in R

The examples detail the steps in both R and Python to generate a random time series graph list,
predict the next graph by one step,
and then compare the predicted graph to the actual graph.

TODO: Change this to loading real graphs instead of generating random graphs.  Real graphs demonstrate practical usage, while random graphs are only useful for writing technical papers.


**TODO:** need to show a self-contained program

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

**TODO:** need to show a self-contained program

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

## Graph Examples

The provided sample dataset consists of 15 graphs in a time series.

**TODO:** the tables are too much for a JOSS paper.

The Table 1\ref{table:Step-Error} is created by loading the first 12 graphs into memory,
then predicting the 13th, 14th and 15th using steps 1, 2, and 3.
The Edge Error is defined as the absolute error ratio of the number of edges in the predicted graph compared to the actual graph; where: (Predicted Edge Count - Actual Edge Count) / Actual Edge Count
The Vertex Error is defined as the absolute error ratio of the number of Vertex in the predicted graph compared to the actual graph; where: (Predicted Vertex Count - Actual Vertex Count) / Actual Vertex Count

|Step|Edge Error|Vertex Error|
|---------|----------:|----------:|
|1        |5.88x10-3         |-4.16x10-3         |
|2        |15.78x10-3         |-14.81x10-3         |
|3        |4.16x10-3         |-30.30x10-3         |

:Table 1. Edge and Vertex Errors per Predicted Step.\label{table:Step-Error}

Testing on the Facebook Dataset [@facebook] can give a larger example, Table 2\ref{table:Step-Compare}[@netseer] compares the Edge and Vertex Error of FBA against the Last Seen graph. The Last Seen graph being the real graph before the prediction. To account for randomisation when generating predicted graphs, each step is re-calculated 10 times and averaged.  
In addition, the reduction percentage between the Last Seen method results and the Netseer FBA adaption method results is shown.  
**--BO: Check how many times the data is recalculated**

|Step|Method|Edge Error|Vertex Error|
|--|--|--:|--:|
|1|Last Seen|50.48×10-3|33.45×10-3|
| |Netseer|13.04×10-3|9.31×10-3|
| |(Reduction)| 74.2% | 72.2%|
|2|Last Seen|95.28×10-3|64.58×10-3|
| |Netseer|23.53×10-3|17.89×10-3|
| |(Reduction)| 75.3% | 72.3%|
|3|Last Seen|137.41×10-3|94.41×10-3|
| |Netseer|30.25×10-3|24.03×10-3|
| |(Reduction)| 78.0% | 74.5%|
|4|Last Seen|179.25×10-3|124.70×10-3|
| |Netseer|33.62×10-3|32.11×10-3|
| |(Reduction)| 81.2% | 74.3%|
|5|Last Seen|226.59×10-3|163.19×10-3|
| |Netseer|28.82×10-3|34.42×10-3|
| |(Reduction)| 87.3% | 78.9%|
:Table 2. Edge and Vertex Errors comparison between Last Seen and Netseer's FBA Adaption.\label{table:Step-Compare}

# Licensing and Availability

The _Netseer_ package is licensed under the **TODO** license,
with the source code available at GitHub (<https://github.com/sevvandi/netseer>).
We are open to feature requests and bug reports, as well as questions and concerns.

# Acknowledgements

This work has been supported in part by the Australian Research Council (ARC) Industrial Transformation Training Centre
in Optimisation Technologies, Integrated Methodologies, and Applications (OPTIMA), Project ID IC200100009.

# References
