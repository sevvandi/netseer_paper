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
# TODO: do we list Ziqi Xu as well?  ORCID: 0000-0003-1748-5801
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
date: xx November 2025
bibliography: paper.bib
---

# Summary

_Netseer_ is an open-source package for both R and Python that models a temporal sequence of graphs and predicts graph structures at future time steps.
The underlying algorithm is a combination of time series modelling combined with an adapted form of Flux Balance Analysis (FBA),
a technique originating from biochemistry used for reconstructing metabolic networks from partial information [@whatIsFlux; @patternsAndDynamics].
_Netseer_ is able to predict both vertices and edges in the context of growing graphs
while having low computational intensity and data requirements.


# Statement of need

Many dynamic processes such as transport, electricity, telecommunication, and social networks
can be represented through an ordered sequence of graphs, known as dynamic graphs.
The graphs evolve over time through the addition and deletion of vertices (nodes) and/or edges (links between nodes) [@repLearning].
Modelling the dynamics can be used for predicting graphs at future time steps,
which in turn facilitates applications such as the detection of anomalous graphs via differences between observed and predicted graphs.
The anomalous graphs may represent events of interest, including network overloads, cyber attacks, and car accidents.

Existing approaches related to graph prediction have notable shortcomings,
including limitations in processing of vertices and edges,
requiring large amounts of training data,
or being computationally expensive.
In the task of _link prediction_ (predicting the presence of links between vertices),
it is assumed that vertices are assumed not to change between consecutive graphs (TODO: ref).
In the task of _network time series prediction_,
attributes of vertices are predicted while the structure of the network is assumed to be fixed and known (TODO: ref).
Contemporary machine learning approaches such as DAMNETS [@damnets] and AGE [@age] 
employ computationally intensive pipelines and require large amounts of training data which may not always be available.

**TODO:** sort out the refs.
[DAMNETS](https://arxiv.org/abs/2203.15009),
[AGE](https://dl.acm.org/doi/10.1007/978-3-030-47426-3_34),
[Meta Study with both](https://dl.acm.org/doi/10.1145/3642970.3655829)--**  



# Implementation

`Netseer` aims to be both computationally efficient and have low training data requirements.
This is achieved via exploiting standard time series modelling in conjunction with an adapted form of FBA.
FBA is a mathematical approach used widely in biochemistry for describing networks of chemical reactions.

Netseer predicts the graph structure in two steps.
First, the vetex degrees at a future time step are predicted using standard time series methods.
The degree forecasts include the degrees of new, unseen vertices.
Then the predicted degrees, which correspond to edges are allocated to the vertices using FBA.
The technical details of the underlying algorithm are given in[@predictingGraphStruc]

# Usage

`Netseer` is provided as R and Python packages, available on CRAN [@netseerR] and PyPI [@netseerPy], respectively.
iGraph compatible graphs are loaded into memory as an ordered list,
then the graph list is used for predictions.
Both the Python and R implementations have methods for generating dummy data,
and the Python implementation has helper functions for loading graphs from local directories.
The dummy data can be generated with various constraints,
such as exponential growth between time-series steps, or linear growth.
(TODO: HUH??? where did this come from?)

<!-- old figure commented out, as it's a mess -->
<!-- ![A time-series graph growing, with a 1 step prediction by netseer.\label{fig:graph_grow}](assets/netseer.svg) -->

TODO: insert figure showing a sequence of real graphs (graph 1, 5, 10, 15), and predicted graph 15

TODO: use plain PDF for the figure; do not use SVG as that slows down compilation and requires conversion

# Examples

Loading a set of graphs from GML files, then predicting a one step into the future.

## R

**TODO:** need to show a self-contained program

``` R
%% using the iGraph package, load the time-series graphs into a list
graphlist = list()
file_paths = list.files(pattern = "^*.gml$")
for (file in file_paths) {
  tempGraph <- read_graph(file, format ="gml")
  graphlist <- append(graphlist, tempGraph)
}

%% predict graph one step ahead
grpred_1 <- predict_graph(graphlist[1:length(graphlist)],h = 1) 

%% predict graph five steps ahead
grpred_2 <- predict_graph(graphlist[1:length(graphlist)],h = 5) 
```

## Python

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
