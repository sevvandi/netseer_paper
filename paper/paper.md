---
title: 'Netseer: A Python and R package for Predicting Graph Structure via Adapted Flux Balance Analysis'
tags:
  - Python
  - R
authors:
  - name: Brodie Oldfield
    orcid: 0009-0000-4500-5006
    equal-contrib: true
    affiliation: 1
  - name: Auth 2
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 1
  - name: Author with no affiliation
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: [1,2]
affiliations:
 - name: Data61, CSIRO, Australia
   index: 1
 - name: Secondary Affil if needed.
   index: 2
date: 17 November 2025
bibliography: paper.bib
---

# Notes

The notes section isn't part of the paper. Just here as style guide for comments etc.  
I'm writing comments like:
**--BO: Comment goes here--**  > I'm using this for general comments/structure  
[//]: # (BO: This is the comment message)   > I'm using this for thoughts/dot points  
Typically, the 2nd comment type should be hidden in HTML rendered markdown but is it still shown in the pdf.
Where BO are my initials. Remember for markdown that 2 spaces at the end of a line = new line.  

[JOSS Guidelines](https://joss.readthedocs.io/en/latest/paper.html):  

- Length: 250-1000 words.

Timeline:

- Alpha > Sep 17. Then everyone iterates over paper.
- Beta > Oct 17. Futher iteration.
- Release Ready > Oct 31. For prep for JOSS release.
- Submit when AJCAI is ready. Could be Nov or Dec.

# Summary

**--BO: Summary intro version 1 --**  
Dynamic processes such as transport, electricity, telecommunication, and social networks can be represented through an ordered sequence of graphs. Such sequences, also known as dynamic graphs, describe how aspects of graphs evolve over time, such as the addition and deletion of vertices (nodes) and edges (links between nodes) [@repLearning]. Modelling the observed dynamics in such sequences can be used for predicting graphs at future time steps. This in turn can facilitate various applications, such as the detection of anomalies (differences between predicted and observed graphs), thereby allowing active response to events of interest (eg. network overloads, cyber attacks, car accidents)

**--BO: Summary intro version 2 --**  
Complex systems, such as transport, electricity, telecommunications, and social networks, are dynamic and ever growing. These systems can be represented as an ordered sequence of graphs, also known as Dynamic graphs. Dynamic graphs often describe how different aspects of graphs evolve over time, such as via the addition or deletion of vertices (nodes), or edges (links between nodes) [@repLearning]. Modelling the changes in dynamic graphs is useful, as it allows for predicting the next graph in the ordered sequence of graphs. This is particularly useful in anomaly detection, as it highlights events of interest such as network overloads, cyber attacks, and car accidents in their respective system.

**--BO: Small description of netseer and FBA--**  
[//]: # (BO: Ordering wise, we could change the order.)  
[//]: # (BO: Currently it's - Introduce Netseer > then issues with normal forecasting > FBA)  
[//]: # (BO: Instead could - Issues > Introduce Netseer > FBA.)  
Our proposed software, `netseer`, combines time-series forecasting with Flux Balance Analysis (FBA) [@whatIsFlux; @patternsAndDynamics] to predict graph structures. Typically, in time-series forecasting the network is assumed to be fixed and known which is inflexible when dealing with dynamic graphs. `netseer` uses FBA, a mathematical approach used widely in biochemistry for describing networks of chemical reactions. We have adapted FBA towards graph prediction[@predictingGraphStruc], which allows for graph prediction involving changes in the number of vertices and edges, something that to our knowledge has not been studied before.  

# Statement of need

**-- Primarily, the purpose of the software.--**  
**-- Where does netseer fit in against related work.--**  
The purpose of `netseer` is to provide a novel yet low resource method of predicting graph structures for fields where modelling the future state of dynamic graphs is important, such as traffic forecasting where available routes and traffic load are constantly changing at different time intervals. With the intent of being as impactful as possible in different fields, `netseer` has been implemented as both a Python package and an R package for flexible adoption.

**--BO:We're comparing to DAMNETS and AGE I believe--**  
**--BO: I'll probably be working on this part for a bit--**  
**--Links to other papers: [DAMNETS](https://arxiv.org/abs/2203.15009), [AGE](https://dl.acm.org/doi/10.1007/978-3-030-47426-3_34), [Meta Study with both](https://dl.acm.org/doi/10.1145/3642970.3655829)--**  
**-AGE Seems to require access. --**  
`netseer` aims to be resource efficient, as it utilises a graph and constraint based methodology using FBA. Other contemporary approaches, such as DAMNETS[@damnets] and AGE[@age] typically use more resource intensive techniques like predictive/generative AI to achieve a predicted graph from a sequence, where they typically use training data and computationally expensive neural networks. As `netseer` doesn't require the use of training data or generative machine learning processes, it achieves the status of a resource efficient alternative.

# Usage

`netseer` has both a Python and R implementation as packages that use an adapted form of Flux Balance Analysis to predict graph structures from an ordered time-series list of graphs. It is published on both CRAN[@netseerR] and PYPI [@netseerPy] under the `netseer` package. `netseer` operates on a load then predict methodology, where iGraph compatible graphs are loaded into memory as an ordered list, then the graph list is used for predictions. Both the Python and R implementations have methods for generating dummy data, and the Python implementation has helper functions for loading graphs from local directories. The dummy data can be generated with different constraints, such as exponential growth between time-series steps, or linear growth.

![A time-series graph growing, with a 1 step prediction by netseer.\label{fig:graph_grow}](assets/netseer.svg)

# Examples

Loading a set of graphs from GML files, then predicting a one step into the future.

## Python Implementation

``` Python
# Load the graphs. 
graph_list = utils.read_graph_list(path_to_graphs)

# Generate the predicted next graph
predicted_graph = predict.predict_graph(graph_list, h=1)
```

`path_to_graphs` is a list of paths to be loaded into memory.  
Increase `h` to predict more steps into the future.

## R Implementation

``` R
%% Using the iGraph package, load the time-series graphs into a list.
graphlist = list()
file_paths = list.files(pattern = "^*.gml$")
for (file in file_paths) {
  tempGraph <- read_graph(file, format ="gml")
  graphlist <- append(graphlist, tempGraph)
}

%% Generate the predicted next graph
grpred <- predict_graph(graphlist[1:length(graphlist)],h = 1) 
```

## Graph Examples

From generating and processing the graphs using the python implementation:

|graph_num|node_count|edge_count|
|---------|----------|----------|
|1        |5         |4         |
|2        |6         |5         |
|3        |7         |6         |
|4        |8         |6         |
|5        |9         |6         |
|6        |10        |7         |
|7        |11        |8         |
|8        |13        |9         |
|9        |15        |10        |
|10        |17        |11        |
|11       |19        |13        |
|12       |21        |15        |
|13       |24        |17        |
|14       |27        |19        |
|15       |30        |21        |
|pred|33        |24        |

*--BO: In paper/assets I've included the 1-15 graphs.*

# Acknowledgements

**--BO: We would add any financial support here, if applicable.--**  
**--BO: Or any general Acknowledgements--**  
This software was financially supported by X with the intent of improving methods for Y.  

# References

**--BO: The cite for netseer is using the arxiv version. Update to full when published circa Nov-Dec--**
