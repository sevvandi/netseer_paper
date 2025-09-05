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
  - name: Author Without ORCID
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
  - name: Author with no affiliation
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 3
affiliations:
 - name: Data61, CSIRO, Australia
   index: 1
 - name: Independent Researcher, Country
   index: 2
date: 14 August 2025
bibliography: paper.bib
---

# Summary

Dynamic processes such as transport, electricity, telecommunication, and social networks can be represented through an ordered sequence of graphs. Such sequences, also known as dynamic graphs, describe how aspects of graphs evolve over time, such as the addition and deletion of vertices (nodes) and edges (links between nodes) [11]. Modelling the observed dynamics in such sequences can be used for predicting graphs at future time steps. This in turn can facilitate various applications, such as the detection of anomalies (differences between predicted and observed graphs), thereby allowing active response to events of interest (eg., network overloads, cyber attacks, car accidents)

Complex systems, such as transport, electricity, telecommunications, and social networks, are dynamic and ever growing. These systems can be represented as an ordered sequence of graphs, also known as Dynamic graphs. Dynamic graphs often describe how different aspects of graphs evolve over time, such as via the addition or deletion of vertices (nodes), or edges (links between nodes). **--Reference to 11 in paper--** Modelling the changes in dynamic graphs is useful, as it allows for predicting the next graph in the ordered sequence of graphs. This is particularly useful in anomaly detection, as it highlights events of interest such as network overloads, cyber attacks, and car accidents in their respective system.

**--Small description of netseer and FBA**
Our proposed software, `netseer`, combines time-series forecasting with Flux Balance Analysis (FBA)**--Cite 20,22--** to predict graph structures. Typically, in time-series forecasting the network is assumed to be fixed and known which is inflexible when dealing with dynamic graphs. `netseer` uses FBA, a mathematical approach used widely in biochemistry for describing networks of chemical reactions. We have adapted FBA towards graph prediction **--Cite original paper?--**, which allows for graph prediction involving changes in the number of vertices and edges, something that to our knowledge has not been studied before.

Common contemporary approaches, such as **X, Y, Z**, typically use more resource intensive techniques like predictive/generative AI to achieve a predicted graph from a sequence. `netseer` however is more resource efficient as it utilises a **--graph and constraint based methodology using FBA???**. Additionally,

# Statement of need

**What goes into the statement of need:  
    - Primarily, the purpose of the software.  
    - Q. Why is Netseer needed?  
    - A. Dynamic systems have important roles in life: E.g. Transport.  
    - Q. Who is the intended audience?  
    - A. Different fields that want to model graph growth.  
    - A2. Telecommunications, Transport, Maybe Banking  
    - Q. Are there products that would benefit from this?**  

`netseer` is an R and Python-based package that uses an adapted form of Flux Balance Analysis to predict graph structures from time-series graphs. It is published on both CRAN[@netseerR] and PYPI [@netseerPy] under the `netseer` package.  

As `netseer` is relevant to professions that occupy differing programming language ecosystems, `netseer` has both an R and Python implementations that act in parity. The underlying logic of the `netseer` packages is available in a publication [@kandanaarachchi2025predictinggraphstructureadapted:2025].  

# Usage

`netseer` processes suitable discrete time-series graphs that are in an ordered sequence (e.g. by time) and grow over time **--Double check that growing is needed. I believe so**.  

**--Short description here of what are some alternatives to netseer. I assume we would talk about how FBA lets us use changing weights in vertices compared to standard approaches.**  

**--Are there any research projects using netseer.**  

![A time-series graph growing, with a 1 step prediction by netseer.\label{fig:graph_grow}](assets/netseer.svg)

# Examples

Predicting a graph 1 step into the future.

``` Python
# Load the graphs. 
graph_list = utils.read_graph_list(path_to_graphs)

# Generate the predicted next graph
predicted_graph = predict.predict)graph(graph_list, h=1)
```

`path_to_graphs` is a list of paths to be loaded.
Increase `h` to predict more steps into the future.

# Acknowledgements

# References

**--Note: The cite for netseer is using the arxiv version.**
